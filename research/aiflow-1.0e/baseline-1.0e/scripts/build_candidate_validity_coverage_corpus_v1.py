#!/usr/bin/env python3
"""Generate project-owned positive contexts for critical homograph tokens."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from character_tensor_v1 import ROOT, _json_lines
import train_independent_formula_context_v1 as independent
import train_masked_context_reranker_v1 as masked
from training_data_guard_v1 import (
    assert_training_entrypoint_arguments_clean,
    assert_training_path_clean,
)

if __name__ == "__main__":
    assert_training_entrypoint_arguments_clean()


SCHEMA = "aiflow-candidate-validity-coverage-corpus/v2"
AUDIT_SCHEMA = "aiflow-candidate-validity-coverage-audit/v2"
SEED = 20260820
FOCUS_TOKENS = (
    "0", "O", "o", r"\mathcal{O}", r"\circ",
    "1", "|", "l", "/", r"\mathbb{1}",
    "x", r"\times", "X", r"\mathcal{X}", r"\chi",
)
DEFAULT_HWR = independent.DEFAULT_HWR
DEFAULT_DIRECT = independent.DEFAULT_DIRECT
DEFAULT_OUTPUT = (
    ROOT / "datasets" / "00_project_owned" / "generated_context"
    / "candidate_validity_coverage_v2.jsonl.gz"
)
DEFAULT_AUDIT = DEFAULT_OUTPUT.with_name("candidate_validity_coverage_v2_audit.json")
VARIABLES = tuple("abcdefghijklmnpqrsuvwxyz")


def _d_path(path: Path, label: str, *, file: bool = True) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{label} must remain on D: {resolved}")
    if file and not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _labels(path: Path) -> list[str]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    labels = list(payload.get("math_labels", []))
    if len(labels) != 372 or len(set(labels)) != 372:
        raise ValueError("expected unified 372-class HWR checkpoint")
    return labels


def _number(value: int) -> list[str]:
    return list(str(abs(int(value))))


def _evaluation_sequences(path: Path) -> set[tuple[str, ...]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in _json_lines(path):
        grouped[str(row["formula_id"])].append(row)
    output = set()
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["context"]["index"]))
        output.add(tuple(str(row["label"]) for row in rows))
    return output


def _template(token: str, index: int) -> tuple[list[str], int, str]:
    a = VARIABLES[index % len(VARIABLES)]
    b = VARIABLES[(index * 7 + 3) % len(VARIABLES)]
    left = _number(100 + index)
    right = _number(9000 + index * 13)
    digit = str((index % 9) + 1)
    variable_like = {"O", "o", "l", "x", "X", r"\mathcal{X}", r"\chi"}
    if token in {"0", "1"}:
        tokens = left + ["+", token, "=", *right]
        return tokens, len(left) + 1, "numeric_operand"
    if token in variable_like:
        tokens = [token, "+", digit, "=", *left, "-", b]
        return tokens, 0, "variable_operand"
    if token == r"\mathcal{O}":
        tokens = [a, "(", b, ")", "=", token, "(", *left, ")"]
        return tokens, 5, "big_o_notation"
    if token == r"\circ":
        tokens = [a, token, b, "(", "x", ")", "=", *right]
        return tokens, 1, "function_composition"
    if token == "|":
        if index % 2:
            tokens = [token, a, "-", *left, token, "=", *right]
            return tokens, 0, "absolute_value_left"
        tokens = [*left, token, *right]
        return tokens, len(left), "divisibility_relation"
    if token == "/":
        tokens = [*right, token, *left, "=", digit]
        return tokens, len(right), "fraction_operator"
    if token == r"\mathbb{1}":
        threshold = _number(index)
        tokens = [token, "(", a, ">", *threshold, ")", "=", digit]
        return tokens, 0, "indicator_function"
    if token == r"\times":
        tokens = [*left, token, a, "=", *right]
        return tokens, len(left), "multiplication_operator"
    raise ValueError(f"missing coverage template for {token}")


def _write(path: Path, rows: list[dict]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for row in rows:
                    text.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_HWR)
    parser.add_argument("--direct-candidates", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--per-token", type=int, default=800)
    args = parser.parse_args()
    if args.per_token < 100:
        parser.error("per-token must be at least 100")
    hwr = _d_path(args.hwr_checkpoint, "HWR checkpoint")
    direct = _d_path(args.direct_candidates, "direct candidates")
    output = _d_path(args.output, "coverage output", file=False)
    audit_path = _d_path(args.audit, "coverage audit", file=False)
    if output.exists() or audit_path.exists():
        parser.error("refusing to overwrite coverage corpus")
    labels = _labels(hwr)
    if set(FOCUS_TOKENS) - set(labels):
        raise ValueError("coverage token outside frozen 372 classes")
    assert_training_path_clean(direct, "project-owned development candidates")
    development_sequences = _evaluation_sequences(direct)
    rows = []
    seen = set()
    excluded = 0
    templates = Counter()
    by_token = Counter()
    for token in FOCUS_TOKENS:
        candidate_index = 0
        attempt_limit = max(args.per_token * 10, args.per_token + 100)
        while by_token[token] < args.per_token and candidate_index < attempt_limit:
            tokens, focus_index, template = _template(token, candidate_index)
            candidate_index += 1
            sequence = tuple(tokens)
            if sequence in development_sequences:
                excluded += 1
                continue
            key = (sequence, focus_index)
            if key in seen:
                continue
            seen.add(key)
            digest = hashlib.sha256(
                (token + "\0" + "\x1f".join(tokens)).encode("utf-8")
            ).hexdigest()
            rows.append({
                "schema": SCHEMA,
                "formula_id": f"owned-coverage::{digest[:24]}",
                "tokens": tokens,
                "relations": ["right"] * (len(tokens) - 1),
                "focus_index": focus_index,
                "focus_token": token,
                "template": template,
                "commercial_training_rights": "project-owned generated formula",
            })
            templates[template] += 1
            by_token[token] += 1
        if by_token[token] != args.per_token:
            raise RuntimeError(
                f"coverage generation exhausted for {token}: "
                f"accepted={by_token[token]} attempts={candidate_index}"
            )
        print(json.dumps({
            "event": "coverage_token_complete",
            "token": token,
            "records": by_token[token],
            "attempts": candidate_index,
        }, ensure_ascii=False), flush=True)
    if any(row["tokens"][row["focus_index"]] != row["focus_token"] for row in rows):
        raise AssertionError("coverage focus contract mismatch")
    if len(rows) != len(FOCUS_TOKENS) * args.per_token:
        raise AssertionError("coverage count mismatch")
    rows.sort(key=lambda row: row["formula_id"])
    _write(output, rows)
    audit = {
        "schema": AUDIT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contracts": {
            "commercial_training_rights": True,
            "project_owned_generated": True,
            "raster_images": False,
            "raw_ink": False,
            "project_owned_dev_exact_sequence_overlap": 0,
            "crohme_used_for_generation_or_filtering": False,
        },
        "generation": {
            "seed": SEED,
            "records": len(rows),
            "per_token": args.per_token,
            "focus_tokens": list(FOCUS_TOKENS),
            "by_token": dict(sorted(by_token.items())),
            "by_template": dict(sorted(templates.items())),
            "project_owned_dev_overlap_excluded": excluded,
        },
        "artifacts": {
            "corpus": str(output),
            "corpus_sha256": _sha256(output),
            "hwr_checkpoint_sha256": masked._sha256(hwr),
            "direct_candidates_sha256": masked._sha256(direct),
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "event": "coverage_complete", "records": len(rows),
        "output": str(output), "sha256": audit["artifacts"]["corpus_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
