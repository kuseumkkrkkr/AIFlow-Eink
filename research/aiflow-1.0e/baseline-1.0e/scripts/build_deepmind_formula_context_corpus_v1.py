#!/usr/bin/env python3
"""Build a commercially usable formula-context corpus from DeepMind Math.

The official Apache-2.0 generator is sampled locally.  Natural-language words
are treated as hard boundaries; only expressions exactly representable by the
frozen AIFlow 372-token output contract are admitted.  Exact project-owned
development token sequences are excluded before the corpus is written.
CROHME is not read and cannot influence corpus generation or filtering.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from character_tensor_v1 import ROOT, _json_lines
from training_data_guard_v1 import (
    assert_training_entrypoint_arguments_clean,
    assert_training_path_clean,
)

if __name__ == "__main__":
    assert_training_entrypoint_arguments_clean()


SCHEMA = "aiflow-deepmind-formula-context-corpus/v2"
AUDIT_SCHEMA = "aiflow-deepmind-formula-context-audit/v2"
SOURCE_REPOSITORY = "https://github.com/google-deepmind/mathematics_dataset"
SOURCE_REVISION = "427f45075f84b8b9774950196ad63867ca20ffb3"
SEED = 20260820

DEFAULT_SOURCE = (
    ROOT / "datasets" / "10_approved_external"
    / "deepmind_mathematics_dataset" / "source"
)
DEFAULT_OUTPUT = (
    ROOT / "datasets" / "10_approved_external"
    / "deepmind_mathematics_dataset" / "derived"
    / "formula_context_v2.jsonl.gz"
)
DEFAULT_AUDIT = DEFAULT_OUTPUT.with_name("formula_context_v2_audit.json")
DEFAULT_HWR = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1\final_all_writers_steps250_lr1e-3"
    r"\project_symbol_head_checkpoint.pt"
)
DEFAULT_DIRECT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1"
    r"\expanded_writer_loo_candidates_r1.jsonl.gz"
)
DEFAULT_DEPS = ROOT / "research" / "python_deps"

LEXEME = re.compile(
    r"!=|<=|>=|\*\*|[A-Za-z]+|\d+(?:\.\d+)?|[+\-*/=<>()[\]{}]|[^\s]"
)
OPERATOR_MAP = {
    "+": "+",
    "-": "-",
    "*": r"\times",
    "/": "/",
    "=": "=",
    "<": "<",
    ">": ">",
    "!=": r"\neq",
    "<=": r"\leq",
    ">=": r"\geq",
    "(": "(",
    ")": ")",
    "[": "[",
    "]": "]",
    "{": r"\{",
    "}": r"\}",
}
BINARY = frozenset({"+", "-", r"\times", "/", "=", "<", ">", r"\neq", r"\leq", r"\geq"})
OPEN_TO_CLOSE = {"(": ")", "[": "]", r"\{": r"\}"}
CLOSE_TO_OPEN = {value: key for key, value in OPEN_TO_CLOSE.items()}


def _event(name: str, **values: object) -> None:
    print(json.dumps({"event": name, **values}, ensure_ascii=False), flush=True)


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


def _labels(checkpoint: Path) -> list[str]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    labels = list(payload.get("math_labels", []))
    if len(labels) != 372 or len(set(labels)) != 372:
        raise ValueError("expected frozen unified 372-class HWR checkpoint")
    return labels


def _formula_sequences(path: Path) -> set[tuple[str, ...]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in _json_lines(path):
        grouped[str(row["formula_id"])].append(row)
    output = set()
    for formula_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["context"]["index"]))
        if [int(row["context"]["index"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"invalid evaluation formula sequence: {formula_id}")
        output.add(tuple(str(row["label"]) for row in rows))
    return output


def _balanced(tokens: list[str]) -> bool:
    stack = []
    for token in tokens:
        if token in OPEN_TO_CLOSE:
            stack.append(token)
        elif token in CLOSE_TO_OPEN:
            if not stack or stack.pop() != CLOSE_TO_OPEN[token]:
                return False
    return not stack


def _valid_formula(tokens: list[str]) -> bool:
    if not 3 <= len(tokens) <= 64 or not _balanced(tokens):
        return False
    if not any(token in BINARY for token in tokens):
        return False
    if tokens[-1] in BINARY or tokens[0] in BINARY - {"-"}:
        return False
    for left, right in zip(tokens, tokens[1:]):
        if left in BINARY and right in BINARY and right != "-":
            return False
    return True


def _extract_formulas(text: str, allowed: set[str]) -> tuple[list[list[str]], Counter]:
    output = []
    current: list[str] = []
    invalid = False
    rejected = Counter()

    def flush() -> None:
        nonlocal current, invalid
        if current:
            if invalid:
                rejected["unsupported_lexeme"] += 1
            elif _valid_formula(current):
                output.append(current)
            else:
                rejected["invalid_structure"] += 1
        current = []
        invalid = False

    for match in LEXEME.finditer(text):
        lexeme = match.group(0)
        if lexeme.isalpha():
            if len(lexeme) == 1 and lexeme in allowed:
                current.append(lexeme)
            else:
                flush()
            continue
        if lexeme[0].isdigit():
            if "." in lexeme:
                invalid = True
            else:
                current.extend(lexeme)
            continue
        if lexeme == "**":
            invalid = True
            continue
        mapped = OPERATOR_MAP.get(lexeme)
        if mapped is not None:
            if mapped not in allowed:
                invalid = True
            else:
                current.append(mapped)
            continue
        flush()
    flush()
    return output, rejected


def _flatten_modules(tree: dict, prefix: str = "") -> dict[str, object]:
    output = {}
    for name, value in tree.items():
        full_name = f"{prefix}__{name}" if prefix else str(name)
        if isinstance(value, dict):
            output.update(_flatten_modules(value, full_name))
        else:
            output[full_name] = value
    return output


def _source_commit(source: Path) -> str:
    command = [
        "git", "-c", f"safe.directory={source.as_posix()}",
        "-C", str(source), "rev-parse", "HEAD",
    ]
    return subprocess.check_output(command, text=True).strip()


def _load_modules(source: Path, deps: Path) -> dict[str, object]:
    sys.path.insert(0, str(deps))
    sys.path.insert(0, str(source))
    from mathematics_dataset.modules import modules  # pylint: disable=import-outside-toplevel

    return _flatten_modules(modules.train(lambda entropy_range: entropy_range))


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for row in rows:
                    text.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(partial, path)


def _self_test(labels: list[str]) -> None:
    allowed = set(labels)
    formulas, rejected = _extract_formulas(
        "Solve -42*r + 27*c = -1167 and 130*r + 4*c = 372 for r.",
        allowed,
    )
    assert formulas == [
        ["-", "4", "2", r"\times", "r", "+", "2", "7", r"\times", "c", "=", "-", "1", "1", "6", "7"],
        ["1", "3", "0", r"\times", "r", "+", "4", r"\times", "c", "=", "3", "7", "2"],
    ]
    unsupported, counts = _extract_formulas("Calculate 1.2 + 3 and x**2 + 1.", allowed)
    assert unsupported == [] and counts["unsupported_lexeme"] == 2
    assert rejected["invalid_structure"] == 1  # trailing isolated r


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--deps", type=Path, default=DEFAULT_DEPS)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_HWR)
    parser.add_argument("--direct-candidates", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--max-formulas", type=int, default=80_000)
    parser.add_argument("--min-formulas", type=int, default=50_000)
    parser.add_argument("--problems-per-module", type=int, default=3_000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.min_formulas <= args.max_formulas:
        parser.error("formula bounds must satisfy 1 <= min <= max")
    if args.problems_per_module < 1:
        parser.error("problems-per-module must be positive")

    source = _d_path(args.source, "DeepMind source", file=False)
    deps = _d_path(args.deps, "local Python dependencies", file=False)
    hwr = _d_path(args.hwr_checkpoint, "HWR checkpoint")
    direct = _d_path(args.direct_candidates, "direct candidates")
    output = _d_path(args.output, "formula corpus output", file=False)
    audit_path = _d_path(args.audit, "formula corpus audit", file=False)
    labels = _labels(hwr)
    if args.self_test:
        _self_test(labels)
        _event("self_test", status="pass")
        return 0
    if output.exists() or audit_path.exists():
        parser.error("refusing to overwrite existing formula-context corpus")

    source_commit = _source_commit(source)
    if source_commit != SOURCE_REVISION:
        raise ValueError(f"DeepMind source revision drift: {source_commit}")
    license_path = source / "LICENSE"
    if not license_path.is_file() or "Apache License" not in license_path.read_text(encoding="utf-8")[:200]:
        raise ValueError("DeepMind Apache-2.0 license evidence missing")

    assert_training_path_clean(source, "commercial DeepMind source")
    assert_training_path_clean(direct, "project-owned development candidates")
    development_sequences = _formula_sequences(direct)
    modules = _load_modules(source, deps)
    if len(modules) != 56:
        raise ValueError(f"DeepMind train module count drift: {len(modules)}")

    random.seed(SEED)
    np.random.seed(SEED)
    allowed = set(labels)
    records = []
    seen: set[tuple[str, ...]] = set()
    module_counts = Counter()
    field_counts = Counter()
    token_counts = Counter()
    rejected = Counter()
    module_errors = Counter()
    overlap_excluded = 0
    sampled_problems = 0
    names = sorted(modules)

    for round_index in range(args.problems_per_module):
        for module_name in names:
            if len(records) >= args.max_formulas:
                break
            try:
                problem = modules[module_name]()
            except Exception as error:  # generator modules are independent; audit bounded failures
                module_errors[f"{module_name}:{type(error).__name__}"] += 1
                if module_errors[f"{module_name}:{type(error).__name__}"] > 25:
                    raise RuntimeError(f"repeated DeepMind generator failure: {module_name}") from error
                continue
            sampled_problems += 1
            for field, text in (("question", str(problem.question)), ("answer", str(problem.answer))):
                formulas, field_rejections = _extract_formulas(text, allowed)
                rejected.update(field_rejections)
                for tokens in formulas:
                    sequence = tuple(tokens)
                    if sequence in development_sequences:
                        overlap_excluded += 1
                        continue
                    if sequence in seen:
                        rejected["duplicate_sequence"] += 1
                        continue
                    seen.add(sequence)
                    digest = hashlib.sha256(
                        (module_name + "\0" + field + "\0" + "\x1f".join(tokens)).encode("utf-8")
                    ).hexdigest()
                    records.append({
                        "schema": SCHEMA,
                        "formula_id": f"deepmind::{digest[:24]}",
                        "tokens": tokens,
                        "relations": ["right"] * (len(tokens) - 1),
                        "source_module": module_name,
                        "source_field": field,
                        "source_repository": SOURCE_REPOSITORY,
                        "source_revision": SOURCE_REVISION,
                        "license": "Apache-2.0",
                    })
                    module_counts[module_name] += 1
                    field_counts[field] += 1
                    token_counts.update(tokens)
                    if len(records) >= args.max_formulas:
                        break
                if len(records) >= args.max_formulas:
                    break
        if len(records) >= args.max_formulas:
            break
        if (round_index + 1) % 250 == 0:
            _event(
                "corpus_progress", rounds=round_index + 1,
                sampled_problems=sampled_problems, unique_formulas=len(records),
            )

    if len(records) < args.min_formulas:
        raise RuntimeError(
            f"insufficient unique formula corpus: {len(records)} < {args.min_formulas}"
        )
    if any(tuple(record["tokens"]) in development_sequences for record in records):
        raise AssertionError("project-owned development sequence leaked into formula corpus")

    records.sort(key=lambda row: row["formula_id"])
    _write_jsonl_gz(output, records)
    audit = {
        "schema": AUDIT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": source_commit,
            "license": "Apache-2.0",
            "license_path": str(license_path),
            "license_sha256": _sha256(license_path),
            "module_count": len(modules),
        },
        "contracts": {
            "commercial_training_rights": True,
            "synthetic_source": True,
            "raster_images": False,
            "raw_ink": False,
            "output_class_count": len(labels),
            "unsupported_expression_fragments_discarded_whole": True,
            "project_owned_dev_exact_sequence_overlap": 0,
            "crohme_used_for_generation_or_filtering": False,
        },
        "generation": {
            "seed": SEED,
            "sampled_problems": sampled_problems,
            "problems_per_module_limit": args.problems_per_module,
            "unique_formulas": len(records),
            "formula_length_min": min(len(row["tokens"]) for row in records),
            "formula_length_max": max(len(row["tokens"]) for row in records),
            "formula_length_mean": sum(len(row["tokens"]) for row in records) / len(records),
            "by_module": dict(sorted(module_counts.items())),
            "by_field": dict(sorted(field_counts.items())),
            "rejections": dict(sorted(rejected.items())),
            "module_errors": dict(sorted(module_errors.items())),
            "project_owned_dev_overlap_excluded": overlap_excluded,
            "token_counts": dict(sorted(token_counts.items())),
        },
        "artifacts": {
            "corpus": str(output),
            "corpus_sha256": _sha256(output),
            "hwr_checkpoint": str(hwr),
            "hwr_checkpoint_sha256": _sha256(hwr),
            "direct_candidates_sha256": _sha256(direct),
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    _event(
        "corpus_complete", output=str(output), formulas=len(records),
        sampled_problems=sampled_problems, sha256=audit["artifacts"]["corpus_sha256"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
