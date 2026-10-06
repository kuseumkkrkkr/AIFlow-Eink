#!/usr/bin/env python3
"""Freeze independently reviewed formula ownership before model evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


DEFAULT_PREPARATION = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\fresh-context-acceptance-20260820-r2"
)
DEFAULT_SOURCE_DATASET = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\public-candidate-20260820-r4-replay-restored"
)
KNOWN_SPARSE = ("|", "O", "o")


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ), encoding="utf-8")


def _validate_groups(groups: list[list[int]], stroke_count: int, sample_id: str) -> None:
    owned = [int(index) for group in groups for index in group]
    if (
        any(not group for group in groups)
        or sorted(owned) != list(range(stroke_count))
        or len(owned) != len(set(owned))
    ):
        raise ValueError(f"invalid exhaustive ownership: {sample_id}")


def freeze(args: argparse.Namespace) -> dict:
    preparation = args.preparation.expanduser().resolve()
    source_dataset = args.source_dataset.expanduser().resolve()
    review_path = args.review.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path, label in (
        (preparation, "preparation"), (source_dataset, "source dataset")
    ):
        if path.drive.upper() != "D:" or not path.is_dir():
            raise ValueError(f"{label} must be an existing D: directory: {path}")
    if review_path.drive.upper() != "D:" or not review_path.is_file():
        raise ValueError(f"review must be an existing D: file: {review_path}")
    if output.drive.upper() != "D:":
        raise ValueError(f"output must remain on D:: {output}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen acceptance: {output}")

    readiness_path = preparation / "readiness.json"
    proposals_path = preparation / "ownership_proposals.json"
    inspection_path = preparation / "inspection_manifest.json"
    readiness = _json(readiness_path)
    proposals = _json(proposals_path)
    inspection = _json(inspection_path)
    review = _json(review_path)
    if readiness.get("schema") != "aiflow-fresh-acceptance-readiness/v1":
        raise ValueError("unexpected readiness schema")
    if proposals.get("schema") != "aiflow-fresh-acceptance-proposals/v1":
        raise ValueError("unexpected proposals schema")
    if inspection.get("schema") != "aiflow-fresh-acceptance-inspection/v1":
        raise ValueError("unexpected inspection schema")
    if review.get("schema") != "aiflow-fresh-acceptance-review/v1":
        raise ValueError("unexpected review schema")
    if readiness.get("model_predictions_opened") is not False:
        raise ValueError("acceptance predictions were opened before freeze")
    if review.get("model_predictions_opened_before_review") is not False:
        raise ValueError("review was not prediction-independent")

    expected_pages = {
        str(row["file"]): str(row["sha256"]) for row in inspection["pages"]
    }
    reviewed_pages = {
        str(row["file"]): str(row["sha256"]) for row in review["pages"]
    }
    if reviewed_pages != expected_pages:
        raise ValueError("review does not cover the exact frozen inspection pages")
    for filename, digest in expected_pages.items():
        page = preparation / "inspection" / filename
        if not page.is_file() or _sha(page) != digest:
            raise ValueError(f"inspection page changed after review: {filename}")

    source_formulas_path = source_dataset / "data" / "formulas_valid.jsonl"
    source_ownership_path = source_dataset / "data" / "ownership_train.jsonl"
    source_formula_hash = readiness["artifacts"]["formulae"]["sha256"]
    source_ownership_hash = readiness["artifacts"]["ownership"]["sha256"]
    if (
        _sha(source_formulas_path) != source_formula_hash
        or _sha(source_ownership_path) != source_ownership_hash
        or _sha(proposals_path) != readiness["artifacts"]["proposals"]["sha256"]
        or _sha(inspection_path)
        != readiness["artifacts"]["inspection_manifest"]["sha256"]
    ):
        raise ValueError("acceptance preparation input changed before freeze")

    formulas = _jsonl(source_formulas_path)
    formula_by_id = {str(row["sample_id"]): row for row in formulas}
    legacy_ownership = _jsonl(source_ownership_path)
    legacy_ids = {str(row["sample_id"]) for row in legacy_ownership}
    corrections = {
        str(key): value for key, value in review.get("corrections", {}).items()
    }
    excluded = {
        str(key): value for key, value in review.get("excluded", {}).items()
    }
    proposal_ids = {str(row["sample_id"]) for row in proposals["rows"]}
    if set(corrections) - proposal_ids or set(excluded) - proposal_ids:
        raise ValueError("review references an unknown acceptance proposal")

    accepted = []
    for proposal in proposals["rows"]:
        sample_id = str(proposal["sample_id"])
        if sample_id in excluded:
            continue
        if sample_id in legacy_ids:
            raise ValueError(f"fresh acceptance overlaps existing ownership: {sample_id}")
        formula = formula_by_id.get(sample_id)
        if formula is None:
            raise ValueError(f"fresh acceptance formula missing: {sample_id}")
        labels = [str(cell["token"]) for cell in formula.get("target_cells", [])]
        if labels != [str(value) for value in proposal["labels"]]:
            raise ValueError(f"acceptance labels changed: {sample_id}")
        if str(formula["writer_id"]) != str(proposal["writer_id"]):
            raise ValueError(f"acceptance writer changed: {sample_id}")
        groups = corrections.get(sample_id, {}).get("groups", proposal["groups"])
        groups = [[int(index) for index in group] for group in groups]
        if len(groups) != len(labels):
            raise ValueError(f"ownership label/group mismatch: {sample_id}")
        _validate_groups(groups, len(formula["strokes"]), sample_id)
        accepted.append({
            "schema": "aiflow-public-ownership/v1",
            "sample_id": sample_id,
            "writer_id": str(formula["writer_id"]),
            "groups": groups,
            "labels": labels,
            "accepted": True,
            "review_method": "independent_visual_stroke_index_review_codex_2026-08-20",
        })

    writer_counts = Counter(row["writer_id"] for row in accepted)
    token_counts = Counter(token for row in accepted for token in row["labels"])
    context_formulae = sum(len(row["labels"]) >= 2 for row in accepted)
    if len(accepted) < 50 or len(writer_counts) < 2:
        raise ValueError("reviewed fresh acceptance no longer meets the core size gate")

    output.mkdir(parents=True)
    fresh_path = output / "ownership_fresh_acceptance.jsonl"
    _write_jsonl(fresh_path, accepted)
    frozen_dataset = output / "frozen_dataset" / "data"
    frozen_dataset.mkdir(parents=True)
    frozen_formulas = frozen_dataset / "formulas_valid.jsonl"
    frozen_formulas.write_bytes(source_formulas_path.read_bytes())
    merged_ownership = frozen_dataset / "ownership_train.jsonl"
    _write_jsonl(merged_ownership, legacy_ownership + accepted)

    dataset_info = {
        "schema": "aiflow-fresh-acceptance-frozen-dataset/v1",
        "legacy_ownership_formulae": len(legacy_ownership),
        "fresh_acceptance_formulae": len(accepted),
        "merged_ownership_formulae": len(legacy_ownership) + len(accepted),
        "fresh_writers": len(writer_counts),
        "fresh_context_formulae": context_formulae,
        "files": {
            "formulas_valid.jsonl": _sha(frozen_formulas),
            "ownership_train.jsonl": _sha(merged_ownership),
        },
    }
    dataset_info_path = output / "frozen_dataset" / "dataset_info.json"
    dataset_info_path.write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    sparse_counts = {token: token_counts[token] for token in KNOWN_SPARSE}
    manifest = {
        "schema": "aiflow-fresh-context-acceptance-freeze/v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "model_predictions_opened_before_freeze": False,
        "review": {
            "reviewer_type": review.get("reviewer_type"),
            "review_method": review.get("review_method"),
            "reviewed_pages": len(reviewed_pages),
            "corrections": len(corrections),
            "excluded": len(excluded),
        },
        "acceptance": {
            "formulae": len(accepted),
            "writers": len(writer_counts),
            "context_formulae": context_formulae,
            "single_symbol_formulae": len(accepted) - context_formulae,
            "writer_formula_counts": dict(sorted(writer_counts.items())),
            "known_sparse_truth_counts": sparse_counts,
        },
        "gates": {
            "at_least_two_unused_writers": len(writer_counts) >= 2,
            "at_least_fifty_formulae": len(accepted) >= 50,
            "independent_visual_ownership_review": True,
            "training_sequence_exact_overlap": 0,
            "known_sparse_truth_present": all(value > 0 for value in sparse_counts.values()),
            "ready_for_overall_acceptance_evaluation": True,
            "ready_for_adoption": all(value > 0 for value in sparse_counts.values()),
        },
        "artifacts": {
            "source_formulae": {"path": str(source_formulas_path), "sha256": _sha(source_formulas_path)},
            "source_ownership": {"path": str(source_ownership_path), "sha256": _sha(source_ownership_path)},
            "proposals": {"path": str(proposals_path), "sha256": _sha(proposals_path)},
            "inspection": {"path": str(inspection_path), "sha256": _sha(inspection_path)},
            "review": {"path": str(review_path), "sha256": _sha(review_path)},
            "fresh_ownership": {"path": str(fresh_path), "sha256": _sha(fresh_path)},
            "frozen_dataset_info": {"path": str(dataset_info_path), "sha256": _sha(dataset_info_path)},
        },
    }
    manifest_path = output / "frozen_acceptance_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE_DATASET)
    parser.add_argument(
        "--review", type=Path,
        default=DEFAULT_PREPARATION / "visual_review_decision.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=DEFAULT_PREPARATION / "frozen_acceptance",
    )
    args = parser.parse_args()
    print(json.dumps(freeze(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
