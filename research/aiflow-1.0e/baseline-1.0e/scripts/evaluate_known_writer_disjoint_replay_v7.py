#!/usr/bin/env python3
"""Run the known writer-disjoint replay diagnostic once; never promote/tune."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from hierarchical_writer_model_v7 import (
    DEFAULT_CHECKPOINT, DEFAULT_FROZEN, SAMPLE_ID, build_raw_catalog,
    fresh_sample_ids, sha256,
)
from train_character_classifier_v1 import InkClassifierV1


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD = ROOT / "artifacts/hierarchical_writer_model_v7_20260823_r1"
DEFAULT_OUTPUT = DEFAULT_BUILD / "known_writer_disjoint_replay_diagnostic"
SCHEMA = "aiflow-known-writer-disjoint-replay-diagnostic/v7"


def fresh_rows(frozen: Path) -> tuple[dict[str, dict], list[dict], dict]:
    fresh_path = frozen / "ownership_fresh_acceptance.jsonl"
    formulas_path = frozen / "frozen_dataset/data/formulas_valid.jsonl"
    ownership_path = frozen / "frozen_dataset/data/ownership_train.jsonl"
    fresh = fresh_sample_ids(fresh_path)
    formulas = {}
    with formulas_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip() and SAMPLE_ID.search(line).group(1) in fresh:
                row = json.loads(line); formulas[str(row["sample_id"])] = row
    ownership = [json.loads(line) for line in fresh_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(formulas) != 53 or len(ownership) != 53:
        raise ValueError("known replay partition must contain 53 formulas")
    return formulas, ownership, {
        "formulas": sha256(formulas_path), "ownership": sha256(fresh_path),
        "merged_ownership": sha256(ownership_path), "fresh_id_exclusion": sha256(fresh_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = [args.frozen.resolve(), args.build.resolve(), args.checkpoint.resolve(), args.output.resolve()]
    if any(path.drive.upper() != "D:" for path in paths):
        parser.error("all inputs and outputs must remain on D:")
    if args.output.exists():
        parser.error(f"single replay receipt already exists: {args.output}")
    build_report = json.loads((args.build / "report.json").read_text(encoding="utf-8"))
    if build_report["schema"] != "aiflow-hierarchical-writer-model/v7":
        raise ValueError("diagnostic requires a frozen v7 build")
    if sha256(args.checkpoint) != build_report["inputs"]["checkpoint"]["sha256"]:
        raise ValueError("checkpoint differs from frozen v7 build")

    formulas, ownership, hashes = fresh_rows(args.frozen)
    labels = [str(value) for value in torch.load(args.checkpoint, map_location="cpu", weights_only=False)["math_labels"]]
    _catalog, features, metadata = build_raw_catalog(
        formulas, ownership, labels, hashes,
        source_domain="project_owned_known_writer_replay_raw_observed",
        evidence_tier="known_writer_disjoint_replay_diagnostic_raw_observed",
    )
    legacy_writers = set(build_report["raw_catalog"]["writer_ids"])
    replay_writers = {row["writer_id"] for row in metadata}
    if legacy_writers & replay_writers:
        raise ValueError("known replay is not writer-disjoint from legacy raw catalog")
    supported = [index for index, row in enumerate(metadata) if row["label_index"] >= 0]
    truth = np.asarray([metadata[index]["label_index"] for index in supported], dtype=np.int64)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = InkClassifierV1(len(labels), None); model.load_state_dict(payload["state_dict"], strict=True); model.eval()
    with torch.no_grad():
        rank = model(torch.from_numpy(features[supported]), "math").argsort(dim=1, descending=True)[:, :5].cpu().numpy()
    per_formula = defaultdict(list)
    for index, expected, predicted in zip(supported, truth, rank, strict=True):
        per_formula[metadata[index]["sample_id"]].append((expected, predicted))
    top1 = float(np.mean(rank[:, 0] == truth)); top5 = float(np.mean(np.any(rank == truth[:, None], axis=1)))
    report = {
        "schema": SCHEMA, "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "status": "known_writer_disjoint_replay_diagnostic_only",
        "scope": {
            "formulae": len(formulas), "writers": sorted(replay_writers), "glyph_groups": len(metadata),
            "classes": len({row["token"] for row in metadata}),
            "label_counts": dict(sorted(Counter(row["token"] for row in metadata).items())),
            "writer_disjoint_from_legacy": True,
        },
        "metrics": {
            "supported_glyphs": len(supported), "unsupported_glyphs": len(metadata) - len(supported),
            "top1": top1, "top5": top5,
            "formula_top1_exact": float(np.mean([all(predicted[0] == expected for expected, predicted in rows) for rows in per_formula.values()])),
            "formula_top5_oracle": float(np.mean([all(expected in predicted for expected, predicted in rows) for rows in per_formula.values()])),
        },
        "provenance": {
            "source_domains": sorted({row["source_domain"] for row in metadata}),
            "evidence_tiers": sorted({row["evidence_tier"] for row in metadata}),
            "ownership_sha256": hashes["ownership"],
            "exact_stroke_partition_validated": True,
        },
        "gates": {
            "known_previously_used_replay": True, "untouched_acceptance": False,
            "selection_performed": False, "tuning_performed": False,
            "promotion_evidence": False, "promotion_ready": False,
            "new_untouched_writer_formula_set_required": True,
            "training_performed": False, "product_runtime_changed": False,
            "checkpoint_changed": False,
        },
        "frozen_build": {"path": str(args.build.resolve()), "report_sha256": sha256(args.build / "report.json")},
        "checkpoint_sha256": sha256(args.checkpoint), "source_hashes": hashes,
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scope": report["scope"], "metrics": report["metrics"], "gates": report["gates"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
