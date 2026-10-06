#!/usr/bin/env python3
"""Direct, hash-checked pairwise audit of two hard-negative configurations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from build_normalized_ink_v1 import ROOT, _sha256
from audit_hwr_architecture_capacity_paired_v1 import (
    _jsonl_train_uji_writer_hashes,
    _paired,
    _writer_bootstrap,
)
from audit_hwr_objective_fresh_writer_rankshift_v1 import _predict
from run_hwr_architecture_capacity_probe_v1 import (
    DEFAULT_CANONICAL_ROOT,
    DEFAULT_CURATED,
    NpyDataset,
    _class_labels,
    _load_writer_map,
)


DEFAULT_SPLIT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "uji_writer_group_split_seed20261088_fresh.json"
DEFAULT_CACHE = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-augmentation-20261002\fresh-writer-paired-cache-seed20261088"
)
DEFAULT_LOW = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "objective_hardnegative_weight005_seed20261088_reused_ce.json"
DEFAULT_HIGH = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "objective_hardnegative_weight01_seed20261088.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "objective_hardnegative_weight005_vs_weight01_seed20261088_pairwise.json"


def _read_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "aiflow-hwr-objective-competition-probe/v2":
        raise ValueError(f"unexpected report schema: {path}")
    if report.get("status") != "completed_exploratory_objective_ablation":
        raise ValueError(f"objective report is not complete: {path}")
    data, experiment = report.get("data", {}), report.get("experiment", {})
    if data.get("crohme_rows") != 0 or data.get("project_owned_holdout_used") is not False:
        raise ValueError(f"report crosses the CROHME/project-holdout boundary: {path}")
    if data.get("cache_audit", {}).get("crohme_rows") != 0 or experiment.get("product_adopted") is not False:
        raise ValueError(f"report is not an explicitly CROHME-free shadow result: {path}")
    if len(report.get("seed_runs", [])) != 1:
        raise ValueError(f"expected one paired seed per report: {path}")
    return report


def _single_arm(report: dict[str, Any], name: str) -> dict[str, Any]:
    arms = report["seed_runs"][0].get("arms", {})
    if name not in arms:
        raise ValueError(f"missing {name} arm")
    return arms[name]


def _verify_pair(low_path: Path, low: dict[str, Any], high_path: Path, high: dict[str, Any], split_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    low_run, high_run = low["seed_runs"][0], high["seed_runs"][0]
    low_exp, high_exp = low.get("experiment", {}), high.get("experiment", {})
    for field in ("architecture_fixed", "sampler_fixed", "optimizer", "precision", "epochs", "eval_batch_size", "seeds"):
        if low_exp.get(field) != high_exp.get(field):
            raise ValueError(f"paired reports differ in fixed experiment field: {field}")
    for field in ("canonical_hwrt_sha256", "canonical_uji_sha256", "curated_uji_sha256", "writer_split_sha256", "validation_rows", "validation_writer_clusters", "validation_present_labels"):
        if low["data"].get(field) != high["data"].get(field):
            raise ValueError(f"paired reports differ in data provenance: {field}")
    low_cache, high_cache = low["data"]["cache_audit"], high["data"]["cache_audit"]
    if low_cache.get("prepared_cache_sha256") != high_cache.get("prepared_cache_sha256"):
        raise ValueError("paired reports were not evaluated on identical prepared arrays")
    low_seed, high_seed = low_run.get("seed"), high_run.get("seed")
    if low_seed != high_seed or low_exp.get("seeds") != [low_seed]:
        raise ValueError("paired reports do not use the same single seed")
    low_ce, high_ce = _single_arm(low, "ce_pk"), _single_arm(high, "ce_pk")
    if low_ce.get("checkpoint_sha256") != high_ce.get("checkpoint_sha256"):
        raise ValueError("CE controls differ; refusing a direct weight-only comparison")
    low_hn, high_hn = _single_arm(low, "ce_plus_hardnegative"), _single_arm(high, "ce_plus_hardnegative")
    low_obj, high_obj = low_hn.get("objective", {}), high_hn.get("objective", {})
    low_weight, high_weight = low_obj.get("hardnegative_weight"), high_obj.get("hardnegative_weight")
    low_gate = low_obj.get("hardnegative_gate", "all_examples")
    high_gate = high_obj.get("hardnegative_gate", "all_examples")
    if not isinstance(low_weight, (int, float)) or not isinstance(high_weight, (int, float)):
        raise ValueError("reports lack numeric hard-negative weights")
    if low_weight > high_weight:
        raise ValueError("candidate report must not use a larger hard-negative weight than reference")
    if low_obj.get("hardnegative_margin") != high_obj.get("hardnegative_margin"):
        raise ValueError("hard-negative margins differ")
    if (low_weight == high_weight) == (low_gate == high_gate):
        raise ValueError("reports must differ in exactly one of weight or eligibility gate")
    split_hash = _sha256(split_path)
    if split_hash != low["data"].get("writer_split_sha256"):
        raise ValueError("writer split hash does not match the reports")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("schema") != "aiflow-hwr-uji-writer-group-split/v1" or split.get("status") != "completed":
        raise ValueError("writer split manifest is invalid")
    if split.get("official_split", {}).get("official_test_writers_used_for_inner_selection") is not False:
        raise ValueError("official UJI test writers were not explicitly excluded")
    if split.get("candidate_protocol", {}).get("crohme_rows") != 0:
        raise ValueError("writer split does not attest CROHME exclusion")
    return low_hn, high_hn


def _verify_sources(report: dict[str, Any], canonical_root: Path, curated: Path, cache_root: Path) -> None:
    data = report["data"]
    for filename, expected in (("hwrt.jsonl.gz", data["canonical_hwrt_sha256"]), ("uji.jsonl.gz", data["canonical_uji_sha256"])):
        if _sha256(canonical_root / filename) != expected:
            raise ValueError(f"canonical source changed: {filename}")
    if _sha256(curated) != data["curated_uji_sha256"]:
        raise ValueError("curated UJI source changed")
    for filename, expected in data["cache_audit"]["prepared_cache_sha256"].items():
        if _sha256(cache_root / filename) != expected:
            raise ValueError(f"prepared cache changed: {filename}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", "--lower-report", dest="lower_report", type=Path, default=DEFAULT_LOW)
    parser.add_argument("--reference-report", "--higher-report", dest="higher_report", type=Path, default=DEFAULT_HIGH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite pairwise audit: {args.output}")
    if args.batch_size < 1 or args.bootstrap_draws < 1000:
        parser.error("batch size must be positive and bootstrap draws at least 1000")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    low_path, high_path = args.lower_report.resolve(), args.higher_report.resolve()
    low, high = _read_report(low_path), _read_report(high_path)
    low_arm, high_arm = _verify_pair(low_path, low, high_path, high, args.split.resolve())
    _verify_sources(low, args.canonical_root.resolve(), args.curated.resolve(), args.cache_root.resolve())
    _verify_sources(high, args.canonical_root.resolve(), args.curated.resolve(), args.cache_root.resolve())

    low_checkpoint, high_checkpoint = Path(low_arm["checkpoint"]).resolve(), Path(high_arm["checkpoint"]).resolve()
    low_digest, high_digest = _sha256(low_checkpoint), _sha256(high_checkpoint)
    if low_digest != low_arm.get("checkpoint_sha256") or high_digest != high_arm.get("checkpoint_sha256"):
        raise ValueError("challenger checkpoint hash changed")
    ce_arm = _single_arm(low, "ce_pk")
    ce_checkpoint = Path(ce_arm["checkpoint"]).resolve()
    ce_digest = _sha256(ce_checkpoint)
    if ce_digest != ce_arm.get("checkpoint_sha256"):
        raise ValueError("shared CE checkpoint hash changed")

    labels = _class_labels(args.canonical_root.resolve())
    validation = NpyDataset(args.cache_root / "validation_features.npy", args.cache_root / "validation_labels.npy")
    if len(validation) != low["data"].get("validation_rows"):
        raise ValueError("validation row count changed")
    device = torch.device(args.device)
    ce_rank, ce_prediction = _predict(ce_checkpoint, validation, labels, device, args.batch_size)
    low_rank, low_prediction = _predict(low_checkpoint, validation, labels, device, args.batch_size)
    high_rank, high_prediction = _predict(high_checkpoint, validation, labels, device, args.batch_size)

    for arm, rank in (("ce_pk", ce_rank), ("low", low_rank), ("high", high_rank)):
        if len(rank) != len(validation):
            raise AssertionError(f"{arm} predictions are not row-aligned")
    expected = low["seed_runs"][0]["paired_comparison"]
    if int((ce_rank == 1).sum()) != expected["top1"]["before_hits"] or int((low_rank == 1).sum()) != expected["top1"]["after_hits"]:
        raise AssertionError("lower-weight ranks disagree with their paired report")
    high_expected = high["seed_runs"][0]["paired_comparison"]
    if int((ce_rank == 1).sum()) != high_expected["top1"]["before_hits"] or int((high_rank == 1).sum()) != high_expected["top1"]["after_hits"]:
        raise AssertionError("higher-weight ranks disagree with their paired report")

    split = json.loads(args.split.read_text(encoding="utf-8"))
    writer_map = _load_writer_map(args.curated.resolve())
    writer_hashes = _jsonl_train_uji_writer_hashes(
        args.canonical_root.resolve(), writer_map,
        set(split["inner_split"]["validation_writer_hashes"]), set(labels),
    )
    if len(writer_hashes) != len(validation) or len(set(writer_hashes)) != 8:
        raise ValueError("writer identities are not aligned to exactly eight validation clusters")

    candidate_name = f"weight_{low_arm['objective']['hardnegative_weight']:.3f}_gate_{low_arm['objective'].get('hardnegative_gate', 'all_examples')}"
    reference_name = f"weight_{high_arm['objective']['hardnegative_weight']:.3f}_gate_{high_arm['objective'].get('hardnegative_gate', 'all_examples')}"
    metrics: dict[str, Any] = {}
    for metric, ranks in (("top1", (high_rank == 1, low_rank == 1)), ("top5", (high_rank <= 5, low_rank <= 5))):
        metrics[metric] = {
            "before": reference_name,
            "after": candidate_name,
            "paired": _paired(ranks[0], ranks[1]),
            "writer_cluster_bootstrap": _writer_bootstrap(ranks[0], ranks[1], writer_hashes, low_run_seed(low), args.bootstrap_draws),
        }

    targets = np.asarray(validation.labels, dtype=np.int64)
    per_class = []
    for class_id in sorted(set(targets.tolist())):
        selected = targets == class_id
        before1, after1 = high_rank[selected] == 1, low_rank[selected] == 1
        before5, after5 = high_rank[selected] <= 5, low_rank[selected] <= 5
        per_class.append({
            "label": labels[class_id], "support": int(selected.sum()),
            "top1_before": int(before1.sum()), "top1_after": int(after1.sum()),
            "top1_delta_hits": int(after1.sum() - before1.sum()),
            "top1_recovered": int((~before1 & after1).sum()), "top1_regressed": int((before1 & ~after1).sum()),
            "top5_before": int(before5.sum()), "top5_after": int(after5.sum()),
            "top5_delta_hits": int(after5.sum() - before5.sum()),
        })
    changed = Counter()
    for index in np.flatnonzero(high_prediction != low_prediction):
        changed[(labels[int(targets[index])], labels[int(high_prediction[index])], labels[int(low_prediction[index])])] += 1

    output = {
        "schema": "aiflow-hwr-hardnegative-pairwise-audit/v2",
        "status": "posthoc_same_writer_pairwise_diagnostic",
        "protocol": {
            "training_performed": False, "model_or_objective_selection_performed": False,
            "crohme_rows": 0, "official_uji_test_scored": False,
            "interpretation_limit": "one consumed internal eight-writer split; direct pairwise evidence is diagnostic, not fresh product acceptance",
        },
        "inputs": {
            "candidate_report": str(low_path), "candidate_report_sha256": _sha256(low_path),
            "reference_report": str(high_path), "reference_report_sha256": _sha256(high_path),
            "writer_split": str(args.split.resolve()), "writer_split_sha256": _sha256(args.split.resolve()),
            "candidate_weight": low_arm["objective"]["hardnegative_weight"],
            "reference_weight": high_arm["objective"]["hardnegative_weight"],
            "candidate_gate": low_arm["objective"].get("hardnegative_gate", "all_examples"),
            "reference_gate": high_arm["objective"].get("hardnegative_gate", "all_examples"),
            "margin": low_arm["objective"]["hardnegative_margin"],
            "shared_ce_checkpoint_sha256": ce_digest,
            "lower_checkpoint_sha256": low_digest, "higher_checkpoint_sha256": high_digest,
        },
        "summary": {
            "rows": len(validation), "writer_clusters": len(set(writer_hashes)),
            "candidate_top1": int((low_rank == 1).sum()), "reference_top1": int((high_rank == 1).sum()),
            "candidate_top5": int((low_rank <= 5).sum()), "reference_top5": int((high_rank <= 5).sum()),
            "top1_metric": metrics["top1"], "top5_metric": metrics["top5"],
            "changed_top1_predictions": int((low_prediction != high_prediction).sum()),
            "largest_top1_gains": sorted((x for x in per_class if x["top1_delta_hits"] > 0), key=lambda x: x["top1_delta_hits"], reverse=True)[:12],
            "largest_top1_losses": sorted((x for x in per_class if x["top1_delta_hits"] < 0), key=lambda x: x["top1_delta_hits"])[:12],
            "most_common_changed_confusions": [
                {"target": t, "reference_prediction": h, "candidate_prediction": l, "rows": n}
                for (t, h, l), n in changed.most_common(30)
            ],
        },
        "per_class": per_class,
        "product_adopted": False,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "event": "hardnegative_weight_pairwise_audit_complete", "report": str(output_path),
        "rows": len(validation), "writer_clusters": len(set(writer_hashes)),
        "top1_delta_pp": metrics["top1"]["paired"]["delta_percentage_points"],
        "top5_delta_pp": metrics["top5"]["paired"]["delta_percentage_points"],
        "crohme_rows": 0, "product_adopted": False,
    }, ensure_ascii=False), flush=True)
    return 0


def low_run_seed(report: dict[str, Any]) -> int:
    seeds = report.get("experiment", {}).get("seeds", [])
    if len(seeds) != 1:
        raise ValueError("expected exactly one seed")
    return int(seeds[0]) + 61000


if __name__ == "__main__":
    raise SystemExit(main())
