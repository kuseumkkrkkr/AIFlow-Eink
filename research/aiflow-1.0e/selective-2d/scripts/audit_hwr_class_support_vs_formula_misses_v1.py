#!/usr/bin/env python3
"""Relate frozen HWR class support to consumed formula-set candidate misses.

Read-only diagnosis only: no training, no threshold/model selection, no CROHME.
The formula cohort is consumed development data and is never acceptance evidence.
"""

from __future__ import annotations

import hashlib
import json
import site
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

_USER_SITE = site.getusersitepackages()
if _USER_SITE in sys.path:
    sys.path.remove(_USER_SITE)

from audit_prompt_bert_context_on_frozen149_v1 import (
    DEFAULT_DATA,
    DEFAULT_SUMMARY,
    _formula_rows,
    _jsonl,
)
from run_hwr_architecture_capacity_probe_v1 import DEFAULT_CANONICAL_ROOT, _class_labels


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "architecture_scale_uji_inner_writer_probe.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "class_support_confusions_20261002_r2.json"
SUPPORT_BINS = ((0, 99, "0-99"), (100, 199, "100-199"), (200, 499, "200-499"), (500, None, "500+"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _support_bin(count: int) -> str:
    for lower, upper, label in SUPPORT_BINS:
        if count >= lower and (upper is None or count <= upper):
            return label
    raise AssertionError(f"support count outside bins: {count}")


def main() -> int:
    import argparse

    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    probe_path, summary_path = args.probe.resolve(), args.summary.resolve()
    data_path, canonical_root, output = args.data.resolve(), args.canonical_root.resolve(), args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")
    for name, path in (("architecture probe", probe_path), ("frozen summary", summary_path), ("formula dataset", data_path)):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")

    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if probe.get("status") != "completed_exploratory_capacity_probe" or probe["data"]["cache"]["crohme_rows"] != 0:
        raise ValueError("capacity probe lacks expected CROHME-excluded provenance")
    if _sha256(canonical_root / "hwrt.jsonl.gz") != probe["data"]["canonical_hwrt_sha256"]:
        raise ValueError("canonical HWRT source hash mismatch")
    if _sha256(canonical_root / "uji.jsonl.gz") != probe["data"]["canonical_uji_sha256"]:
        raise ValueError("canonical UJI source hash mismatch")
    cache = probe["data"]["cache"]["cache"]
    train_labels_path = Path(cache["train"]["labels"])
    validation_labels_path = Path(cache["validation"]["labels"])
    expected_hashes = probe["data"]["cache"]["prepared_cache_sha256"]
    if _sha256(train_labels_path) != expected_hashes["train_labels.npy"]:
        raise ValueError("train label cache hash mismatch")
    if _sha256(validation_labels_path) != expected_hashes["validation_labels.npy"]:
        raise ValueError("validation label cache hash mismatch")

    labels = _class_labels(canonical_root)
    if len(labels) != 372:
        raise ValueError("expected frozen 372-class vocabulary")
    train_ids = np.load(train_labels_path, allow_pickle=False).astype(np.int64, copy=False)
    validation_ids = np.load(validation_labels_path, allow_pickle=False).astype(np.int64, copy=False)
    if train_ids.size != probe["data"]["cache"]["rows"]["train"]:
        raise ValueError("train cache row count mismatch")
    if validation_ids.size != probe["data"]["cache"]["rows"]["validation"]:
        raise ValueError("validation cache row count mismatch")
    if np.any(train_ids < 0) or np.any(train_ids >= len(labels)) or np.any(validation_ids < 0) or np.any(validation_ids >= len(labels)):
        raise ValueError("class ID outside frozen vocabulary")
    train_counts = np.bincount(train_ids, minlength=len(labels))
    validation_counts = np.bincount(validation_ids, minlength=len(labels))

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("crohme_training_or_tuning") is not False:
        raise ValueError("frozen formula summary lacks CROHME exclusion attestation")
    if _sha256(data_path) != summary["inputs"]["formulas_valid_sha256"]:
        raise ValueError("frozen formula dataset hash mismatch")
    raw_rows = _jsonl(data_path)
    if any("crohme" in str(row.get("source_partition", "")).casefold() for row in raw_rows):
        raise ValueError("CROHME source row found in diagnostic corpus")
    raw_by_id = {str(row["sample_id"]): row for row in raw_rows}
    rows, targets = _formula_rows(summary, raw_by_id)
    label_to_id = {label: index for index, label in enumerate(labels)}

    per_class: dict[str, Counter] = defaultdict(Counter)
    oov_labels: Counter[str] = Counter()
    top1_confusions: Counter[tuple[str, str]] = Counter()
    top1_rivals_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    top5_miss_detail = []
    rows_by_formula: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_formula[str(row["formula_id"])].append(row)
    exact_group_formulas = 0
    grouping_failure_formulas = 0
    exact_group_tokens = 0
    in_vocab_tokens = 0
    all_top1_hits = all_top5_hits = 0
    top1_hits = top5_hits = 0
    for formula_id, target in targets.items():
        if not target["group_exact"]:
            grouping_failure_formulas += 1
            continue
        exact_group_formulas += 1
        sequence = sorted(rows_by_formula[formula_id], key=lambda row: int(row["context"]["index"]))
        if len(sequence) != len(target["tokens"]):
            raise ValueError(f"group-exact token alignment mismatch: {formula_id}")
        for row, truth in zip(sequence, target["tokens"], strict=True):
            truth = str(truth)
            candidates = [str(token) for token in row["final_topk"]]
            if len(candidates) != 5:
                raise ValueError("expected five HWR candidates per exact-group token")
            top1_hit = candidates[0] == truth
            top5_hit = truth in candidates
            all_top1_hits += int(top1_hit)
            all_top5_hits += int(top5_hit)
            exact_group_tokens += 1
            if not top1_hit:
                top1_confusions[(truth, candidates[0])] += 1
                top1_rivals_by_class[truth][candidates[0]] += 1
            if truth not in label_to_id:
                oov_labels[truth] += 1
                if not top5_hit:
                    top5_miss_detail.append({"label": truth, "train_support": None, "top5": candidates})
                continue
            support = int(train_counts[label_to_id[truth]])
            stats = per_class[truth]
            stats["tokens"] += 1
            stats["top1_hits"] += int(candidates[0] == truth)
            stats["top5_hits"] += int(truth in candidates)
            stats["train_support"] = support
            stats["validation_support"] = int(validation_counts[label_to_id[truth]])
            if top5_hit:
                stats[f"target_rank_{candidates.index(truth) + 1}"] += 1
            in_vocab_tokens += 1
            top1_hits += int(top1_hit)
            top5_hits += int(top5_hit)
            if not top5_hit:
                top5_miss_detail.append({"label": truth, "train_support": support, "top5": candidates})

    grouped = {
        name: {"observed_tokens": 0, "top1_hits": 0, "top5_hits": 0, "observed_classes": set()}
        for _, _, name in SUPPORT_BINS
    }
    for label, stats in per_class.items():
        support_bin = _support_bin(int(stats["train_support"]))
        aggregate = grouped[support_bin]
        aggregate["observed_tokens"] += int(stats["tokens"])
        aggregate["top1_hits"] += int(stats["top1_hits"])
        aggregate["top5_hits"] += int(stats["top5_hits"])
        aggregate["observed_classes"].add(label)

    support_distribution = {
        "rows": int(train_ids.size),
        "classes": len(labels),
        "observed_classes": int(np.count_nonzero(train_counts)),
        "minimum": int(train_counts.min()),
        "median": float(np.median(train_counts)),
        "p10": float(np.quantile(train_counts, 0.10)),
        "maximum": int(train_counts.max()),
        "classes_below_50": int(np.count_nonzero(train_counts < 50)),
        "validation_rows": int(validation_ids.size),
        "validation_observed_classes": int(np.count_nonzero(validation_counts)),
        "validation_min_positive": int(validation_counts[validation_counts > 0].min()),
        "validation_max": int(validation_counts.max()),
    }
    by_support_bin = {}
    for _, _, name in SUPPORT_BINS:
        stats = grouped[name]
        count = int(stats["observed_tokens"])
        by_support_bin[name] = {
            "observed_classes": len(stats["observed_classes"]),
            "tokens": count,
            "top1_hits": int(stats["top1_hits"]),
            "top1_rate": float(stats["top1_hits"] / count) if count else None,
            "top5_hits": int(stats["top5_hits"]),
            "top5_rate": float(stats["top5_hits"] / count) if count else None,
        }
    class_rows = []
    for label, stats in per_class.items():
        count = int(stats["tokens"])
        class_rows.append({
            "label": label,
            "train_support": int(stats["train_support"]),
            "validation_support": int(stats["validation_support"]),
            "support_bin": _support_bin(int(stats["train_support"])),
            "exact_group_tokens": count,
            "top1_hits": int(stats["top1_hits"]),
            "top1_rate": float(stats["top1_hits"] / count),
            "top5_hits": int(stats["top5_hits"]),
            "top5_misses": int(count - stats["top5_hits"]),
            "top5_rate": float(stats["top5_hits"] / count),
            "top1_rivals": dict(sorted(top1_rivals_by_class[label].items(), key=lambda item: (-item[1], item[0]))),
            "target_rank_histogram": {
                str(rank): int(stats[f"target_rank_{rank}"])
                for rank in range(1, 6)
                if stats[f"target_rank_{rank}"]
            },
        })
    class_rows.sort(key=lambda item: (item["train_support"], -item["top5_misses"], item["label"]))

    report = {
        "schema": "aiflow-hwr-class-support-vs-formula-misses/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "consumed_development_diagnostic_only",
        "protocol": {
            "training_performed": False,
            "model_or_threshold_selection": False,
            "crohme_rows_loaded": 0,
            "official_uji_test_scored": False,
            "project_owned_holdout_trained_on": False,
            "formula_set_consumed_development_data": True,
        },
        "provenance": {
            "architecture_probe": str(probe_path),
            "architecture_probe_sha256": _sha256(probe_path),
            "train_label_cache_sha256": _sha256(train_labels_path),
            "validation_label_cache_sha256": _sha256(validation_labels_path),
            "canonical_hwrt_sha256": probe["data"]["canonical_hwrt_sha256"],
            "canonical_uji_sha256": probe["data"]["canonical_uji_sha256"],
            "summary": str(summary_path),
            "summary_sha256": _sha256(summary_path),
            "formula_data_sha256": _sha256(data_path),
        },
        "training_support": support_distribution,
        "formula_diagnostic": {
            "formulas": len(targets),
            "group_exact_formulas": exact_group_formulas,
            "grouping_failure_formulas": grouping_failure_formulas,
            "exact_group_tokens": exact_group_tokens,
            "in_vocabulary_tokens": in_vocab_tokens,
            "in_vocabulary_top1_hits": top1_hits,
            "in_vocabulary_top1_rate": top1_hits / in_vocab_tokens if in_vocab_tokens else None,
            "in_vocabulary_top5_hits": top5_hits,
            "in_vocabulary_top5_misses": in_vocab_tokens - top5_hits,
            "in_vocabulary_top5_rate": top5_hits / in_vocab_tokens if in_vocab_tokens else None,
            "all_tokens_top1_hits": all_top1_hits,
            "all_tokens_top5_hits": all_top5_hits,
            "all_tokens_top5_misses": exact_group_tokens - all_top5_hits,
            "all_tokens_top5_rate": all_top5_hits / exact_group_tokens if exact_group_tokens else None,
            "out_of_vocabulary_targets": dict(sorted(oov_labels.items())),
            "top1_confusions": [
                {"target": target, "predicted_top1": predicted, "count": int(count)}
                for (target, predicted), count in sorted(top1_confusions.items(), key=lambda item: (-item[1], item[0]))[:30]
            ],
            "top5_miss_details": top5_miss_detail,
            "by_training_support_bin": by_support_bin,
            "observed_classes": class_rows,
        },
        "decision": {
            "causal_claim": False,
            "next_use": "Use only to prioritize future collection/augmentation hypotheses; verify on untouched writer/formula acceptance before selecting changes.",
            "promotion_eligible": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "hwr_class_support_formula_miss_audit_complete",
        "report": str(output),
        "train_classes": support_distribution["observed_classes"],
        "train_support_min_median_max": [support_distribution["minimum"], support_distribution["median"], support_distribution["maximum"]],
        "validation_classes": support_distribution["validation_observed_classes"],
        "exact_group_formula_count": exact_group_formulas,
        "top5_misses": exact_group_tokens - all_top5_hits,
        "crohme_rows_loaded": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
