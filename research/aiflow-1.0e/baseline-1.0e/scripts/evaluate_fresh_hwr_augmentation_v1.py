#!/usr/bin/env python3
"""Compare a frozen augmented HWR candidate on untouched project writers."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "vertical_slash": ("1", "|", "/"),
    "circle": ("0", "O", "o"),
    "cross": ("x", "\\times"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _d_path(path: Path, kind: str, *, must_exist: bool = True) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{kind} must remain on D:: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"missing {kind}: {resolved}")
    return resolved


def _rows(path: Path) -> dict[str, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    selected = {
        str(row["record_id"]): row for row in rows
        if row.get("evaluation_partition") == "new_writer"
    }
    if len(selected) != 186:
        raise ValueError(f"expected 186 untouched glyphs, got {len(selected)}")
    return selected


def _metrics(rows: dict[str, dict]) -> dict:
    values = list(rows.values())
    top1 = [row["final_topk"][0] == row["label"] for row in values]
    top5 = [row["label"] in row["final_topk"] for row in values]
    formulae: dict[str, list[dict]] = defaultdict(list)
    writers: dict[str, list[dict]] = defaultdict(list)
    labels: dict[str, list[dict]] = defaultdict(list)
    for row in values:
        formulae[str(row["formula_id"])].append(row)
        writers[str(row["writer_group"])].append(row)
        labels[str(row["label"])].append(row)

    def score(subset: list[dict]) -> dict:
        return {
            "records": len(subset),
            "top1": sum(row["final_topk"][0] == row["label"] for row in subset) / len(subset),
            "top5": sum(row["label"] in row["final_topk"] for row in subset) / len(subset),
        }

    by_writer = {key: score(value) for key, value in sorted(writers.items())}
    by_label = {key: score(value) for key, value in sorted(labels.items())}
    by_family = {}
    for family, members in FAMILIES.items():
        subset = [row for row in values if row["label"] in members]
        if subset:
            by_family[family] = score(subset)
    return {
        "records": len(values), "writers": len(writers), "formulas": len(formulae),
        "top1": sum(top1) / len(values), "top5": sum(top5) / len(values),
        "outside_top5": len(values) - sum(top5),
        "formula_exact": sum(
            all(row["final_topk"][0] == row["label"] for row in sequence)
            for sequence in formulae.values()
        ) / len(formulae),
        "writer_macro_top1": sum(row["top1"] for row in by_writer.values()) / len(by_writer),
        "writer_macro_top5": sum(row["top5"] for row in by_writer.values()) / len(by_writer),
        "strict_macro_top1": sum(row["top1"] for row in by_label.values()) / len(by_label),
        "strict_macro_top5": sum(row["top5"] for row in by_label.values()) / len(by_label),
        "by_writer": by_writer, "by_label": by_label, "by_family": by_family,
    }


def _self_test() -> None:
    rows = {
        "a": {"record_id": "a", "label": "1", "final_topk": ["1", "|", "/", "I", "l"], "formula_id": "f", "writer_group": "w"},
        "b": {"record_id": "b", "label": "0", "final_topk": ["O", "0", "o", "\\circ", "6"], "formula_id": "g", "writer_group": "w"},
    }
    metrics = _metrics(rows)
    assert metrics["top1"] == 0.5 and metrics["top5"] == 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-build", type=Path)
    parser.add_argument("--augmentation-report", type=Path)
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass"}))
        return 0
    if any(value is None for value in (
        args.baseline, args.candidate, args.candidate_build,
        args.augmentation_report, args.freeze_manifest, args.output,
    )):
        parser.error("all input and output paths are required")

    baseline_path = _d_path(args.baseline, "baseline candidate cache")
    candidate_path = _d_path(args.candidate, "augmented candidate cache")
    build_path = _d_path(args.candidate_build, "candidate build report")
    training_path = _d_path(args.augmentation_report, "augmentation report")
    freeze_path = _d_path(args.freeze_manifest, "fresh acceptance freeze")
    output = _d_path(args.output, "output", must_exist=False)
    if output.exists():
        parser.error(f"refusing to overwrite output: {output}")

    baseline = _rows(baseline_path)
    candidate = _rows(candidate_path)
    if set(baseline) != set(candidate):
        raise ValueError("baseline/candidate untouched record coverage differs")
    for key in baseline:
        immutable = ("label", "formula_id", "writer_group", "evaluation_partition")
        if any(baseline[key].get(name) != candidate[key].get(name) for name in immutable):
            raise ValueError(f"untouched truth contract changed: {key}")

    build = json.loads(build_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    checkpoint = training.get("checkpoint", {})
    if build.get("checkpoints", {}).get("product_checkpoint_sha256") != checkpoint.get("sha256"):
        raise ValueError("fresh candidate cache does not use the frozen augmented checkpoint")
    selection = training.get("selection_policy", {})
    if selection.get("selection_used_fresh_acceptance") or selection.get("selection_used_crohme"):
        raise ValueError("augmented checkpoint selection is not untouched")
    if freeze.get("training_performed") or not freeze.get("gates", {}).get("ready_for_overall_acceptance_evaluation"):
        raise ValueError("fresh acceptance freeze contract is invalid")

    before = _metrics(baseline)
    after = _metrics(candidate)
    changes = []
    for record_id in sorted(candidate):
        old, new = baseline[record_id], candidate[record_id]
        old_ok = old["final_topk"][0] == old["label"]
        new_ok = new["final_topk"][0] == new["label"]
        if old["final_topk"] != new["final_topk"]:
            changes.append({
                "record_id": record_id, "formula_id": new["formula_id"],
                "writer_group": new["writer_group"], "truth": new["label"],
                "baseline_top1": old["final_topk"][0], "candidate_top1": new["final_topk"][0],
                "improved": not old_ok and new_ok, "regressed": old_ok and not new_ok,
                "baseline_top5_hit": old["label"] in old["final_topk"],
                "candidate_top5_hit": new["label"] in new["final_topk"],
            })
    improved = sum(row["improved"] for row in changes)
    regressed = sum(row["regressed"] for row in changes)
    family_top5_nonregression = all(
        after["by_family"][name]["top5"] >= row["top5"]
        for name, row in before["by_family"].items()
    )
    gates = {
        "top1_nonregression": after["top1"] >= before["top1"],
        "top5_nonregression": after["top5"] >= before["top5"],
        "formula_exact_nonregression": after["formula_exact"] >= before["formula_exact"],
        "strict_macro_top5_nonregression": after["strict_macro_top5"] >= before["strict_macro_top5"],
        "family_top5_nonregression": family_top5_nonregression,
        "top1_regressions_zero": regressed == 0,
        "external_technical_gate": bool(training.get("external_technical_nonregression", {}).get("passed")),
        "training_boundary_zero_evaluation_rows": bool(training.get("training_data_guard", {}).get("passed")),
    }
    report = {
        "schema": "aiflow-fresh-hwr-augmentation-acceptance/v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "selection_used_fresh_acceptance": False,
        "baseline": before, "candidate": after,
        "delta": {
            "top1": after["top1"] - before["top1"],
            "top5": after["top5"] - before["top5"],
            "formula_exact": after["formula_exact"] - before["formula_exact"],
            "outside_top5": after["outside_top5"] - before["outside_top5"],
            "changed_candidate_sets": len(changes), "improved_top1": improved,
            "regressed_top1": regressed,
        },
        "gates": gates, "passed": all(gates.values()), "changes": changes,
        "inputs": {
            "baseline": str(baseline_path), "baseline_sha256": _sha256(baseline_path),
            "candidate": str(candidate_path), "candidate_sha256": _sha256(candidate_path),
            "candidate_build_sha256": _sha256(build_path),
            "augmentation_report_sha256": _sha256(training_path),
            "freeze_manifest_sha256": _sha256(freeze_path),
            "checkpoint_sha256": checkpoint.get("sha256"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "event": "fresh_hwr_augmentation_acceptance_complete", "passed": report["passed"],
        "top1": after["top1"], "top5": after["top5"],
        "improved": improved, "regressed": regressed, "output": str(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
