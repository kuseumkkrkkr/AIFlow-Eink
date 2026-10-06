#!/usr/bin/env python3
"""Post-freeze comparison of two raw formula-complete validation runs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from audit_replay_protocol_v1 import _crohme_sample


SCHEMA = "aiflow-final-grouping-delta-analysis/v1"


def _rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _bin(strokes: int) -> str:
    if strokes < 8:
        return "01-07"
    if strokes < 16:
        return "08-15"
    if strokes < 24:
        return "16-23"
    if strokes < 32:
        return "24-31"
    return "32+"


def _count_error(row: dict) -> str:
    if row["grouping_exact"]:
        return "exact"
    if int(row["groups"]) < int(row["truth_groups"]):
        return "under_grouped_overmerge_proxy"
    if int(row["groups"]) > int(row["truth_groups"]):
        return "over_grouped_oversplit_proxy"
    return "equal_count_wrong_partition"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--crohme", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_dir = args.baseline.resolve(); candidate_dir = args.candidate.resolve()
    crohme = args.crohme.resolve(); output = args.output.resolve()
    if any(path.drive.upper() != "D:" for path in (baseline_dir, candidate_dir, crohme, output)):
        parser.error("all paths must remain on D:")
    if output.exists() or any(not path.exists() for path in (baseline_dir, candidate_dir, crohme)):
        parser.error("inputs must exist and output must be new")
    baseline_report = json.loads((baseline_dir / "validation_report.json").read_text(encoding="utf-8"))
    candidate_report = json.loads((candidate_dir / "validation_report.json").read_text(encoding="utf-8"))
    if any(candidate_report.get(key) is not False for key in (
        "training_performed", "selection_performed", "threshold_tuning_performed",
    )) or int(candidate_report.get("crohme_gradient_updates", -1)) != 0:
        raise ValueError("candidate is not a frozen validation-only run")
    baseline_rows = {row["formula_id"]: row for row in _rows(baseline_dir / "formula_results.jsonl.gz")}
    candidate_rows = {row["formula_id"]: row for row in _rows(candidate_dir / "formula_results.jsonl.gz")}
    if set(baseline_rows) != set(candidate_rows) or len(candidate_rows) != 769:
        raise ValueError("validation row coverage differs")
    details = []; bins: dict[str, Counter] = {}; long_errors = {
        "baseline": Counter(), "candidate": Counter(),
    }
    for formula_id in sorted(candidate_rows):
        sample = _crohme_sample(crohme / formula_id)
        if sample is None:
            raise ValueError(f"missing source sample: {formula_id}")
        strokes = len(sample["strokes"]); band = _bin(strokes)
        baseline = baseline_rows[formula_id]; candidate = candidate_rows[formula_id]
        bins.setdefault(band, Counter())["formulas"] += 1
        bins[band]["baseline_exact"] += bool(baseline["grouping_exact"])
        bins[band]["candidate_exact"] += bool(candidate["grouping_exact"])
        bins[band]["improved"] += bool(candidate["grouping_exact"] and not baseline["grouping_exact"])
        bins[band]["regressed"] += bool(baseline["grouping_exact"] and not candidate["grouping_exact"])
        if strokes >= 16:
            long_errors["baseline"][_count_error(baseline)] += 1
            long_errors["candidate"][_count_error(candidate)] += 1
        if bool(baseline["grouping_exact"]) != bool(candidate["grouping_exact"]):
            details.append({
                "formula_id": formula_id, "strokes": strokes,
                "truth_groups": int(candidate["truth_groups"]),
                "baseline_groups": int(baseline["groups"]),
                "candidate_groups": int(candidate["groups"]),
                "change": "improved" if candidate["grouping_exact"] else "regressed",
                "baseline_count_error": _count_error(baseline),
                "candidate_count_error": _count_error(candidate),
            })
    improved = [row for row in details if row["change"] == "improved"]
    regressed = [row for row in details if row["change"] == "regressed"]
    bin_rows = []
    for band in ("01-07", "08-15", "16-23", "24-31", "32+"):
        values = bins.get(band, Counter()); total = max(values["formulas"], 1)
        bin_rows.append({
            "stroke_band": band, "formulas": values["formulas"],
            "baseline_exact": values["baseline_exact"],
            "baseline_rate": values["baseline_exact"] / total,
            "candidate_exact": values["candidate_exact"],
            "candidate_rate": values["candidate_exact"] / total,
            "improved": values["improved"], "regressed": values["regressed"],
        })
    report = {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "post_freeze_analysis_only",
        "training_performed": False, "selection_performed": False,
        "threshold_tuning_performed": False, "gradient_updates": 0,
        "overall": {
            "formulas": 769,
            "baseline_grouping_exact": baseline_report["scores"]["grouping_exact"],
            "candidate_grouping_exact": candidate_report["scores"]["grouping_exact"],
            "absolute_point_delta": (
                candidate_report["scores"]["grouping_exact"]
                - baseline_report["scores"]["grouping_exact"]
            ),
            "improved_formulas": len(improved), "regressed_formulas": len(regressed),
            "net_exact_formulas": len(improved) - len(regressed),
        },
        "by_stroke_band": bin_rows,
        "long_formula_count_error_proxy": {
            name: dict(sorted(values.items())) for name, values in long_errors.items()
        },
        "changed_formulas": details,
        "formula_scores": {
            "baseline": baseline_report["scores"],
            "candidate": candidate_report["scores"],
        },
        "contracts": candidate_report["contracts"],
        "interpretation": [
            "Only formulas with at least 16 input strokes were eligible for the new grouping blend.",
            "Group-count direction is a proxy; equal-count wrong partitions require stroke-level ownership review.",
            "This analysis occurred after the single frozen validation run and was not used to tune the candidate.",
        ],
    }
    output.mkdir(parents=True)
    json_path = output / "grouping_delta_analysis.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    lines = [
        "# Final frozen grouping delta", "",
        "- Training/selection/threshold tuning on validation data: **0**",
        f"- Grouping exact: **{baseline_report['scores']['grouping_exact']:.2%} → {candidate_report['scores']['grouping_exact']:.2%}**",
        f"- Improved/regressed/net formulas: **{len(improved)} / {len(regressed)} / {len(improved)-len(regressed):+d}**",
        f"- Strict formula proxy: **{baseline_report['scores']['strict_group_layout_relation_character_exact']:.2%} → {candidate_report['scores']['strict_group_layout_relation_character_exact']:.2%}**",
        "", "## Stroke-count bands", "",
        "| Input strokes | N | Baseline exact | Candidate exact | Improved | Regressed |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bin_rows:
        lines.append(
            f"| {row['stroke_band']} | {row['formulas']} | {row['baseline_rate']:.2%} | "
            f"{row['candidate_rate']:.2%} | {row['improved']} | {row['regressed']} |"
        )
    lines.extend([
        "", "## Decision", "",
        "The conservative long-formula blend improves grouping modestly, but does not improve strict formula exactness. "
        "It remains shadow-only; a larger fresh, project-owned long/2D grouping corpus is required before product promotion.",
    ])
    (output / "GROUPING_DELTA_ANALYSIS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output), "overall": report["overall"],
        "long_formula_count_error_proxy": report["long_formula_count_error_proxy"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
