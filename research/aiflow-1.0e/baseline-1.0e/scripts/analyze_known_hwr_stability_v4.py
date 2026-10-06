#!/usr/bin/env python3
"""이미 관측된 186개 글자에서 v4 변화를 진단하되 승인 근거로 만들지 않는다."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

SCHEMA = "aiflow-known-hwr-stability-diagnostic/v4"
FAMILIES = {
    "vertical_slash": ("1", "|", "/"),
    "circle": ("0", "O", "o"),
    "cross": ("x", "\\times"),
}


def _sha256(path: Path) -> str:
    """입력·출력 계보 검증용 SHA-256을 스트리밍으로 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _d_path(path: Path, kind: str, *, must_exist: bool = True) -> Path:
    """진단 파일을 D: 내부로 제한하고 존재 계약을 확인한다."""

    resolved = path.resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{kind} must remain on D:: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"missing {kind}: {resolved}")
    return resolved


def _rows(path: Path) -> dict[str, dict]:
    """전체 후보 캐시에서 이미 관측된 new_writer 186글자만 읽는다."""

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    selected = {
        str(row["record_id"]): row
        for row in rows
        if row.get("evaluation_partition") == "new_writer"
    }
    if len(selected) != 186:
        raise ValueError(f"expected 186 known diagnostic glyphs, got {len(selected)}")
    return selected


def _metrics(rows: dict[str, dict]) -> dict:
    """문자·식·writer·클래스·동형계열 지표를 동일 분할에서 계산한다."""

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
        """한 하위 분할의 문자 Top-1과 Top-5를 계산한다."""

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
        "records": len(values),
        "writers": len(writers),
        "formulas": len(formulae),
        "top1": sum(top1) / len(values),
        "top5": sum(top5) / len(values),
        "outside_top5": len(values) - sum(top5),
        "formula_exact": sum(
            all(row["final_topk"][0] == row["label"] for row in sequence)
            for sequence in formulae.values()
        ) / len(formulae),
        "writer_macro_top1": sum(row["top1"] for row in by_writer.values()) / len(by_writer),
        "writer_macro_top5": sum(row["top5"] for row in by_writer.values()) / len(by_writer),
        "strict_macro_top1": sum(row["top1"] for row in by_label.values()) / len(by_label),
        "strict_macro_top5": sum(row["top5"] for row in by_label.values()) / len(by_label),
        "by_writer": by_writer,
        "by_label": by_label,
        "by_family": by_family,
    }


def _paired(base: dict[str, dict], candidate: dict[str, dict]) -> tuple[dict, list[dict]]:
    """동일 레코드의 Top-1·Top-5 개선과 회귀를 분리해 센다."""

    summary = {
        "changed_candidate_sets": 0,
        "top1_improved": 0,
        "top1_regressed": 0,
        "top5_rescued": 0,
        "top5_lost": 0,
    }
    changes = []
    for record_id in sorted(base):
        old, new = base[record_id], candidate[record_id]
        old_ok = old["final_topk"][0] == old["label"]
        new_ok = new["final_topk"][0] == new["label"]
        old_top5 = old["label"] in old["final_topk"]
        new_top5 = new["label"] in new["final_topk"]
        if old["final_topk"] != new["final_topk"]:
            summary["changed_candidate_sets"] += 1
            changes.append({
                "record_id": record_id,
                "formula_id": new["formula_id"],
                "writer_group": new["writer_group"],
                "truth": new["label"],
                "baseline_top1": old["final_topk"][0],
                "candidate_top1": new["final_topk"][0],
                "improved": not old_ok and new_ok,
                "regressed": old_ok and not new_ok,
                "baseline_top5_hit": old_top5,
                "candidate_top5_hit": new_top5,
            })
        summary["top1_improved"] += int(not old_ok and new_ok)
        summary["top1_regressed"] += int(old_ok and not new_ok)
        summary["top5_rescued"] += int(not old_top5 and new_top5)
        summary["top5_lost"] += int(old_top5 and not new_top5)
    return summary, changes


def main() -> int:
    """계보를 검증하고 오염 표시가 고정된 진단 보고서를 작성한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--stability-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_path = _d_path(args.baseline, "known baseline candidate cache")
    candidate_path = _d_path(args.candidate, "known v4 candidate cache")
    build_path = _d_path(args.candidate_build, "known v4 candidate build")
    stability_path = _d_path(args.stability_report, "v4 stability report")
    output = _d_path(args.output, "known diagnostic output", must_exist=False)
    if output.exists():
        parser.error(f"refusing to overwrite output: {output}")

    baseline = _rows(baseline_path)
    candidate = _rows(candidate_path)
    if set(baseline) != set(candidate):
        raise ValueError("known diagnostic record coverage differs")
    immutable = ("label", "formula_id", "writer_group", "evaluation_partition")
    for record_id in baseline:
        if any(baseline[record_id].get(key) != candidate[record_id].get(key) for key in immutable):
            raise ValueError(f"known diagnostic truth contract changed: {record_id}")

    build = json.loads(build_path.read_text(encoding="utf-8"))
    stability = json.loads(stability_path.read_text(encoding="utf-8"))
    checkpoint = stability.get("checkpoint", {})
    if build.get("checkpoints", {}).get("product_checkpoint_sha256") != checkpoint.get("sha256"):
        raise ValueError("known diagnostic cache does not use the frozen v4 checkpoint")
    policy = stability.get("selection_policy", {})
    if policy.get("selection_used_known_fresh_acceptance") or policy.get("selection_used_crohme"):
        raise ValueError("v4 selection boundary is contaminated in its own manifest")

    before, after = _metrics(baseline), _metrics(candidate)
    paired, changes = _paired(baseline, candidate)
    family_top5_nonregression = all(
        after["by_family"][name]["top5"] >= row["top5"]
        for name, row in before["by_family"].items()
    )
    diagnostic_checks = {
        "top1_nonregression": after["top1"] >= before["top1"],
        "top5_nonregression": after["top5"] >= before["top5"],
        "formula_exact_nonregression": after["formula_exact"] >= before["formula_exact"],
        "strict_macro_top5_nonregression": after["strict_macro_top5"] >= before["strict_macro_top5"],
        "family_top5_nonregression": family_top5_nonregression,
        "top1_regressions_zero": paired["top1_regressed"] == 0,
    }
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "acceptance_valid": False,
        "contaminated_by_prior_observation": True,
        "usable_for_model_selection": False,
        "usable_for_product_promotion": False,
        "reason": (
            "the same 186 glyphs were opened during earlier v3 evaluation; "
            "this comparison is a regression diagnostic only"
        ),
        "baseline": before,
        "candidate": after,
        "delta": {
            "top1": after["top1"] - before["top1"],
            "top5": after["top5"] - before["top5"],
            "formula_exact": after["formula_exact"] - before["formula_exact"],
            "strict_macro_top5": after["strict_macro_top5"] - before["strict_macro_top5"],
            **paired,
        },
        "diagnostic_checks": diagnostic_checks,
        "all_diagnostic_checks_passed": all(diagnostic_checks.values()),
        "changes": changes,
        "inputs": {
            "baseline": str(baseline_path),
            "baseline_sha256": _sha256(baseline_path),
            "candidate": str(candidate_path),
            "candidate_sha256": _sha256(candidate_path),
            "candidate_build": str(build_path),
            "candidate_build_sha256": _sha256(build_path),
            "stability_report": str(stability_path),
            "stability_report_sha256": _sha256(stability_path),
            "checkpoint_sha256": checkpoint.get("sha256"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({
        "event": "known_hwr_stability_diagnostic_complete",
        "acceptance_valid": False,
        "top1": after["top1"],
        "top5": after["top5"],
        "formula_exact": after["formula_exact"],
        "improved": paired["top1_improved"],
        "regressed": paired["top1_regressed"],
        "output": str(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
