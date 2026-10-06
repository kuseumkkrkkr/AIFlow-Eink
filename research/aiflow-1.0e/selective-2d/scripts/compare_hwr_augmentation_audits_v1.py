#!/usr/bin/env python3
"""Compare two augmentation-only audits bound to identical training-pool rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from statistics import fmean
from typing import Any


GEOMETRY_METRICS = (
    "pairwise_xy_rms_mean",
    "pairwise_xy_rms_p95",
    "affine_aligned_residual_xy_rms_mean",
    "affine_aligned_residual_xy_rms_p95",
    "identical_view_pair_fraction",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _visual_sample_summary(report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    image_qa = report.get("image_sample_verification") or {}
    class_count = int(report.get("class_count", -1))
    entries = image_qa.get("images") or []
    label_entries = [entry for entry in entries if isinstance(entry, dict) and "label" in entry]
    page_entries = [entry for entry in entries if isinstance(entry, dict) and "page_image" in entry]
    labels = [str(entry["label"]) for entry in label_entries]
    if (
        len(labels) != class_count
        or len(set(labels)) != class_count
        or int(image_qa.get("classes_visually_sampled", 0)) != class_count
        or not page_entries
    ):
        raise ValueError(f"visual sample manifest does not cover each class exactly once: {report_path}")

    labels_by_page: dict[int, int] = {}
    for entry in label_entries:
        try:
            page = int(entry["page"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"visual sample entry has no valid page index: {report_path}") from exc
        labels_by_page[page] = labels_by_page.get(page, 0) + 1

    pages: dict[int, dict[str, Any]] = {}
    for entry in page_entries:
        image_path = Path(str(entry["page_image"]))
        if not image_path.is_absolute():
            image_path = report_path.parent / image_path
        image_path = image_path.resolve()
        match = re.search(r"_(\d+)$", image_path.stem)
        if not match:
            raise ValueError(f"visual sample page filename has no page index: {image_path}")
        page = int(match.group(1))
        if page in pages:
            raise ValueError(f"duplicate visual sample page {page}: {report_path}")
        if not image_path.is_file():
            raise FileNotFoundError(f"visual sample page is missing: {image_path}")
        with image_path.open("rb") as stream:
            header = stream.read(24)
        if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            raise ValueError(f"visual sample page is not a valid PNG: {image_path}")
        width, height = struct.unpack(">II", header[16:24])
        if width < 1 or height < 1 or int(entry.get("classes", -1)) != labels_by_page.get(page, 0):
            raise ValueError(f"visual sample page dimensions or class count do not match its manifest: {image_path}")
        pages[page] = {
            "path": str(image_path),
            "sha256": _sha256(image_path),
            "bytes": image_path.stat().st_size,
            "width": width,
            "height": height,
            "classes": int(entry["classes"]),
        }
    if set(pages) != set(labels_by_page) or sum(page["classes"] for page in pages.values()) != class_count:
        raise ValueError(f"visual sample pages do not cover the declared class set: {report_path}")
    return {
        "classes_visually_sampled": class_count,
        "panels_per_class": int(image_qa.get("panels_per_class", 0)),
        "page_count": len(pages),
        "pages": [{"page": page, **pages[page]} for page in sorted(pages)],
    }


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "aiflow-hwr-affine-distillation-experiment/v1":
        raise ValueError(f"unsupported audit schema: {path}")
    if report.get("status") != "pass":
        raise ValueError(f"augmentation data-unit audit did not pass: {path}")
    if report.get("heldout_rows_used") != 0 or report.get("crohme_rows") != 0:
        raise ValueError(f"restricted rows were used by the audit: {path}")
    if report.get("heldout_used_for_selection") is not False or report.get("product_adopted") is not False:
        raise ValueError(f"audit is not diagnostic-only: {path}")
    validation = report.get("data_unit_validation") or {}
    qa_rows = int(validation.get("unique_rows", 0))
    view_count = int(validation.get("augmented_views_per_row", 0))
    if (
        qa_rows < 1
        or view_count < 1
        or int(validation.get("augmented_data_units_checked", -1)) != qa_rows * view_count
        or int(validation.get("unit_failure_total", -1)) != 0
        or validation.get("all_labels_immutable") is not True
        or validation.get("all_nonspatial_channels_identical") is not True
        or validation.get("stroke_count_and_start_positions_preserved") is not True
        or validation.get("finite_xy_in_unit_square") is not True
    ):
        raise ValueError(f"audit data-unit integrity gates are incomplete: {path}")
    image_qa = report.get("image_sample_verification") or {}
    if int(image_qa.get("classes_visually_sampled", 0)) != int(report.get("class_count", -1)):
        raise ValueError(f"audit did not visually cover the full class vocabulary: {path}")
    report["_verified_visual_sample_manifest"] = _visual_sample_summary(report, path)
    report["_verified_geometry_diversity"] = _geometry_diversity_summary(report, path)
    robustness_views = (report.get("teacher_robustness_on_qa_rows") or {}).get("views") or []
    if len(robustness_views) != view_count or any(
        int(view.get("overall", {}).get("rows", -1)) != qa_rows
        for view in robustness_views
    ):
        raise ValueError(f"teacher stability rows do not match the audited data units: {path}")
    provenance = report.get("provenance") or {}
    required = (
        "audit_script_sha256",
        "prepared_manifest_sha256",
        "teacher_checkpoint_sha256",
        "training_cache_sha256",
        "qa_row_indices_sha256",
        "qa_selection_seed",
    )
    if any(not provenance.get(key) for key in required):
        raise ValueError(f"audit is missing paired provenance: {path}")
    return report


def _same_source_rows(left: dict[str, Any], right: dict[str, Any]) -> bool:
    fields = (
        "audit_script_sha256",
        "prepared_manifest_sha256",
        "teacher_checkpoint_sha256",
        "training_cache_sha256",
        "qa_row_indices_sha256",
        "qa_selection_seed",
    )
    return all(left["provenance"][key] == right["provenance"][key] for key in fields)


def _parameter_differences(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, Any]]:
    left_parameters = left.get("parameters") or {}
    right_parameters = right.get("parameters") or {}
    if not isinstance(left_parameters, dict) or not isinstance(right_parameters, dict):
        raise ValueError("paired augmentation audits must contain parameter objects")
    differences = {}
    for name in sorted(set(left_parameters) | set(right_parameters)):
        left_value = left_parameters.get(name)
        right_value = right_parameters.get(name)
        if left_value != right_value:
            differences[name] = {"baseline": left_value, "candidate": right_value}
    return differences


def _view_metrics(report: dict[str, Any]) -> dict[int, dict[str, float]]:
    result = {}
    for view in report["teacher_robustness_on_qa_rows"]["views"]:
        index = int(view["view_index"])
        overall = view["overall"]
        result[index] = {
            "teacher_top1_label_accuracy": float(overall["augmented_teacher_top1_label_accuracy"]),
            "teacher_top5_label_accuracy": float(overall["augmented_teacher_top5_label_accuracy"]),
            "teacher_top1_agreement": float(overall["teacher_top1_agreement_with_source"]),
        }
    if not result:
        raise ValueError("audit contains no teacher robustness views")
    return result


def _geometry_diversity_summary(report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    diversity = report.get("augmentation_diversity_geometry_only") or {}
    validation = report["data_unit_validation"]
    qa_rows = int(validation["unique_rows"])
    view_count = int(validation["augmented_views_per_row"])
    expected_pairs = view_count * (view_count - 1) // 2
    overall = diversity.get("overall") or {}
    families = diversity.get("by_class_family") or {}
    if (
        diversity.get("status") != "measured"
        or int(diversity.get("view_count", -1)) != view_count
        or int(diversity.get("pairs_per_source_row", -1)) != expected_pairs
        or int(overall.get("source_rows", -1)) != qa_rows
    ):
        raise ValueError(f"geometry diversity summary does not match audited rows/views: {report_path}")
    if set(families) != set(report.get("class_family_coverage", {})):
        raise ValueError(f"geometry diversity does not cover every class family: {report_path}")

    def metrics(summary: dict[str, Any]) -> dict[str, float]:
        result = {name: float(summary[name]) for name in GEOMETRY_METRICS}
        if any(not math.isfinite(value) for value in result.values()):
            raise ValueError(f"geometry diversity contains a non-finite metric: {report_path}")
        if any(result[name] < 0.0 for name in GEOMETRY_METRICS[:-1]):
            raise ValueError(f"geometry diversity contains a negative distance: {report_path}")
        if not 0.0 <= result["identical_view_pair_fraction"] <= 1.0:
            raise ValueError(f"geometry duplicate fraction is outside [0, 1]: {report_path}")
        return result

    family_summary = {}
    for family, value in families.items():
        if int(value.get("source_rows", 0)) < 1:
            raise ValueError(f"geometry diversity family has no QA rows ({family}): {report_path}")
        family_summary[family] = {
            "source_rows": int(value["source_rows"]),
            **metrics(value),
        }
    if sum(value["source_rows"] for value in family_summary.values()) != qa_rows:
        raise ValueError(f"geometry family row counts do not cover the QA cohort: {report_path}")
    return {
        "scope": diversity.get("scope"),
        "view_count": view_count,
        "pairs_per_source_row": expected_pairs,
        "overall": metrics(overall),
        "by_class_family": family_summary,
    }


def _arm_summary(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    views = _view_metrics(report)
    parameter_counts = {name: 0 for name in ("rotation", "axis_scale", "shear", "elastic")}
    any_wide_rows = 0
    for view in report["data_unit_validation"]["per_view_transform_diagnostics"]:
        severity = view.get("severity_mixture") or {}
        any_wide_rows += int(severity.get("wide_selected_any_parameter_rows", 0))
        for name, count in (severity.get("wide_selected_rows_by_parameter") or {}).items():
            if name in parameter_counts:
                parameter_counts[name] += int(count)
    return {
        "report": str(path.resolve()),
        "report_sha256": _sha256(path),
        "augmentation_engine": report["augmentation_engine"],
        "parameters": report["parameters"],
        "data_unit_failures": int(report["data_unit_validation"]["unit_failure_total"]),
        "augmented_data_units_checked": int(report["data_unit_validation"]["augmented_data_units_checked"]),
        "wide_selected_rows_by_parameter_total": parameter_counts,
        "wide_selected_any_parameter_rows_total_across_views": any_wide_rows,
        "teacher_view_metrics": views,
        "teacher_view_metric_means": {
            metric: fmean(view[metric] for view in views.values())
            for metric in (
                "teacher_top1_label_accuracy",
                "teacher_top5_label_accuracy",
                "teacher_top1_agreement",
            )
        },
        "geometry_diversity": report["_verified_geometry_diversity"],
        "verified_visual_sample_manifest": report["_verified_visual_sample_manifest"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-parameter-difference", action="append", default=[], metavar="PARAMETER",
        help="explicitly declare a parameter changed by this comparison; all other audited parameters must match",
    )
    args = parser.parse_args()
    baseline_path = args.baseline.resolve()
    candidate_path = args.candidate.resolve()
    output_path = args.output.resolve()
    if baseline_path == candidate_path or output_path in {baseline_path, candidate_path}:
        parser.error("baseline, candidate, and output must be three distinct paths")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite comparison report: {output_path}")

    baseline = _load_report(baseline_path)
    candidate = _load_report(candidate_path)
    if not _same_source_rows(baseline, candidate):
        raise ValueError("paired audits do not share identical prepared data, teacher, and QA rows")
    parameter_differences = _parameter_differences(baseline, candidate)
    parameter_names = set((baseline.get("parameters") or {})) | set((candidate.get("parameters") or {}))
    unknown_allowlist = set(args.allow_parameter_difference) - parameter_names
    if unknown_allowlist:
        raise ValueError(f"parameter allowlist names are absent from both audits: {sorted(unknown_allowlist)}")
    unapproved_parameter_differences = set(parameter_differences) - set(args.allow_parameter_difference)
    if unapproved_parameter_differences:
        raise ValueError(
            "paired audits differ in parameters not explicitly declared as experiment factors: "
            f"{sorted(unapproved_parameter_differences)}"
        )
    if baseline["augmentation_engine"] != candidate["augmentation_engine"]:
        raise ValueError("paired audits use different augmentation engines")
    if baseline["class_count"] != candidate["class_count"]:
        raise ValueError("paired audits cover different class vocabularies")
    if baseline["training_pool_rows"] != candidate["training_pool_rows"]:
        raise ValueError("paired audits use different training-pool sizes")
    if baseline["data_unit_validation"]["unique_rows"] != candidate["data_unit_validation"]["unique_rows"]:
        raise ValueError("paired audits use different QA row counts")

    baseline_views = _view_metrics(baseline)
    candidate_views = _view_metrics(candidate)
    if set(baseline_views) != set(candidate_views):
        raise ValueError("paired audits contain different augmentation view indices")
    baseline_arm = _arm_summary(baseline_path, baseline)
    candidate_arm = _arm_summary(candidate_path, candidate)
    metric_names = (
        "teacher_top1_label_accuracy",
        "teacher_top5_label_accuracy",
        "teacher_top1_agreement",
    )
    comparison = {
        "schema": "aiflow-hwr-augmentation-audit-pair/v2",
        "status": "paired_augmentation_integrity_teacher_stability_geometry_and_visual_assets_compared",
        "comparison_script_sha256": _sha256(Path(__file__).resolve()),
        "paired_provenance": {
            **baseline["provenance"],
            "identical_across_arms": True,
        },
        "paired_parameter_contract": {
            "allowed_changed_parameters": sorted(set(args.allow_parameter_difference)),
            "changed_parameters": parameter_differences,
            "all_other_parameters_identical": True,
        },
        "scope": {
            "class_count": int(baseline["class_count"]),
            "training_pool_rows": int(baseline["training_pool_rows"]),
            "qa_rows": int(baseline["data_unit_validation"]["unique_rows"]),
            "views_per_row": int(baseline["data_unit_validation"]["augmented_views_per_row"]),
            "crohme_rows": 0,
            "heldout_rows_used": 0,
        },
        "baseline": baseline_arm,
        "candidate": candidate_arm,
        "candidate_minus_baseline_mean": {
            metric: candidate_arm["teacher_view_metric_means"][metric]
            - baseline_arm["teacher_view_metric_means"][metric]
            for metric in metric_names
        },
        "candidate_minus_baseline_geometry_diversity": {
            "overall": {
                metric: candidate_arm["geometry_diversity"]["overall"][metric]
                - baseline_arm["geometry_diversity"]["overall"][metric]
                for metric in GEOMETRY_METRICS
            },
            "by_class_family": {
                family: {
                    metric: candidate_arm["geometry_diversity"]["by_class_family"][family][metric]
                    - baseline_arm["geometry_diversity"]["by_class_family"][family][metric]
                    for metric in GEOMETRY_METRICS
                }
                for family in baseline_arm["geometry_diversity"]["by_class_family"]
            },
        },
        "interpretation_boundary": "augmentation integrity, frozen-teacher stability, and visual asset integrity only; no student training, recognition-performance claim, model selection, or product promotion",
        "product_adopted": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(comparison, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "event": "augmentation_audits_paired",
        "status": comparison["status"],
        "same_qa_rows": True,
        "teacher_metric_delta": comparison["candidate_minus_baseline_mean"],
        "geometry_metric_delta": comparison["candidate_minus_baseline_geometry_diversity"]["overall"],
        "output": str(output_path),
        "product_adopted": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
