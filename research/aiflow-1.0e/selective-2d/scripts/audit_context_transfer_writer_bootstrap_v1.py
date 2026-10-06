#!/usr/bin/env python3
"""Writer-cluster audit of saved context-transfer predictions on Fast groups."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "aiflow-hwr-context-transfer-writer-bootstrap/v2"
BASELINE_STAGE = "fast_fence_infix_equation"
CHALLENGER_STAGE = "context_fence_infix_equation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _writer_group(writer_id: str) -> str:
    payload = json.dumps(
        ["project_owned_writer", writer_id], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _ink_signatures(formula: dict[str, Any]) -> dict[str, str]:
    strokes = sorted(formula["strokes"], key=lambda row: int(row["order"]))
    raw_xy: list[list[list[float]]] = []
    raw_xyt: list[list[list[float]]] = []
    all_xy: list[tuple[float, float]] = []
    for stroke in strokes:
        xy, xyt = [], []
        for point in stroke["points"]:
            x, y = float(point["x"]), float(point["y"])
            xy.append([x, y])
            xyt.append([x, y, float(point["t_ms"])])
            all_xy.append((x, y))
        raw_xy.append(xy)
        raw_xyt.append(xyt)
    left = min(point[0] for point in all_xy)
    top = min(point[1] for point in all_xy)
    scale = max(max(point[0] for point in all_xy) - left, max(point[1] for point in all_xy) - top, 1e-12)
    normalized = [
        [[round((float(point["x"]) - left) / scale, 6), round((float(point["y"]) - top) / scale, 6)]
         for point in stroke["points"]]
        for stroke in strokes
    ]

    def digest(value: Any) -> str:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    return {
        "raw_xy": digest(raw_xy),
        "raw_xyt": digest(raw_xyt),
        "normalized_xy": digest(normalized),
    }


def _paired_writer_bootstrap(
    rows: list[dict[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    writers = sorted({row["writer_id"] for row in rows})
    if not writers:
        raise ValueError("writer-cluster bootstrap requires at least one writer")
    grouped = {writer: [row for row in rows if row["writer_id"] == writer] for writer in writers}
    deltas = np.empty(iterations, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for index in range(iterations):
        sampled = rng.choice(writers, size=len(writers), replace=True)
        numerator = denominator = 0
        for writer in sampled:
            cluster = grouped[str(writer)]
            denominator += len(cluster)
            numerator += sum(int(row["challenger_exact"]) - int(row["baseline_exact"]) for row in cluster)
        deltas[index] = numerator / denominator
    baseline = sum(row["baseline_exact"] for row in rows)
    challenger = sum(row["challenger_exact"] for row in rows)
    return {
        "formulas": len(rows),
        "writers": len(writers),
        "baseline_exact": baseline,
        "challenger_exact": challenger,
        "transitions": _transitions(rows),
        "delta_rate_pp_point_estimate": (challenger - baseline) / len(rows) * 100,
        "delta_rate_pp_95_interval": [
            float(np.quantile(deltas, 0.025) * 100), float(np.quantile(deltas, 0.975) * 100)
        ],
        "writer_rows": [
            {
                "writer_id": writer,
                "formulas": len(grouped[writer]),
                "baseline_exact": sum(row["baseline_exact"] for row in grouped[writer]),
                "challenger_exact": sum(row["challenger_exact"] for row in grouped[writer]),
                "delta_exact": sum(
                    int(row["challenger_exact"]) - int(row["baseline_exact"]) for row in grouped[writer]
                ),
            }
            for writer in writers
        ],
    }


def _transitions(rows: list[dict[str, Any]]) -> dict[str, int]:
    result = {"both_exact": 0, "baseline_only": 0, "challenger_only": 0, "both_wrong": 0}
    for row in rows:
        baseline, challenger = row["baseline_exact"], row["challenger_exact"]
        key = "both_exact" if baseline and challenger else "baseline_only" if baseline else "challenger_only" if challenger else "both_wrong"
        result[key] += 1
    return result


def audit(
    context_report_path: Path,
    context_model_report_path: Path,
    trace_summary_path: Path,
    formulas_path: Path,
    context_checkpoint_path: Path,
    training_candidates_path: Path,
    training_formulas_path: Path,
    acceptance_ownership_path: Path,
    acceptance_manifest_path: Path,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    context_report = _json(context_report_path)
    context_model_report = _json(context_model_report_path)
    trace_summary = _json(trace_summary_path)
    formulas = _jsonl(formulas_path)
    training_formulas = _jsonl(training_formulas_path)
    training_candidates = _jsonl_gz(training_candidates_path)
    acceptance_ownership = _jsonl(acceptance_ownership_path)
    acceptance_manifest = _json(acceptance_manifest_path)
    formula_by_id = {str(row["sample_id"]): row for row in formulas}
    if len(formula_by_id) != len(formulas):
        raise ValueError("duplicate sample_id in formula source")
    training_formula_by_id = {str(row["sample_id"]): row for row in training_formulas}
    if len(training_formula_by_id) != len(training_formulas):
        raise ValueError("duplicate sample_id in training formula source")

    expected_formula_hash = trace_summary["reproducibility"]["inputs"]["dataset_files"]["data/formulas_valid.jsonl"]
    formula_hash = _sha256(formulas_path)
    if formula_hash != expected_formula_hash:
        raise ValueError("formula source hash differs from layer trace summary")
    expected_context_hash = context_report["checkpoints"]["context_checkpoint_sha256"]
    context_hash = _sha256(context_checkpoint_path)
    if context_hash != expected_context_hash:
        raise ValueError("context checkpoint hash differs from context-transfer report")
    if context_hash != context_model_report["checkpoint"]["sha256"]:
        raise ValueError("context checkpoint hash differs from its training report")
    crohme_excluded = "never used for training or selection" in context_model_report["data_admission"].get("crohme", "")
    if not crohme_excluded:
        raise ValueError("context model report does not verify the CROHME training/selection exclusion")
    expected_training_candidates_hash = context_model_report["provenance"]["direct_candidate_sha256"]
    training_candidates_hash = _sha256(training_candidates_path)
    if training_candidates_hash != expected_training_candidates_hash:
        raise ValueError("direct training candidate hash differs from context model report")
    acceptance_ownership_hash = _sha256(acceptance_ownership_path)
    if acceptance_ownership_hash != acceptance_manifest["artifacts"]["fresh_ownership"]["sha256"]:
        raise ValueError("frozen acceptance ownership hash differs from manifest")
    if formula_hash != acceptance_manifest["artifacts"]["source_formulae"]["sha256"]:
        raise ValueError("evaluation formula source hash differs from frozen acceptance manifest")
    expected_hwr_hash = context_report["checkpoints"]["current_hwr_sha256"]
    trace_hwr_hash = trace_summary["runtime"]["checkpoint_sha256"]
    if trace_hwr_hash != expected_hwr_hash:
        raise ValueError("current HWR checkpoint differs between context and trace reports")

    cases = context_report["actual_fast_partition_formula_cases"]
    case_by_id = {str(row["sample_id"]): row for row in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("duplicate sample_id in context-transfer formula cases")
    if set(case_by_id) - set(formula_by_id):
        raise ValueError("context-transfer formulas are missing from raw formula source")

    accepted_ownership_by_id = {
        str(row["sample_id"]): row for row in acceptance_ownership if row.get("accepted")
    }
    if len(accepted_ownership_by_id) != len(acceptance_ownership):
        raise ValueError("frozen fresh-acceptance ownership file contains unaccepted rows")
    accepted_ids = set(accepted_ownership_by_id)
    if len(accepted_ids) != int(acceptance_manifest["acceptance"]["formulae"]):
        raise ValueError("fresh-acceptance formula count differs from frozen manifest")
    if accepted_ids - set(case_by_id):
        raise ValueError("frozen acceptance formulas are missing from saved context-transfer cases")

    rows = []
    for sample_id, case in case_by_id.items():
        writer_id = str(formula_by_id[sample_id].get("writer_id", ""))
        if not writer_id:
            raise ValueError(f"missing writer_id for {sample_id}")
        stages = case["stage_scores"]
        for stage in (BASELINE_STAGE, CHALLENGER_STAGE):
            if int(stages[stage]["candidate_violations"]) != 0:
                raise AssertionError(f"candidate preservation violation in {sample_id}:{stage}")
        rows.append({
            "sample_id": sample_id,
            "writer_id": writer_id,
            "baseline_exact": bool(stages[BASELINE_STAGE]["formula_exact"]),
            "challenger_exact": bool(stages[CHALLENGER_STAGE]["formula_exact"]),
            "fast_group_exact": bool(case["fast_group_exact"]),
            "baseline_token_hits": (
                None if stages[BASELINE_STAGE]["token_hits_if_aligned"] is None
                else int(stages[BASELINE_STAGE]["token_hits_if_aligned"])
            ),
            "challenger_token_hits": (
                None if stages[CHALLENGER_STAGE]["token_hits_if_aligned"] is None
                else int(stages[CHALLENGER_STAGE]["token_hits_if_aligned"])
            ),
            "candidate_violations": int(stages[CHALLENGER_STAGE]["candidate_violations"]),
        })
    if len(rows) != int(context_report["data"]["formulas"]):
        raise AssertionError("case count does not reconcile with context report")

    writers = sorted({row["writer_id"] for row in rows})

    base_exact = sum(row["baseline_exact"] for row in rows)
    challenger_exact = sum(row["challenger_exact"] for row in rows)
    token_paired_rows = [
        row for row in rows
        if row["baseline_token_hits"] is not None and row["challenger_token_hits"] is not None
    ]
    base_tokens = sum(row["baseline_token_hits"] for row in token_paired_rows)
    challenger_tokens = sum(row["challenger_token_hits"] for row in token_paired_rows)
    baseline_token_rows = sum(row["baseline_token_hits"] is not None for row in rows)
    challenger_token_rows = sum(row["challenger_token_hits"] is not None for row in rows)
    saved = context_report["actual_fast_partition_counterfactual"]["stages"]
    if base_exact != int(saved[BASELINE_STAGE]["formula_exact_all_149"]):
        raise AssertionError("baseline exact count does not reconcile")
    if challenger_exact != int(saved[CHALLENGER_STAGE]["formula_exact_all_149"]):
        raise AssertionError("challenger exact count does not reconcile")

    overall_bootstrap = _paired_writer_bootstrap(rows, iterations=iterations, seed=seed)

    training_writer_groups = {str(row["writer_group"]) for row in training_candidates}
    evaluation_writer_groups = {writer: _writer_group(writer) for writer in writers}
    overlapping_writer_ids = sorted(
        writer for writer, group in evaluation_writer_groups.items() if group in training_writer_groups
    )
    disjoint_rows = [row for row in rows if row["writer_id"] not in set(overlapping_writer_ids)]
    disjoint_bootstrap = _paired_writer_bootstrap(
        disjoint_rows, iterations=iterations, seed=seed + 1
    ) if disjoint_rows else None
    training_formula_rows: dict[str, list[tuple[int, str]]] = {}
    training_groups_by_formula: dict[str, set[str]] = {}
    for train_row in training_candidates:
        formula_id = str(train_row["formula_id"])
        context = train_row.get("context", {})
        training_formula_rows.setdefault(formula_id, []).append((int(context["index"]), str(train_row["label"])))
        training_groups_by_formula.setdefault(formula_id, set()).add(str(train_row["writer_group"]))
    training_targets = {
        formula_id: [label for _, label in sorted(tokens)]
        for formula_id, tokens in training_formula_rows.items()
    }
    for formula_id, tokens in training_formula_rows.items():
        indices = sorted(index for index, _ in tokens)
        lengths = {
            int(train_row["context"]["length"])
            for train_row in training_candidates if str(train_row["formula_id"]) == formula_id
        }
        if indices != list(range(len(tokens))) or lengths != {len(tokens)}:
            raise AssertionError(f"training target sequence indices do not reconcile for {formula_id}")
    if any(len(groups) != 1 for groups in training_groups_by_formula.values()):
        raise AssertionError("a training formula ID maps to multiple writer groups")
    missing_training_formula_ids = sorted(set(training_targets) - set(training_formula_by_id))
    if missing_training_formula_ids:
        raise ValueError(f"training candidate IDs missing from training formula source: {missing_training_formula_ids[:5]}")
    candidate_target_mismatches = sorted(
        formula_id for formula_id, target in training_targets.items()
        if target != [str(cell["token"]) for cell in training_formula_by_id[formula_id]["target_cells"]]
    )
    candidate_writer_mismatches = sorted(
        formula_id for formula_id in training_targets
        if _writer_group(str(training_formula_by_id[formula_id]["writer_id"]))
        not in training_groups_by_formula[formula_id]
    )
    training_ink = {
        formula_id: _ink_signatures(training_formula_by_id[formula_id])
        for formula_id in training_targets
    }
    evaluation_ink = {
        str(row["sample_id"]): _ink_signatures(row)
        for row in formulas if str(row["sample_id"]) in case_by_id
    }
    evaluation_targets = {
        str(row["sample_id"]): [str(cell["token"]) for cell in row["target_cells"]]
        for row in formulas if str(row["sample_id"]) in case_by_id
    }
    training_formula_ids = set(training_targets)
    evaluation_formula_ids = {str(row["sample_id"]) for row in rows}
    shared_formula_ids = sorted(training_formula_ids & evaluation_formula_ids)
    same_target_sequence = sorted(
        formula_id for formula_id in shared_formula_ids
        if training_targets[formula_id] == evaluation_targets[formula_id]
    )
    different_target_sequence = sorted(set(shared_formula_ids) - set(same_target_sequence))
    evaluation_id_to_writer_group = {
        str(row["sample_id"]): evaluation_writer_groups[row["writer_id"]]
        for row in rows
    }
    same_id_same_writer = sorted(
        formula_id for formula_id in shared_formula_ids
        if any(
            str(train_row["formula_id"]) == formula_id
            and str(train_row["writer_group"]) == evaluation_id_to_writer_group[formula_id]
            for train_row in training_candidates
        )
    )
    ink_overlap = {}
    ink_pairs_by_signature: dict[str, list[tuple[str, str]]] = {}
    for signature_name in ("raw_xy", "raw_xyt", "normalized_xy"):
        train_by_signature: dict[str, list[str]] = defaultdict(list)
        evaluation_by_signature: dict[str, list[str]] = defaultdict(list)
        for formula_id, signatures in training_ink.items():
            train_by_signature[signatures[signature_name]].append(formula_id)
        for formula_id, signatures in evaluation_ink.items():
            evaluation_by_signature[signatures[signature_name]].append(formula_id)
        matching_signatures = set(train_by_signature) & set(evaluation_by_signature)
        pairs = sorted(
            (train_id, evaluation_id)
            for signature in matching_signatures
            for train_id in train_by_signature[signature]
            for evaluation_id in evaluation_by_signature[signature]
        )
        ink_pairs_by_signature[signature_name] = pairs
        ink_overlap[signature_name] = {
            "matching_signature_count": len(matching_signatures),
            "matching_formula_pair_count": len(pairs),
            "matching_formula_pairs": [{"training_id": left, "evaluation_id": right} for left, right in pairs],
        }
    exact_ink_pairs = ink_pairs_by_signature["raw_xy"]
    exact_ink_eval_ids = {evaluation_id for _, evaluation_id in exact_ink_pairs}
    exact_ink_pair_details = [
        {
            "training_id": training_id,
            "evaluation_id": evaluation_id,
            "target_sequence_equal": training_targets[training_id] == evaluation_targets[evaluation_id],
            "training_writer_id": str(training_formula_by_id[training_id]["writer_id"]),
            "evaluation_writer_id": str(formula_by_id[evaluation_id]["writer_id"]),
            "writer_id_equal": (
                str(training_formula_by_id[training_id]["writer_id"])
                == str(formula_by_id[evaluation_id]["writer_id"])
            ),
        }
        for training_id, evaluation_id in exact_ink_pairs
    ]
    raw_ink_clean_rows = [row for row in rows if row["sample_id"] not in exact_ink_eval_ids]
    writer_clean_rows = [row for row in rows if row["writer_id"] not in set(overlapping_writer_ids)]
    ink_and_writer_clean_rows = [
        row for row in raw_ink_clean_rows if row["writer_id"] not in set(overlapping_writer_ids)
    ]
    duplicate_ink_rows = [row for row in rows if row["sample_id"] in exact_ink_eval_ids]
    acceptance_rows = [row for row in rows if row["sample_id"] in accepted_ids]
    if len(acceptance_rows) != len(accepted_ids):
        raise AssertionError("not all frozen acceptance formulas have paired saved predictions")
    acceptance_writer_counts = {
        str(writer): int(count)
        for writer, count in acceptance_manifest["acceptance"]["writer_formula_counts"].items()
    }
    actual_acceptance_writer_counts: dict[str, int] = {}
    for sample_id, ownership in accepted_ownership_by_id.items():
        formula = formula_by_id[sample_id]
        expected_tokens = [str(cell["token"]) for cell in formula["target_cells"]]
        if str(ownership["writer_id"]) != str(formula["writer_id"]):
            raise AssertionError(f"frozen acceptance writer mismatch: {sample_id}")
        if [str(token) for token in ownership["labels"]] != expected_tokens:
            raise AssertionError(f"frozen acceptance target mismatch: {sample_id}")
        actual_acceptance_writer_counts[str(ownership["writer_id"])] = (
            actual_acceptance_writer_counts.get(str(ownership["writer_id"]), 0) + 1
        )
    acceptance_ink_overlap = {}
    for signature_name in ("raw_xy", "raw_xyt", "normalized_xy"):
        training_signatures = {signatures[signature_name] for signatures in training_ink.values()}
        matches = sorted(
            sample_id for sample_id in accepted_ids
            if evaluation_ink[sample_id][signature_name] in training_signatures
        )
        acceptance_ink_overlap[signature_name] = matches
    acceptance_bootstrap = _paired_writer_bootstrap(
        acceptance_rows, iterations=iterations, seed=seed + 2
    )
    clean_slices = {
        "all_saved_evaluation": rows,
        "writer_disjoint_only": writer_clean_rows,
        "exact_ink_disjoint_only": raw_ink_clean_rows,
        "exact_ink_and_writer_disjoint": ink_and_writer_clean_rows,
        "exact_ink_duplicate_subset": duplicate_ink_rows,
    }
    clean_slice_bootstraps = {
        name: _paired_writer_bootstrap(subset, iterations=iterations, seed=seed + 10 + index)
        for index, (name, subset) in enumerate(clean_slices.items())
    }

    training_identity = context_report["checkpoints"].get("training_identity_audit", {})
    verification_checks = {
        "formula_source_hash_matches_trace_summary": formula_hash == expected_formula_hash,
        "training_candidates_hash_matches_training_report": training_candidates_hash == expected_training_candidates_hash,
        "all_training_candidate_formula_ids_found": not missing_training_formula_ids,
        "candidate_targets_match_training_formula_source": not candidate_target_mismatches,
        "candidate_writer_groups_match_training_formula_source": not candidate_writer_mismatches,
        "context_checkpoint_hash_matches_report": context_hash == expected_context_hash,
        "context_checkpoint_hash_matches_training_report": context_hash == context_model_report["checkpoint"]["sha256"],
        "training_report_excludes_crohme": crohme_excluded,
        "current_hwr_hash_matches_trace_summary": trace_hwr_hash == expected_hwr_hash,
        "149_unique_formula_cases": len(case_by_id) == len(cases) == 149,
        "all_formula_cases_have_writer": len(rows) == len(case_by_id),
        "candidate_violations_zero": sum(row["candidate_violations"] for row in rows) == 0,
        "aggregate_counts_reconcile": (
            base_exact == int(saved[BASELINE_STAGE]["formula_exact_all_149"])
            and challenger_exact == int(saved[CHALLENGER_STAGE]["formula_exact_all_149"])
        ),
        "writer_disjoint_subset_excludes_training_writers": all(
            evaluation_writer_groups[row["writer_id"]] not in training_writer_groups for row in disjoint_rows
        ),
        "formula_id_overlap_audit_reconciles_training_report": (
            len(shared_formula_ids) == int(training_identity.get("shared_formula_id_strings", -1))
            and len(same_target_sequence)
            == int(training_identity.get("same_ordered_target_sequences_on_shared_ids", -1))
            and len(different_target_sequence)
            == int(training_identity.get("shared_ids_with_different_target_sequences", -1))
        ),
        "frozen_acceptance_manifest_hashes_match": (
            formula_hash == acceptance_manifest["artifacts"]["source_formulae"]["sha256"]
            and acceptance_ownership_hash == acceptance_manifest["artifacts"]["fresh_ownership"]["sha256"]
        ),
        "frozen_acceptance_ids_and_targets_reconcile": (
            len(acceptance_rows) == len(accepted_ids)
            and actual_acceptance_writer_counts == acceptance_writer_counts
        ),
        "frozen_acceptance_has_no_exact_or_normalized_ink_duplicates": all(
            not matches for matches in acceptance_ink_overlap.values()
        ),
    }
    return {
        "schema": SCHEMA,
        "scope": "paired writer-cluster audit of saved project-owned context shadow predictions on actual Fast groups; no inference, training, CROHME, or promotion",
        "inputs": {
            "context_transfer_report": {"path": str(context_report_path), "sha256": _sha256(context_report_path)},
            "context_model_report": {"path": str(context_model_report_path), "sha256": _sha256(context_model_report_path)},
            "trace_summary": {"path": str(trace_summary_path), "sha256": _sha256(trace_summary_path)},
            "formula_source": {"path": str(formulas_path), "sha256": formula_hash},
            "training_candidates": {"path": str(training_candidates_path), "sha256": training_candidates_hash},
            "training_formula_source": {"path": str(training_formulas_path), "sha256": _sha256(training_formulas_path)},
            "frozen_acceptance_ownership": {"path": str(acceptance_ownership_path), "sha256": acceptance_ownership_hash},
            "frozen_acceptance_manifest": {"path": str(acceptance_manifest_path), "sha256": _sha256(acceptance_manifest_path)},
            "current_hwr_sha256": trace_hwr_hash,
            "context_checkpoint": {"path": str(context_checkpoint_path), "sha256": context_hash},
        },
        "arms": {"baseline": BASELINE_STAGE, "challenger": CHALLENGER_STAGE},
        "policy": {
            "selection_independent": False,
            "training_data": context_model_report["data_admission"]["project"],
            "training_excluded_crohme": crohme_excluded,
            "raw_ink_overlap_auditable": training_identity.get("raw_ink_overlap_auditable"),
            "writer_id_overlap_auditable": training_identity.get("writer_id_overlap_auditable"),
            "overlapping_formula_id_strings": training_identity.get("shared_formula_id_strings"),
            "overlapping_ids_with_same_target_sequence": training_identity.get("same_ordered_target_sequences_on_shared_ids"),
            "writer_overlap_audit": {
                "training_writer_groups": len(training_writer_groups),
                "evaluation_writers": len(writers),
                "overlapping_evaluation_writer_ids": overlapping_writer_ids,
                "writer_disjoint_evaluation_ids": sorted(set(writers) - set(overlapping_writer_ids)),
            },
            "formula_id_overlap_audit": {
                "shared_formula_id_strings": len(shared_formula_ids),
                "same_ordered_target_sequences": len(same_target_sequence),
                "different_ordered_target_sequences": len(different_target_sequence),
                "same_id_same_writer_group": len(same_id_same_writer),
                "different_target_examples": [
                    {
                        "formula_id": formula_id,
                        "training_target": training_targets[formula_id],
                        "evaluation_target": evaluation_targets[formula_id],
                    }
                    for formula_id in different_target_sequence[:8]
                ],
            },
            "training_candidates_contain_raw_strokes": all("strokes" in row for row in training_candidates),
            "candidate_to_source_reconciliation": {
                "candidate_formula_count": len(training_targets),
                "missing_source_formula_ids": missing_training_formula_ids,
                "target_sequence_mismatch_count": len(candidate_target_mismatches),
                "writer_group_mismatch_count": len(candidate_writer_mismatches),
            },
            "raw_ink_overlap_audit": {
                "training_formulas_with_raw_strokes": len(training_ink),
                "evaluation_formulas_with_raw_strokes": len(evaluation_ink),
                "signature_method": "ordered stroke trajectories: exact x/y, exact x/y/time, and bbox-normalized x/y rounded to 1e-6; cross-ID comparisons included",
                **ink_overlap,
                "exact_raw_xy_pair_details": exact_ink_pair_details,
            },
            "evaluation_independent_of_context_training": False,
            "frozen_acceptance_subset": {
                "frozen_at": acceptance_manifest["frozen_at"],
                "model_predictions_opened_before_freeze": acceptance_manifest["model_predictions_opened_before_freeze"],
                "model_training_performed_before_freeze": acceptance_manifest["training_performed"],
                "formulas": len(acceptance_rows),
                "writer_formula_counts": actual_acceptance_writer_counts,
                "candidate_violations": sum(
                    int(case_by_id[sample_id]["stage_scores"][CHALLENGER_STAGE]["candidate_violations"])
                    for sample_id in accepted_ids
                ),
                "fast_group_exact": sum(bool(case_by_id[sample_id]["fast_group_exact"]) for sample_id in accepted_ids),
                "raw_ink_duplicate_ids_by_signature": acceptance_ink_overlap,
                "paired_strict_exact": acceptance_bootstrap,
                "selection_independence_after_freeze": False,
                "interpretation": "the frozen subset excludes exact training ink, but the saved predictions have since been inspected in development; treat as post-hoc diagnostic, not a new acceptance claim",
            },
            "acceptance_readiness": {
                "eligible": False,
                "blockers": [
                    f"{len(exact_ink_eval_ids)} evaluation formulas are exact raw-ink copies of training formulas",
                    f"{len(overlapping_writer_ids)} evaluation writer IDs overlap training writer groups",
                    "the evaluation cohort was inspected during development",
                    "approximate near-duplicate trajectories are not covered by exact-signature matching",
                ],
            },
            "interpretation_limit": "all exact ink copies are excluded in the exact-ink-disjoint slice; these repeated-split results remain exploratory because this evaluation cohort has been inspected during development, and signature matching does not rule out approximate near-duplicates",
        },
        "summary": {
            "formulas": len(rows),
            "writers": len(writers),
            "fast_group_exact_formulas": sum(row["fast_group_exact"] for row in rows),
            "formula_exact": {
                "baseline": base_exact,
                "challenger": challenger_exact,
                "transitions": _transitions(rows),
            },
            "token_hits_on_paired_alignable_rows": {
                "formulas": len(token_paired_rows),
                "baseline_formula_rows_with_aligned_tokens": baseline_token_rows,
                "challenger_formula_rows_with_aligned_tokens": challenger_token_rows,
                "baseline": base_tokens,
                "challenger": challenger_tokens,
                "delta": challenger_tokens - base_tokens,
            },
            "candidate_violations": sum(row["candidate_violations"] for row in rows),
            "writer_cluster_bootstrap": {
                "method": "paired writer-cluster bootstrap; writers sampled with replacement, formula-weighted rate difference",
                "iterations": iterations,
                "seed": seed,
                **overall_bootstrap,
            },
            "writer_disjoint_subset": {
                "bootstrap_seed": seed + 1,
                **(disjoint_bootstrap or {"formulas": 0, "writers": 0}),
            },
            "contamination_sensitivity_slices": {
                "bootstrap_method": "paired writer-cluster bootstrap; formula-weighted rate difference; exploratory sensitivity only",
                "iterations": iterations,
                "slices": clean_slice_bootstraps,
            },
            "frozen_acceptance_subset": {
                "formulas": len(acceptance_rows),
                "writers": len(actual_acceptance_writer_counts),
                "writer_formula_counts": actual_acceptance_writer_counts,
                "group_exact": sum(bool(case_by_id[sample_id]["fast_group_exact"]) for sample_id in accepted_ids),
                "strict_exact": acceptance_bootstrap,
                "raw_ink_duplicate_counts": {
                    signature: len(matches) for signature, matches in acceptance_ink_overlap.items()
                },
            },
            "writer_rows": overall_bootstrap["writer_rows"],
        },
        "verification": {
            **verification_checks,
            "all_exact_ink_pairs_have_matching_target_sequence": all(
                row["target_sequence_equal"] for row in exact_ink_pair_details
            ),
            "all_exact_ink_pairs_have_different_writer_metadata": all(
                not row["writer_id_equal"] for row in exact_ink_pair_details
            ),
            "exact_ink_duplicate_formula_count": len(exact_ink_eval_ids),
            "exact_ink_disjoint_formula_count": len(raw_ink_clean_rows),
            "exact_ink_duplicates_absent": len(exact_ink_eval_ids) == 0,
            "raw_ink_and_writer_independence_pass": (
                len(exact_ink_eval_ids) == 0 and not overlapping_writer_ids
            ),
            "all_provenance_reconciliation_checks_pass": all(verification_checks.values()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-report", type=Path, required=True)
    parser.add_argument("--context-model-report", type=Path, required=True)
    parser.add_argument("--trace-summary", type=Path, required=True)
    parser.add_argument("--formulas", type=Path, required=True)
    parser.add_argument("--context-checkpoint", type=Path, required=True)
    parser.add_argument("--training-candidates", type=Path, required=True)
    parser.add_argument("--training-formulas", type=Path, required=True)
    parser.add_argument("--acceptance-ownership", type=Path, required=True)
    parser.add_argument("--acceptance-manifest", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 100:
        parser.error("--iterations must be at least 100")
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    report = audit(
        args.context_report,
        args.context_model_report,
        args.trace_summary,
        args.formulas,
        args.context_checkpoint,
        args.training_candidates,
        args.training_formulas,
        args.acceptance_ownership,
        args.acceptance_manifest,
        iterations=args.iterations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": report["summary"], "verification": report["verification"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
