#!/usr/bin/env python3
"""Probe HWR Top-1/Top-5 token-multiset exact-cover reachability.

The ONNX export has no embedded class vocabulary. This diagnostic recovers
only writer-stable output-index mappings from accepted project-owned groups.
This is a proxy, not a strict bound or accuracy measure: unknown output classes
are omitted, while order and spatial relations are ignored. It performs only
forward inference and never updates model weights.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from evaluate_48hz_prefix_v1 import INPUT_MODE
from evaluate_joint_hwr_grouping_v1 import _candidate_tensor
from stroke_grouping_v1 import DEFAULT_LATTICE_CONFIG, build_lattice
from train_character_classifier_v1 import apply_input_mode
from train_project_owned_grouping_v1 import _samples
from validate_public_dataset09 import validate as validate_dataset


SCHEMA = "aiflow-transcript-only-hwr-reachability/v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "hf-dataset"
DEFAULT_ONNX = ROOT / "artifacts" / "mobile_hwr_fp32_20260914" / "hwr.onnx"
INFERENCE_BATCH_SIZE = 128


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_onnx(path: Path):
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as error:
        raise RuntimeError("reachability audit requires onnxruntime") from error
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = {row.name: row for row in session.get_outputs()}
    if len(inputs) != 1 or inputs[0].shape[1:] != [128, 5]:
        raise ValueError("expected the frozen [N,128,5] HWR ONNX input")
    if "logits" not in outputs or outputs["logits"].shape[-1] != 372:
        raise ValueError("expected the frozen 372-class HWR logits output")
    return session


def _infer_logits(session, tensors: list[np.ndarray]) -> np.ndarray:
    if not tensors:
        return np.empty((0, 372), dtype=np.float32)
    values = np.stack(tensors).astype(np.float32, copy=False)
    values = apply_input_mode(values, INPUT_MODE).astype(np.float32, copy=False)
    chunks = [
        session.run(
            ["logits"], {"points": values[start:start + INFERENCE_BATCH_SIZE]},
        )[0]
        for start in range(0, len(values), INFERENCE_BATCH_SIZE)
    ]
    result = np.concatenate(chunks, axis=0)
    if result.shape != (len(tensors), 372) or not np.isfinite(result).all():
        raise ValueError("HWR ONNX returned invalid logits")
    return result


def _recover_stable_label_indices(
    metadata: list[tuple[str, str]], logits: np.ndarray,
) -> tuple[dict[str, int], dict[str, Any]]:
    by_writer: dict[str, list[int]] = defaultdict(list)
    writers_by_label: dict[str, set[str]] = defaultdict(set)
    for index, (writer, label) in enumerate(metadata):
        by_writer[writer].append(index)
        writers_by_label[label].add(writer)
    writers = sorted(by_writer)
    assignments: dict[str, list[int | None]] = defaultdict(list)
    fold_rows = []
    cv_scored = cv_top1 = cv_top5 = cv_unscored = 0

    for held_writer in writers:
        train_indices = [
            index
            for writer, indices in by_writer.items()
            if writer != held_writer
            for index in indices
        ]
        train_by_label: dict[str, list[int]] = defaultdict(list)
        for index in train_indices:
            train_by_label[metadata[index][1]].append(index)

        mapping = {
            label: Counter(int(np.argmax(logits[index])) for index in indices)
            .most_common(1)[0][0]
            for label, indices in train_by_label.items()
        }
        labels_by_index: dict[int, list[str]] = defaultdict(list)
        for label, output_index in mapping.items():
            labels_by_index[output_index].append(label)
        collided = {
            label
            for labels in labels_by_index.values() if len(labels) > 1
            for label in labels
        }
        unique_mapping = {
            label: output_index
            for label, output_index in mapping.items()
            if label not in collided
        }

        for label in writers_by_label:
            assignments[label].append(unique_mapping.get(label))

        test_indices = by_writer[held_writer]
        scored = [
            index for index in test_indices
            if metadata[index][1] in unique_mapping
        ]
        top5_indices = np.argsort(-logits, axis=1)[:, :5]
        top1_hits = sum(
            int(np.argmax(logits[index]) == unique_mapping[metadata[index][1]])
            for index in scored
        )
        top5_hits = sum(
            int(unique_mapping[metadata[index][1]] in top5_indices[index])
            for index in scored
        )
        cv_scored += len(scored)
        cv_unscored += len(test_indices) - len(scored)
        cv_top1 += top1_hits
        cv_top5 += top5_hits
        fold_rows.append({
            "held_writer_groups": len(test_indices),
            "scored_groups": len(scored),
            "ambiguous_label_mappings": len(collided),
            "top1_index_hits": top1_hits,
            "top5_index_hits": top5_hits,
        })

    # Require agreement in every held-writer fold and support from at least
    # two writers; partial agreement is not enough to name an output class.
    stable = {
        label: values[0]
        for label, values in assignments.items()
        if len(writers_by_label[label]) >= 2
        and len(values) == len(writers)
        and all(value is not None for value in values)
        and len(set(values)) == 1
    }
    labels_by_index = defaultdict(list)
    for label, output_index in stable.items():
        labels_by_index[output_index].append(label)
    collisions = {
        label
        for labels in labels_by_index.values() if len(labels) > 1
        for label in labels
    }
    for label in collisions:
        stable.pop(label, None)

    summary = {
        "gold_groups": len(metadata),
        "gold_labels": len(writers_by_label),
        "writer_folds": len(writers),
        "cross_writer_scored_groups": cv_scored,
        "cross_writer_unscored_groups": cv_unscored,
        "cross_writer_top1_index_consistency": cv_top1 / max(cv_scored, 1),
        "cross_writer_top5_index_consistency": cv_top5 / max(cv_scored, 1),
        "stable_labels": len(stable),
        "labels_rejected_for_collisions": len(collisions),
        "folds": fold_rows,
    }
    return stable, summary


def _held_writer_label_indices(
    metadata: list[tuple[str, str]],
    logits: np.ndarray,
    held_writer: str,
) -> tuple[dict[str, int], int]:
    by_writer_label: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, (writer, label) in enumerate(metadata):
        if writer != held_writer:
            by_writer_label[writer][label].append(index)

    train_writers = sorted(by_writer_label)
    label_modes: dict[str, list[int]] = defaultdict(list)
    for writer in train_writers:
        for label, indices in by_writer_label[writer].items():
            label_modes[label].append(
                Counter(int(np.argmax(logits[index])) for index in indices)
                .most_common(1)[0][0]
            )

    mapping = {
        label: modes[0]
        for label, modes in label_modes.items()
        if len(modes) == len(train_writers)
        and len(modes) >= 2
        and len(set(modes)) == 1
    }
    labels_by_index: dict[int, list[str]] = defaultdict(list)
    for label, output_index in mapping.items():
        labels_by_index[output_index].append(label)
    collisions = {
        label
        for labels in labels_by_index.values() if len(labels) > 1
        for label in labels
    }
    for label in collisions:
        mapping.pop(label, None)
    return mapping, len(collisions)


def _solve_labeled_exact_cover(
    candidates: list[dict[str, Any]],
    candidate_labels: list[tuple[str, ...]],
    stroke_count: int,
    target_tokens: list[str],
    *,
    cap: int = 2,
) -> tuple[int, tuple[tuple[int, str], ...] | None]:
    if cap < 2:
        raise ValueError("solution-count cap must be at least two")
    vocabulary = sorted(Counter(target_tokens))
    label_position = {label: index for index, label in enumerate(vocabulary)}
    initial_counts = tuple(Counter(target_tokens)[label] for label in vocabulary)
    label_options_by_mask: dict[int, set[int]] = defaultdict(set)
    for candidate, labels in zip(candidates, candidate_labels, strict=True):
        indices = tuple(sorted({int(value) for value in candidate["source_indices"]}))
        if not indices or indices[0] < 0 or indices[-1] >= stroke_count:
            raise ValueError("candidate contains an invalid stroke index")
        mask = sum(1 << value for value in indices)
        options = tuple(sorted({
            label_position[label] for label in labels if label in label_position
        }))
        label_options_by_mask[mask].update(options)
    by_stroke: list[list[tuple[int, int]]] = [
        [] for _ in range(stroke_count)
    ]
    for mask, option_set in sorted(label_options_by_mask.items()):
        options = sorted(option_set)
        if not options:
            continue
        for value in range(stroke_count):
            if mask & (1 << value):
                by_stroke[value].extend((mask, label_index) for label_index in options)

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def visit(
        remaining: int, token_counts: tuple[int, ...],
    ) -> tuple[int, tuple[tuple[int, str], ...] | None]:
        if remaining == 0:
            return (1, ()) if not any(token_counts) else (0, None)
        if sum(token_counts) > remaining.bit_count():
            return 0, None
        pivot = (remaining & -remaining).bit_length() - 1
        solutions = 0
        unique_path: tuple[tuple[int, str], ...] | None = None
        for mask, label_index in by_stroke[pivot]:
            if mask & remaining != mask:
                continue
            if token_counts[label_index] == 0:
                continue
            next_counts = list(token_counts)
            next_counts[label_index] -= 1
            child_count, child_path = visit(remaining ^ mask, tuple(next_counts))
            if child_count and unique_path is None and child_path is not None:
                unique_path = ((mask, vocabulary[label_index]), *child_path)
            solutions += child_count
            if solutions >= cap:
                return cap, None
        return solutions, unique_path if solutions == 1 else None

    return visit((1 << stroke_count) - 1, initial_counts)


def _count_labeled_exact_covers(
    candidates: list[dict[str, Any]],
    candidate_labels: list[tuple[str, ...]],
    stroke_count: int,
    target_tokens: list[str],
    *,
    cap: int = 2,
) -> int:
    return _solve_labeled_exact_cover(
        candidates, candidate_labels, stroke_count, target_tokens, cap=cap,
    )[0]


def _has_labeled_exact_cover(
    candidates: list[dict[str, Any]],
    candidate_labels: list[tuple[str, ...]],
    stroke_count: int,
    target_tokens: list[str],
) -> bool:
    return bool(_count_labeled_exact_covers(
        candidates, candidate_labels, stroke_count, target_tokens,
    ))


def audit(dataset_root: Path, onnx_path: Path) -> dict[str, Any]:
    root = dataset_root.resolve()
    model_path = onnx_path.resolve()
    if any("crohme" in part.lower() for part in (*root.parts, *model_path.parts)):
        raise ValueError("project-owned data and the frozen mobile HWR export are required")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    validation = validate_dataset(root)
    samples, _dataset = _samples(root)
    formulas = _read_jsonl(root / "data" / "formulas_valid.jsonl")
    ownership = [
        row for row in _read_jsonl(root / "data" / "ownership_train.jsonl")
        if row.get("accepted")
    ]
    owned_label_support = Counter(
        str(label) for row in ownership for label in row.get("labels", [])
    )
    ownership_by_id = {str(row["sample_id"]): row for row in ownership}
    formulas_by_id = {str(row["sample_id"]): row for row in formulas}
    if set(ownership_by_id) - set(formulas_by_id):
        raise ValueError("ownership annotation references a missing formula")
    weak_formulas = [
        row for row in formulas
        if str(row["sample_id"]) not in ownership_by_id
    ]

    tensors: list[np.ndarray] = []
    gold_metadata: list[tuple[str, str]] = []
    for sample in samples:
        annotation = ownership_by_id[sample.sample_id]
        if len(sample.truth) != len(annotation["labels"]):
            raise ValueError("owned group count differs from accepted labels")
        for group, label in zip(sample.truth, annotation["labels"], strict=True):
            tensors.append(_candidate_tensor(sample, sorted(group)))
            gold_metadata.append((sample.writer, str(label)))
    gold_count = len(tensors)

    owned_candidate_rows: list[tuple[Any, dict[str, Any], list[dict[str, Any]], int]] = []
    for sample in samples:
        formula = formulas_by_id[sample.sample_id]
        formula_for_cv = dict(formula)
        formula_for_cv["_accepted_group_labels"] = ownership_by_id[sample.sample_id]["labels"]
        strokes = sorted(formula["strokes"], key=lambda row: int(row["order"]))
        candidates = build_lattice(strokes, **DEFAULT_LATTICE_CONFIG)
        start = len(tensors)
        candidate_sample = SimpleNamespace(sample_id=sample.sample_id, strokes=strokes)
        tensors.extend(
            _candidate_tensor(candidate_sample, candidate["source_indices"])
            for candidate in candidates
        )
        owned_candidate_rows.append((sample, formula_for_cv, candidates, start))

    weak_rows: list[tuple[dict[str, Any], list[dict[str, Any]], int]] = []
    for formula in weak_formulas:
        strokes = sorted(formula["strokes"], key=lambda row: int(row["order"]))
        candidates = build_lattice(strokes, **DEFAULT_LATTICE_CONFIG)
        start = len(tensors)
        sample = SimpleNamespace(sample_id=str(formula["sample_id"]), strokes=strokes)
        tensors.extend(
            _candidate_tensor(sample, candidate["source_indices"])
            for candidate in candidates
        )
        weak_rows.append((formula, candidates, start))

    session = _load_onnx(model_path)
    logits = _infer_logits(session, tensors)
    stable_mapping, mapping_summary = _recover_stable_label_indices(
        gold_metadata, logits[:gold_count],
    )
    index_to_label = {index: label for label, index in stable_mapping.items()}
    pseudo_cv = _audit_pseudo_ownership_cross_validation(
        owned_candidate_rows, gold_metadata, logits[:gold_count], logits,
    )
    all_target_tokens = {
        str(cell["token"])
        for formula in formulas
        for cell in formula.get("target_cells") or []
    }
    unmapped_target_tokens = sorted(all_target_tokens - set(stable_mapping))

    top1_reachable = 0
    top5_reachable = 0
    top1_solution_counts: Counter[str] = Counter()
    top5_solution_counts: Counter[str] = Counter()
    review_candidates: list[dict[str, Any]] = []
    fully_mappable = 0
    unmapped_occurrences: Counter[str] = Counter()
    candidate_counts: list[int] = []
    for formula, candidates, start in weak_rows:
        target_tokens = [
            str(cell["token"]) for cell in formula.get("target_cells") or []
        ]
        missing = set(target_tokens) - set(stable_mapping)
        if missing:
            unmapped_occurrences.update(label for label in target_tokens if label in missing)
            continue

        fully_mappable += 1
        candidate_counts.append(len(candidates))
        formula_logits = logits[start:start + len(candidates)]
        top5_indices = np.argsort(-formula_logits, axis=1)[:, :5]
        top1_labels = [
            (index_to_label[index],) if index in index_to_label else ()
            for index in top5_indices[:, 0].tolist()
        ]
        top5_labels = [
            tuple(sorted({
                index_to_label[int(index)]
                for index in row
                if int(index) in index_to_label
            }))
            for row in top5_indices
        ]
        top1_solutions, _top1_path = _solve_labeled_exact_cover(
            candidates, top1_labels, len(formula["strokes"]), target_tokens,
        )
        top5_solutions, top5_path = _solve_labeled_exact_cover(
            candidates, top5_labels, len(formula["strokes"]), target_tokens,
        )
        top1_reachable += int(top1_solutions > 0)
        top5_reachable += int(top5_solutions > 0)
        top1_solution_counts[
            "zero" if top1_solutions == 0 else "unique" if top1_solutions == 1 else "two_or_more"
        ] += 1
        top5_solution_counts[
            "zero" if top5_solutions == 0 else "unique" if top5_solutions == 1 else "two_or_more"
        ] += 1
        if top5_solutions == 1:
            if top5_path is None:
                raise AssertionError("unique Top-5 cover is missing its proposed ownership groups")
            review_candidates.append({
                "sample_id": str(formula["sample_id"]),
                "review_status": "unverified_manual_review_only",
                "target_tokens": target_tokens,
                "proposed_groups": [
                    {
                        "stroke_indices": [
                            index for index in range(len(formula["strokes"]))
                            if mask & (1 << index)
                        ],
                        "token": label,
                    }
                    for mask, label in top5_path
                ],
                "candidate_count": len(candidates),
                "mapped_top5_exact_cover_solutions": 1,
            })

    return {
        "schema": SCHEMA,
        "status": "diagnostic_proxy",
        "training_performed": False,
        "files_written": False,
        "crohme_rows": 0,
        "source": {
            "dataset_validation": validation,
            "valid_formulas": len(formulas),
            "ownership_labeled_formulas": len(ownership),
            "transcript_only_formulas": len(weak_formulas),
            "formulas_sha256": _sha256(root / "data" / "formulas_valid.jsonl"),
            "ownership_sha256": _sha256(root / "data" / "ownership_train.jsonl"),
            "onnx_sha256": _sha256(model_path),
            "input_mode": INPUT_MODE,
            "input_shape": [128, 5],
            "output_classes": 372,
        },
        "vocabulary_recovery": {
            **mapping_summary,
            "target_token_classes": len(all_target_tokens),
            "target_classes_without_stable_mapping": unmapped_target_tokens,
            "accepted_owned_groups": sum(owned_label_support.values()),
            "owned_symbol_classes": len(owned_label_support),
            "target_classes_without_owned_groups": sorted(
                all_target_tokens - set(owned_label_support)
            ),
            "owned_group_support_by_label": dict(sorted(owned_label_support.items())),
        },
        "pseudo_ownership_cross_validation": pseudo_cv,
        "transcript_only_reachability": {
            "fully_mappable_formulas": fully_mappable,
            "unmappable_formulas": len(weak_formulas) - fully_mappable,
            "unmapped_token_occurrences": dict(sorted(unmapped_occurrences.items())),
            "candidate_count_min_median_max": {
                "min": min(candidate_counts) if candidate_counts else None,
                "median": float(np.median(candidate_counts)) if candidate_counts else None,
                "max": max(candidate_counts) if candidate_counts else None,
            },
            "top1_token_multiset_exact_cover_reachable": top1_reachable,
            "top5_token_multiset_exact_cover_reachable": top5_reachable,
            "top1_exact_cover_solution_count_capped_at_2": dict(sorted(top1_solution_counts.items())),
            "top5_exact_cover_solution_count_capped_at_2": dict(sorted(top5_solution_counts.items())),
            "top5_unique_cover_review_candidates": top5_solution_counts["unique"],
            "pseudo_ownership_review_candidates": review_candidates,
            "top1_rate_on_fully_mappable": top1_reachable / max(fully_mappable, 1),
            "top5_rate_on_fully_mappable": top5_reachable / max(fully_mappable, 1),
        },
        "limits": [
            "output-index labels are inferred from accepted exact-group labels, not a paired checkpoint vocabulary",
            "writer folds validate mapping stability only; they are not an independent HWR accuracy evaluation",
            "unmapped output classes are omitted, which can undercount candidate reachability",
            "token multiset exact cover ignores reading order, spatial relations, and decoder validity, which can overcount usable candidates",
            "a unique mapped Top-5 cover is only a pseudo-ownership review lead, not a verified annotation",
            "the opposing omissions mean this is neither a strict upper nor lower bound, and not formula accuracy or product evidence",
        ],
    }


def _audit_pseudo_ownership_cross_validation(
    owned_candidate_rows: list[tuple[Any, dict[str, Any], list[dict[str, Any]], int]],
    gold_metadata: list[tuple[str, str]],
    gold_logits: np.ndarray,
    all_logits: np.ndarray,
) -> dict[str, Any]:
    """Score the proposal protocol while hiding each formula writer's labels."""
    formulas_by_writer: dict[str, list[tuple[Any, dict[str, Any], list[dict[str, Any]], int]]] = defaultdict(list)
    for row in owned_candidate_rows:
        formulas_by_writer[row[0].writer].append(row)

    totals = Counter()
    fold_rows = []
    for fold_index, held_writer in enumerate(sorted(formulas_by_writer), start=1):
        mapping, collisions = _held_writer_label_indices(
            gold_metadata, gold_logits, held_writer,
        )
        index_to_label = {index: label for label, index in mapping.items()}
        fold = Counter()
        for sample, formula, candidates, start in formulas_by_writer[held_writer]:
            target_tokens = [
                str(cell["token"]) for cell in formula.get("target_cells") or []
            ]
            annotation = Counter(
                str(label) for label in formula.get("_accepted_group_labels", [])
            )
            if annotation and annotation != Counter(target_tokens):
                raise ValueError("accepted ownership labels disagree with formula target tokens")
            if not target_tokens or not set(target_tokens).issubset(mapping):
                fold["unmappable_formulas"] += 1
                continue

            fold["mappable_formulas"] += 1
            candidate_logits = all_logits[start:start + len(candidates)]
            if candidate_logits.shape != (len(candidates), 372):
                raise ValueError("owned candidate logits do not align with the candidate lattice")
            top5_indices = np.argsort(-candidate_logits, axis=1)[:, :5]
            top1_labels = [
                (index_to_label[int(row[0])],)
                if int(row[0]) in index_to_label else ()
                for row in top5_indices
            ]
            top5_labels = [
                tuple(sorted({
                    index_to_label[int(index)] for index in row
                    if int(index) in index_to_label
                }))
                for row in top5_indices
            ]
            gold_pairs = {
                (tuple(sorted(int(index) for index in group)), str(label))
                for group, label in zip(
                    sample.truth, formula["_accepted_group_labels"], strict=True,
                )
            }
            if Counter(label for _group, label in gold_pairs) != Counter(target_tokens):
                raise ValueError("accepted ownership group labels disagree with target-token multiset")

            for name, labels in (("top1", top1_labels), ("top5", top5_labels)):
                solution_count, path = _solve_labeled_exact_cover(
                    candidates, labels, len(formula["strokes"]), target_tokens,
                )
                if solution_count > 0:
                    fold[f"{name}_reachable"] += 1
                if solution_count == 1:
                    if path is None:
                        raise AssertionError("unique cross-validation proposal has no path")
                    fold[f"{name}_unique_proposals"] += 1
                    proposed_pairs = {
                        (
                            tuple(index for index in range(len(formula["strokes"])) if mask & (1 << index)),
                            label,
                        )
                        for mask, label in path
                    }
                    if proposed_pairs == gold_pairs:
                        fold[f"{name}_exact_group_label_matches"] += 1
                else:
                    fold[f"{name}_ambiguous_or_absent"] += 1
        for key, value in fold.items():
            totals[key] += value
        totals["held_writer_folds"] += 1
        totals["held_writer_label_collisions"] += collisions
        totals["held_writer_formulas"] += len(formulas_by_writer[held_writer])
        fold_rows.append({
            "fold": fold_index,
            "held_writer_formulas": len(formulas_by_writer[held_writer]),
            "mappable_formulas": fold["mappable_formulas"],
            "unmappable_formulas": fold["unmappable_formulas"],
            "label_index_collisions": collisions,
            "top1_unique_proposals": fold["top1_unique_proposals"],
            "top1_exact_group_label_matches": fold["top1_exact_group_label_matches"],
            "top5_unique_proposals": fold["top5_unique_proposals"],
            "top5_exact_group_label_matches": fold["top5_exact_group_label_matches"],
        })

    top1_unique = totals["top1_unique_proposals"]
    top5_unique = totals["top5_unique_proposals"]
    return {
        "status": "diagnostic_protocol_validation",
        "writer_labels_held_out_from_mapping": True,
        "training_performed": False,
        "held_writer_folds": totals["held_writer_folds"],
        "held_writer_formulas": totals["held_writer_formulas"],
        "mappable_formulas": totals["mappable_formulas"],
        "unmappable_formulas": totals["unmappable_formulas"],
        "held_writer_label_collisions": totals["held_writer_label_collisions"],
        "top1_reachable_formulas": totals["top1_reachable"],
        "top1_unique_proposals": top1_unique,
        "top1_exact_group_label_matches": totals["top1_exact_group_label_matches"],
        "top1_precision_among_unique_proposals": totals["top1_exact_group_label_matches"] / max(top1_unique, 1),
        "top5_reachable_formulas": totals["top5_reachable"],
        "top5_unique_proposals": top5_unique,
        "top5_exact_group_label_matches": totals["top5_exact_group_label_matches"],
        "top5_precision_among_unique_proposals": totals["top5_exact_group_label_matches"] / max(top5_unique, 1),
        "per_writer_fold_aggregates": fold_rows,
        "limits": [
            "the frozen HWR checkpoint may have seen the accepted ownership data during earlier development",
            "writer-held-out mapping validates proposal-label recovery only, not independent HWR generalization",
            "unique means unique under the mapped token-multiset exact-cover constraint, not unique spatial interpretation",
            "no pseudo-labels are accepted for training; all proposed labels remain human-review-only",
        ],
    }


def _self_test() -> None:
    held_meta = [
        (writer, label)
        for writer in ("w1", "w2", "w3")
        for label in ("a", "b")
    ]
    held_logits = np.zeros((len(held_meta), 372), dtype=np.float32)
    for row, (_writer, label) in enumerate(held_meta):
        held_logits[row, 0 if label == "a" else 1] = 1.0
    held_mapping, held_collisions = _held_writer_label_indices(
        held_meta, held_logits, "w1",
    )
    assert held_mapping == {"a": 0, "b": 1} and held_collisions == 0
    collision_logits = held_logits.copy()
    for row, (_writer, label) in enumerate(held_meta):
        collision_logits[row, 0 if label == "a" else 1] = 0.0
        collision_logits[row, 0] = 1.0
    collision_mapping, collision_count = _held_writer_label_indices(
        held_meta, collision_logits, "w1",
    )
    assert collision_mapping == {} and collision_count == 2

    candidates = [
        {"source_indices": [0]},
        {"source_indices": [1]},
        {"source_indices": [0, 1]},
    ]
    assert _has_labeled_exact_cover(
        candidates, [("a",), ("+",), ("a", "+")], 2, ["a", "+"],
    )
    assert _count_labeled_exact_covers(
        candidates, [("a",), ("+",), ("a", "+")], 2, ["a", "+"],
    ) == 1
    unique_count, unique_path = _solve_labeled_exact_cover(
        candidates, [("a",), ("+",), ("a", "+")], 2, ["a", "+"],
    )
    assert unique_count == 1 and unique_path is not None
    assert sorted(label for _mask, label in unique_path) == ["+", "a"]
    assert not _has_labeled_exact_cover(
        candidates, [("a",), ("+",), ("a", "+")], 2, ["a", "a", "+"],
    )
    multiple_count, multiple_path = _solve_labeled_exact_cover(
        [{"source_indices": [0]}, {"source_indices": [1]}],
        [("a", "+"), ("a", "+")], 2, ["a", "+"],
    )
    assert multiple_count == 2 and multiple_path is None
    assert _count_labeled_exact_covers(
        [{"source_indices": [0]}, {"source_indices": [0]}, {"source_indices": [1]}],
        [("a",), ("a",), ("+",)], 2, ["a", "+"],
    ) == 1


def _write_weak_ownership_review(path: Path, report: dict[str, Any]) -> Path:
    output_path = path.resolve()
    artifacts_root = (ROOT / "artifacts").resolve()
    try:
        output_path.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError("review output must remain under this worktree's artifacts directory") from error
    if any("crohme" in part.lower() for part in output_path.parts):
        raise ValueError("CROHME-named output paths are forbidden")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(output_path.parent)

    payload = {
        "schema": "aiflow-weak-ownership-review/v1",
        "status": "unverified_manual_review_only",
        "training_use_allowed": False,
        "manual_review_required": True,
        "selection_method": "exactly one exact-cover assignment using stable-mapped HWR Top-5 labels and formula token multiset",
        "source": report["source"],
        "vocabulary_recovery": report["vocabulary_recovery"],
        "limits": [
            "output-index labels are inferred from accepted ownership groups rather than a paired checkpoint vocabulary",
            "unique cover ignores formula reading order, spatial relations, and decoder validity",
            "candidates are proposals only and must not enter ownership training until a human verifies them",
        ],
        "candidates": report["transcript_only_reachability"]["pseudo_ownership_review_candidates"],
    }
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument(
        "--review-output", type=Path,
        help="write unique Top-5 weak-ownership proposals under artifacts/; human review only, never training data",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass", "training_performed": False}))
        return 0
    report = audit(args.dataset_root, args.onnx)
    if args.review_output is not None:
        output_path = _write_weak_ownership_review(args.review_output, report)
        report["files_written"] = True
        report["review_artifact_path"] = str(output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
