#!/usr/bin/env python3
"""Microscope trace of the frozen online HWR pipeline, formula by formula.

The report follows raw ink through canonicalization, 128x5 model input, every
Transformer block, pooling, 372-way logits, grouping, and the strict decoder.
It is diagnostic only: no fitting, threshold selection, external collection,
or CROHME access is performed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import torch

import raw_formula_context_runtime_v1 as raw_runtime_module
from build_normalized_ink_v1 import SourceSample, _canonicalize
from character_tensor_v1 import CHANNELS
from evaluate_48hz_prefix_v1 import INPUT_MODE, _load_model, _prefix_tensor, resample_direct_48hz
from evaluate_joint_hwr_grouping_v1 import (
    _candidate_tensor, _probabilities,
)
from formula_layout_v1 import infer_formula_layout, selected_layout_evidence_rows
from selective_decoder_v1 import decode_selective_partition
from stroke_grouping_v1 import build_lattice, candidate_features, enumerate_partitions
from stroke_grouping_v1 import build_lattice, candidate_features, enumerate_partitions
from train_character_classifier_v1 import apply_input_mode
from train_project_owned_grouping_v1 import Sample
from semantic_fence_guard_v1 import apply_semantic_fence_guard
from semantic_infix_guard_v1 import apply_semantic_infix_guard


SCHEMA = "aiflow-hwr-pipeline-microscope/v1"
BATCH_SIZE = 128
TRACE_CODE_FILES = (
    "build_normalized_ink_v1.py",
    "character_tensor_v1.py",
    "evaluate_48hz_prefix_v1.py",
    "evaluate_joint_hwr_grouping_v1.py",
    "formula_layout_v1.py",
    "raw_formula_context_runtime_v1.py",
    "selective_2d_anytime_v1.py",
    "selective_decoder_v1.py",
    "semantic_fence_guard_v1.py",
    "semantic_infix_guard_v1.py",
    "stroke_grouping_v1.py",
    "train_character_classifier_v1.py",
    "trace_hwr_pipeline_layers_v1.py",
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trace_code_hashes() -> dict[str, str]:
    scripts_dir = Path(__file__).resolve().parent
    return {
        f"scripts/{name}": _sha256(scripts_dir / name)
        for name in TRACE_CODE_FILES
    }


def _stable_fingerprint(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _array_summary(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    clean = array[finite]
    return {
        "shape": list(array.shape), "elements": int(array.size),
        "finite_elements": int(finite.sum()), "finite_fraction": float(finite.mean()) if array.size else 1.0,
        "min": float(clean.min()) if clean.size else None,
        "max": float(clean.max()) if clean.size else None,
        "mean": float(clean.mean()) if clean.size else None,
        "sum": float(clean.sum()) if clean.size else None,
        "std": float(clean.std()) if clean.size else None,
        "rms": float(np.sqrt(np.mean(np.square(clean)))) if clean.size else None,
        "zero_fraction": float(np.mean(clean == 0)) if clean.size else None,
    }


def _channel_summary(values: np.ndarray) -> dict[str, dict[str, Any]]:
    return {name: _array_summary(values[..., index]) for index, name in enumerate(CHANNELS)}


def _torch_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"unsupported layer output: {type(output).__name__}")


def _logit_probe(stage: str, logits: np.ndarray, entry: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    order = np.argsort(-values)
    top1_index = int(order[0])
    vocab = entry["label_vocab"]
    target = entry.get("target_label")
    target_index = vocab.index(target) if target in vocab else None
    target_rank = None
    target_margin = None
    target_logit = None
    best_other_token = None
    best_other_logit = None
    if target_index is not None:
        target_rank = int(np.flatnonzero(order == target_index)[0]) + 1
        masked = values.copy()
        masked[target_index] = -np.inf
        best_other_index = int(np.argmax(masked))
        target_logit = float(values[target_index])
        best_other_logit = float(values[best_other_index])
        target_margin = target_logit - best_other_logit
        best_other_token = str(vocab[best_other_index])
    return {
        "stage": stage,
        "predicted_top1": str(vocab[top1_index]),
        "target_in_vocab": target_index is not None,
        "target_rank": target_rank,
        "target_logit": target_logit,
        "best_other_token": best_other_token,
        "best_other_logit": best_other_logit,
        "target_minus_best_other_logit": target_margin,
    }


def _append_intermediate_logit_probe(
    stage: str, hidden: torch.Tensor, model: torch.nn.Module,
    entries: list[dict[str, Any]],
) -> None:
    selected = [
        (index, entry) for index, entry in enumerate(entries)
        if entry.get("phase") == "oracle_group_hwr"
    ]
    if not selected:
        return
    indices = torch.tensor([index for index, _ in selected], device=hidden.device)
    entries = [entry for _, entry in selected]
    hidden = hidden.detach().float().index_select(0, indices)
    pool_logits = torch.nn.functional.linear(
        hidden, model.pool_score.weight.float(),
        model.pool_score.bias.float() if model.pool_score.bias is not None else None,
    ).squeeze(-1)
    weights = torch.softmax(pool_logits, dim=1)
    pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
    logits = torch.nn.functional.linear(
        pooled, model.math_head.weight.float(),
        model.math_head.bias.float() if model.math_head.bias is not None else None,
    ).detach().cpu().numpy()
    if logits.shape[0] != len(entries):
        raise AssertionError(f"logit-probe batch mismatch at {stage}")
    for entry, row_logits in zip(entries, logits, strict=True):
        entry.setdefault("layer_logit_lens", []).append(
            _logit_probe(stage, row_logits, entry),
        )


def _final_probe_parity(entry: dict[str, Any]) -> dict[str, Any]:
    encoder_output = next((
        row for row in entry.get("layer_logit_lens", [])
        if row.get("stage") == "encoder.output"
    ), None)
    final_head = entry.get("final_head_probe")
    if encoder_output is None or final_head is None:
        return {"available": False, "passed": False}
    margin_a = encoder_output.get("target_minus_best_other_logit")
    margin_b = final_head.get("target_minus_best_other_logit")
    margin_delta = (
        abs(float(margin_a) - float(margin_b))
        if margin_a is not None and margin_b is not None else None
    )
    passed = (
        encoder_output.get("predicted_top1") == final_head.get("predicted_top1")
        and encoder_output.get("target_rank") == final_head.get("target_rank")
        and (
            margin_delta is None or margin_delta <= 1e-5
        )
    )
    return {
        "available": True,
        "passed": passed,
        "target_rank_matches": encoder_output.get("target_rank") == final_head.get("target_rank"),
        "top1_matches": encoder_output.get("predicted_top1") == final_head.get("predicted_top1"),
        "target_margin_abs_delta": margin_delta,
    }


def _raw_group_stats(strokes: list[dict], group: list[int]) -> dict[str, Any]:
    selected = [strokes[index] for index in group]
    points = [point for stroke in selected for point in stroke["points"]]
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    times = [float(point.get("t_ms", 0.0)) for point in points]
    duplicate_time_steps = sum(a == b for a, b in zip(times, times[1:]))
    return {
        "stroke_indices": group,
        "stroke_count": len(selected),
        "raw_point_count": len(points),
        "raw_points_per_stroke": [len(stroke["points"]) for stroke in selected],
        "single_point_stroke_count": sum(len(stroke["points"]) == 1 for stroke in selected),
        "raw_duplicate_adjacent_timestamps": int(duplicate_time_steps),
        "raw_time_first_ms": times[0] if times else None,
        "raw_time_last_ms": times[-1] if times else None,
        "raw_time_span_ms": times[-1] - times[0] if len(times) > 1 else 0.0,
        "raw_bbox": {
            "left": min(xs), "top": min(ys), "right": max(xs), "bottom": max(ys),
        } if points else None,
    }


def _candidate_input(sample: Sample, row: dict[str, Any], trace_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    group = [int(value) for value in row["source_indices"]]
    raw_strokes = [
        [(float(point["x"]), float(point["y"]), float(point.get("t_ms", 0.0)))
         for point in sample.strokes[index]["points"]]
        for index in group
    ]
    record = _canonicalize(SourceSample(
        "project_owned_group_candidate", f"{sample.sample_id}:{','.join(map(str, group))}",
        "?", "grouping_candidate", "candidate_only", raw_strokes,
    ))
    replay_strokes = resample_direct_48hz(record)
    replay_tensor = np.asarray(_prefix_tensor(replay_strokes), dtype=np.float32)
    reference_tensor = np.asarray(_candidate_tensor(sample, group), dtype=np.float32)
    same_as_runtime = bool(np.array_equal(replay_tensor, reference_tensor))
    if not same_as_runtime:
        raise AssertionError(f"diagnostic preprocessing diverged from runtime: {trace_id}")
    return replay_tensor, {
        **_raw_group_stats(sample.strokes, group),
        "canonical_point_count": int(record["point_count"]),
        "canonical_stroke_count": int(record["stroke_count"]),
        "canonical_time_kind": record["normalization"]["time_kind"],
        "canonical_duration_ms": record["transform"]["duration_ms"],
        "resampled_points_per_stroke": [int(len(stroke)) for stroke in replay_strokes],
        "resampled_point_count": int(sum(len(stroke) for stroke in replay_strokes)),
        "tensor": _array_summary(replay_tensor),
        "tensor_channels_before_input_mode": _channel_summary(replay_tensor),
        "preprocessing_matches_runtime_bitwise": same_as_runtime,
    }


def _activation_rows(name: str, value: torch.Tensor, entries: list[dict[str, Any]]) -> None:
    array = value.detach().float().cpu().numpy()
    if array.shape[0] != len(entries):
        raise AssertionError(f"layer batch mismatch at {name}")
    for index, entry in enumerate(entries):
        summary = _array_summary(array[index])
        entry["network_layers"][name] = summary
        if name == "pool.score_logits":
            weights = torch.softmax(value.float().squeeze(-1), dim=1).detach().cpu().numpy()[index]
            entropy = float(-(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum())
            entry["pool_attention"] = {
                "weight_sum": float(weights.sum()),
                "entropy_nats": entropy,
                "effective_points": float(math.exp(entropy)),
                "max_weight": float(weights.max()),
                "max_weight_position": int(weights.argmax()),
                "first_quarter_mass": float(weights[:32].sum()),
                "middle_half_mass": float(weights[32:96].sum()),
                "last_quarter_mass": float(weights[96:].sum()),
            }
    if name == "math_head":
        for index, entry in enumerate(entries):
            entry["final_head_probe"] = _logit_probe(name, array[index], entry)


def _install_hooks(model, pending: dict[str, Any], active: dict[str, list[dict[str, Any]]]) -> list[Any]:
    handles = []

    def attach(name: str, module: torch.nn.Module) -> None:
        def hook(_module, _inputs, output):
            value = _torch_output(output)
            if name == "math_head":
                entries = pending.get("entries")
                if entries is None:
                    return
                _activation_rows(name, value, entries)
                pending["entries"] = None
                return
            entries = active.get("entries")
            if entries is not None:
                _activation_rows(name, value, entries)
                if name == "encoder.output" or (
                    name.startswith("encoder.block_") and name.endswith(".output")
                ):
                    _append_intermediate_logit_probe(name, value, model, entries)
        handles.append(module.register_forward_hook(hook))

    attach("input_projection.linear", model.input_projection[0])
    attach("input_projection.layer_norm", model.input_projection[1])
    attach("input_projection.gelu", model.input_projection[2])
    attach("input_projection.output", model.input_projection)
    for layer_index, layer in enumerate(model.encoder.layers):
        prefix = f"encoder.block_{layer_index}"
        attach(prefix + ".self_attention", layer.self_attn)
        attach(prefix + ".norm1", layer.norm1)
        attach(prefix + ".feed_forward_in", layer.linear1)
        attach(prefix + ".feed_forward_out", layer.linear2)
        attach(prefix + ".norm2", layer.norm2)
        attach(prefix + ".output", layer)
    attach("encoder.output", model.encoder)
    attach("pool.score_logits", model.pool_score)
    attach("math_head", model.math_head)

    def encoder_input_hook(_module, inputs):
        entries = active.get("entries")
        if entries is not None:
            _append_intermediate_logit_probe(
                "encoder.input", _torch_output(inputs), model, entries,
            )

    handles.append(model.encoder.register_forward_pre_hook(encoder_input_hook))
    return handles


def _layer_deltas(entry: dict[str, Any]) -> None:
    previous = entry["model_input"]["rms"]
    for name, summary in entry["network_layers"].items():
        if name == "math_head":
            continue
        current = summary.get("rms")
        summary["rms_gain_from_previous_traced_stage"] = (
            float(current / previous) if current is not None and previous not in (None, 0.0) else None
        )
        previous = current


def _set_logits_and_probabilities(
    entries: list[dict[str, Any]], probabilities: np.ndarray, diagnostic_top_k: int = 0,
) -> None:
    if len(entries) != len(probabilities):
        raise AssertionError("candidate/probability row count mismatch")
    for entry, values in zip(entries, probabilities, strict=True):
        order_all = np.argsort(-values)
        order = order_all[:5]
        target = entry.get("target_label")
        vocab = entry["label_vocab"]
        target_rank_full = int(np.flatnonzero(order_all == vocab.index(target))[0]) + 1 if target in vocab else None
        target_rank_top5 = target_rank_full if target_rank_full is not None and target_rank_full <= 5 else None
        target_probability = float(values[vocab.index(target)]) if target in vocab else None
        top5_probabilities = [float(values[int(index)]) for index in order]
        entropy = float(-(values * np.log(np.clip(values, 1e-12, 1.0))).sum() / math.log(len(values)))
        target_log_probability = math.log(max(target_probability, 1e-12)) if target_probability is not None else None
        entry["prediction"] = {
            "top1": vocab[int(order[0])],
            "top1_probability": top5_probabilities[0],
            "top1_top2_probability_margin": top5_probabilities[0] - top5_probabilities[1],
            "top5": [vocab[int(index)] for index in order],
            "top5_probabilities": top5_probabilities,
            "top5_probability_mass": float(sum(top5_probabilities)),
            "normalized_entropy": entropy,
            "target": target,
            "target_rank": target_rank_full,
            "target_rank_top5": target_rank_top5,
            "target_probability": target_probability,
            "target_nll": -target_log_probability if target_log_probability is not None else None,
            "target_in_top5": target_rank_top5 is not None,
            "target_is_top1": target_rank_full == 1,
            "full_probability_distribution": _array_summary(values),
        }
        if diagnostic_top_k and entry.get("phase") == "oracle_group_hwr":
            selected = order_all[:diagnostic_top_k]
            entry["prediction"]["diagnostic_topk"] = {
                "k": diagnostic_top_k,
                "tokens": [vocab[int(index)] for index in selected],
                "probabilities": [float(values[int(index)]) for index in selected],
            }
        parity = _final_probe_parity(entry)
        if entry.get("phase") == "oracle_group_hwr" and not parity["passed"]:
            encoder_output_probe = next((
                row for row in entry.get("layer_logit_lens", [])
                if row.get("stage") == "encoder.output"
            ), None)
            raise AssertionError(
                "intermediate logit probe diverged at final encoder output: "
                f"{entry['trace_id']}; parity={json.dumps(parity, sort_keys=True)}; "
                + "encoder_output=" + json.dumps(encoder_output_probe, sort_keys=True)
                + "; final_head=" + json.dumps(entry.get("final_head_probe"), sort_keys=True)
            )
        entry["layer_logit_lens_final_head_parity"] = (
            parity if entry.get("phase") == "oracle_group_hwr"
            else {"available": False, "passed": None}
        )
        logits = entry["network_layers"].get("math_head")
        if logits:
            logits["softmax_probability_sum"] = float(values.sum())


def _prepare_trace_entry(
    sample: Sample, row: dict[str, Any], *, phase: str, call_index: int,
    label_vocab: list[str], target_label: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    group = [int(value) for value in row["source_indices"]]
    trace_id = f"{sample.sample_id}:{phase}:{call_index}:{','.join(map(str, group))}"
    tensor, preprocessing = _candidate_input(sample, row, trace_id)
    model_input = apply_input_mode(tensor[None, ...], INPUT_MODE)[0]
    entry = {
        "trace_id": trace_id, "formula_id": sample.sample_id,
        "phase": phase, "classifier_call_index": call_index,
        "stroke_indices": group, "target_label": target_label,
        "label_vocab": label_vocab,
        "preprocessing": preprocessing,
        "model_input": _array_summary(model_input),
        "model_input_channels": _channel_summary(model_input),
        "network_layers": {},
    }
    return model_input, entry


def _run_entries(entries_with_inputs: list[tuple[np.ndarray, dict[str, Any]]], model, device, active) -> torch.Tensor:
    if not entries_with_inputs:
        return torch.empty((0, 128), dtype=torch.float32)
    inputs = np.stack([row[0] for row in entries_with_inputs]).astype(np.float32, copy=False)
    entries = [row[1] for row in entries_with_inputs]
    outputs = []
    for start in range(0, len(entries), BATCH_SIZE):
        chunk_entries = entries[start:start + BATCH_SIZE]
        batch = torch.from_numpy(inputs[start:start + BATCH_SIZE]).to(device)
        active["entries"] = chunk_entries
        with torch.inference_mode():
            embedding = model.encode(batch)
            outputs.append(embedding.detach().cpu())
        active["entries"] = None
        for index, entry in enumerate(chunk_entries):
            entry["network_layers"]["pooled_embedding"] = _array_summary(embedding[index])
            if entry["model_input"]["shape"] != [128, 5] or entry["model_input"]["finite_fraction"] != 1.0:
                raise AssertionError(f"invalid model input: {entry['trace_id']}")
            if "encoder.output" not in entry["network_layers"] or "pool.score_logits" not in entry["network_layers"]:
                raise AssertionError(f"missing neural layer trace: {entry['trace_id']}")
            _layer_deltas(entry)
    return torch.cat(outputs, dim=0)


def _hist(values: list[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items(), key=lambda row: str(row[0]))}


def _safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _layer_logit_lens_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a shared-final-head probe; intermediate ranks are not causal metrics."""
    stage_names = [
        row["stage"] for row in (entries[0].get("layer_logit_lens") or [])
    ] if entries else []
    final_misses = [
        entry for entry in entries
        if not bool((entry.get("prediction") or {}).get("target_is_top1"))
    ]
    by_stage = {}
    transitions = []
    previous_rows = None
    for stage in stage_names:
        rows = [
            (entry, next((
                probe for probe in entry.get("layer_logit_lens", [])
                if probe.get("stage") == stage
            ), None))
            for entry in entries
        ]
        rows = [(entry, probe) for entry, probe in rows if probe is not None]
        ranks = [str(probe["target_rank"] or "missing") for _, probe in rows]
        margins = [
            float(probe["target_minus_best_other_logit"])
            for _, probe in rows
            if probe.get("target_minus_best_other_logit") is not None
        ]
        miss_margins = [
            float(probe["target_minus_best_other_logit"])
            for entry, probe in rows
            if not bool((entry.get("prediction") or {}).get("target_is_top1"))
            and probe.get("target_minus_best_other_logit") is not None
        ]
        by_stage[stage] = {
            "candidate_rows": len(rows),
            "target_rank_histogram": _hist(ranks),
            "target_top1_rows": sum(probe.get("target_rank") == 1 for _, probe in rows),
            "mean_target_minus_best_other_logit": _safe_mean(margins),
            "final_hwr_miss_rows": len(final_misses),
            "mean_margin_on_final_hwr_misses": _safe_mean(miss_margins),
            "positive_margin_on_final_hwr_misses": sum(value > 0 for value in miss_margins),
        }
        if previous_rows is not None:
            previous_by_id = {entry["trace_id"]: probe for entry, probe in previous_rows}
            transitions_count: Counter[str] = Counter()
            for entry, probe in rows:
                previous = previous_by_id.get(entry["trace_id"])
                if previous is None or previous.get("target_rank") is None or probe.get("target_rank") is None:
                    transitions_count["rank_unavailable"] += 1
                    continue
                was_correct = previous["target_rank"] == 1
                now_correct = probe["target_rank"] == 1
                transitions_count[
                    "stayed_correct" if was_correct and now_correct else
                    "became_wrong" if was_correct else
                    "became_correct" if now_correct else "stayed_wrong"
                ] += 1
            transitions.append({
                "from": previous_rows[0][1]["stage"],
                "to": stage,
                "counts": dict(sorted(transitions_count.items())),
            })
        previous_rows = rows
    parity_rows = [entry.get("layer_logit_lens_final_head_parity") or {} for entry in entries]
    return {
        "method": "shared_final_pool_and_math_head_on_intermediate_encoder_hidden",
        "interpretation": "diagnostic_only_not_causal_or_promotion_evidence",
        "candidate_rows": len(entries),
        "final_head_parity_rows": sum(bool(row.get("passed")) for row in parity_rows),
        "stage_summaries": by_stage,
        "adjacent_stage_target_rank_transitions": transitions,
    }


def _semantic_guard_shadow(
    formula_id: str, symbols: list[dict[str, Any]], target_tokens: list[str],
) -> dict[str, Any]:
    """Trace fixed Top-5 semantic guards without loading a context checkpoint."""
    if len(symbols) != len(target_tokens):
        raise AssertionError(f"semantic guard symbol/label mismatch: {formula_id}")
    boxes = [dict(row["geometry"] or {}) for row in symbols]
    if not boxes or any(not box for box in boxes):
        raise AssertionError(f"semantic guard geometry missing: {formula_id}")
    left = min(float(box["left"]) for box in boxes)
    right = max(float(box["right"]) for box in boxes)
    top = min(float(box["top"]) for box in boxes)
    bottom = max(float(box["bottom"]) for box in boxes)
    width, height = max(right - left, 1e-8), max(bottom - top, 1e-8)
    rows = []
    for index, (symbol, box) in enumerate(zip(symbols, boxes, strict=True)):
        center_x = (float(box["left"]) + float(box["right"])) / 2.0
        center_y = (float(box["top"]) + float(box["bottom"])) / 2.0
        rows.append({
            "record_id": f"{formula_id}:{index}",
            "formula_id": formula_id,
            "final_topk": list(symbol["hwr_topk"]),
            "final_topk_probabilities": list(symbol["hwr_topk_probabilities"]),
            "context": {"index": index, "length": len(symbols)},
            "geometry": {
                **box,
                "center_x": (center_x - left) / width,
                "center_y": (center_y - top) / height,
                "width_rel": (float(box["right"]) - float(box["left"])) / width,
                "height_rel": (float(box["bottom"]) - float(box["top"])) / height,
            },
        })
    top1 = {row["record_id"]: row["final_topk"][0] for row in rows}
    after_fence, fence_audit = apply_semantic_fence_guard(rows, top1)
    after_infix, infix_audit = apply_semantic_infix_guard(
        rows, after_fence, minimum_probability_ratio=0.0,
    )
    stage_tokens = {
        "hwr_top1": [top1[row["record_id"]] for row in rows],
        "after_fence_guard": [after_fence[row["record_id"]] for row in rows],
        "after_infix_guard": [after_infix[row["record_id"]] for row in rows],
    }
    candidate_preservation = all(
        token in row["final_topk"]
        for stage in stage_tokens.values()
        for token, row in zip(stage, rows, strict=True)
    )
    if not candidate_preservation:
        raise AssertionError(f"semantic guard invented an HWR token: {formula_id}")
    return {
        "scope": "oracle ownership order; guard-only shadow; no context checkpoint",
        "input_top1_tokens": stage_tokens["hwr_top1"],
        "stage_tokens": stage_tokens,
        "exact_by_stage": {
            stage: tokens == target_tokens for stage, tokens in stage_tokens.items()
        },
        "changed_glyphs": {
            "fence": sum(a != b for a, b in zip(stage_tokens["hwr_top1"], stage_tokens["after_fence_guard"], strict=True)),
            "infix": sum(a != b for a, b in zip(stage_tokens["after_fence_guard"], stage_tokens["after_infix_guard"], strict=True)),
        },
        "candidate_preservation": candidate_preservation,
        "fence_audit": fence_audit,
        "infix_audit": infix_audit,
    }


def _decoder_from_symbols(formula_id: str, groups: list[list[int]], symbols: list[dict[str, Any]], stroke_count: int) -> dict[str, Any]:
    return decode_selective_partition(formula_id, groups, symbols, stroke_count=stroke_count)


def _relation_signature(rows: list[dict[str, Any]]) -> set[tuple[int, int, str]]:
    output = set()
    for row in rows:
        parent = str(row.get("parent", ""))
        child = str(row.get("child", ""))
        if ":" not in parent or ":" not in child:
            continue
        output.add((int(parent.rsplit(":", 1)[1]), int(child.rsplit(":", 1)[1]), str(row.get("type", "")).lower()))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition-ranker", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-top-k", type=int, default=0,
        help="optionally retain this many oracle-group HWR candidates for a shadow decoder audit (0 disables)",
    )
    args = parser.parse_args()
    if args.diagnostic_top_k < 0 or args.diagnostic_top_k > 372:
        parser.error("--diagnostic-top-k must be between 0 and 372")
    device = torch.device("cpu")
    torch.set_num_threads(1)
    model, labels, model_report = _load_model(args.checkpoint, device)
    if len(labels) != 372:
        raise AssertionError(f"expected 372 classifier classes, got {len(labels)}")
    ranker_payload = joblib.load(args.partition_ranker)
    grouping_model = ranker_payload["grouping_model"]
    runtime = object.__new__(raw_runtime_module.RawFormulaContextRuntimeV1)
    for name, value in {
        "ranker_payload": ranker_payload, "hwr": model, "device": device, "labels": labels,
    }.items():
        object.__setattr__(runtime, name, value)

    active: dict[str, Any] = {"entries": None}
    pending: dict[str, Any] = {"entries": None}
    handles = _install_hooks(model, pending, active)
    per_formula_calls: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    call_counter: Counter[str] = Counter()

    def traced_candidate_embeddings(samples: list[Sample], current_model, current_device):
        prepared: list[tuple[np.ndarray, dict[str, Any]]] = []
        slices: dict[str, slice] = {}
        offset = 0
        for sample in samples:
            call_index = call_counter[sample.sample_id]
            call_counter[sample.sample_id] += 1
            phase = f"selective_hwr_call_{call_index}"
            current_entries = []
            for row in sample.candidates:
                group = [int(value) for value in row["source_indices"]]
                trace, target = target_lookup.get(sample.sample_id, {}).get(frozenset(group), (None, None))
                tensor_input, entry = _prepare_trace_entry(
                    sample, row, phase=phase, call_index=call_index,
                    label_vocab=labels, target_label=target,
                )
                entry["target_available_for_group"] = trace is not None
                prepared.append((tensor_input, entry)); current_entries.append(entry)
            per_formula_calls[sample.sample_id].append(current_entries)
            slices[sample.sample_id] = slice(offset, offset + len(current_entries))
            offset += len(current_entries)
        embeddings = _run_entries(prepared, current_model, current_device, active)
        pending["entries"] = [entry for _, entry in prepared]
        return embeddings, slices

    original_runtime_embeddings = raw_runtime_module._candidate_embeddings
    original_runtime_probabilities = raw_runtime_module._probabilities
    raw_runtime_module._candidate_embeddings = traced_candidate_embeddings

    def traced_probabilities(head, embeddings, current_device):
        entries = pending.get("entries")
        probabilities = original_runtime_probabilities(head, embeddings, current_device)
        if entries is None:
            raise AssertionError("classifier probability output has no matching candidate trace")
        _set_logits_and_probabilities(entries, probabilities, args.diagnostic_top_k)
        pending["entries"] = None
        return probabilities

    raw_runtime_module._probabilities = traced_probabilities
    try:
        formulas_path = args.dataset_root / "data" / "formulas_valid.jsonl"
        ownership_path = args.dataset_root / "data" / "ownership_train.jsonl"
        formula_source_rows = _read_rows(formulas_path)
        ownership_source_rows = _read_rows(ownership_path)
        formulas = {str(row["sample_id"]): row for row in formula_source_rows}
        ownership_rows = [row for row in ownership_source_rows if row.get("accepted")]
        target_lookup: dict[str, dict[frozenset[int], tuple[int, str]]] = {}
        samples: list[Sample] = []
        for annotation in ownership_rows:
            sample_id = str(annotation["sample_id"])
            source = formulas[sample_id]
            strokes = sorted(source["strokes"], key=lambda row: int(row["order"]))
            groups = [[int(value) for value in group] for group in annotation["groups"]]
            target_tokens = [str(value) for value in annotation["labels"]]
            if len(groups) != len(target_tokens):
                raise AssertionError(f"ownership labels/groups mismatch: {sample_id}")
            flat = [index for group in groups for index in group]
            if sorted(flat) != list(range(len(strokes))) or len(set(flat)) != len(flat):
                raise AssertionError(f"ownership exact cover failed: {sample_id}")
            target_lookup[sample_id] = {
                frozenset(group): (index, target_tokens[index]) for index, group in enumerate(groups)
            }
            candidates = [{"source_indices": group} for group in groups]
            samples.append(Sample(sample_id, str(annotation.get("writer_id", "")), strokes, tuple(), candidates, np.empty((0, 0))))

        # Oracle-group pass: isolate HWR and structural decoding from grouping.
        oracle_entries_by_id: dict[str, list[dict[str, Any]]] = {}
        oracle_probabilities_by_id: dict[str, np.ndarray] = {}
        for sample in samples:
            candidates = []
            for row in sample.candidates:
                group = [int(value) for value in row["source_indices"]]
                _index, target = target_lookup[sample.sample_id][frozenset(group)]
                tensor_input, entry = _prepare_trace_entry(
                    sample, row, phase="oracle_group_hwr", call_index=0,
                    label_vocab=labels, target_label=target,
                )
                candidates.append((tensor_input, entry))
            embeddings = _run_entries(candidates, model, device, active)
            pending["entries"] = [entry for _, entry in candidates]
            probabilities = _probabilities(model.math_head, embeddings, device)
            pending["entries"] = None
            _set_logits_and_probabilities(
                [entry for _, entry in candidates], probabilities, args.diagnostic_top_k,
            )
            oracle_entries_by_id[sample.sample_id] = [entry for _, entry in candidates]
            oracle_probabilities_by_id[sample.sample_id] = probabilities

        # Runtime selective grouping pass. The monkey-patched embedding seam
        # returns the same tensors/embeddings while recording every classifier call.
        selective_by_id: dict[str, dict[str, Any]] = {}
        for sample in samples:
            source = formulas[sample.sample_id]
            selective_by_id[sample.sample_id] = runtime.selective_grouping_preview(source)

        # Attach probabilities returned by the real preview to the captured
        # layer traces; candidate group IDs are the stable join key.
        for sample in samples:
            preview = selective_by_id[sample.sample_id]
            final_call_entries = per_formula_calls[sample.sample_id]
            symbol_entries: dict[tuple[int, ...], dict[str, Any]] = {}
            for call_entries in final_call_entries:
                for entry in call_entries:
                    symbol_entries[tuple(entry["stroke_indices"])] = entry
            for symbol in preview["symbols"]:
                key = tuple(int(value) for value in symbol["stroke_indices"])
                entry = symbol_entries.get(key)
                if entry is None:
                    raise AssertionError(f"selected symbol lacks captured neural trace: {sample.sample_id}:{key}")
                prediction = entry.get("prediction")
                if prediction is None or prediction["top5"] != [str(value) for value in symbol["hwr_topk"]]:
                    raise AssertionError(f"runtime Top-5 differs from captured classifier logits: {sample.sample_id}:{key}")
                entry["runtime_top5_matches_captured_logits"] = True

        # Per-formula data and contract checks.
        formula_traces = []
        layer_aggregates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        runtime_layer_aggregates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        label_confusions: Counter[tuple[str, str]] = Counter()
        missing_vocab_targets: Counter[str] = Counter()
        target_ranks: Counter[str] = Counter()
        length_stats: dict[int, dict[str, int]] = defaultdict(lambda: {"formulas": 0, "top1_exact": 0, "top5_complete": 0})
        checks_total: Counter[str] = Counter()
        checks_passed: Counter[str] = Counter()
        token_total = token_top1 = token_top5 = 0
        for sample, annotation in zip(samples, ownership_rows, strict=True):
            sample_id = sample.sample_id
            source = formulas[sample_id]
            target_groups = [list(map(int, group)) for group in annotation["groups"]]
            target_tokens = [str(value) for value in annotation["labels"]]
            oracle_entries = oracle_entries_by_id[sample_id]
            oracle_predictions = [entry["prediction"] for entry in oracle_entries]
            oracle_top1_exact = [row["top1"] for row in oracle_predictions] == target_tokens
            oracle_top5_complete = all(row["target_in_top5"] for row in oracle_predictions)
            oracle_symbols = []
            for entry in oracle_entries:
                prediction = entry["prediction"]
                oracle_symbols.append({
                    "stroke_indices": entry["stroke_indices"],
                    "hwr_topk": prediction["top5"],
                    "hwr_topk_probabilities": prediction["top5_probabilities"],
                    "geometry": entry["preprocessing"]["raw_bbox"],
                })
            oracle_decoder = decode_selective_partition(
                sample_id, target_groups, oracle_symbols, stroke_count=len(sample.strokes),
            )
            semantic_guard_trace = _semantic_guard_shadow(
                sample_id, oracle_symbols, target_tokens,
            )
            oracle_decoder_tokens = oracle_decoder.get("tokens") or []
            oracle_decoder_exact = bool(oracle_decoder.get("accepted") and oracle_decoder_tokens == target_tokens)
            oracle_decoder_changed = bool(oracle_decoder.get("accepted") and oracle_decoder_tokens != [row["top1"] for row in oracle_predictions])
            target_relation_signature = {
                (int(edge["from"]), int(edge["to"]), str(edge["type"]).lower())
                for edge in (source.get("target_relations") or [])
            }
            oracle_relation_signature = _relation_signature(oracle_decoder.get("relations") or [])
            oracle_relation_exact = oracle_relation_signature == target_relation_signature
            token_total += len(target_tokens)
            token_top1 += sum(bool(row["target_is_top1"]) for row in oracle_predictions)
            token_top5 += sum(bool(row["target_in_top5"]) for row in oracle_predictions)
            for row in oracle_predictions:
                target_ranks[str(row["target_rank"] or "missing")] += 1
                if row["target_rank"] is None:
                    missing_vocab_targets[str(row["target"])] += 1
                if row["top1"] != row["target"]:
                    label_confusions[(str(row["target"]), str(row["top1"]))] += 1
            length_stats[len(target_tokens)]["formulas"] += 1
            length_stats[len(target_tokens)]["top1_exact"] += int(oracle_top1_exact)
            length_stats[len(target_tokens)]["top5_complete"] += int(oracle_top5_complete)

            preview = selective_by_id[sample_id]
            selected_groups = [[int(value) for value in group] for group in preview["groups"]]
            lattice = build_lattice(sample.strokes, temporal_window=6, spatial_neighbors=4)
            lattice_probabilities = grouping_model.predict_proba(candidate_features(lattice, sample.strokes))[:, 1]
            clipped_probabilities = np.clip(lattice_probabilities, 1e-6, 1.0 - 1e-6)
            lattice_logits = np.log(clipped_probabilities / (1.0 - clipped_probabilities))
            lattice_by_group = {
                frozenset(row["source_indices"]): (row, float(probability), float(logit))
                for row, probability, logit in zip(lattice, lattice_probabilities, lattice_logits, strict=True)
            }
            top_partitions = enumerate_partitions(
                lattice, lattice_logits.tolist(), len(sample.strokes), top_n=32,
                beam_width=32, options_per_stroke=64,
            )
            if not top_partitions:
                raise AssertionError(f"grouping selector produced no exact cover: {sample_id}")
            fast_groups = [sorted(group) for group in top_partitions[0][1]]
            fast_group_exact = _set_groups(fast_groups) == _set_groups(target_groups)
            fast_group_score = float(sum(lattice_by_group[frozenset(group)][2] for group in fast_groups))
            target_keys = [frozenset(group) for group in target_groups]
            target_groups_in_lattice = all(key in lattice_by_group for key in target_keys)
            target_partition_score = float(sum(lattice_by_group[key][2] for key in target_keys)) if target_groups_in_lattice else None
            target_partition_rank = next((
                index + 1 for index, row in enumerate(top_partitions)
                if _set_groups([list(group) for group in row[1]]) == _set_groups(target_groups)
            ), None)
            fast_ranked_groups_match_preview = (
                _set_groups(fast_groups) == _set_groups(selected_groups) if preview["route"] == "fast" else None
            )
            group_candidate_rows = []
            for target_index, target_group in enumerate(target_groups):
                item = lattice_by_group.get(frozenset(target_group))
                group_candidate_rows.append({
                    "target_group_index": target_index,
                    "stroke_indices": target_group,
                    "target_token": target_tokens[target_index],
                    "present_in_fast_lattice": item is not None,
                    "grouping_probability_if_present": item[1] if item else None,
                    "grouping_logit_if_present": item[2] if item else None,
                })
            target_by_group = target_lookup[sample_id]
            selected_group_exact = _set_groups(selected_groups) == _set_groups(target_groups)
            selected_tokens = preview["decoder"].get("tokens") or []
            output_order_target_tokens = [
                target_lookup[sample_id][frozenset(group)][1]
                for group in selected_groups if frozenset(group) in target_lookup[sample_id]
            ]
            selected_symbol_rows = []
            for ordinal, symbol in enumerate(preview["symbols"]):
                group_key = frozenset(int(value) for value in symbol["stroke_indices"])
                truth_row = target_by_group.get(group_key)
                entry = next((row for row in per_formula_calls[sample_id][-1] if row["stroke_indices"] == sorted(group_key)), None)
                if entry is None:
                    entry = next((row for call_rows in reversed(per_formula_calls[sample_id]) for row in call_rows if row["stroke_indices"] == sorted(group_key)), None)
                selected_symbol_rows.append({
                    "ordinal": ordinal,
                    "stroke_indices": sorted(group_key),
                    "group_exact_with_ownership": truth_row is not None,
                    "target_token_if_group_exact": truth_row[1] if truth_row is not None else None,
                    "selected_token": selected_tokens[ordinal] if ordinal < len(selected_tokens) else None,
                    "group_probability": lattice_by_group[group_key][1] if group_key in lattice_by_group else None,
                    "group_score_source": "frozen_fast_lattice" if group_key in lattice_by_group else "local_candidate_not_in_fast_lattice",
                    "hwr_trace_id": entry["trace_id"] if entry else None,
                    "runtime_top5_matches_captured_logits": bool(entry.get("runtime_top5_matches_captured_logits")) if entry else False,
                    "hwr_target_rank_if_group_exact": (
                        entry["prediction"]["target_rank"] if truth_row is not None and entry and entry.get("prediction") else None
                    ),
                    "preprocessing": entry["preprocessing"] if entry else None,
                    "model_input_channels": entry["model_input_channels"] if entry else None,
                    "pool_attention": entry.get("pool_attention") if entry else None,
                    "network_layers": entry["network_layers"] if entry else None,
                    "hwr_prediction": entry.get("prediction") if entry else None,
                })
            decoder = preview["decoder"]
            selected_top1_exact = bool(
                selected_group_exact and len(selected_symbol_rows) == len(target_tokens)
                and all(row["hwr_prediction"] and row["hwr_prediction"]["top1"] == row["target_token_if_group_exact"] for row in selected_symbol_rows)
            )
            exact_selected_tokens = bool(
                selected_group_exact and len(selected_tokens) == len(output_order_target_tokens) == len(target_tokens)
                and selected_tokens == output_order_target_tokens
            )
            point_counts = [len(stroke["points"]) for stroke in sample.strokes]
            raw_formula_points = sum(point_counts)
            checks = {
                "raw_stroke_orders_contiguous": [int(stroke["order"]) for stroke in sample.strokes] == list(range(len(sample.strokes))),
                "ownership_exact_cover": sorted(index for group in target_groups for index in group) == list(range(len(sample.strokes))),
                "oracle_candidate_count_matches_target_tokens": len(oracle_entries) == len(target_tokens),
                "all_preprocessing_bitwise_runtime_parity": all(entry["preprocessing"]["preprocessing_matches_runtime_bitwise"] for entry in oracle_entries),
                "all_tensor_contracts_128x5_finite": all(entry["model_input"]["shape"] == [128, 5] and entry["model_input"]["finite_fraction"] == 1.0 for entry in oracle_entries),
                "uniform_time_transform_and_stroke_starts_valid": all(
                    abs(entry["model_input_channels"]["observed"]["min"] - 1.0) < 1e-8
                    and abs(entry["model_input_channels"]["observed"]["max"] - 1.0) < 1e-8
                    and abs(entry["model_input_channels"]["delta_t"]["sum"] - 1.0) < 1e-5
                    and int(round(entry["model_input_channels"]["stroke_start"]["sum"])) == entry["preprocessing"]["stroke_count"]
                    for entry in oracle_entries
                ),
                "all_encoder_and_head_outputs_finite": all(all(stage["finite_fraction"] == 1.0 for stage in entry["network_layers"].values()) for entry in oracle_entries),
                "layer_logit_lens_final_head_parity": all(
                    bool(entry.get("layer_logit_lens_final_head_parity", {}).get("passed"))
                    for entry in oracle_entries
                ),
                "all_probabilities_sum_to_one": all(abs(entry["network_layers"]["math_head"]["softmax_probability_sum"] - 1.0) <= 1e-5 for entry in oracle_entries),
                "top5_labels_unique_and_length_5": all(len(entry["prediction"]["top5"]) == 5 and len(set(entry["prediction"]["top5"])) == 5 for entry in oracle_entries),
                "selected_partition_exact_cover": sorted(index for group in selected_groups for index in group) == list(range(len(sample.strokes))) and len({index for group in selected_groups for index in group}) == len(sample.strokes),
                "decoder_tokens_stay_in_hwr_top5": len(selected_tokens) == len(preview["symbols"]) and all(token in symbol["hwr_topk"] for token, symbol in zip(selected_tokens, preview["symbols"], strict=True)),
                "semantic_guard_stages_preserve_top5": semantic_guard_trace["candidate_preservation"],
                "runtime_top5_matches_captured_logits": all(
                    bool(row.get("runtime_top5_matches_captured_logits")) for row in selected_symbol_rows
                ),
            }
            for name, passed in checks.items():
                checks_total[name] += 1
                checks_passed[name] += int(passed)
            for entry in oracle_entries:
                for name, metrics in entry["network_layers"].items():
                    layer_aggregates[name].append(metrics)
            for symbol_row in selected_symbol_rows:
                for name, metrics in (symbol_row.get("network_layers") or {}).items():
                    runtime_layer_aggregates[name].append(metrics)

            # Reconstruct the public layout stage from the selected decoder tokens,
            # retaining the decoder's exact Top-5 candidate probabilities.
            layout_audit = None
            if decoder.get("accepted"):
                layout_rows = []
                for index, symbol in enumerate(preview["symbols"]):
                    layout_rows.append({
                        "record_id": f"{sample_id}:{index}", "formula_id": sample_id,
                        "final_topk": symbol["hwr_topk"],
                        "final_topk_probabilities": symbol["hwr_topk_probabilities"],
                        "geometry": symbol["geometry"],
                    })
                evidence, evidence_audit = selected_layout_evidence_rows(
                    layout_rows,
                    {f"{sample_id}:{index}": str(token) for index, token in enumerate(selected_tokens)},
                )
                layout = infer_formula_layout(evidence)
                layout_audit = {
                    "relation_edges": layout["edges"],
                    "evidence_audit": evidence_audit,
                    "edge_count": len(layout["edges"]),
                }
            formula_traces.append({
                "sample_id": sample_id,
                "source": {
                    "stroke_count": len(sample.strokes),
                    "raw_point_count": raw_formula_points,
                    "raw_points_per_stroke": point_counts,
                    "single_point_strokes": [index for index, count in enumerate(point_counts) if count == 1],
                    "target_tokens": target_tokens,
                    "target_grouping": target_groups,
                    "target_relations": source.get("target_relations") or [],
                },
                "oracle_group_hwr": {
                    "symbols": [
                        {key: value for key, value in entry.items() if key != "label_vocab"}
                        for entry in oracle_entries
                    ],
                    "token_count": len(target_tokens),
                    "top1_token_hits": sum(bool(row["target_is_top1"]) for row in oracle_predictions),
                    "top5_token_hits": sum(bool(row["target_in_top5"]) for row in oracle_predictions),
                    "top1_formula_exact": oracle_top1_exact,
                    "top5_formula_complete": oracle_top5_complete,
                    "decoder": {
                        "accepted": bool(oracle_decoder.get("accepted")),
                        "reason": oracle_decoder.get("reason"),
                        "tokens": oracle_decoder_tokens,
                        "latex": oracle_decoder.get("latex"),
                        "relations": oracle_decoder.get("relations") or [],
                        "exact_if_ordered_by_target_cells": oracle_decoder_exact,
                        "changed_any_top1_token": oracle_decoder_changed,
                        "relation_signature": [list(edge) for edge in sorted(oracle_relation_signature)],
                        "target_relation_signature": [list(edge) for edge in sorted(target_relation_signature)],
                        "relations_exact": oracle_relation_exact if target_relation_signature else None,
                    },
                },
                "semantic_guard_shadow": semantic_guard_trace,
                "grouping": {
                    "route": preview["route"],
                    "groups": selected_groups,
                    "group_exact": selected_group_exact,
                    "selected_hwr_top1_exact_by_group": selected_top1_exact,
                    "fast_groups": fast_groups,
                    "fast_group_exact": fast_group_exact,
                    "fast_ranked_groups_match_preview": fast_ranked_groups_match_preview,
                    "fast_lattice_top32_partition_count": len(top_partitions),
                    "fast_lattice_candidate_count": len(lattice),
                    "fast_lattice_candidates_by_group_size": _hist([len(row["source_indices"]) for row in lattice]),
                    "target_group_candidate_count": sum(row["present_in_fast_lattice"] for row in group_candidate_rows),
                    "target_group_candidate_recall": sum(row["present_in_fast_lattice"] for row in group_candidate_rows) / max(len(group_candidate_rows), 1),
                    "target_partition_score": target_partition_score,
                    "fast_partition_score": fast_group_score,
                    "target_partition_rank_in_top32": target_partition_rank,
                    "target_group_candidates": group_candidate_rows,
                    "candidate_groups_fast": preview["audit"].get("candidate_groups_fast"),
                    "candidate_groups_local": preview["audit"].get("candidate_groups_local"),
                    "skip_reason": preview["audit"].get("skip_reason"),
                    "region_budget_violations": preview["audit"].get("region_budget_violations"),
                    "risk_reasons": preview["audit"].get("risk_reasons"),
                    "risk_strokes": preview["audit"].get("risk_strokes"),
                    "fast_incumbent_groups": preview["audit"].get("fast_incumbent_groups"),
                    "local_region_seed_strokes": preview["audit"].get("local_region_seed_strokes"),
                    "local_region_strokes": preview["audit"].get("local_region_strokes"),
                    "locked_fast_groups": preview["audit"].get("locked_fast_groups"),
                    "local_processed_stroke_fraction": preview["audit"].get("local_processed_stroke_fraction"),
                    "local_requested_stroke_fraction": preview["audit"].get("local_requested_stroke_fraction"),
                    "region_fraction_exception": preview["audit"].get("region_fraction_exception"),
                    "partition_schedule_completed": preview["audit"].get("partition_schedule_completed"),
                    "incumbent_score": preview["audit"].get("incumbent_score"),
                    "winner_score": preview["audit"].get("winner_score"),
                    "winner_score_delta": preview["audit"].get("winner_score_delta"),
                    "selected_symbols": selected_symbol_rows,
                },
                "layout_and_decoder": {
                    "decoder_accepted": bool(decoder.get("accepted")),
                    "decoder_reason": decoder.get("reason"),
                    "decoder_tokens": selected_tokens,
                    "decoder_latex": preview.get("formula_latex"),
                    "decoder_joint_score": decoder.get("joint_token_relation_score"),
                    "decoder_relation_log_score": decoder.get("relation_log_score"),
                    "layout": layout_audit,
                    "selected_token_sequence_exact_if_groups_correct": exact_selected_tokens,
                    "expected_tokens_in_selected_group_order": output_order_target_tokens if selected_group_exact else None,
                },
                "checks": checks,
            })

        all_oracle_entries = [entry for sample_id in oracle_entries_by_id for entry in oracle_entries_by_id[sample_id]]
        all_oracle_predictions = [entry["prediction"] for entry in all_oracle_entries]
        wrong_rows = [entry for entry in all_oracle_entries if not entry["prediction"]["target_is_top1"]]
        confidence_correct = [entry["prediction"]["top1_probability"] for entry in all_oracle_entries if entry["prediction"]["target_is_top1"]]
        confidence_wrong = [entry["prediction"]["top1_probability"] for entry in wrong_rows]
        input_channel_transition = {}
        for channel in CHANNELS:
            before = [entry["preprocessing"]["tensor_channels_before_input_mode"][channel] for entry in all_oracle_entries]
            after = [entry["model_input_channels"][channel] for entry in all_oracle_entries]
            input_channel_transition[channel] = {
                "before_input_mode_mean_of_candidate_means": _safe_mean([float(row["mean"]) for row in before]),
                "before_input_mode_mean_candidate_std": _safe_mean([float(row["std"]) for row in before]),
                "after_input_mode_mean_of_candidate_means": _safe_mean([float(row["mean"]) for row in after]),
                "after_input_mode_mean_candidate_std": _safe_mean([float(row["std"]) for row in after]),
                "after_input_mode_mean_min": _safe_mean([float(row["min"]) for row in after]),
                "after_input_mode_mean_max": _safe_mean([float(row["max"]) for row in after]),
            }
        confusion_pairs = [
            {"target": target, "top1": predicted, "count": count}
            for (target, predicted), count in label_confusions.most_common(25)
        ]
        selective_group_exact = sum(bool(trace["grouping"]["group_exact"]) for trace in formula_traces)
        selective_decoder_exact = sum(bool(trace["layout_and_decoder"]["selected_token_sequence_exact_if_groups_correct"]) for trace in formula_traces)
        fast_group_exact_count = sum(bool(trace["grouping"]["fast_group_exact"]) for trace in formula_traces)
        target_group_count = sum(len(trace["source"]["target_grouping"]) for trace in formula_traces)
        target_group_candidates_present = sum(trace["grouping"]["target_group_candidate_count"] for trace in formula_traces)
        target_partition_rank_histogram = _hist([
            trace["grouping"]["target_partition_rank_in_top32"] or ">32/not_enumerated"
            for trace in formula_traces
        ])
        oracle_decoder_exact_count = sum(bool(trace["oracle_group_hwr"]["decoder"]["exact_if_ordered_by_target_cells"]) for trace in formula_traces)
        oracle_decoder_recovered = sum(
            bool(trace["oracle_group_hwr"]["decoder"]["exact_if_ordered_by_target_cells"])
            and not bool(trace["oracle_group_hwr"]["top1_formula_exact"])
            for trace in formula_traces
        )
        oracle_decoder_regressed = sum(
            not bool(trace["oracle_group_hwr"]["decoder"]["exact_if_ordered_by_target_cells"])
            and bool(trace["oracle_group_hwr"]["top1_formula_exact"])
            for trace in formula_traces
        )
        structural_traces = [trace for trace in formula_traces if trace["source"]["target_relations"]]
        semantic_guard_stages = ("hwr_top1", "after_fence_guard", "after_infix_guard")
        semantic_guard_summary = {}
        for stage in semantic_guard_stages:
            exact_count = sum(
                bool(trace["semantic_guard_shadow"]["exact_by_stage"][stage])
                for trace in formula_traces
            )
            token_hits = sum(
                token == target
                for trace in formula_traces
                for token, target in zip(
                    trace["semantic_guard_shadow"]["stage_tokens"][stage],
                    trace["source"]["target_tokens"], strict=True,
                )
            )
            changed = sum(
                trace["semantic_guard_shadow"]["stage_tokens"][stage]
                != trace["semantic_guard_shadow"]["stage_tokens"]["hwr_top1"]
                for trace in formula_traces
            )
            stage_glyphs_changed = sum(
                token != baseline
                for trace in formula_traces
                for token, baseline in zip(
                    trace["semantic_guard_shadow"]["stage_tokens"][stage],
                    trace["semantic_guard_shadow"]["stage_tokens"]["hwr_top1"], strict=True,
                )
            )
            stage_glyphs_recovered = sum(
                token == target and baseline != target
                for trace in formula_traces
                for token, baseline, target in zip(
                    trace["semantic_guard_shadow"]["stage_tokens"][stage],
                    trace["semantic_guard_shadow"]["stage_tokens"]["hwr_top1"],
                    trace["source"]["target_tokens"], strict=True,
                )
            )
            stage_glyphs_regressed = sum(
                token != target and baseline == target
                for trace in formula_traces
                for token, baseline, target in zip(
                    trace["semantic_guard_shadow"]["stage_tokens"][stage],
                    trace["semantic_guard_shadow"]["stage_tokens"]["hwr_top1"],
                    trace["source"]["target_tokens"], strict=True,
                )
            )
            recovered = sum(
                bool(trace["semantic_guard_shadow"]["exact_by_stage"][stage])
                and not bool(trace["semantic_guard_shadow"]["exact_by_stage"]["hwr_top1"])
                for trace in formula_traces
            )
            regressed = sum(
                not bool(trace["semantic_guard_shadow"]["exact_by_stage"][stage])
                and bool(trace["semantic_guard_shadow"]["exact_by_stage"]["hwr_top1"])
                for trace in formula_traces
            )
            semantic_guard_summary[stage] = {
                "token_hits": token_hits,
                "token_accuracy": token_hits / token_total,
                "formula_exact": exact_count,
                "formula_exact_rate": exact_count / len(samples),
                "changed_formulas_vs_hwr_top1": changed,
                "changed_glyphs_vs_hwr_top1": stage_glyphs_changed,
                "recovered_glyphs_vs_hwr_top1": stage_glyphs_recovered,
                "regressed_glyphs_vs_hwr_top1": stage_glyphs_regressed,
                "recovered_vs_hwr_top1": recovered,
                "regressed_vs_hwr_top1": regressed,
            }
        formula_source_sha256 = _sha256(formulas_path)
        ownership_source_sha256 = _sha256(ownership_path)
        checkpoint_sha256 = _sha256(args.checkpoint)
        partition_ranker_sha256 = _sha256(args.partition_ranker)
        trace_script_sha256 = _sha256(Path(__file__).resolve())
        code_hashes = _trace_code_hashes()
        reproducibility_inputs = {
            "dataset_files": {
                "data/formulas_valid.jsonl": formula_source_sha256,
                "data/ownership_train.jsonl": ownership_source_sha256,
            },
            "artifacts": {
                "hwr_checkpoint_sha256": checkpoint_sha256,
                "partition_ranker_sha256": partition_ranker_sha256,
            },
            "code_sha256": code_hashes,
            "runtime": {
                "python": sys.version.split()[0],
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "device": str(device),
                "input_mode": INPUT_MODE,
                "diagnostic_top_k": args.diagnostic_top_k,
            },
        }
        reproducibility_fingerprint = _stable_fingerprint(reproducibility_inputs)
        summary = {
            "schema": SCHEMA,
            "scope": "frozen project-owned diagnostic only; no training; no CROHME; no product promotion",
            "reproducibility": {
                "schema": "aiflow-hwr-microscope-run-fingerprint/v1",
                "sha256": reproducibility_fingerprint,
                "inputs": reproducibility_inputs,
            },
            "runtime": {"device": str(device), "input_mode": INPUT_MODE, "input_shape": [128, 5], "class_count": len(labels),
                        "diagnostic_top_k": args.diagnostic_top_k,
                        "model_contract": {
                            "schema": model_report.get("schema"), "mode": model_report.get("mode"),
                            "head_mode": model_report.get("head_mode"), "input_contract": model_report.get("input_contract"),
                            "calibration_steps": model_report.get("calibration", {}).get("steps"),
                        },
                        "checkpoint_sha256": checkpoint_sha256,
                        "partition_ranker_sha256": partition_ranker_sha256,
                        "trace_script_sha256": trace_script_sha256},
            "dataset": {"formula_count": len(samples), "source_formula_rows": len(formulas), "oracle_target_tokens": token_total,
                        "raw_strokes": sum(len(sample.strokes) for sample in samples),
                        "raw_points": sum(sum(len(stroke["points"]) for stroke in sample.strokes) for sample in samples),
                        "source_files": {
                            "formulas_valid.jsonl": {
                                "path_relative_to_dataset_root": "data/formulas_valid.jsonl",
                                "sha256": formula_source_sha256,
                                "rows": len(formula_source_rows),
                            },
                            "ownership_train.jsonl": {
                                "path_relative_to_dataset_root": "data/ownership_train.jsonl",
                                "sha256": ownership_source_sha256,
                                "rows": len(ownership_source_rows),
                                "accepted_rows": len(ownership_rows),
                            },
                        }},
            "oracle_group_hwr": {
                "top1_token_hits": token_top1, "token_top1_accuracy": token_top1 / token_total,
                "top5_token_hits": token_top5, "token_top5_recall": token_top5 / token_total,
                "formula_top1_exact": sum(bool(trace["oracle_group_hwr"]["top1_formula_exact"]) for trace in formula_traces),
                "formula_top1_exact_rate": sum(bool(trace["oracle_group_hwr"]["top1_formula_exact"]) for trace in formula_traces) / len(samples),
                "formula_top5_complete": sum(bool(trace["oracle_group_hwr"]["top5_formula_complete"]) for trace in formula_traces),
                "formula_top5_complete_rate": sum(bool(trace["oracle_group_hwr"]["top5_formula_complete"]) for trace in formula_traces) / len(samples),
                "target_rank_histogram": {str(rank): int(count) for rank, count in sorted(target_ranks.items())},
                "target_labels_absent_from_372_class_vocabulary": dict(missing_vocab_targets),
                "input_channel_transition": input_channel_transition,
                "target_rank_2_to_5_tokens": sum(count for rank, count in target_ranks.items() if rank in {"2", "3", "4", "5"}),
                "target_rank_below_top5_or_missing_tokens": sum(count for rank, count in target_ranks.items() if rank == "missing" or (rank.isdigit() and int(rank) > 5)),
                "mean_top1_confidence_correct": _safe_mean(confidence_correct),
                "mean_top1_confidence_wrong": _safe_mean(confidence_wrong),
                "mean_target_nll": _safe_mean([float(row["target_nll"]) for row in all_oracle_predictions if row["target_nll"] is not None]),
                "strict_decoder_formula_exact": oracle_decoder_exact_count,
                "strict_decoder_recovered_over_top1": oracle_decoder_recovered,
                "strict_decoder_regressed_from_top1": oracle_decoder_regressed,
                "strict_decoder_changed_token_formula_count": sum(bool(trace["oracle_group_hwr"]["decoder"]["changed_any_top1_token"]) for trace in formula_traces),
                "target_relation_formula_count": len(structural_traces),
                "target_relation_exact_count": sum(bool(trace["oracle_group_hwr"]["decoder"]["relations_exact"]) for trace in structural_traces),
                "common_target_to_top1_confusions": confusion_pairs,
                "formula_exact_by_token_count": {str(size): values for size, values in sorted(length_stats.items())},
                "captured_network_layer_row_count": {name: len(values) for name, values in layer_aggregates.items()},
                "network_layer_activation_summary": {
                    name: {
                        "candidate_rows": len(rows),
                        "mean_rms": _safe_mean([float(row["rms"]) for row in rows if row.get("rms") is not None]),
                        "mean_abs_max": _safe_mean([max(abs(float(row["min"])), abs(float(row["max"]))) for row in rows if row.get("min") is not None and row.get("max") is not None]),
                        "nonfinite_rows": sum(row["finite_fraction"] < 1.0 for row in rows),
                    } for name, rows in layer_aggregates.items()
                },
                "layer_logit_lens_diagnostic": _layer_logit_lens_summary(all_oracle_entries),
                "runtime_selected_network_layer_row_count": {name: len(values) for name, values in runtime_layer_aggregates.items()},
                "runtime_selected_network_layer_activation_summary": {
                    name: {
                        "candidate_rows": len(rows),
                        "mean_rms": _safe_mean([float(row["rms"]) for row in rows if row.get("rms") is not None]),
                        "nonfinite_rows": sum(row["finite_fraction"] < 1.0 for row in rows),
                    } for name, rows in runtime_layer_aggregates.items()
                },
            },
            "selective_grouping": {
                "fast_group_exact": fast_group_exact_count,
                "group_exact": selective_group_exact,
                "group_exact_rate": selective_group_exact / len(samples),
                "decoder_token_exact_given_exact_groups": selective_decoder_exact,
                "target_group_candidate_count": target_group_candidates_present,
                "target_group_count": target_group_count,
                "target_group_candidate_recall": target_group_candidates_present / max(target_group_count, 1),
                "target_partition_rank_in_fast_top32": target_partition_rank_histogram,
                "fast_ranked_partition_matches_runtime_preview": sum(trace["grouping"]["fast_ranked_groups_match_preview"] is True for trace in formula_traces),
                "local_2d_formula_wins": sum(trace["grouping"]["route"] == "local_2d" for trace in formula_traces),
                "structural_target_count": sum(bool(trace["source"]["target_relations"]) for trace in formula_traces),
                "structural_scout_recall": (
                    sum("structural_layout" in (trace["grouping"]["risk_reasons"] or []) for trace in formula_traces if trace["source"]["target_relations"])
                    / max(sum(bool(trace["source"]["target_relations"]) for trace in formula_traces), 1)
                ),
            },
            "semantic_guard_shadow": {
                "scope": "oracle ownership groups/order; syntax guards only; no context checkpoint; shadow metrics only",
                "minimum_probability_ratio": 0.0,
                "product_default_enabled": False,
                "stages": semantic_guard_summary,
                "candidate_preservation_rate": (
                    sum(bool(trace["semantic_guard_shadow"]["candidate_preservation"]) for trace in formula_traces)
                    / len(formula_traces)
                ),
                "fence_changed_glyphs": sum(
                    int(trace["semantic_guard_shadow"]["changed_glyphs"]["fence"])
                    for trace in formula_traces
                ),
                "infix_changed_glyphs": sum(
                    int(trace["semantic_guard_shadow"]["changed_glyphs"]["infix"])
                    for trace in formula_traces
                ),
            },
            "verification": {
                "oracle_top1_token_hits_match_previous_470": token_top1 == 470,
                "oracle_top5_token_hits_match_previous_562": token_top5 == 562,
                "oracle_top1_formula_count_matches_previous_76": sum(bool(trace["oracle_group_hwr"]["top1_formula_exact"]) for trace in formula_traces) == 76,
                "oracle_top5_formula_count_matches_previous_137": sum(bool(trace["oracle_group_hwr"]["top5_formula_complete"]) for trace in formula_traces) == 137,
                "oracle_layer_logit_lens_matches_final_head": all(
                    bool(entry.get("layer_logit_lens_final_head_parity", {}).get("passed"))
                    for entry in all_oracle_entries
                ),
                "checks_total": dict(checks_total),
                "checks_passed": dict(checks_passed),
                "all_checks_pass": checks_total == checks_passed,
            },
            "top1_confusions": confusion_pairs,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = args.output_dir / "formula_traces.jsonl"
        trace_path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in formula_traces), encoding="utf-8")
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in summary.items() if key not in {"oracle_group_hwr", "verification", "top1_confusions"}}, ensure_ascii=False))
        print(json.dumps({"oracle_group_hwr": summary["oracle_group_hwr"], "verification": summary["verification"], "top_confusions": confusion_pairs[:10]}, ensure_ascii=False))
    finally:
        raw_runtime_module._candidate_embeddings = original_runtime_embeddings
        raw_runtime_module._probabilities = original_runtime_probabilities
        for handle in handles:
            handle.remove()


def _set_groups(groups: list[list[int]]) -> set[frozenset[int]]:
    return {frozenset(group) for group in groups}


if __name__ == "__main__":
    main()
