#!/usr/bin/env python3
"""Export and numerically audit the frozen mini formula-context HWR reranker.

The 149-formula corpus is inference-only, consumed development diagnostics.
CROHME is neither loaded nor used.  This script does not train or tune.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import site
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# This workstation has a conflicting user-site huggingface-hub.  Import the
# model/training modules against the system site first, then expose the optional
# user-site ORT wheel after Transformers is already resolved.
_USER_SITE = site.getusersitepackages()
if _USER_SITE in sys.path:
    sys.path.remove(_USER_SITE)

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from torch import nn

from audit_prompt_bert_context_on_frozen149_v1 import (
    DEFAULT_DATA,
    DEFAULT_SUMMARY,
    _formula_metrics,
    _formula_rows,
    _jsonl,
    _sha256,
)
from run_prompt_mini_lm_distillation_v1 import (
    DEFAULT_OUTPUT as DEFAULT_DISTILL_OUTPUT,
    HIDDEN,
    LAYERS,
    MAX_POSITIONS,
    MiniFormulaLM,
)
from train_masked_context_reranker_v1 import (
    RELATIONS,
    _formulae,
    _fused_predictions,
    _spatial_relation,
)
from train_prompt_context_reranker_v1 import _strict_lock

if _USER_SITE and _USER_SITE not in sys.path:
    sys.path.append(_USER_SITE)
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_onnx_20261002"
MAX_FP32_ABS_ERROR = 1e-4


class ManualMiniFormulaLM(nn.Module):
    """ONNX-friendly equivalent of MiniFormulaLM's pre-norm Transformer."""

    def __init__(self, model: MiniFormulaLM) -> None:
        super().__init__()
        self.token_embedding = model.token_embedding
        self.position_embedding = model.position_embedding
        self.layers = model.encoder.layers
        self.output_norm = model.output_norm
        self.classifier = model.classifier
        self.heads = int(model.encoder.layers[0].self_attn.num_heads)
        self.hidden = HIDDEN

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> torch.Tensor:
        sequence_length = input_ids.shape[1]
        positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        batch, length, hidden = x.shape
        head_dim = hidden // self.heads
        valid_keys = attention_mask.to(torch.bool).unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            normalized = layer.norm1(x)
            qkv = F.linear(normalized, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
            query, key, value = qkv.chunk(3, dim=-1)
            query = query.reshape(batch, length, self.heads, head_dim).transpose(1, 2)
            key = key.reshape(batch, length, self.heads, head_dim).transpose(1, 2)
            value = value.reshape(batch, length, self.heads, head_dim).transpose(1, 2)
            scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim ** -0.5)
            scores = torch.where(valid_keys, scores, torch.full_like(scores, -1.0e4))
            attended = torch.matmul(torch.softmax(scores, dim=-1), value)
            attended = attended.transpose(1, 2).contiguous().reshape(batch, length, hidden)
            x = x + F.linear(attended, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias)
            normalized = layer.norm2(x)
            feedforward = F.linear(normalized, layer.linear1.weight, layer.linear1.bias)
            feedforward = F.gelu(feedforward)
            x = x + F.linear(feedforward, layer.linear2.weight, layer.linear2.bias)
        batch_indices = torch.arange(batch, device=input_ids.device)
        masked = x[batch_indices, mask_positions]
        return self.classifier(self.output_norm(masked))


def _current_inputs(rows: list[dict], model: MiniFormulaLM, labels: list[str]) -> tuple[dict, list[dict]]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    relation_to_index = {relation: index for index, relation in enumerate(RELATIONS)}
    examples = []
    for formula_id, sequence in _formulae(rows).items():
        for target_index, target in enumerate(sequence):
            ids = [model.cls_id]
            mask_position = -1
            for index, row in enumerate(sequence):
                if index:
                    ids.append(len(labels) + relation_to_index[_spatial_relation(sequence[index - 1], row)])
                if index == target_index:
                    mask_position = len(ids)
                    ids.append(model.mask_id)
                else:
                    token = str(row["final_topk"][0])
                    if token not in label_to_index:
                        raise ValueError(f"Fast HWR token outside student vocabulary: {token}")
                    ids.append(label_to_index[token])
            ids.append(model.sep_id)
            if mask_position < 0 or len(ids) > MAX_POSITIONS:
                raise ValueError(f"invalid student sequence length for {formula_id}")
            examples.append({"record_id": str(target["record_id"]), "input_ids": ids, "mask_position": mask_position})
    if {item["record_id"] for item in examples} != {str(row["record_id"]) for row in rows}:
        raise AssertionError("inference example coverage mismatch")
    width = max(len(item["input_ids"]) for item in examples)
    input_ids = np.full((len(examples), width), model.pad_id, dtype=np.int64)
    attention = np.zeros((len(examples), width), dtype=np.bool_)
    positions = np.empty((len(examples),), dtype=np.int64)
    for index, item in enumerate(examples):
        length = len(item["input_ids"])
        input_ids[index, :length] = item["input_ids"]
        attention[index, :length] = True
        positions[index] = item["mask_position"]
    return {"input_ids": input_ids, "attention_mask": attention, "mask_positions": positions}, examples


def _predict_torch(model: nn.Module, inputs: dict[str, np.ndarray], batch_size: int = 128) -> np.ndarray:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(inputs["input_ids"]), batch_size):
            end = min(start + batch_size, len(inputs["input_ids"]))
            args = (
                torch.from_numpy(inputs["input_ids"][start:end]),
                torch.from_numpy(inputs["attention_mask"][start:end]),
                torch.from_numpy(inputs["mask_positions"][start:end]),
            )
            outputs.append(model(*args).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _predict_ort(session: ort.InferenceSession, inputs: dict[str, np.ndarray], batch_size: int = 128) -> np.ndarray:
    outputs = []
    for start in range(0, len(inputs["input_ids"]), batch_size):
        end = min(start + batch_size, len(inputs["input_ids"]))
        batch = {key: value[start:end] for key, value in inputs.items()}
        outputs.append(session.run(["logits"], batch)[0])
    return np.concatenate(outputs, axis=0)


def _product_metrics(rows: list[dict], targets: dict, logits: np.ndarray, labels: list[str], lambda_value: float) -> dict:
    context = {
        str(row["record_id"]): values.astype(np.float32) - np.logaddexp.reduce(values.astype(np.float32))
        for row, values in zip(rows, logits, strict=True)
    }
    predictions = _fused_predictions(rows, context, labels, lambda_value)
    locked = _strict_lock(rows, predictions)
    return _formula_metrics(rows, targets, locked)


def _latency_ms(session: ort.InferenceSession, inputs: dict[str, np.ndarray]) -> dict:
    batch = {key: value[: min(32, len(value))] for key, value in inputs.items()}
    for _ in range(5):
        session.run(["logits"], batch)
    measurements = []
    for _ in range(40):
        start = time.perf_counter()
        session.run(["logits"], batch)
        measurements.append(1000.0 * (time.perf_counter() - start))
    return {
        "batch_size": len(batch["input_ids"]),
        "median_batch_ms_host_not_mobile": statistics.median(measurements),
        "p95_batch_ms_host_not_mobile": float(np.percentile(measurements, 95)),
        "iterations": len(measurements),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DISTILL_OUTPUT / "mini_formula_lm.pt")
    parser.add_argument("--distillation-report", type=Path, default=DEFAULT_DISTILL_OUTPUT / "mini_formula_lm_distillation_report.json")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    checkpoint, distill_report_path = args.checkpoint.resolve(), args.distillation_report.resolve()
    summary_path, data_path, output = args.summary.resolve(), args.data.resolve(), args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing output directory: {output}")
    for name, path in (("checkpoint", checkpoint), ("distillation report", distill_report_path), ("summary", summary_path), ("formula data", data_path)):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")

    distill_report = json.loads(distill_report_path.read_text(encoding="utf-8"))
    protocol = distill_report["protocol"]
    if protocol.get("crohme_rows_loaded") != 0 or protocol.get("crohme_training_or_tuning") is not False:
        raise ValueError("checkpoint report does not attest CROHME exclusion")
    if _sha256(checkpoint) != distill_report["student"]["checkpoint_sha256"]:
        raise ValueError("student checkpoint hash mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("crohme_training_or_tuning") is not False:
        raise ValueError("evaluation summary lacks CROHME exclusion attestation")
    if _sha256(data_path) != summary["inputs"]["formulas_valid_sha256"]:
        raise ValueError("frozen formula dataset hash mismatch")
    raw = _jsonl(data_path)
    if any("crohme" in str(row.get("source_partition", "")).casefold() for row in raw):
        raise ValueError("CROHME row encountered in diagnostic data")
    raw_by_id = {str(row["sample_id"]): row for row in raw}
    rows, targets = _formula_rows(summary, raw_by_id)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema") != "aiflow-prompt-mini-lm-distillation/v1":
        raise ValueError("unsupported checkpoint schema")
    labels = [str(label) for label in payload["labels"]]
    if len(labels) != 372 or payload["relations"] != list(RELATIONS):
        raise ValueError("student label/relation contract mismatch")
    model = MiniFormulaLM(len(labels), len(RELATIONS), int(payload["max_positions"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    manual = ManualMiniFormulaLM(model).eval()
    inputs, examples = _current_inputs(rows, model, labels)

    torch_original = _predict_torch(model, inputs)
    torch_manual = _predict_torch(manual, inputs)
    manual_error = float(np.max(np.abs(torch_original - torch_manual)))
    if manual_error > MAX_FP32_ABS_ERROR:
        raise AssertionError(f"manual ONNX wrapper disagrees with PyTorch model: {manual_error}")

    output.mkdir(parents=True, exist_ok=False)
    fp32_path = output / "mini_formula_lm_fp32.onnx"
    sample_ids = torch.from_numpy(inputs["input_ids"][:2])
    sample_attention = torch.from_numpy(inputs["attention_mask"][:2])
    sample_positions = torch.from_numpy(inputs["mask_positions"][:2])
    torch.onnx.export(
        manual,
        (sample_ids, sample_attention, sample_positions),
        str(fp32_path),
        input_names=["input_ids", "attention_mask", "mask_positions"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "mask_positions": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    graph = onnx.load(str(fp32_path))
    onnx.checker.check_model(graph)
    graph = onnx.shape_inference.infer_shapes(graph)
    onnx.save(graph, str(fp32_path))

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    fp32_session = ort.InferenceSession(str(fp32_path), sess_options=options, providers=["CPUExecutionProvider"])
    ort_fp32 = _predict_ort(fp32_session, inputs)
    fp32_error = float(np.max(np.abs(torch_original - ort_fp32)))
    if fp32_error > MAX_FP32_ABS_ERROR:
        raise AssertionError(f"FP32 ONNX Runtime parity failed: max_abs_error={fp32_error}")

    lambda_value = float(protocol["fusion_lambda_frozen_from_teacher"])
    fp32_metrics = _product_metrics(rows, targets, ort_fp32, labels, lambda_value)
    quantized_path = output / "mini_formula_lm_dynamic_int8.onnx"
    quantization = {"status": "failed"}
    int8_logits = None
    try:
        quantize_dynamic(str(fp32_path), str(quantized_path), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
        int8_session = ort.InferenceSession(str(quantized_path), sess_options=options, providers=["CPUExecutionProvider"])
        int8_logits = _predict_ort(int8_session, inputs)
        int8_abs_error = float(np.max(np.abs(ort_fp32 - int8_logits)))
        int8_argmax_agreement = float(np.mean(ort_fp32.argmax(axis=1) == int8_logits.argmax(axis=1)))
        int8_metrics = _product_metrics(rows, targets, int8_logits, labels, lambda_value)
        quantization = {
            "status": "diagnostic_only",
            "weight_type": "QInt8 dynamic",
            "max_abs_logit_error_vs_fp32": int8_abs_error,
            "token_argmax_agreement_vs_fp32": int8_argmax_agreement,
            "formula_exact_metrics": int8_metrics,
            "model_bytes": quantized_path.stat().st_size,
            "host_latency": _latency_ms(int8_session, inputs),
        }
    except Exception as exc:  # preserve FP32 export and report the quantizer limitation
        quantization["error"] = f"{type(exc).__name__}: {exc}"
        if quantized_path.exists():
            quantized_path.unlink()

    report = {
        "schema": "aiflow-mini-formula-lm-onnx-parity/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "shadow_only_consumed_development_diagnostic",
        "protocol": {
            "training_performed": False,
            "crohme_rows_loaded": 0,
            "crohme_training_or_tuning": False,
            "formula_examples": len(examples),
            "candidate_selection_or_threshold_tuning": False,
            "evaluation_warning": "The frozen 149-formula development set is consumed data; this is parity evidence only, not independent product acceptance.",
        },
        "provenance": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "distillation_report": str(distill_report_path),
            "distillation_report_sha256": _sha256(distill_report_path),
            "summary_sha256": _sha256(summary_path),
            "formula_data_sha256": _sha256(data_path),
        },
        "contract": {
            "inputs": {"input_ids": "int64[N,S]", "attention_mask": "bool[N,S]", "mask_positions": "int64[N]"},
            "output": "float32[N,372] logits",
            "dynamic_batch": True,
            "dynamic_sequence": True,
            "opset": 17,
            "max_positions": int(payload["max_positions"]),
        },
        "fp32": {
            "status": "passed",
            "manual_wrapper_max_abs_error_vs_pytorch": manual_error,
            "onnxruntime_max_abs_error_vs_pytorch": fp32_error,
            "max_abs_error_gate": MAX_FP32_ABS_ERROR,
            "model_bytes": fp32_path.stat().st_size,
            "pytorch_checkpoint_bytes": checkpoint.stat().st_size,
            "onnxruntime_version": ort.__version__,
            "formula_exact_metrics_diagnostic_only": fp32_metrics,
            "host_latency": _latency_ms(fp32_session, inputs),
        },
        "int8": quantization,
        "decision": {
            "automatic_default_replacement": False,
            "android_latency_verified": False,
            "next_gate": "fresh writer/formula-disjoint acceptance and real-device Android ORT profiling",
        },
    }
    (output / "mini_formula_lm_onnx_parity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({
        "event": "mini_formula_lm_onnx_parity_complete",
        "report": str(output / "mini_formula_lm_onnx_parity_report.json"),
        "fp32_max_abs_error": fp32_error,
        "manual_wrapper_max_abs_error": manual_error,
        "fp32_model_bytes": fp32_path.stat().st_size,
        "int8_status": quantization["status"],
        "int8_max_abs_error": quantization.get("max_abs_logit_error_vs_fp32"),
        "crohme_rows_loaded": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
