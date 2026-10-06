#!/usr/bin/env python3
"""Microscope-test dynamic batch/sequence ONNX inputs for the frozen mini LM."""

from __future__ import annotations

import hashlib
import json
import site
import sys
from datetime import datetime, timezone
from pathlib import Path

# Resolve the model's Transformers dependencies from the system site before
# the exporter exposes this machine's separate user-site ONNX Runtime wheel.
_USER_SITE = site.getusersitepackages()
if _USER_SITE in sys.path:
    sys.path.remove(_USER_SITE)

import numpy as np
import torch

from export_mini_formula_lm_onnx_v1 import ManualMiniFormulaLM
import run_prompt_mini_lm_distillation_v1 as mini_lm_trainer
from run_prompt_mini_lm_distillation_v1 import DEFAULT_OUTPUT as DISTILL_DIR

if _USER_SITE and _USER_SITE not in sys.path:
    sys.path.append(_USER_SITE)
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_onnx_20261002"
DEFAULT_OUTPUT = MODEL_DIR / "dynamic_shape_contract_report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile(model: MiniFormulaLM, batch: int, symbols: int, relation_count: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    width = 2 * symbols + 1
    input_ids = np.full((batch, width), model.pad_id, dtype=np.int64)
    attention = np.zeros((batch, width), dtype=np.bool_)
    mask_positions = np.empty(batch, dtype=np.int64)
    for row in range(batch):
        count = symbols if row == 0 or symbols == 1 else max(1, symbols - row % min(symbols, 5))
        sequence = [model.cls_id]
        for position in range(count):
            if position:
                relation_id = model.pad_id - relation_count + int(rng.integers(0, relation_count))
                sequence.append(relation_id)
            if position == count // 2:
                mask_positions[row] = len(sequence)
                sequence.append(model.mask_id)
            else:
                sequence.append(int(rng.integers(0, 372)))
        sequence.append(model.sep_id)
        input_ids[row, :len(sequence)] = sequence
        attention[row, :len(sequence)] = True
    return {"input_ids": input_ids, "attention_mask": attention, "mask_positions": mask_positions}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DISTILL_DIR / "mini_formula_lm.pt")
    parser.add_argument("--fp32-onnx", type=Path, default=MODEL_DIR / "mini_formula_lm_fp32.onnx")
    parser.add_argument("--int8-onnx", type=Path, default=MODEL_DIR / "mini_formula_lm_dynamic_int8.onnx")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = [args.checkpoint.resolve(), args.fp32_onnx.resolve(), args.int8_onnx.resolve()]
    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")
    for path in paths:
        if not path.is_file():
            parser.error(f"missing model input: {path}")

    payload = torch.load(paths[0], map_location="cpu", weights_only=True)
    layers = int(payload["layers"])
    mini_lm_trainer.LAYERS = layers
    model = mini_lm_trainer.MiniFormulaLM(
        len(payload["labels"]), len(payload["relations"]), int(payload["max_positions"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    reference = ManualMiniFormulaLM(model).eval()
    fp32 = ort.InferenceSession(str(paths[1]), providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(str(paths[2]), providers=["CPUExecutionProvider"])

    profiles = [(1, 1), (2, 8), (8, 32), (1, 64), (2, 64), (8, 64)]
    results = []
    overall_max_error = 0.0
    int8_argmax_matches = int8_total = 0
    with torch.inference_mode():
        for index, (batch, symbols) in enumerate(profiles):
            arrays = _profile(model, batch, symbols, len(payload["relations"]), seed=20261020 + index)
            expected = reference(
                torch.from_numpy(arrays["input_ids"]),
                torch.from_numpy(arrays["attention_mask"]),
                torch.from_numpy(arrays["mask_positions"]),
            ).cpu().numpy()
            actual = fp32.run(["logits"], arrays)[0]
            compressed = int8.run(["logits"], arrays)[0]
            error = float(np.max(np.abs(expected - actual)))
            if not np.isfinite(actual).all() or actual.shape != (batch, 372):
                raise AssertionError(f"invalid FP32 output shape/value for profile {(batch, symbols)}")
            if error > 1e-4:
                raise AssertionError(f"FP32 parity exceeded 1e-4 for profile {(batch, symbols)}: {error}")
            overall_max_error = max(overall_max_error, error)
            matches = int(np.count_nonzero(expected.argmax(axis=1) == compressed.argmax(axis=1)))
            int8_argmax_matches += matches
            int8_total += batch
            results.append({
                "batch": batch,
                "symbols_max": symbols,
                "sequence_length": 2 * symbols + 1,
                "valid_lengths": arrays["attention_mask"].sum(axis=1).astype(int).tolist(),
                "fp32_max_abs_error": error,
                "fp32_output_shape": list(actual.shape),
                "int8_max_abs_error": float(np.max(np.abs(actual - compressed))),
                "int8_argmax_matches": matches,
                "int8_argmax_total": batch,
            })

    report = {
        "schema": "aiflow-mini-formula-lm-onnx-dynamic-shape-audit/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_fp32_dynamic_shape_contract",
        "protocol": {
            "training_performed": False,
            "development_or_acceptance_rows_loaded": 0,
            "crohme_rows_loaded": 0,
            "synthetic_smoke_inputs_only": True,
            "profiles": len(profiles),
        },
        "provenance": {
            "checkpoint_sha256": _sha256(paths[0]),
            "fp32_onnx_sha256": _sha256(paths[1]),
            "int8_onnx_sha256": _sha256(paths[2]),
            "onnxruntime_version": ort.__version__,
        },
        "fp32": {
            "max_abs_error_across_profiles": overall_max_error,
            "tolerance": 1e-4,
            "shape_profiles": results,
        },
        "int8": {
            "argmax_agreement_across_synthetic_rows": int8_argmax_matches / int8_total,
            "argmax_matches": int8_argmax_matches,
            "rows": int8_total,
            "status": "diagnostic_only_no_accuracy_claim",
        },
        "decision": {
            "dynamic_batch_and_sequence_supported": True,
            "android_execution_verified": False,
            "promotion_eligible": False,
        },
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "mini_formula_lm_dynamic_shape_audit_complete",
        "report": str(output),
        "profiles": len(profiles),
        "fp32_max_abs_error": overall_max_error,
        "int8_argmax_agreement_synthetic_only": int8_argmax_matches / int8_total,
        "crohme_rows_loaded": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
