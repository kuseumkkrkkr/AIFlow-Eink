#!/usr/bin/env python3
"""Export and verify the one-layer prompt-only mini-LM challenger."""

from __future__ import annotations

from pathlib import Path

import export_mini_formula_lm_onnx_v1 as exporter
import run_prompt_mini_lm_distillation_v1 as trainer


ROOT = Path(__file__).resolve().parents[1]
TRAINED_DIR = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_distill_20261003_one_layer"
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_onnx_20261002" / "mini_lm_one_layer_onnx_20261003"


def main() -> int:
    trainer.LAYERS = 1
    exporter.DEFAULT_DISTILL_OUTPUT = TRAINED_DIR
    exporter.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    return exporter.main()


if __name__ == "__main__":
    raise SystemExit(main())
