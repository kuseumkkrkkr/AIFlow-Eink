#!/usr/bin/env python3
"""Train a one-layer formula-context challenger on the owned prompt corpus only.

The trainer's prompt validation split selects the checkpoint. The consumed
149-formula set remains inference-only diagnostic data; CROHME is never loaded.
"""

from __future__ import annotations

from pathlib import Path

import run_prompt_mini_lm_distillation_v1 as trainer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_distill_20261003_one_layer"


def main() -> int:
    trainer.LAYERS = 1
    trainer.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    return trainer.main()


if __name__ == "__main__":
    raise SystemExit(main())
