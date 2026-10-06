#!/usr/bin/env python3
"""Prepare offline TexTeller encoder targets for the serial bridge.

This process intentionally ends after writing targets.  Decoder tuning is run
in a separate process so the ViT and decoder allocations never coexist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from serial_ink_teller_offline_teacher_10e import OfflineTexTellerEncoder, render_formula
import train_serial_ink_teller_10e as serial


DEFAULT_OUTPUT = serial.ROOT / "artifacts" / "serial_ink_teller_teacher_targets_20260831.npz"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--texteller", type=Path, default=serial.DEFAULT_TEXTELLER)
    parser.add_argument("--formulas", type=Path, default=serial.DEFAULT_FORMULAS)
    parser.add_argument("--candidates", type=Path, default=serial.DEFAULT_CANDIDATES)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing target file: {args.output}")
    records = serial._load_formula_records(args.formulas)
    rows = serial._load_candidate_rows(args.candidates)
    examples, candidate_ids = serial._examples(rows, records, include_all_raw=True)
    print(json.dumps({"event": "teacher_target_start", "formulas": len(examples), "eval_formulas": len(candidate_ids), "device": "cpu"}, ensure_ascii=False), flush=True)
    teacher = OfflineTexTellerEncoder(args.texteller, torch.device("cpu"))
    values = teacher.encode([render_formula(records[example["formula_id"]]) for example in examples], args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, formula_ids=np.asarray([example["formula_id"] for example in examples]), teacher_memory=values)
    meta = {"schema": "aiflow-1.0e-serial-ink-teller-teacher-targets/v1", "texteller_revision": serial.TEXTELLER_REVISION, "formula_count": len(examples), "memory_shape": list(values.shape), "used_for_fit_only": True, "raster_runtime": False, "output": str(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "teacher_target_complete", **meta}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
