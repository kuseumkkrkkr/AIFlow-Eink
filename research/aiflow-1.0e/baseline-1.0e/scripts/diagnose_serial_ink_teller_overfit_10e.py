#!/usr/bin/env python3
"""One-formula optimization diagnostic for the raster-free serial path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import train_serial_ink_teller_10e as serial


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formula-id", default="aiflow_0100")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--output", type=Path, default=serial.ROOT / "artifacts" / "serial_ink_teller_overfit_20260831")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    device = torch.device(args.device)
    rows = serial._load_candidate_rows(serial.DEFAULT_CANDIDATES)
    formulas = serial._load_formula_records(serial.DEFAULT_FORMULAS)
    examples, _ = serial._examples(rows, formulas)
    selected = [example for example in examples if example["formula_id"] == args.formula_id]
    if len(selected) != 1:
        raise SystemExit(f"expected one formula, found {len(selected)} for {args.formula_id}")
    tokenizer = serial.AutoTokenizer.from_pretrained(str(serial.DEFAULT_TEXTELLER), local_files_only=True)
    decoder = serial._load_decoder_only(serial.DEFAULT_TEXTELLER, device)
    serial._enable_decoder_cross_attention(decoder)
    online = serial.OnlinePrior.from_checkpoints(serial.DEFAULT_AI_FLOW_BASE, serial.DEFAULT_AI_FLOW_ADAPTER).to(device)
    bridge = serial.SerialBridge().to(device)
    for parameter in online.trajectory_encoder.parameters():
        parameter.requires_grad_(False)
    serial._train(online, bridge, decoder, tokenizer, selected, args.epochs, 1, 1701, device, 1e-4, 1e-2, 2e-4)
    prediction = serial._predict(online, bridge, decoder, tokenizer, selected, device, 24)[0]
    args.output.mkdir(parents=True)
    result = {
        "schema": "aiflow-1.0e-serial-ink-teller-overfit-diagnostic/v1",
        "formula_id": args.formula_id,
        "epochs": args.epochs,
        "device": str(device),
        "raster_encoder_executed": False,
        "prediction": prediction,
        "interpretation": "optimization_or_decoder_contract_not_closed" if not prediction["exact"] else "single_example_optimization_can_close",
    }
    (args.output / "overfit_diagnostic.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
