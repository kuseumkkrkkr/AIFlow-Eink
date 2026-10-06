#!/usr/bin/env python3
"""Run a small, reproducible hyperparameter loop for the serial ink path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_serial_ink_teller_10e.py"
DEFAULT_OUTPUT = ROOT / "artifacts" / "serial_ink_teller_training_loop_20260830"

CONFIGS = (
    {"name": "aux_e20_lr1e-3", "epochs": 20, "online_lr": 1e-5, "bridge_lr": 1e-3, "decoder_lr": 2e-5},
    {"name": "aux_e40_lr3e-3", "epochs": 40, "online_lr": 2e-5, "bridge_lr": 3e-3, "decoder_lr": 1e-4},
    {"name": "aux_e60_lr5e-3", "epochs": 60, "online_lr": 5e-5, "bridge_lr": 5e-3, "decoder_lr": 2e-4},
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--only", nargs="*", choices=[config["name"] for config in CONFIGS])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [config for config in CONFIGS if not args.only or config["name"] in args.only]
    summary = {"schema": "aiflow-1.0e-serial-ink-teller-training-loop/v1", "configs": []}
    for config in selected:
        output = args.output / config["name"]
        command = [
            sys.executable, "-u", str(TRAIN),
            "--epochs", str(config["epochs"]),
            "--batch-size", "8",
            "--max-length", "24",
            "--online-lr", str(config["online_lr"]),
            "--bridge-lr", str(config["bridge_lr"]),
            "--decoder-lr", str(config["decoder_lr"]),
            "--include-all-raw",
            "--output", str(output),
            "--device", args.device,
        ]
        stdout_path = output.with_suffix(".stdout.log")
        stderr_path = output.with_suffix(".stderr.log")
        output.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"event": "loop_config_start", "config": config, "output": str(output)}, ensure_ascii=False), flush=True)
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
        item = {"config": config, "output": str(output), "returncode": result.returncode}
        evaluation_path = output / "evaluation.json"
        if evaluation_path.exists():
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            item["aggregate"] = evaluation.get("aggregate")
            item["baseline"] = evaluation.get("baseline")
            item["writer_loo"] = evaluation.get("writer_loo")
        summary["configs"].append(item)
        print(json.dumps({"event": "loop_config_complete", **item}, ensure_ascii=False), flush=True)
    summary["selected"] = max(
        (item for item in summary["configs"] if item.get("aggregate")),
        key=lambda item: item["aggregate"].get("formula_exact_rate", 0.0),
        default=None,
    )
    (args.output / "loop_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "loop_complete", "summary": str(args.output / 'loop_summary.json'), "selected": summary["selected"]}, ensure_ascii=False), flush=True)
    return 0 if all(item["returncode"] == 0 for item in summary["configs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
