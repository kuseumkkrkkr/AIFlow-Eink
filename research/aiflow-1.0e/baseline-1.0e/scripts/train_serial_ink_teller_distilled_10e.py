#!/usr/bin/env python3
"""Offline-distilled serial AIFlow 1.0e experiment.

The TexTeller ViT encoder is used only while fitting the bridge.  It is never
saved in or called by the resulting runtime path: ordered online ink is fed to
the AIFlow prior, the distilled bridge, and the TexTeller decoder only.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

try:
    import train_serial_ink_teller_10e as serial
except ModuleNotFoundError:
    from scripts import train_serial_ink_teller_10e as serial


DEFAULT_OUTPUT = serial.ROOT / "artifacts" / "serial_ink_teller_distilled_10e_20260831"
DEFAULT_TEACHER_TARGETS = serial.ROOT / "artifacts" / "serial_ink_teller_teacher_targets_20260831.npz"
SCHEMA = "aiflow-1.0e-serial-ink-teller-distilled/v1"
TEACHER_GRID = 4


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _distill(
    online: serial.OnlinePrior,
    bridge: serial.SerialBridge,
    train_examples: list[dict],
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    online_lr: float,
    bridge_lr: float,
) -> float:
    trainable = [parameter for parameter in list(online.parameters()) + list(bridge.parameters()) if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": list(online.parameters()), "lr": online_lr},
        {"params": list(bridge.parameters()), "lr": bridge_lr},
    ], weight_decay=1e-4)
    rng = random.Random(seed)
    online.train()
    bridge.train()
    last_loss = float("nan")
    for _ in range(epochs):
        order = list(range(len(train_examples)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            selected = [train_examples[index] for index in order[start:start + batch_size]]
            features, mask = serial._batch(selected, device)
            encoded, mask = serial._serial_tokens(online(features), mask)
            prediction = bridge(encoded)
            target = torch.from_numpy(np.stack([example["teacher_memory"] for example in selected])).to(device)
            loss = F.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite distillation loss: {loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
    online.eval()
    bridge.eval()
    return last_loss


def _load_teacher_targets(examples: list[dict], path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"offline teacher target file missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        formula_ids = payload["formula_ids"].tolist()
        teacher_memory = payload["teacher_memory"].copy()
    by_id = {str(formula_id): memory for formula_id, memory in zip(formula_ids, teacher_memory)}
    missing = [example["formula_id"] for example in examples if example["formula_id"] not in by_id]
    if missing:
        raise KeyError(f"teacher targets missing {missing[:3]}")
    for example in examples:
        example["teacher_memory"] = np.asarray(by_id[example["formula_id"]], dtype=np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--teacher-targets", type=Path, default=DEFAULT_TEACHER_TARGETS)
    parser.add_argument("--texteller", type=Path, default=serial.DEFAULT_TEXTELLER)
    parser.add_argument("--candidates", type=Path, default=serial.DEFAULT_CANDIDATES)
    parser.add_argument("--formulas", type=Path, default=serial.DEFAULT_FORMULAS)
    parser.add_argument("--ai-flow-base", type=Path, default=serial.DEFAULT_AI_FLOW_BASE)
    parser.add_argument("--ai-flow-adapter", type=Path, default=serial.DEFAULT_AI_FLOW_ADAPTER)
    parser.add_argument("--distill-epochs", type=int, default=20)
    parser.add_argument("--decoder-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--teacher-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--online-lr", type=float, default=1e-5)
    parser.add_argument("--bridge-lr", type=float, default=1e-3)
    parser.add_argument("--decoder-lr", type=float, default=2e-5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    _set_seed(args.seed)
    rows = serial._load_candidate_rows(args.candidates)
    records = serial._load_formula_records(args.formulas)
    examples, candidate_ids = serial._examples(rows, records, include_all_raw=True)
    candidate_writers = sorted({example["writer_group"] for example in examples if example["is_eval"]})
    tokenizer = AutoTokenizer.from_pretrained(str(args.texteller), local_files_only=True)
    print(json.dumps({"event": "decoder_load_start", "schema": SCHEMA, "device": str(device), "train_formulas": len(examples), "eval_formulas": len(candidate_ids), "raster_runtime": False}, ensure_ascii=False), flush=True)
    decoder = serial._load_decoder_only(args.texteller, device)
    print(json.dumps({"event": "decoder_only_complete", "decoder_device": str(device), "raster_runtime": False}, ensure_ascii=False), flush=True)
    print(json.dumps({"event": "offline_teacher_targets_load", "target_file": str(args.teacher_targets)}, ensure_ascii=False), flush=True)
    _load_teacher_targets(examples, args.teacher_targets)
    print(json.dumps({"event": "offline_teacher_targets_complete", "target_shape": [len(examples), TEACHER_GRID * TEACHER_GRID, 768], "raster_runtime": False}, ensure_ascii=False), flush=True)
    serial._enable_decoder_cross_attention(decoder)
    initial_cross_attention = serial._cross_attention_state(decoder)
    online_template = serial.OnlinePrior.from_checkpoints(args.ai_flow_base, args.ai_flow_adapter)
    for parameter in online_template.trajectory_encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in online_template.online.parameters():
        parameter.requires_grad_(True)
    online_template.to(device)
    folds = []
    all_predictions = []
    for fold_index, held_writer in enumerate(candidate_writers):
        serial._restore_cross_attention_state(decoder, initial_cross_attention)
        train = [example for example in examples if example["writer_group"] != held_writer]
        test = [example for example in examples if example["is_eval"] and example["writer_group"] == held_writer]
        online = serial.OnlinePrior()
        online.load_state_dict(online_template.state_dict())
        online.to(device)
        bridge = serial.SerialBridge().to(device)
        distill_loss = _distill(online, bridge, train, args.distill_epochs, args.batch_size, args.seed + fold_index, device, args.online_lr, args.bridge_lr)
        serial._train(online, bridge, decoder, tokenizer, train, args.decoder_epochs, args.batch_size, args.seed + 100 + fold_index, device, args.online_lr, args.bridge_lr, args.decoder_lr)
        predictions = serial._predict(online, bridge, decoder, tokenizer, test, device, args.max_length)
        all_predictions.extend(predictions)
        folds.append({"held_writer": held_writer, "formula_count": len(test), "distill_loss": distill_loss, "exact": sum(int(row["exact"]) for row in predictions), "exact_rate": sum(int(row["exact"]) for row in predictions) / max(1, len(predictions))})
        del online, bridge
        if device.type == "cuda":
            torch.cuda.empty_cache()
    serial._restore_cross_attention_state(decoder, initial_cross_attention)
    refit_online = serial.OnlinePrior()
    refit_online.load_state_dict(online_template.state_dict())
    refit_online.to(device)
    refit_bridge = serial.SerialBridge().to(device)
    refit_loss = _distill(refit_online, refit_bridge, examples, args.distill_epochs, args.batch_size, args.seed + 1000, device, args.online_lr, args.bridge_lr)
    serial._train(refit_online, refit_bridge, decoder, tokenizer, examples, args.decoder_epochs, args.batch_size, args.seed + 1100, device, args.online_lr, args.bridge_lr, args.decoder_lr)
    exact = sum(int(row["exact"]) for row in all_predictions)
    args.output.mkdir(parents=True)
    checkpoint = args.output / "serial_bridge_distilled.pt"
    torch.save({
        "schema": SCHEMA,
        "texteller_revision": serial.TEXTELLER_REVISION,
        "teacher_used_offline_only": True,
        "raster_runtime": False,
        "ai_flow_base": str(args.ai_flow_base),
        "ai_flow_adapter": str(args.ai_flow_adapter),
        "online_state_dict": refit_online.online.state_dict(),
        "bridge_state_dict": refit_bridge.state_dict(),
        "decoder_cross_attention_state_dict": serial._cross_attention_state(decoder),
        "input_contract": {"source": "ordered raw strokes", "channels": 19, "max_events": serial.MAX_EVENTS},
        "output_contract": {"decoder": "TexTeller decoder-only", "free_form_latex": True},
    }, checkpoint)
    evaluation = {
        "schema": SCHEMA,
        "status": "shadow_only",
        "runtime_contract": {"serial": ["ordered_online_ink", "AIFlow_online_prior", "distilled_bridge", "TexTeller_decoder"], "raster_runtime": False, "teacher_runtime": False, "candidate_only": False, "free_form_decoder": True},
        "teacher_contract": {"model": "TexTeller ViT encoder", "used_during_fit_only": True, "target_memory_tokens": TEACHER_GRID * TEACHER_GRID, "runtime_saved": False},
        "data": {"train_formulas": len(examples), "eval_formulas": len(candidate_ids), "writer_count": len(candidate_writers), "raw_stroke_input": True, "external_new_data": False},
        "training": {"distill_epochs": args.distill_epochs, "decoder_epochs": args.decoder_epochs, "batch_size": args.batch_size, "online_lr": args.online_lr, "bridge_lr": args.bridge_lr, "decoder_lr": args.decoder_lr, "refit_distill_loss": refit_loss},
        "baseline": serial._baseline_metrics(rows, records),
        "writer_loo": folds,
        "aggregate": {"formula_exact": exact, "formula_count": len(all_predictions), "formula_exact_rate": exact / max(1, len(all_predictions))},
        "checkpoint": str(checkpoint),
    }
    (args.output / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    with serial.gzip.open(args.output / "writer_loo_predictions.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in all_predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "distilled_serial_complete", "output": str(args.output), "aggregate": evaluation["aggregate"], "baseline": evaluation["baseline"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
