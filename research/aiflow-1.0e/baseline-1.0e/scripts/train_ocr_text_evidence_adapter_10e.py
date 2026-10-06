#!/usr/bin/env python3
"""AIFlow 1.0e follow-up: frozen OCR text evidence + candidate adapter.

The existing 1.0 HWR path is frozen.  TexTeller is used only as an external
teacher: its pooled visual embedding and decoded formula tokens become
evidence for a small candidate-only scorer.  The scorer may select an existing
HWR Top-k candidate, but it cannot create tokens, regroup strokes, or delete
rows.  This script is intentionally separate from the first 1.0e artifact.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from train_ocr_decision_adapter_10e import (
    DEFAULT_CANDIDATES,
    DEFAULT_FORMULAS,
    DEFAULT_TEXTELLER,
    FrozenOcrEncoder,
    OcrDecisionAdapter,
    TEXTELLER_REVISION,
    _batch,
    _json_lines,
    _load_candidates,
    _load_formula_records,
    _metrics,
    _render_formula,
    _row_numeric,
    _set_seed,
    _train,
    _predict,
)

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "ocr_text_evidence_adapter_10e_20260830"
SCHEMA = "aiflow-1.0e-ocr-text-evidence-adapter/v1"
TEXT_FEATURE_NAMES = (
    "teacher_exact_position",
    "teacher_near_position",
    "teacher_global_presence",
    "teacher_position_similarity",
    "teacher_length_ratio",
)


class FrozenOcrTextTeacher(FrozenOcrEncoder):
    """TexTeller encoder plus frozen decoded text; no gradients anywhere."""

    def __init__(self, model_path: Path, device: torch.device):
        super().__init__(model_path, device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )

    @torch.inference_mode()
    def decode(self, images: list, batch_size: int = 1, max_length: int = 96) -> list[str]:
        texts: list[str] = []
        for start in range(0, len(images), batch_size):
            batch = torch.cat(
                [self._tensor(image) for image in images[start:start + batch_size]],
                dim=0,
            ).to(self.device)
            generated = self.model.generate(
                pixel_values=batch,
                num_beams=1,
                max_length=max_length,
                do_sample=False,
            )
            texts.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return texts


def _teacher_tokens(text: str) -> list[str]:
    """Convert decoded LaTeX into comparable atomic symbols.

    This is evidence extraction only.  It is deliberately conservative: TeX
    layout commands are ignored and no new AIFlow token is emitted.
    """
    text = re.sub(r"\\(?:left|right|begin\{[^}]+\}|end\{[^}]+\})", "", text)
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\[", "").replace(r"\]", "")
    aliases = {
        r"\times": r"\times", r"\cdot": r"\cdot", r"\div": r"\div",
        r"\pm": r"\pm", r"\mp": r"\mp", r"\leq": r"\leq",
        r"\geq": r"\geq", r"\neq": r"\neq", r"\mathbb{1}": r"\mathbb{1}",
        r"\mathcal{O}": r"\mathcal{O}", r"\mathcal{X}": r"\mathcal{X}",
        r"\chi": r"\chi", r"\circ": r"\circ", r"\mid": r"\mid",
        r"\frac": "/", r"\over": "/",
    }
    commands = sorted(aliases, key=len, reverse=True)
    tokens: list[str] = []
    index = 0
    while index < len(text):
        if text[index].isspace() or text[index] in "{}^_":
            index += 1
            continue
        matched = False
        for command in commands:
            if text.startswith(command, index):
                tokens.append(aliases[command])
                index += len(command)
                matched = True
                break
        if matched:
            continue
        if text[index] == "\\":
            match = re.match(r"\\[A-Za-z]+", text[index:])
            index += len(match.group(0)) if match else 1
            continue
        if text[index] in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ()+-=/< >|[]":
            value = text[index]
            if value != " ":
                tokens.append(value)
        index += 1
    return tokens


def _text_features(rows: list[dict], teacher_tokens: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """Build candidate-specific evidence from the frozen teacher sequence."""
    by_formula: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_formula[str(row["formula_id"])].append(row)
    output: dict[str, np.ndarray] = {}
    for formula_id, formula_rows in by_formula.items():
        tokens = teacher_tokens.get(formula_id, [])
        token_count = len(tokens)
        row_count = len(formula_rows)
        values: list[list[list[float]]] = []
        for row_index, row in enumerate(formula_rows):
            expected = ((row_index + 0.5) / max(1, row_count)) * token_count - 0.5
            per_candidate: list[list[float]] = []
            for candidate in row["final_topk"]:
                positions = [i for i, token in enumerate(tokens) if token == str(candidate)]
                if positions:
                    distance = min(abs(position - expected) for position in positions)
                    similarity = math.exp(-distance / 2.0)
                else:
                    distance = float("inf")
                    similarity = 0.0
                per_candidate.append([
                    float(distance <= 0.75),
                    float(distance <= 2.5),
                    float(bool(positions)),
                    similarity,
                    token_count / max(1, row_count),
                ])
            values.append(per_candidate)
        output[formula_id] = np.asarray(values, dtype=np.float32)
    return output


def _make_examples(rows: list[dict], formula_features: dict[str, np.ndarray], text_features: dict[str, np.ndarray], token_to_id: dict[str, int]) -> list[dict]:
    examples: list[dict] = []
    by_formula_index: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        formula_id = str(row["formula_id"])
        position = by_formula_index[formula_id]
        by_formula_index[formula_id] += 1
        if formula_id not in formula_features or formula_id not in text_features:
            raise KeyError(f"missing external evidence for {formula_id}")
        numeric = []
        for candidate_index, candidate in enumerate(row["final_topk"]):
            numeric.append(
                _row_numeric(row, candidate_index)
                + text_features[formula_id][position][candidate_index].tolist()
            )
        candidates = [str(token) for token in row["final_topk"]]
        target = str(row["label"])
        examples.append({
            "record_id": str(row.get("record_id", "")),
            "formula_id": formula_id,
            "writer_group": str(row["writer_group"]),
            "numeric": np.asarray(numeric, dtype=np.float32),
            "token_ids": np.asarray([token_to_id[token] for token in candidates], dtype=np.int64),
            "mask": np.ones(len(candidates), dtype=bool),
            "ocr": formula_features[formula_id],
            "target": candidates.index(target) if target in candidates else -1,
            "baseline": 0,
            "candidates": candidates,
            "label": target,
        })
    return examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--formulas", type=Path, default=DEFAULT_FORMULAS)
    parser.add_argument("--ocr-model", type=Path, default=DEFAULT_TEXTELLER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-length", type=int, default=96)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    _set_seed(args.seed)
    device = torch.device(args.device)
    rows = _load_candidates(args.candidates, args.limit)
    formulas = _load_formula_records(args.formulas)
    formula_ids = list(dict.fromkeys(str(row["formula_id"]) for row in rows))
    missing = [formula_id for formula_id in formula_ids if formula_id not in formulas]
    if missing:
        raise ValueError(f"formula raster source is missing {len(missing)} formula IDs")
    labels = sorted({str(token) for row in rows for token in row["final_topk"]})
    token_to_id = {token: index for index, token in enumerate(labels)}
    print(json.dumps({"event": "external_teacher_load", "schema": SCHEMA, "model": str(args.ocr_model), "revision": TEXTELLER_REVISION, "formula_count": len(formula_ids), "record_count": len(rows), "device": str(device)}, ensure_ascii=False), flush=True)
    teacher = FrozenOcrTextTeacher(args.ocr_model, device)
    images = [_render_formula(formulas[formula_id]) for formula_id in formula_ids]
    formula_features_array = teacher.encode(images)
    decoded = teacher.decode(images, max_length=args.max_length)
    formula_features = dict(zip(formula_ids, formula_features_array))
    teacher_tokens = dict(zip(formula_ids, [_teacher_tokens(text) for text in decoded]))
    text_features = _text_features(rows, teacher_tokens)
    examples = _make_examples(rows, formula_features, text_features, token_to_id)
    by_writer: defaultdict[str, list[dict]] = defaultdict(list)
    for example in examples:
        by_writer[example["writer_group"]].append(example)
    all_predictions: list[dict] = []
    folds: list[dict] = []
    for fold_index, writer in enumerate(sorted(by_writer)):
        train = [example for group, values in by_writer.items() if group != writer for example in values]
        test = list(by_writer[writer])
        model = OcrDecisionAdapter(examples[0]["numeric"].shape[-1], len(labels), int(formula_features_array.shape[-1])).to(device)
        _train(model, train, args.epochs, args.seed + fold_index, device)
        predictions = _predict(model, test, device)
        all_predictions.extend(predictions)
        folds.append({"held_writer": writer, **_metrics(predictions)})
    refit = OcrDecisionAdapter(examples[0]["numeric"].shape[-1], len(labels), int(formula_features_array.shape[-1])).to(device)
    _train(refit, examples, args.epochs, args.seed + 1000, device)
    args.output.mkdir(parents=True)
    checkpoint = args.output / "ocr_text_evidence_adapter.pt"
    torch.save({
        "schema": SCHEMA,
        "model_family": "frozen_texteller_visual_and_decoded_text_candidate_adapter",
        "texteller_revision": TEXTELLER_REVISION,
        "numeric_size": int(examples[0]["numeric"].shape[-1]),
        "ocr_size": int(formula_features_array.shape[-1]),
        "labels": labels,
        "state_dict": refit.state_dict(),
        "candidate_contract": {"top_k_only": True, "token_creation": False, "row_deletion": False, "stroke_regrouping": False},
    }, checkpoint)
    evaluation = {
        "schema": SCHEMA,
        "model": {"name": "OleehyO/TexTeller", "revision": TEXTELLER_REVISION, "weights_frozen": True, "evidence": ["encoder_mean_pool", "decoded_formula_tokens"]},
        "data": {"candidate_path": str(args.candidates), "formula_path": str(args.formulas), "formula_count": len(formula_ids), "record_count": len(rows)},
        "training": {"epochs": args.epochs, "seed": args.seed, "trainable_adapter_parameters": sum(parameter.numel() for parameter in refit.parameters() if parameter.requires_grad), "external_trainable_parameters": 0, "text_features": list(TEXT_FEATURE_NAMES)},
        "teacher_outputs": {formula_id: {"text": decoded[index], "tokens": teacher_tokens[formula_id]} for index, formula_id in enumerate(formula_ids)},
        "writer_loo": folds,
        "aggregate": _metrics(all_predictions),
        "status": "shadow_only",
        "checkpoint": str(checkpoint),
    }
    (args.output / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    with gzip.open(args.output / "writer_loo_predictions.jsonl.gz", "wt", encoding="utf-8") as stream:
        for prediction in all_predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "complete", "output": str(args.output), "aggregate": evaluation["aggregate"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
