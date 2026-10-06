#!/usr/bin/env python3
"""Measure frozen HWR and strict-decoder ceilings with ownership groups fixed.

This is a diagnostic only: it never fits a model, selects a threshold, or reads
CROHME.  Giving the ownership groups to the recognizer deliberately removes
grouping error so the remaining HWR Top-5 and decoder headroom is visible.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from evaluate_48hz_prefix_v1 import _load_model
from evaluate_joint_hwr_grouping_v1 import _candidate_embeddings
from selective_decoder_v1 import decode_selective_partition
from train_project_owned_grouping_v1 import Sample


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _geometry(strokes: list[dict], group: list[int]) -> dict[str, float]:
    points = [point for index in group for point in strokes[index]["points"]]
    xs, ys = [float(point["x"]) for point in points], [float(point["y"]) for point in points]
    return {"left": min(xs), "top": min(ys), "right": max(xs), "bottom": max(ys)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    formula_rows = {str(row["sample_id"]): row for row in _rows(args.dataset_root / "data" / "formulas_valid.jsonl")}
    ownership = [row for row in _rows(args.dataset_root / "data" / "ownership_train.jsonl") if row.get("accepted")]
    samples: list[Sample] = []
    truth: dict[str, tuple[list[list[int]], list[str]]] = {}
    for annotation in ownership:
        source = formula_rows[str(annotation["sample_id"])]
        strokes = sorted(source["strokes"], key=lambda row: int(row["order"]))
        groups = [[int(value) for value in group] for group in annotation["groups"]]
        labels = [str(value) for value in annotation["labels"]]
        if len(groups) != len(labels) or sorted(index for group in groups for index in group) != list(range(len(strokes))):
            raise ValueError(f"invalid ownership partition: {annotation['sample_id']}")
        candidates = [{"source_indices": group} for group in groups]
        samples.append(Sample(str(annotation["sample_id"]), str(annotation["writer_id"]), strokes, tuple(), candidates, np.empty((0, 0))))
        truth[str(annotation["sample_id"])] = (groups, labels)
    device = torch.device("cpu")
    model, label_vocab, _ = _load_model(args.checkpoint, device)
    embeddings, slices = _candidate_embeddings(samples, model, device)
    with torch.inference_mode():
        probabilities = model.math_head(embeddings.to(device)).softmax(dim=1).cpu().numpy()
    records: list[dict] = []
    for sample in samples:
        groups, labels = truth[sample.sample_id]
        values = probabilities[slices[sample.sample_id]]
        order = np.argsort(-values, axis=1)[:, :5]
        symbols = []
        for group, row_order, row_probability in zip(groups, order, values, strict=True):
            symbols.append({
                "stroke_indices": group,
                "hwr_topk": [label_vocab[int(index)] for index in row_order],
                "hwr_topk_probabilities": [float(row_probability[int(index)]) for index in row_order],
                "geometry": _geometry(sample.strokes, group),
            })
        decoded = decode_selective_partition(sample.sample_id, groups, symbols, stroke_count=len(sample.strokes))
        top1 = [row["hwr_topk"][0] for row in symbols]
        top5_oracle = all(label in row["hwr_topk"] for label, row in zip(labels, symbols, strict=True))
        decoder_exact = bool(decoded.get("accepted") and decoded.get("tokens") == labels)
        records.append({"sample_id": sample.sample_id, "writer_id": sample.writer, "truth_tokens": labels,
                        "top1_tokens": top1, "top5_oracle": top5_oracle, "decoder": decoded,
                        "decoder_token_exact": decoder_exact})
    def count(key: str) -> int:
        return sum(bool(row[key]) for row in records)
    payload = {
        "schema": "aiflow-oracle-group-decoder-ceiling/v1",
        "scope": "ownership groups fixed; frozen HWR; no CROHME; diagnostic only",
        "formulas": len(records),
        "top1_token_exact": count("top1_tokens"),
        "top5_oracle": count("top5_oracle"),
        "decoder_token_exact": count("decoder_token_exact"),
        "decoder_accepted": sum(bool(row["decoder"].get("accepted")) for row in records),
        "records": records,
    }
    # Top-1 exact is list equality, not merely truthiness of the token list.
    payload["top1_token_exact"] = sum(row["top1_tokens"] == row["truth_tokens"] for row in records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
