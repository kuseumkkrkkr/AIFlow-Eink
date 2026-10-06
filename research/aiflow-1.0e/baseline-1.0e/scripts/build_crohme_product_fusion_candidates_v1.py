#!/usr/bin/env python3
"""Build a research-only CROHME Top-k cache from two frozen product heads.

CROHME remains a noncommercial repeated diagnostic.  The script performs no
training and requires an existing audited Top-5 fusion cache as its template.
For wider candidate retrieval, the first five tokens and probabilities must
replay exactly (within floating-point tolerance) before output is admitted.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import torch

from audit_replay_protocol_v1 import DEFAULT_CROHME
from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import (
    DEFAULT_PRODUCT, INPUT_MODE, _crohme_items, _load_model, _prefix_tensor,
    _sha256,
)
from evaluate_homograph_context_reranker_v1 import _d_path, _write_rows
from train_character_classifier_v1 import apply_input_mode


SCHEMA = "aiflow-crohme-product-head-fusion-candidates/v1"


def _assert_shared_encoder(old_model, new_model) -> None:
    for name, value in old_model.state_dict().items():
        if name.startswith("math_head."):
            continue
        if not torch.equal(value.detach().cpu(), new_model.state_dict()[name].detach().cpu()):
            raise ValueError(f"old and new product HWR encoders differ: {name}")


@torch.inference_mode()
def _score(
    items: list[dict], old_model, new_model, labels: list[str], weight: float,
    top_k: int, device: torch.device, batch_size: int,
) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        raw = np.stack([_prefix_tensor(item["strokes"]) for item in batch]).astype(
            np.float32, copy=False,
        )
        embeddings = old_model.encode(
            torch.from_numpy(apply_input_mode(raw, INPUT_MODE)).to(device)
        )
        probabilities = (
            (1.0 - weight) * old_model.math_head(embeddings).softmax(dim=1)
            + weight * new_model.math_head(embeddings).softmax(dim=1)
        )
        values, indices = probabilities.topk(top_k, dim=1)
        for item, token_indices, token_probabilities in zip(
            batch, indices.cpu().tolist(), values.cpu().tolist(), strict=True,
        ):
            output[str(item["record_id"])] = {
                "final_topk": [labels[index] for index in token_indices],
                "final_topk_probabilities": [float(value) for value in token_probabilities],
            }
    return output


def _metrics(rows: list[dict]) -> dict:
    ranks = Counter()
    formulae: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        truth = str(row["label"])
        candidates = [str(value) for value in row["final_topk"]]
        rank = candidates.index(truth) + 1 if truth in candidates else None
        ranks[str(rank) if rank is not None else "outside"] += 1
        formulae[str(row["formula_id"])].append(row)
    width = len(rows[0]["final_topk"])
    return {
        "records": len(rows),
        "formulas": len(formulae),
        "top1_count": ranks.get("1", 0),
        "top1": ranks.get("1", 0) / len(rows),
        "top5_count": sum(ranks.get(str(rank), 0) for rank in range(1, 6)),
        "top5": sum(ranks.get(str(rank), 0) for rank in range(1, 6)) / len(rows),
        "topk_count": len(rows) - ranks.get("outside", 0),
        "topk": 1.0 - ranks.get("outside", 0) / len(rows),
        "truth_rank_counts": dict(sorted(ranks.items())),
        "formula_topk_oracle_count": sum(
            all(str(row["label"]) in row["final_topk"] for row in sequence)
            for sequence in formulae.values()
        ),
        "candidate_width": width,
    }


def _self_test() -> None:
    rows = [
        {"record_id": "a", "formula_id": "f", "label": "1", "final_topk": ["1", "x"]},
        {"record_id": "b", "formula_id": "f", "label": "2", "final_topk": ["x", "2"]},
    ]
    report = _metrics(rows)
    assert report["top1_count"] == 1
    assert report["topk_count"] == 2
    assert report["formula_topk_oracle_count"] == 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-product-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--new-product-checkpoint", type=Path)
    parser.add_argument("--template-candidates", type=Path)
    parser.add_argument("--crohme", type=Path, default=DEFAULT_CROHME)
    parser.add_argument("--weight", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print('{"self_test":"pass"}')
        return 0
    if any(value is None for value in (
        args.new_product_checkpoint, args.template_candidates,
        args.output, args.report_output,
    )):
        parser.error("new checkpoint, template, output, and report paths are required")
    if not math.isfinite(args.weight) or not 0.0 <= args.weight <= 1.0:
        parser.error("--weight must be in [0,1]")
    if not 5 <= args.top_k <= 20:
        parser.error("--top-k must be in [5,20]")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    old_path = _d_path(args.old_product_checkpoint, "old product checkpoint")
    new_path = _d_path(args.new_product_checkpoint, "new product checkpoint")
    template_path = _d_path(args.template_candidates, "template candidate cache")
    crohme_path = _d_path(args.crohme, "CROHME root")
    output_path = _d_path(args.output, "candidate output")
    report_path = _d_path(args.report_output, "report output")
    if not all(path.exists() for path in (old_path, new_path, template_path, crohme_path)):
        parser.error("one or more fusion inputs are missing")
    if output_path.exists() or report_path.exists():
        parser.error("refusing to overwrite CROHME fusion evidence")
    resolved_device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if resolved_device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(resolved_device)
    old_model, labels, _ = _load_model(old_path, device)
    new_model, new_labels, _ = _load_model(new_path, device)
    if new_labels != labels:
        raise ValueError("old and new product HWR vocabularies differ")
    _assert_shared_encoder(old_model, new_model)
    items, coverage, _, _ = _crohme_items(crohme_path, set(labels))
    template_rows = list(_json_lines(template_path))
    if {str(row["record_id"]) for row in template_rows} != {
        str(item["record_id"]) for item in items
    }:
        raise ValueError("template and CROHME item coverage differ")
    scores = _score(
        items, old_model, new_model, labels, args.weight, args.top_k,
        device, args.batch_size,
    )
    output_rows = []
    maximum_probability_difference = 0.0
    prefix_token_changes = 0
    for template in template_rows:
        record_id = str(template["record_id"])
        score = scores[record_id]
        template_tokens = [str(value) for value in template["final_topk"]]
        template_probabilities = [float(value) for value in template["final_topk_probabilities"]]
        if len(template_tokens) != 5:
            raise ValueError(f"template is not Top-5: {record_id}")
        prefix_token_changes += score["final_topk"][:5] != template_tokens
        maximum_probability_difference = max(
            maximum_probability_difference,
            max(abs(left - right) for left, right in zip(
                score["final_topk_probabilities"][:5], template_probabilities, strict=True,
            )),
        )
        output_rows.append({
            **template,
            **score,
            "hwr_policy": "old_new_product_probability_fusion",
            "hwr_fusion_weight": float(args.weight),
            "hwr_candidate_width": int(args.top_k),
            "candidate_status": "research_recall_only",
        })
    if prefix_token_changes or maximum_probability_difference > 1e-7:
        raise ValueError(
            "wider retrieval failed to reproduce audited Top-5 prefix: "
            f"tokens={prefix_token_changes}, probability_delta={maximum_probability_difference}"
        )
    _write_rows(output_path, output_rows)
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "product_validation": False,
        "dataset": "CROHME CC BY-NC repeated diagnostic only",
        "configuration": {
            "weight": float(args.weight), "top_k": int(args.top_k),
            "input_mode": INPUT_MODE,
        },
        "top5_replay": {
            "prefix_token_changes": prefix_token_changes,
            "maximum_probability_difference": maximum_probability_difference,
        },
        "coverage": coverage,
        "metrics": _metrics(output_rows),
        "inputs": {
            "old_product_checkpoint_sha256": _sha256(old_path),
            "new_product_checkpoint_sha256": _sha256(new_path),
            "template_candidates_sha256": _sha256(template_path),
        },
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output_path), "output_sha256": _sha256(output_path),
        "report": str(report_path), "report_sha256": _sha256(report_path),
        "metrics": report["metrics"], "top5_replay": report["top5_replay"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
