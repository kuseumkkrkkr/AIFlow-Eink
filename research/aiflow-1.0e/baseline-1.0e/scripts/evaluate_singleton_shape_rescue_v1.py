#!/usr/bin/env python3
"""Evaluate the singleton-only auxiliary HWR rescue and freeze its shadow config."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from calibrate_project_punctuation_v1 import _collision_free_eval_indices
from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import DEFAULT_PRODUCT, INPUT_MODE, _load_model, _sha256
from evaluate_formula_sequence_guard_v1 import _scope
from finalize_formula_context_v1 import DEFAULT_CONTEXT
from singleton_shape_rescue_v1 import (
    DEFAULT_CONFIGURATION, SCHEMA as RESCUE_SCHEMA,
    _self_test as rescue_self_test, apply_singleton_shape_rescue,
)
from train_character_classifier_v1 import apply_input_mode
import train_masked_context_reranker_v1 as masked


SCHEMA = "aiflow-singleton-shape-rescue-evaluation/v1"
CONFIG_SCHEMA = "aiflow-singleton-shape-rescue-runtime-config/v1"


def _predictions(path: Path) -> dict[str, str]:
    return {
        str(row["record_id"]): str(row["finalized_top1"])
        for row in _json_lines(path)
    }


def _external_holdout(
    cache_dir: Path, old_checkpoint: Path, new_checkpoint: Path,
    device: torch.device, batch_size: int,
) -> dict:
    manifest = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    train_info = manifest["sets"]["math_train"]
    eval_info = manifest["sets"]["math_eval"]
    train = np.load(cache_dir / train_info["features"], mmap_mode="r")
    features = np.load(cache_dir / eval_info["features"], mmap_mode="r")
    truth = np.load(cache_dir / eval_info["labels"], mmap_mode="r")
    kept, collision_audit = _collision_free_eval_indices(train, features, INPUT_MODE)
    indices = kept.numpy()
    old, labels, _ = _load_model(old_checkpoint, device)
    new, new_labels, _ = _load_model(new_checkpoint, device)
    if labels != new_labels:
        raise ValueError("singleton rescue product vocabularies differ")
    new_state = new.state_dict()
    for name, value in old.state_dict().items():
        if not name.startswith("math_head.") and not torch.equal(
            value.detach().cpu(), new_state[name].detach().cpu(),
        ):
            raise ValueError(f"singleton rescue product encoders differ: {name}")
    thresholds = DEFAULT_CONFIGURATION["token_confidence_thresholds"]
    maximum_ranks = DEFAULT_CONFIGURATION["token_candidate_maximum_ranks"]
    maximum_rank = max(maximum_ranks.values())
    weight = float(DEFAULT_CONFIGURATION["auxiliary_weight"])
    baseline_correct = challenger_correct = improved = regressed = changed = 0
    selected_truth = Counter()
    selected_prediction = Counter()
    candidate_violations = 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            take = indices[start:start + batch_size]
            raw = apply_input_mode(np.asarray(features[take]), INPUT_MODE)
            embeddings = old.encode(torch.from_numpy(raw).to(device))
            old_probabilities = old.math_head(embeddings).softmax(dim=1)
            new_probabilities = new.math_head(embeddings).softmax(dim=1)
            auxiliary = (
                (1.0 - weight) * old_probabilities + weight * new_probabilities
            )
            confidences, auxiliary_topk = auxiliary.topk(maximum_rank, dim=1)
            old_top5 = old_probabilities.topk(5, dim=1).indices
            baseline = old_top5[:, 0]
            expected = torch.from_numpy(
                np.asarray(truth[take], dtype=np.int64)
            ).to(device)
            challenger = baseline.clone()
            for index, (candidate_indices, candidate_confidences) in enumerate(zip(
                auxiliary_topk.tolist(), confidences.tolist(),
            )):
                ranked = []
                for rank, (token_index, confidence) in enumerate(
                    zip(candidate_indices, candidate_confidences), start=1,
                ):
                    token = labels[token_index]
                    threshold = thresholds.get(token)
                    if (
                        threshold is None or rank > maximum_ranks[token]
                        or float(confidence) < float(threshold)
                    ):
                        continue
                    ranked.append((token_index, token, float(confidence), rank))
                if not ranked:
                    continue
                original_top5 = old_top5[index].tolist()
                preserved = []
                for item in ranked:
                    if item[0] not in original_top5:
                        candidate_violations += 1
                    else:
                        preserved.append(item)
                if not preserved:
                    continue
                token_index, token, _, _ = max(
                    preserved, key=lambda item: (item[2], -item[3], item[1]),
                )
                challenger[index] = token_index
                selected_truth[labels[int(expected[index])]] += 1
                selected_prediction[token] += 1
            before = baseline == expected
            after = challenger == expected
            baseline_correct += int(before.sum())
            challenger_correct += int(after.sum())
            improved += int((~before & after).sum())
            regressed += int((before & ~after).sum())
            changed += int((challenger != baseline).sum())
    return {
        "scope": "fixed collision-free external math holdout as standalone glyph proxy",
        "records": len(indices),
        "baseline_correct": baseline_correct,
        "baseline_top1": baseline_correct / len(indices),
        "challenger_correct": challenger_correct,
        "challenger_top1": challenger_correct / len(indices),
        "changed": changed,
        "improved": improved,
        "regressed": regressed,
        "candidate_violations": candidate_violations,
        "selected_truth": dict(sorted(selected_truth.items())),
        "selected_prediction": dict(sorted(selected_prediction.items())),
        "collision_audit": collision_audit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-candidates", type=Path)
    parser.add_argument("--direct-finalized", type=Path)
    parser.add_argument("--direct-auxiliary", type=Path)
    parser.add_argument("--crohme-candidates", type=Path)
    parser.add_argument("--external-cache", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--new-product-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        rescue_self_test()
        print('{"self_test":"pass"}')
        return 0
    required = (
        args.direct_candidates, args.direct_finalized, args.direct_auxiliary,
        args.crohme_candidates, args.external_cache, args.new_product_checkpoint,
        args.output, args.config_output,
    )
    if any(value is None for value in required):
        parser.error("direct, CROHME, external, checkpoint, and output paths are required")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    paths = [Path(value).expanduser().resolve() for value in required]
    (
        direct_candidates, direct_finalized, direct_auxiliary,
        crohme_candidates, external_cache, new_product_checkpoint,
        output, config_output,
    ) = paths
    file_inputs = (
        direct_candidates, direct_finalized, direct_auxiliary,
        crohme_candidates, new_product_checkpoint,
    )
    if not all(path.is_file() for path in file_inputs):
        parser.error("singleton shape rescue input file is missing")
    if not (external_cache / "cache_manifest.json").is_file():
        parser.error("singleton shape rescue external cache is missing")
    if output.exists() or config_output.exists():
        parser.error("refusing to overwrite singleton shape rescue evidence")
    context_checkpoint = Path(args.checkpoint).expanduser().resolve()
    hwr_checkpoint = Path(args.hwr_checkpoint).expanduser().resolve()
    direct_rows = list(_json_lines(direct_candidates))
    baseline = _predictions(direct_finalized)
    auxiliary_rows = list(_json_lines(direct_auxiliary))
    challenger, rescue_audit = apply_singleton_shape_rescue(
        direct_rows, baseline, auxiliary_rows, DEFAULT_CONFIGURATION,
    )
    truth = {str(row["record_id"]): str(row["label"]) for row in direct_rows}
    changes = []
    glyph_improved = glyph_regressed = 0
    for change in rescue_audit["changes"]:
        record_id = str(change["record_id"])
        expected = truth[record_id]
        outcome = (
            "improved" if baseline[record_id] != expected and challenger[record_id] == expected
            else "regressed" if baseline[record_id] == expected and challenger[record_id] != expected
            else "neutral"
        )
        glyph_improved += outcome == "improved"
        glyph_regressed += outcome == "regressed"
        changes.append({**change, "truth": expected, "outcome": outcome})
    direct = {
        "evaluation": _scope(direct_rows, baseline, challenger),
        "audit": {**rescue_audit, "changes": changes},
        "glyph_improved": glyph_improved,
        "glyph_regressed": glyph_regressed,
    }
    crohme_rows = list(_json_lines(crohme_candidates))
    crohme_singletons = sum(
        len(sequence) == 1 for sequence in masked._formulae(crohme_rows).values()
    )
    external = _external_holdout(
        external_cache, hwr_checkpoint, new_product_checkpoint,
        torch.device(args.device), args.batch_size,
    )
    formula_outcomes = direct["evaluation"]["formula_outcomes"]
    runtime_admitted = bool(
        formula_outcomes["improved"] > 0
        and formula_outcomes["regressed"] == 0
        and glyph_regressed == 0
        and external["improved"] > 0
        and external["regressed"] == 0
        and external["candidate_violations"] == 0
        and crohme_singletons == 0
    )
    gate = {
        "runtime_admitted": runtime_admitted,
        "status": "shadow_runtime_only",
        "criteria": {
            "direct_formula_improvement": formula_outcomes["improved"] > 0,
            "direct_regressions_zero": formula_outcomes["regressed"] == glyph_regressed == 0,
            "external_improvement": external["improved"] > 0,
            "external_regressions_zero": external["regressed"] == 0,
            "candidate_preservation": external["candidate_violations"] == 0,
            "crohme_has_no_applicable_singletons": crohme_singletons == 0,
        },
        "limits": (
            "thresholds were selected on repeatedly observed project-owned formulas; "
            "external holdout is glyph-only and CROHME has no singleton formula; "
            "untouched project-owned formula acceptance remains required"
        ),
    }
    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "training_performed": False,
        "configuration": DEFAULT_CONFIGURATION,
        "gate": gate,
        "direct_project_owned_writer_loo": direct,
        "external_collision_free_holdout": external,
        "crohme_noncommercial_diagnostic": {
            "formulas": len(masked._formulae(crohme_rows)),
            "singleton_formulas": crohme_singletons,
            "applicable_changes": 0,
            "product_validation": False,
        },
        "inputs": {
            "direct_candidates_sha256": _sha256(direct_candidates),
            "direct_finalized_sha256": _sha256(direct_finalized),
            "direct_auxiliary_sha256": _sha256(direct_auxiliary),
            "crohme_candidates_sha256": _sha256(crohme_candidates),
            "context_checkpoint_sha256": _sha256(context_checkpoint),
            "hwr_checkpoint_sha256": _sha256(hwr_checkpoint),
            "new_product_checkpoint_sha256": _sha256(new_product_checkpoint),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    config = {
        "schema": CONFIG_SCHEMA,
        "rescue_schema": RESCUE_SCHEMA,
        "generated_at": generated_at,
        "status": "shadow_runtime_only",
        "configuration": DEFAULT_CONFIGURATION,
        "checkpoint_sha256": _sha256(context_checkpoint),
        "hwr_checkpoint_sha256": _sha256(hwr_checkpoint),
        "new_product_checkpoint_sha256": _sha256(new_product_checkpoint),
        "direct_auxiliary_sha256": _sha256(direct_auxiliary),
        "gate": gate,
        "evidence": {"path": str(output), "sha256": _sha256(output)},
    }
    config_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "report": str(output), "report_sha256": _sha256(output),
        "config": str(config_output), "config_sha256": _sha256(config_output),
        "gate": gate,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
