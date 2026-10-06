#!/usr/bin/env python3
"""Fixed-policy transfer audit for the all-position synthetic-relation mini-LM.

This is a post-hoc diagnostic on the already-consumed 149-formula development
cohort. It performs no training or threshold selection, never reads CROHME,
and only reweights existing HWR Top-5 candidates.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import audit_mini_lm_selective_decoder_shadow_v1 as gate_audit
import audit_prompt_bert_context_on_frozen149_v1 as context_audit
import export_mini_formula_lm_onnx_v1 as export
import run_prompt_mini_lm_distillation_v1 as trainer
from selective_decoder_v1 import decode_selective_partition


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928"
AUGMENTATION_DIR = ARTIFACTS / "mini_lm_relation_aug_20261003_allmask"
BASE_DIR = ARTIFACTS / "mini_lm_distill_20261002_r2"
DEFAULT_OUTPUT = AUGMENTATION_DIR / "allmask_relation_selective_transfer_shadow_seedpair.json"
SEEDS = (20261003, 20261004)
FIXED_CONTEXT_WEIGHT = gate_audit.FROZEN_CONTEXT_WEIGHT


def _sha256(path: Path) -> str:
    return export._sha256(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verified_model(
    *, checkpoint: Path, report: dict, report_path: Path,
    labels: list[str], relation_ids: list[str], arm: str | None,
) -> tuple[trainer.MiniFormulaLM, dict]:
    checkpoint_hash = _sha256(checkpoint)
    if report.get("protocol", {}).get("crohme_rows_loaded") != 0:
        raise ValueError(f"{report_path}: CROHME exclusion not attested")
    if report["protocol"].get("crohme_training_or_tuning") is not False:
        raise ValueError(f"{report_path}: CROHME training/tuning exclusion not attested")

    if report.get("schema") == "aiflow-prompt-mini-lm-distillation/v1":
        expected_hash = report["student"]["checkpoint_sha256"]
        architecture = report["student"]["architecture"]
        model_kind = "frozen_prompt_distillation_baseline"
    elif report.get("schema") == "aiflow-mini-formula-lm-relation-augmentation-probe/v1":
        if report["protocol"].get("consumed_149_formula_set_loaded") is not False:
            raise ValueError("synthetic training report loaded the consumed evaluation cohort")
        if report["protocol"].get("synthetic_test_used_for_selection") is not False:
            raise ValueError("synthetic test split was used for model selection")
        if report["protocol"].get("synthetic_train_mask_policy") != "all_positions":
            raise ValueError("expected the all-position synthetic training checkpoint")
        if arm is None or arm not in report.get("checkpoints", {}):
            raise ValueError("all-position report must identify the exact checkpoint arm")
        entry = report["checkpoints"][arm]
        if Path(entry["path"]).resolve() != checkpoint.resolve():
            raise ValueError("augmentation report points to a different checkpoint")
        expected_hash = entry["sha256"]
        architecture = report["architecture"]
        model_kind = "all_position_synthetic_relation_augmentation"
        if report.get("data", {}).get("cross_split_formula_overlap") != 0:
            raise ValueError("synthetic formula splits overlap")
        if report.get("data", {}).get("synthetic_prompt_formula_overlap") != 0:
            raise ValueError("synthetic formulas overlap the prompt corpus")
    else:
        raise ValueError(f"unsupported checkpoint report schema: {report.get('schema')}")

    if checkpoint_hash != expected_hash:
        raise ValueError(f"checkpoint/report SHA-256 mismatch: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema") not in {
        "aiflow-prompt-mini-lm-distillation/v1",
        "aiflow-mini-formula-lm-relation-augmentation-probe/v1",
    }:
        raise ValueError("unsupported mini-LM checkpoint schema")
    if [str(value) for value in payload.get("labels", [])] != labels:
        raise ValueError("mini-LM label order differs from the frozen HWR contract")
    if [str(value) for value in payload.get("relations", [])] != relation_ids:
        raise ValueError("mini-LM relation vocabulary differs from the frozen contract")
    layers = int(architecture["layers"])
    if int(payload.get("layers", -1)) != layers:
        raise ValueError("checkpoint/report layer count mismatch")

    trainer.LAYERS = layers
    model = trainer.MiniFormulaLM(len(labels), len(relation_ids), int(payload["max_positions"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, {
        "kind": model_kind,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "layers": layers,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_bytes": checkpoint.stat().st_size,
    }


def _predict_scores(model: trainer.MiniFormulaLM, inputs: dict, examples: list[dict], rows: list[dict], labels: list[str]) -> dict:
    logits = export._predict_torch(model, inputs).astype(np.float32, copy=False)
    if logits.shape != (len(examples), len(labels)) or not np.isfinite(logits).all():
        raise AssertionError("mini-LM emitted invalid logits")
    logp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    row_by_id = {str(row["record_id"]): row for row in rows}
    label_to_index = {label: index for index, label in enumerate(labels)}
    scores = {}
    for index, example in enumerate(examples):
        record_id = str(example["record_id"])
        candidates = [str(token) for token in row_by_id[record_id]["final_topk"]]
        scores[record_id] = {token: float(logp[index, label_to_index[token]]) for token in candidates}
    return scores


def _target_relation_tags(inputs: dict, examples: list[dict], labels: list[str], relation_ids: list[str]) -> dict:
    """Read only the explicit relation markers adjacent to each masked input."""
    tags_by_record = {}
    class_count = len(labels)
    input_ids = inputs["input_ids"]
    for index, example in enumerate(examples):
        position = int(example["mask_position"])
        sequence = input_ids[index]
        tags = []
        for neighbor in (position - 1, position + 1):
            if 0 <= neighbor < len(sequence):
                value = int(sequence[neighbor])
                if class_count <= value < class_count + len(relation_ids):
                    tags.append(relation_ids[value - class_count])
        tags_by_record[str(example["record_id"])] = sorted(set(tags))
    return tags_by_record


def _decode_arms(rows: list[dict], targets: dict, raw: list[dict], scores: dict, summary: dict) -> dict:
    rows_by_id = {str(row["record_id"]): row for row in rows}
    raw_by_id = {str(row["sample_id"]): row for row in raw}
    summary_by_id = {
        str(record["sample_id"]): record for record in summary["records"]
    }
    predictions = {name: {} for name in ("fast", "decoder", "mini_lm_unlocked", "mini_lm_top1_agreement_lock")}
    accepted = {name: 0 for name in predictions if name != "fast"}

    for formula_id, target in targets.items():
        fast = summary_by_id[formula_id]["hwr_tournament"]["fast"]
        selected = list(fast["selected_symbols"])
        groups = [[int(value) for value in group] for group in fast["groups"]]
        symbols = []
        fallback = []
        for ordinal, symbol in enumerate(selected):
            record_id = f"{formula_id}:{ordinal}"
            row = rows_by_id[record_id]
            topk = [str(value) for value in symbol["hwr_topk"]]
            if topk != [str(value) for value in row["final_topk"]]:
                raise AssertionError("saved HWR Top-5 differs across formula/token records")
            fallback.append(topk[0])
            predictions["fast"][record_id] = topk[0]
            symbols.append({
                "stroke_indices": [int(value) for value in symbol["stroke_indices"]],
                "hwr_topk": topk,
                "hwr_topk_probabilities": [float(value) for value in symbol["hwr_topk_probabilities"]],
                "geometry": dict(row["geometry"]),
            })

        stroke_count = len(raw_by_id[formula_id]["strokes"])
        base = decode_selective_partition(formula_id, groups, symbols, stroke_count=stroke_count)
        unlocked = decode_selective_partition(
            formula_id, groups, symbols, stroke_count=stroke_count,
            context_log_probabilities=scores, context_weight=FIXED_CONTEXT_WEIGHT,
        )
        case = {
            "formula_id": formula_id,
            "groups": groups,
            "symbols": symbols,
            "stroke_count": stroke_count,
        }
        locked_tokens, locked, _ = gate_audit._decode_with_override_margin_gate(
            case, scores, margin_threshold=None,
        )
        base_tokens = [str(value) for value in base["tokens"]] if base.get("accepted") else fallback
        unlocked_tokens = [str(value) for value in unlocked["tokens"]] if unlocked.get("accepted") else fallback
        outputs = {
            "decoder": (base, base_tokens),
            "mini_lm_unlocked": (unlocked, unlocked_tokens),
            "mini_lm_top1_agreement_lock": (locked, [str(value) for value in locked_tokens]),
        }
        for name, (result, tokens) in outputs.items():
            accepted[name] += int(bool(result.get("accepted")))
            if len(tokens) != len(symbols):
                raise AssertionError(f"{name}: output token count differs from selected groups")
            for ordinal, (token, symbol) in enumerate(zip(tokens, symbols, strict=True)):
                if token not in symbol["hwr_topk"]:
                    raise AssertionError(f"{name}: emitted a token outside HWR Top-5")
                predictions[name][f"{formula_id}:{ordinal}"] = token

    return {"predictions": predictions, "accepted_structural_decodes": accepted}


def _compare_predictions(rows: list[dict], targets: dict, before: dict, after: dict) -> dict:
    rows_by_formula: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_formula.setdefault(str(row["formula_id"]), []).append(row)
    recovered = regressed = changed = improved_tokens = regressed_tokens = 0
    transitions: dict[str, int] = {}
    deltas = {}
    by_target: dict[str, dict[str, int]] = {}
    changed_formula_details = []
    for formula_id, sequence in sorted(rows_by_formula.items()):
        sequence.sort(key=lambda row: int(row["context"]["index"]))
        target = targets[formula_id]
        truth = [str(value) for value in target["tokens"]] if target["group_exact"] else []
        before_tokens = [str(before[str(row["record_id"])]) for row in sequence]
        after_tokens = [str(after[str(row["record_id"])]) for row in sequence]
        before_exact = bool(target["group_exact"] and before_tokens == truth)
        after_exact = bool(target["group_exact"] and after_tokens == truth)
        transition = ("exact" if before_exact else "wrong") + "_to_" + ("exact" if after_exact else "wrong")
        transitions[transition] = transitions.get(transition, 0) + 1
        deltas[formula_id] = int(after_exact) - int(before_exact)
        recovered += int(not before_exact and after_exact)
        regressed += int(before_exact and not after_exact)

        token_changes = []
        for ordinal, (row, old, new) in enumerate(zip(sequence, before_tokens, after_tokens, strict=True)):
            if old == new:
                continue
            changed += 1
            expected = truth[ordinal] if target["group_exact"] else None
            if expected is not None:
                improved = old != expected and new == expected
                regressed_token = old == expected and new != expected
                improved_tokens += int(improved)
                regressed_tokens += int(regressed_token)
                counts = by_target.setdefault(expected, {"changed": 0, "improved": 0, "regressed": 0})
                counts["changed"] += 1
                counts["improved"] += int(improved)
                counts["regressed"] += int(regressed_token)
            token_changes.append({
                "ordinal": ordinal,
                "truth": expected,
                "baseline": old,
                "allmask": new,
                "hwr_top5": [str(value) for value in row["final_topk"]],
            })
        if token_changes:
            changed_formula_details.append({
                "formula_id": formula_id,
                "writer_hash": hashlib.sha256(str(target["writer_id"]).encode("utf-8")).hexdigest()[:12],
                "group_exact": bool(target["group_exact"]),
                "baseline_formula_exact": before_exact,
                "allmask_formula_exact": after_exact,
                "token_changes": token_changes,
            })
    return {
        "formula_exact_recovered": recovered,
        "formula_exact_regressed": regressed,
        "formula_exact_delta": recovered - regressed,
        "formula_exact_transitions": transitions,
        "writer_cluster_bootstrap_delta_pp_95_ci": context_audit._writer_bootstrap(deltas, targets)["delta_formula_exact_pp_95_ci"],
        "changed_tokens": changed,
        "improved_tokens": improved_tokens,
        "regressed_tokens": regressed_tokens,
        "by_target_class": by_target,
        "changed_formula_details": changed_formula_details,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--augmentation-report", type=Path, default=AUGMENTATION_DIR / "relation_augmentation_report.json")
    parser.add_argument("--base-checkpoint", type=Path, default=BASE_DIR / "mini_formula_lm.pt")
    parser.add_argument("--base-report", type=Path, default=BASE_DIR / "mini_formula_lm_distillation_report.json")
    parser.add_argument("--summary", type=Path, default=context_audit.DEFAULT_SUMMARY)
    parser.add_argument("--data", type=Path, default=context_audit.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing output: {output}")

    augmentation_path = args.augmentation_report.resolve()
    summary_path, data_path = args.summary.resolve(), args.data.resolve()
    for name, path in (("augmentation report", augmentation_path), ("base checkpoint", args.base_checkpoint.resolve()), ("base report", args.base_report.resolve()), ("summary", summary_path), ("formula data", data_path)):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")

    augmentation = _read_json(augmentation_path)
    if augmentation.get("protocol", {}).get("synthetic_train_mask_policy") != "all_positions":
        raise ValueError("expected all-position training report")
    if augmentation.get("protocol", {}).get("crohme_rows_loaded") != 0:
        raise ValueError("augmentation report includes CROHME rows")

    summary = _read_json(summary_path)
    if summary.get("crohme_training_or_tuning") is not False or summary.get("product_default_enabled") is not False:
        raise ValueError("frozen summary violates CROHME/product-off contract")
    raw = context_audit._jsonl(data_path)
    if any("crohme" in str(row.get("source_partition", "")).casefold() for row in raw):
        raise ValueError("CROHME row found in requested diagnostic input")
    raw_by_id = {str(row["sample_id"]): row for row in raw}
    rows, targets = context_audit._formula_rows(summary, raw_by_id)
    if len(targets) != 149:
        raise ValueError(f"expected consumed 149-formula diagnostic; found {len(targets)}")

    base_report_path = args.base_report.resolve()
    base_report = _read_json(base_report_path)
    fixed_lambda = float(base_report["protocol"]["fusion_lambda_frozen_from_teacher"])
    if fixed_lambda != FIXED_CONTEXT_WEIGHT:
        raise ValueError("the fixed audit weight differs from the frozen teacher fusion weight")

    base_checkpoint = args.base_checkpoint.resolve()
    base_model, base_meta = _verified_model(
        checkpoint=base_checkpoint, report=base_report, report_path=base_report_path,
        labels=[str(value) for value in torch.load(base_checkpoint, map_location="cpu", weights_only=True)["labels"]],
        relation_ids=[str(value) for value in torch.load(base_checkpoint, map_location="cpu", weights_only=True)["relations"]],
        arm=None,
    )
    labels = [str(value) for value in torch.load(base_checkpoint, map_location="cpu", weights_only=True)["labels"]]
    relation_ids = [str(value) for value in torch.load(base_checkpoint, map_location="cpu", weights_only=True)["relations"]]
    inputs, examples = export._current_inputs(rows, base_model, labels)
    if len(examples) != len(rows):
        raise AssertionError("Fast-context input coverage mismatch")

    models = [("frozen_prompt_baseline", base_model, base_meta)]
    for seed in SEEDS:
        arm = f"prompt_plus_synthetic_2d_seed{seed}"
        entry = augmentation["checkpoints"].get(arm)
        if not isinstance(entry, dict):
            raise ValueError(f"all-position checkpoint arm missing: {arm}")
        checkpoint = Path(entry["path"]).resolve()
        model, metadata = _verified_model(
            checkpoint=checkpoint, report=augmentation, report_path=augmentation_path,
            labels=labels, relation_ids=relation_ids, arm=arm,
        )
        metadata["seed"] = seed
        metadata["training_examples"] = augmentation["arms"]["prompt_plus_synthetic_2d"][str(seed)]["training_examples"]
        models.append((f"allmask_seed{seed}", model, metadata))

    arm_reports = {}
    arm_predictions = {}
    arm_scores = {}
    for name, model, metadata in models:
        scores = _predict_scores(model, inputs, examples, rows, labels)
        decoded = _decode_arms(rows, targets, raw, scores, summary)
        metrics = {
            key: export._formula_metrics(rows, targets, values)
            for key, values in decoded["predictions"].items()
        }
        if any(metric["candidate_preservation_rate"] != 1.0 for metric in metrics.values()):
            raise AssertionError(f"{name}: Top-5 candidate preservation failed")
        if any(len(values) != len(rows) for values in decoded["predictions"].values()):
            raise AssertionError(f"{name}: incomplete token prediction coverage")
        arm_reports[name] = {
            "model": metadata,
            "accepted_structural_decodes": decoded["accepted_structural_decodes"],
            "metrics": metrics,
        }
        arm_predictions[name] = decoded["predictions"]
        arm_scores[name] = scores

    target_relations = _target_relation_tags(inputs, examples, labels, relation_ids)
    relation_selective_router = {
        "policy": "use all-position model scores only if the masked token has an adjacent non-right relation marker; otherwise use frozen prompt baseline scores",
        "relation_source": "deterministic above/below/superscript/subscript/etc markers already present in the masked Fast-HWR context input",
        "records_by_relation_source": {},
        "formulas_using_all_position_scores": 0,
        "arms": {},
    }
    for seed in SEEDS:
        source_model = f"allmask_seed{seed}"
        routed_name = f"relation_selective_seed{seed}"
        merged_scores = {}
        source_counts = {"frozen_prompt_baseline": 0, "allmask_nonright": 0}
        formulas_with_nonright = set()
        for row in rows:
            record_id = str(row["record_id"])
            relations = target_relations[record_id]
            use_allmask = any(relation != "right" for relation in relations)
            if use_allmask:
                source_counts["allmask_nonright"] += 1
                formulas_with_nonright.add(str(row["formula_id"]))
                merged_scores[record_id] = arm_scores[source_model][record_id]
            else:
                source_counts["frozen_prompt_baseline"] += 1
                merged_scores[record_id] = arm_scores["frozen_prompt_baseline"][record_id]
        routed_decode = _decode_arms(rows, targets, raw, merged_scores, summary)
        routed_metrics = {
            key: export._formula_metrics(rows, targets, values)
            for key, values in routed_decode["predictions"].items()
        }
        if any(metric["candidate_preservation_rate"] != 1.0 for metric in routed_metrics.values()):
            raise AssertionError(f"{routed_name}: Top-5 candidate preservation failed")
        arm_reports[routed_name] = {
            "model": {"kind": "deterministic_relation_selective_fusion", "base_allmask_seed": seed},
            "accepted_structural_decodes": routed_decode["accepted_structural_decodes"],
            "metrics": routed_metrics,
        }
        arm_predictions[routed_name] = routed_decode["predictions"]
        relation_selective_router["records_by_relation_source"][str(seed)] = source_counts
        relation_selective_router["formulas_using_all_position_scores"] = max(
            relation_selective_router["formulas_using_all_position_scores"], len(formulas_with_nonright),
        )
        relation_selective_router["arms"][routed_name] = {
            "source_counts": source_counts,
            "formulas_using_all_position_scores": len(formulas_with_nonright),
            "comparison_vs_frozen_prompt_unlocked": _compare_predictions(
                rows, targets,
                arm_predictions["frozen_prompt_baseline"]["mini_lm_unlocked"],
                routed_decode["predictions"]["mini_lm_unlocked"],
            ),
        }

    reference = arm_predictions["frozen_prompt_baseline"]["mini_lm_unlocked"]
    comparisons = {
        name: _compare_predictions(
            rows, targets, reference,
            predictions["mini_lm_unlocked"],
        )
        for name, predictions in arm_predictions.items()
        if name.startswith("allmask_seed")
    }

    report = {
        "schema": "aiflow-mini-lm-allmask-hwr-transfer-shadow/v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "posthoc_consumed_development_transfer_diagnostic",
        "protocol": {
            "training_performed": False,
            "threshold_selection_performed": False,
            "threshold_policy": "fixed unconditional Top-5 context and fixed HWR-Top-1-agreement lock only; no margin sweep",
            "context_weight": FIXED_CONTEXT_WEIGHT,
            "context_weight_source": "frozen baseline teacher lambda; not retuned for all-position checkpoints",
            "model_input_contains_truth": False,
            "candidate_policy": "existing HWR Top-5 only; no new classes or regrouping",
            "grouping_mutations": 0,
            "crohme_rows_loaded": 0,
            "consumed_formula_count": len(targets),
            "writer_count": len({str(target["writer_id"]) for target in targets.values()}),
            "product_adopted": False,
            "warning": "The 149-formula cohort is consumed development data. The base context weight was historically calibrated on overlapping development material. This transfer replay is not independent handwriting acceptance or Android latency evidence.",
        },
        "provenance": {
            "augmentation_report": str(augmentation_path),
            "augmentation_report_sha256": _sha256(augmentation_path),
            "base_report": str(base_report_path),
            "base_report_sha256": _sha256(base_report_path),
            "summary_sha256": _sha256(summary_path),
            "formula_data_sha256": _sha256(data_path),
            "audit_script_sha256": _sha256(Path(__file__).resolve()),
        },
        "inference": {
            "formulas": len(targets),
            "selected_groups": len(rows),
            "fast_group_exact_formulas": sum(bool(target["group_exact"]) for target in targets.values()),
            "fixed_context_policy_arms": list(arm_reports),
            "arms": arm_reports,
            "allmask_vs_frozen_prompt_unlocked": comparisons,
            "relation_selective_router": relation_selective_router,
        },
        "decision": {
            "independent_acceptance": False,
            "android_latency_verified": False,
            "automatic_default_replacement": False,
            "next_gate": "repeat on a fresh project-owned formula cohort from unseen writers, then export/profile on Android",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "mini_lm_allmask_hwr_transfer_shadow_complete",
        "report": str(output),
        "formulas": len(targets),
        "crohme_rows": 0,
        "arms": {
            name: {
                policy: value["reranked_formula_exact"]
                for policy, value in arm_reports[name]["metrics"].items()
            }
            for name in arm_reports
        },
        "product_adopted": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
