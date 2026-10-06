"""Prepare, then one-shot evaluate the frozen global prototype scorer reserve."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from build_cleanroom_writer_style_bank_v5 import sha256
from evaluate_writer_candidate_promoter_extension_v9 import _class_sets, _global_map, _outer_cache, _parent_overlap
from hierarchical_writer_model_v7 import DEFAULT_CHECKPOINT


ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "artifacts/global_prototype_shape_scorer_v9_20260823_r3_frozen"
SCORER_AUDIT = ROOT / "reports/GLOBAL_PROTOTYPE_SHAPE_SCORER_V9_R3_INDEPENDENT_AUDIT.json"
OLD_BANK = ROOT / "artifacts/cleanroom_writer_style_v5_20260823_smoke32_r4_r2"
EXT_BANK = ROOT / "artifacts/cleanroom_writer_style_v5_20260823_extension32_r4_r1"
OLD_CACHE = ROOT / "artifacts/writer_adaptation_v9_frozen_cache_20260823_r1.npz"
CONSUMED_CACHE = ROOT / "artifacts/writer_adaptation_v9_extension32_frozen_cache_20260823_r1.npz"
EXT_ADMISSION = ROOT / "reports/WRITER_STYLE_V5_EXTENSION32_ADMISSION_RECEIPT.json"
CONSUMED_R1 = ROOT / "artifacts/writer_candidate_promoter_v9_extension_outer_20260823_r1"
INVALID_R2 = ROOT / "artifacts/writer_candidate_promoter_v9_extension_outer_20260823_r2"
HELPER_SCRIPT = ROOT / "scripts/evaluate_writer_candidate_promoter_extension_v9.py"
DEFAULT_OUTPUT = ROOT / "artifacts/global_prototype_shape_scorer_v9_reserve_outer_20260823_r4"
DEFAULT_CACHE = ROOT / "artifacts/writer_adaptation_v9_global_prototype_reserve_cache_20260823_r4.npz"


def _scalar(payload: np.lib.npyio.NpzFile, name: str) -> str:
    return str(np.asarray(payload[name]).item())


def _verify_inputs(checkpoint: Path) -> dict:
    checkpoint_hash = sha256(checkpoint)
    old_bank_file = OLD_BANK / "synthetic_writer_style_bank_v5_r4.npz"
    ext_bank_file = EXT_BANK / "synthetic_writer_style_bank_v5_r4.npz"
    old_bank_hash, ext_bank_hash = sha256(old_bank_file), sha256(ext_bank_file)
    with np.load(OLD_CACHE, allow_pickle=False) as payload:
        if _scalar(payload, "bank_sha256") != old_bank_hash or _scalar(payload, "checkpoint_sha256") != checkpoint_hash:
            raise ValueError("old cache embedded bank/checkpoint hash mismatch")
    with np.load(CONSUMED_CACHE, allow_pickle=False) as payload:
        if _scalar(payload, "bank_sha256") != ext_bank_hash or _scalar(payload, "checkpoint_sha256") != checkpoint_hash:
            raise ValueError("consumed cache embedded bank/checkpoint hash mismatch")

    freeze = json.loads((SCORER / "freeze_receipt.json").read_text(encoding="utf-8"))
    audit = json.loads(SCORER_AUDIT.read_text(encoding="utf-8"))
    if freeze.get("status") != "GLOBAL_PROTOTYPE_SCORER_FROZEN_RESERVE_UNOPENED":
        raise ValueError("scorer freeze status mismatch")
    for key, path in {
        "development_report_sha256": SCORER / "development_report.json",
        "prototype_sha256": SCORER / "global_class_prototypes.npz",
        "config_sha256": SCORER / "frozen_config.json",
        "generator_source_snapshot_sha256": SCORER / "freeze_global_prototype_shape_scorer_v9.source.py",
    }.items():
        if freeze.get(key) != sha256(path):
            raise ValueError(f"scorer freeze hash mismatch: {key}")
    audit_gates = audit.get("gates", {})
    positive_audit_gates = (
        "live_generator_equals_snapshot", "old_fit20_dev4_disjoint",
        "alpha_selected_before_consumed_diagnostic", "writer_residual_disabled",
        "any_unsupported_top5_identity", "candidate_set_exact",
        "independent_metrics_match", "independent_reproduction_exact",
    )
    closed_audit_gates = ("reserve_loaded", "legacy_loaded", "known_replay_loaded", "crohme_loaded", "mathwriting_loaded")
    if audit.get("status") != "INDEPENDENT_GLOBAL_PROTOTYPE_SCORER_AUDIT_PASSED" or not all(audit_gates.get(key) is True for key in positive_audit_gates) or not all(audit_gates.get(key) is False for key in closed_audit_gates):
        raise ValueError("independent scorer audit did not pass")
    if audit.get("decision", {}).get("r3_prepare_allowed") is not True or audit.get("decision", {}).get("r3_open_allowed") is not False:
        raise ValueError("independent scorer decision does not allow prepare-only")
    audit_hashes = audit.get("hashes", {})
    for key, path in {
        "artifact_report": SCORER / "development_report.json",
        "artifact_receipt": SCORER / "freeze_receipt.json",
        "prototype": SCORER / "global_class_prototypes.npz",
        "config": SCORER / "frozen_config.json",
        "source_snapshot": SCORER / "freeze_global_prototype_shape_scorer_v9.source.py",
    }.items():
        if audit_hashes.get(key) != sha256(path):
            raise ValueError(f"independent scorer audit live hash mismatch: {key}")

    admission = json.loads(EXT_ADMISSION.read_text(encoding="utf-8"))
    if admission.get("status") != "INDEPENDENT_EXTENSION_ADMISSION_PASSED" or not all(admission.get("gates", {}).values()):
        raise ValueError("extension independent admission did not pass")
    if admission.get("extension_bank_sha256") != ext_bank_hash or admission.get("extension_report_sha256") != sha256(EXT_BANK / "report.json"):
        raise ValueError("extension admission does not match live bank/report")

    r1_receipt = json.loads((CONSUMED_R1 / "outer_open_receipt.json").read_text(encoding="utf-8"))
    r1_report = json.loads((CONSUMED_R1 / "outer_report.json").read_text(encoding="utf-8"))
    r2_receipt = json.loads((INVALID_R2 / "outer_open_receipt.json").read_text(encoding="utf-8"))
    if r1_report.get("status") != "SYNTHETIC_EXTENSION_OUTER_REJECTED":
        raise ValueError("consumed r1 rejection evidence missing")
    if list(r1_receipt.get("sealed_reserve_local_indices", [])) != list(r2_receipt.get("outer_local_indices", [])):
        raise ValueError("invalid r2 does not identify the original sealed reserve")
    if (INVALID_R2 / "outer_report.json").exists() or DEFAULT_CACHE.exists():
        raise ValueError("invalid r2 or new reserve cache indicates prior opening")
    mapping = _global_map(EXT_BANK)
    outer = [int(value) for value in r2_receipt["outer_local_indices"]]
    if [mapping[value] for value in outer] != r2_receipt["outer_global_writer_ids"]:
        raise ValueError("reserve local/global writer mapping mismatch")
    if set(outer) & set(r1_receipt["outer_local_indices"]) or len(set(outer)) != 16:
        raise ValueError("reserve overlaps consumed r1 or has wrong size")
    return {
        "outer_local_indices": outer,
        "outer_global_writer_ids": [mapping[value] for value in outer],
        "consumed_r1_global_writer_ids": r1_receipt["outer_global_writer_ids"],
        "checkpoint_sha256": checkpoint_hash,
        "old_bank_sha256": old_bank_hash,
        "old_report_sha256": sha256(OLD_BANK / "report.json"),
        "old_metadata_sha256": sha256(OLD_BANK / "synthetic_writer_style_bank_v5_r4.metadata.jsonl.gz"),
        "old_latents_sha256": sha256(OLD_BANK / "writer_latents.json"),
        "extension_bank_sha256": ext_bank_hash,
        "extension_report_sha256": sha256(EXT_BANK / "report.json"),
        "extension_metadata_sha256": sha256(EXT_BANK / "synthetic_writer_style_bank_v5_r4.metadata.jsonl.gz"),
        "extension_latents_sha256": sha256(EXT_BANK / "writer_latents.json"),
        "old_cache_sha256": sha256(OLD_CACHE),
        "consumed_cache_sha256": sha256(CONSUMED_CACHE),
        "scorer_freeze_receipt_sha256": sha256(SCORER / "freeze_receipt.json"),
        "scorer_audit_sha256": sha256(SCORER_AUDIT),
        "extension_admission_sha256": sha256(EXT_ADMISSION),
        "consumed_r1_receipt_sha256": sha256(CONSUMED_R1 / "outer_open_receipt.json"),
        "consumed_r1_report_sha256": sha256(CONSUMED_R1 / "outer_report.json"),
        "invalid_r2_prepared_receipt_sha256": sha256(INVALID_R2 / "outer_open_receipt.json"),
        "helper_script_sha256": sha256(HELPER_SCRIPT),
    }


def _expected(checkpoint: Path) -> dict:
    verified = _verify_inputs(checkpoint)
    return {
        "schema": "aiflow-global-prototype-reserve-one-shot/v9",
        "status": "PREPARED_RESERVE_UNOPENED",
        "scorer": str(SCORER), "extension": str(EXT_BANK), "checkpoint": str(checkpoint),
        **verified,
        "boundary": {
            "reserve_adapter_predictions_scoring_tuning_opened": False,
            "bank_construction_labels_hwr_admission_known": True,
            "legacy_opened": False, "known_replay_opened": False,
            "crohme_opened": False, "mathwriting_opened": False,
            "product_promotion": False,
        },
    }


def _apply(embeddings: np.ndarray, logits: np.ndarray, prototypes: np.ndarray, admitted: np.ndarray, alpha: float) -> tuple[np.ndarray, list[dict], int]:
    norm = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    top5 = np.argsort(logits, axis=1)[:, -5:][:, ::-1]
    prediction = top5[:, 0].copy(); trace = []; identity = 0
    for row, candidates in enumerate(top5):
        if not np.all(admitted[candidates]):
            identity += 1; continue
        values = logits[row, candidates].astype(np.float64)
        z = (values - values.mean()) / max(float(values.std()), 1e-6)
        score = z + alpha * (norm[row] @ prototypes[candidates].T)
        selected = int(candidates[int(np.argmax(score))])
        prediction[row] = selected
        if selected != int(candidates[0]):
            trace.append({"row": row, "baseline_top5": candidates.astype(int).tolist(), "baseline": int(candidates[0]), "selected": selected})
    return prediction, trace, identity


def _metrics(labels: np.ndarray, logits: np.ndarray, prediction: np.ndarray, writers: np.ndarray, splits: np.ndarray, mapping: dict[int, str], allowed: set[int] | None = None) -> dict:
    query = np.flatnonzero(splits == 1)
    if allowed is not None:
        query = query[np.isin(labels[query], np.asarray(sorted(allowed), np.int64))]
    baseline = np.argmax(logits[query], axis=1); adapted = prediction[query]; truth = labels[query]
    states = {}; violations = 0
    for writer in sorted(set(writers[query].tolist())):
        local = np.flatnonzero(writers[query] == writer); old = baseline[local]; new = adapted[local]; expected = truth[local]
        chunks = [local[start:start + 8] for start in range(0, len(local), 8)]
        states[mapping[writer]] = {
            "rows": len(local), "baseline_correct": int(np.sum(old == expected)), "adapted_correct": int(np.sum(new == expected)),
            "changed": int(np.sum(old != new)), "improved": int(np.sum((old != expected) & (new == expected))), "regressed": int(np.sum((old == expected) & (new != expected))),
            "pseudo_formulae": len(chunks),
            "baseline_pseudo_exact": int(sum(bool(np.all(baseline[chunk] == truth[chunk])) for chunk in chunks)),
            "adapted_pseudo_exact": int(sum(bool(np.all(adapted[chunk] == truth[chunk])) for chunk in chunks)),
        }
        for index in query[local]:
            violations += int(int(prediction[index]) not in np.argsort(logits[index])[-5:])
    rows = sum(state["rows"] for state in states.values()); formulae = sum(state["pseudo_formulae"] for state in states.values())
    return {
        "rows": rows, "writers": len(states),
        "baseline_top1": sum(state["baseline_correct"] for state in states.values()) / rows,
        "adapted_top1": sum(state["adapted_correct"] for state in states.values()) / rows,
        "pseudo_formula_contract": "writer query rows in deterministic bank order, chunked by 8; diagnostic only",
        "pseudo_formulae": formulae,
        "baseline_pseudo_exact": sum(state["baseline_pseudo_exact"] for state in states.values()) / formulae,
        "adapted_pseudo_exact": sum(state["adapted_pseudo_exact"] for state in states.values()) / formulae,
        "changed": sum(state["changed"] for state in states.values()), "improved": sum(state["improved"] for state in states.values()), "regressed": sum(state["regressed"] for state in states.values()),
        "candidate_set_violations": violations, "per_global_writer": states,
        "writer_top1_regressions": [writer for writer, state in states.items() if state["adapted_correct"] < state["baseline_correct"]],
        "writer_pseudo_exact_regressions": [writer for writer, state in states.items() if state["adapted_pseudo_exact"] < state["baseline_pseudo_exact"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--prepare", action="store_true"); parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.open:
        parser.error("choose exactly one of --prepare or --open")
    receipt_path = args.output / "reserve_open_receipt.json"
    if args.prepare:
        if args.output.exists() or args.cache.exists():
            parser.error("prepare output/cache must not exist")
        expected = _expected(args.checkpoint)
        args.output.mkdir(parents=True)
        snapshot = args.output / "evaluate_global_prototype_reserve_v9.source.py"
        snapshot.write_bytes(Path(__file__).read_bytes())
        expected["opener_script_sha256"] = sha256(Path(__file__))
        expected["opener_source_snapshot_sha256"] = sha256(snapshot)
        receipt_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "status": expected["status"], "outer": expected["outer_global_writer_ids"]}, ensure_ascii=False))
        return 0

    if not receipt_path.exists() or args.cache.exists() or (args.output / "outer_report.json").exists() or (args.output / "changed_prediction_trace.jsonl").exists():
        raise ValueError("prepare missing or one-shot reserve already opened/interrupted")
    actual = json.loads(receipt_path.read_text(encoding="utf-8")); expected = _expected(args.checkpoint)
    snapshot = args.output / "evaluate_global_prototype_reserve_v9.source.py"
    expected["opener_script_sha256"] = sha256(Path(__file__)); expected["opener_source_snapshot_sha256"] = sha256(snapshot)
    if actual != expected or expected["opener_script_sha256"] != expected["opener_source_snapshot_sha256"]:
        raise ValueError("prepared receipt/source/live input drift")

    arrays, cache_gate = _outer_cache(EXT_BANK, args.checkpoint, args.cache, actual["outer_local_indices"], [])
    with np.load(SCORER / "global_class_prototypes.npz", allow_pickle=False) as payload:
        prototypes = np.asarray(payload["prototypes"], np.float32); admitted = np.asarray(payload["admitted"], bool)
    alpha = float(json.loads((SCORER / "frozen_config.json").read_text(encoding="utf-8"))["alpha"])
    prediction, trace, identity = _apply(arrays["embeddings"], arrays["logits"], prototypes, admitted, alpha)
    mapping = _global_map(EXT_BANK)
    full = _metrics(arrays["labels"], arrays["logits"], prediction, arrays["writers"], arrays["splits"], mapping)
    old_classes, new_classes, class_audit = _class_sets(OLD_BANK, EXT_BANK)
    common = _metrics(arrays["labels"], arrays["logits"], prediction, arrays["writers"], arrays["splits"], mapping, old_classes & new_classes)
    new_only = _metrics(arrays["labels"], arrays["logits"], prediction, arrays["writers"], arrays["splits"], mapping, new_classes - old_classes)
    def safe(result: dict) -> bool:
        return not result["writer_top1_regressions"] and not result["writer_pseudo_exact_regressions"] and result["adapted_top1"] >= result["baseline_top1"] and result["adapted_pseudo_exact"] >= result["baseline_pseudo_exact"] and result["candidate_set_violations"] == 0
    new_only_identity = new_only["changed"] == 0 and safe(new_only)
    passed = safe(full) and safe(common) and full["adapted_top1"] > full["baseline_top1"] and new_only_identity
    trace_path = args.output / "changed_prediction_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as stream:
        for row in trace:
            row.update({"global_writer_id": mapping[int(arrays["writers"][row["row"]])], "truth": int(arrays["labels"][row["row"]]), "adapted_in_baseline_top5": row["selected"] in row["baseline_top5"]})
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema": "aiflow-global-prototype-reserve-outer/v9", "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "SYNTHETIC_RESERVE_OUTER_PASS" if passed else "SYNTHETIC_RESERVE_OUTER_REJECTED",
        "full": full, "common_class_sensitivity": common, "extension_only": new_only,
        "class_set_audit": class_audit, "identity_unsupported_top5": identity, "cache_gate": cache_gate,
        "parent_overlap": _parent_overlap(OLD_BANK, EXT_BANK, set(actual["outer_global_writer_ids"])),
        "gates": {"all_writer_top1_pseudo_exact_nonregression": safe(full), "aggregate_top1_positive": full["adapted_top1"] > full["baseline_top1"], "common_class_nonregression": safe(common), "extension_only_identity": new_only_identity, "candidate_set_exact": full["candidate_set_violations"] == 0, "legacy_opened": False, "known_replay_opened": False, "crohme_opened": False, "mathwriting_opened": False, "product_promotion": False},
        "boundary": "synthetic known-parent-manifold check; pseudo formula exact is not real formula exact or promotion evidence",
        "hashes": {"prepare_receipt_sha256": sha256(receipt_path), "cache_sha256": sha256(args.cache), "trace_sha256": sha256(trace_path), "opener_script_sha256": sha256(Path(__file__))},
    }
    report_path = args.output / "outer_report.json"; report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"], "full": full}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
