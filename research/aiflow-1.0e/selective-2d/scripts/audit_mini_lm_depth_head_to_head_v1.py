#!/usr/bin/env python3
"""Paired consumed-development comparison of mini formula LM depths."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "group_mean_geometry_prior_shadow_20261001" / "summary.json"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paired(before: np.ndarray, after: np.ndarray) -> dict:
    if before.shape != after.shape:
        raise ValueError("paired depth outcomes do not align")
    recovered = int((~before & after).sum())
    regressed = int((before & ~after).sum())
    discordant = recovered + regressed
    if discordant:
        tail = min(recovered, regressed)
        p_value = min(1.0, float(Fraction(
            2 * sum(math.comb(discordant, i) for i in range(tail + 1)),
            2**discordant,
        )))
    else:
        p_value = 1.0
    return {
        "rows": int(len(before)),
        "one_layer_hits": int(before.sum()),
        "two_layer_hits": int(after.sum()),
        "delta_two_minus_one_percentage_points": float((after.mean() - before.mean()) * 100.0) if len(before) else 0.0,
        "one_layer_only_exact": regressed,
        "two_layer_only_exact": recovered,
        "exact_to_exact": int((before & after).sum()),
        "wrong_to_wrong": int((~before & ~after).sum()),
        "two_sided_exact_mcnemar_p": p_value,
    }


def _writer_bootstrap(before: np.ndarray, after: np.ndarray, writers: list[str], seed: int, draws: int) -> dict:
    if len(before) != len(after) or len(before) != len(writers):
        raise ValueError("writer assignments do not align with formula outcomes")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, writer in enumerate(writers):
        groups[writer].append(index)
    keys = sorted(groups)
    deltas = np.asarray([int(after[groups[key]].sum()) - int(before[groups[key]].sum()) for key in keys], dtype=np.float64)
    counts = np.asarray([len(groups[key]) for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected = rng.integers(0, len(keys), size=len(keys))
        samples[draw] = deltas[selected].sum() / counts[selected].sum() * 100.0
    return {
        "writer_clusters": len(keys),
        "draws": draws,
        "seed": seed,
        "delta_two_minus_one_percentage_points_percentile_95_interval": [
            float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)),
        ],
        "bootstrap_fraction_two_layer_better": float((samples > 0).mean()),
        "per_writer": [
            {
                "writer_hash": key,
                "formulas": int(counts[index]),
                "one_layer_exact": int(before[groups[key]].sum()),
                "two_layer_exact": int(after[groups[key]].sum()),
            }
            for index, key in enumerate(keys)
        ],
        "interpretation_limit": "nine writers from consumed development data; not independent formula or device acceptance",
    }


def _load(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "aiflow-mini-lm-selective-decoder-shadow/v4":
        raise ValueError(f"unsupported shadow report schema: {path}")
    if report.get("protocol", {}).get("crohme_rows_loaded") != 0:
        raise ValueError(f"report does not attest CROHME exclusion: {path}")
    if report.get("protocol", {}).get("per_formula_predictions_included") is not True:
        raise ValueError(f"report lacks per-formula paired outputs: {path}")
    return report


def _candidate_reachability(summary_path: Path, expected_summary_sha256: str, formula_rows: dict[str, dict]) -> dict:
    """Separate visual Top-5 ceiling from context-reranking residuals."""
    if _sha256(summary_path) != expected_summary_sha256:
        raise ValueError("candidate-coverage summary hash differs from paired shadow reports")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("crohme_training_or_tuning") is not False:
        raise ValueError("candidate-coverage summary does not attest CROHME exclusion")
    records = {str(row["sample_id"]): row for row in summary.get("records", [])}
    if records.keys() != formula_rows.keys():
        raise ValueError("candidate-coverage summary formula set differs from paired reports")

    grouping_wrong = candidate_missing = top5_complete = 0
    by_depth = {
        name: {
            "top5_complete_exact_formulas": 0,
            "top5_complete_wrong_formulas": 0,
            "wrong_tokens": 0,
            "wrong_token_target_rank_counts": {str(rank): 0 for rank in range(1, 6)},
        }
        for name in ("one_layer", "two_layer")
    }
    for formula_id, result in formula_rows.items():
        fast = records[formula_id]["hwr_tournament"]["fast"]
        if bool(fast["group_exact"]) != bool(result["group_exact"]):
            raise ValueError(f"formula {formula_id} grouping status differs from the saved HWR summary")
        if not result["group_exact"]:
            grouping_wrong += 1
            continue
        truth = [str(token) for token in fast["target_tokens"]]
        symbols = fast["selected_symbols"]
        if len(truth) != len(symbols):
            raise ValueError(f"formula {formula_id} target/group counts differ despite exact grouping")
        ranks: list[int | None] = []
        for token, symbol in zip(truth, symbols, strict=True):
            candidates = [str(value) for value in symbol["hwr_topk"]]
            rank = candidates.index(token) + 1 if token in candidates else None
            ranks.append(rank)
        complete = all(rank is not None for rank in ranks)
        if complete != bool(fast["hwr_top5_complete_if_groups_exact"]):
            raise ValueError(f"formula {formula_id} Top-5 completeness disagrees with HWR summary")
        if not complete:
            candidate_missing += 1
            for name in by_depth:
                if bool(result[name]["formula_exact"]):
                    raise AssertionError(f"formula {formula_id} was exact despite its target being absent from Top-5")
            continue
        top5_complete += 1
        for name, out in (("one_layer", result["one_layer"]), ("two_layer", result["two_layer"])):
            exact = bool(out["formula_exact"])
            if exact != (out["tokens"] == truth):
                raise AssertionError(f"formula {formula_id} exact flag disagrees with its token output")
            if exact:
                by_depth[name]["top5_complete_exact_formulas"] += 1
                continue
            by_depth[name]["top5_complete_wrong_formulas"] += 1
            for token, prediction, rank in zip(truth, out["tokens"], ranks, strict=True):
                if str(token) == str(prediction):
                    continue
                by_depth[name]["wrong_tokens"] += 1
                by_depth[name]["wrong_token_target_rank_counts"][str(rank)] += 1

    return {
        "formulas": len(formula_rows),
        "grouping_wrong_formulas": grouping_wrong,
        "group_exact_top5_missing_formulas": candidate_missing,
        "group_exact_top5_complete_formulas": top5_complete,
        "by_depth": by_depth,
        "interpretation": "grouping errors and absent target candidates cannot be repaired by any Top-5-preserving language reranker",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-layer-report", type=Path, required=True)
    parser.add_argument("--two-layer-report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("mini_lm_decoder", "mini_lm_margin_gated"), default="mini_lm_margin_gated")
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        parser.error("bootstrap-draws must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite head-to-head report: {args.output}")

    one, two = _load(args.one_layer_report.resolve()), _load(args.two_layer_report.resolve())
    p1, p2 = one["protocol"], two["protocol"]
    if int(p1["mini_lm_layers"]) != 1 or int(p2["mini_lm_layers"]) != 2:
        raise ValueError("expected one-layer and two-layer models in that order")
    for field in ("consumed_formula_count", "writer_count", "candidate_policy", "context_weight", "threshold_grid"):
        if p1[field] != p2[field]:
            raise ValueError(f"reports use different {field}")
    for field in ("summary_sha256", "formula_data_sha256"):
        if one["provenance"][field] != two["provenance"][field]:
            raise ValueError(f"reports use different frozen inputs: {field}")
    if one["inference"]["fast_group_exact_formulas"] != two["inference"]["fast_group_exact_formulas"]:
        raise ValueError("reports use different fast grouping outcomes")
    for report in (one, two):
        arm_metrics = report["inference"]["arms"][args.arm]
        if arm_metrics["candidate_preservation_rate"] != 1.0 or arm_metrics["grouping_mutations"] != 0:
            raise ValueError("depth report violates candidate-preservation or grouping invariants")

    rows1 = {str(row["formula_id"]): row for row in one["inference"]["per_formula_predictions"]}
    rows2 = {str(row["formula_id"]): row for row in two["inference"]["per_formula_predictions"]}
    if not rows1 or rows1.keys() != rows2.keys() or len(rows1) != int(p1["consumed_formula_count"]):
        raise ValueError("per-formula outcome sets do not align or are incomplete")
    formula_ids = sorted(rows1)
    combined_rows = {}
    for formula_id in formula_ids:
        a, b = rows1[formula_id], rows2[formula_id]
        if a["group_exact"] != b["group_exact"]:
            raise ValueError(f"formula {formula_id} grouping outcome differs between depth reports")
        combined_rows[formula_id] = {
            "group_exact": bool(a["group_exact"]),
            "one_layer": a["arms"][args.arm],
            "two_layer": b["arms"][args.arm],
        }
    candidate_reachability = _candidate_reachability(
        args.summary.resolve(), one["provenance"]["summary_sha256"], combined_rows,
    )
    before, after, writer_hashes = [], [], []
    changed_details = []
    token_rows = token_one_hits = token_two_hits = token_changed = token_recovered = token_regressed = 0
    for formula_id in formula_ids:
        a, b = rows1[formula_id], rows2[formula_id]
        for field in ("writer_hash", "group_exact", "truth_tokens"):
            if a[field] != b[field]:
                raise ValueError(f"formula {formula_id} differs in {field}")
        if args.arm not in a["arms"] or args.arm not in b["arms"]:
            raise ValueError(f"formula {formula_id} lacks requested arm {args.arm}")
        out1, out2 = a["arms"][args.arm], b["arms"][args.arm]
        correct1, correct2 = bool(out1["formula_exact"]), bool(out2["formula_exact"])
        expected1 = bool(a["group_exact"] and out1["tokens"] == a["truth_tokens"])
        expected2 = bool(b["group_exact"] and out2["tokens"] == b["truth_tokens"])
        if correct1 != expected1 or correct2 != expected2:
            raise AssertionError(f"formula exact flag disagrees with tokens for {formula_id}")
        before.append(correct1)
        after.append(correct2)
        writer_hashes.append(str(a["writer_hash"]))
        if out1["tokens"] != out2["tokens"]:
            changed_details.append({
                "formula_id": formula_id,
                "writer_hash": a["writer_hash"],
                "group_exact": bool(a["group_exact"]),
                "one_layer_exact": correct1,
                "two_layer_exact": correct2,
                "one_layer_tokens": out1["tokens"],
                "two_layer_tokens": out2["tokens"],
                "truth_tokens": a["truth_tokens"],
            })
        if a["group_exact"]:
            truth = a["truth_tokens"]
            if len(out1["tokens"]) != len(truth) or len(out2["tokens"]) != len(truth):
                raise ValueError(f"formula {formula_id} token count differs from exact grouping")
            for target_token, token1, token2 in zip(truth, out1["tokens"], out2["tokens"], strict=True):
                hit1, hit2 = token1 == target_token, token2 == target_token
                token_rows += 1
                token_one_hits += int(hit1)
                token_two_hits += int(hit2)
                token_changed += int(token1 != token2)
                token_recovered += int(not hit1 and hit2)
                token_regressed += int(hit1 and not hit2)

    before_array, after_array = np.asarray(before, dtype=bool), np.asarray(after, dtype=bool)
    formula_pair = _paired(before_array, after_array)
    report = {
        "schema": "aiflow-mini-lm-depth-head-to-head/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "consumed_development_matched_depth_diagnostic",
        "protocol": {
            "training_performed": False,
            "threshold_selection_performed_in_source_reports": True,
            "crohme_rows_loaded": 0,
            "product_adopted": False,
            "warning": "Both thresholds were selected by nested writer-LOFO on this consumed 149-formula cohort; this is not independent acceptance.",
            "arm": args.arm,
        },
        "provenance": {
            "one_layer_report_sha256": _sha256(args.one_layer_report.resolve()),
            "two_layer_report_sha256": _sha256(args.two_layer_report.resolve()),
            "summary_sha256": one["provenance"]["summary_sha256"],
            "candidate_coverage_summary_sha256": _sha256(args.summary.resolve()),
            "formula_data_sha256": one["provenance"]["formula_data_sha256"],
            "one_layer_checkpoint_sha256": one["provenance"]["checkpoint_sha256"],
            "two_layer_checkpoint_sha256": two["provenance"]["checkpoint_sha256"],
        },
        "models": {
            "one_layer": {"layers": int(p1["mini_lm_layers"]), "candidate_preservation_rate": one["inference"]["arms"][args.arm]["candidate_preservation_rate"], "grouping_mutations": one["inference"]["arms"][args.arm]["grouping_mutations"]},
            "two_layer": {"layers": int(p2["mini_lm_layers"]), "candidate_preservation_rate": two["inference"]["arms"][args.arm]["candidate_preservation_rate"], "grouping_mutations": two["inference"]["arms"][args.arm]["grouping_mutations"]},
        },
        "comparison": {
            "formulas": len(formula_ids),
            "writers": len(set(writer_hashes)),
            "group_exact_formulas": int(sum(bool(rows1[formula_id]["group_exact"]) for formula_id in formula_ids)),
            "formula_exact": {
                **formula_pair,
                "writer_cluster_bootstrap": _writer_bootstrap(before_array, after_array, writer_hashes, args.seed, args.bootstrap_draws),
            },
            "exact_group_token_comparison": {
                "tokens": token_rows,
                "one_layer_hits": token_one_hits,
                "two_layer_hits": token_two_hits,
                "delta_hits": token_two_hits - token_one_hits,
                "one_layer_top1": token_one_hits / token_rows if token_rows else 0.0,
                "two_layer_top1": token_two_hits / token_rows if token_rows else 0.0,
                "predictions_changed": token_changed,
                "one_layer_wrong_two_layer_correct": token_recovered,
                "one_layer_correct_two_layer_wrong": token_regressed,
            },
            "candidate_reachability": candidate_reachability,
            "changed_formula_details": changed_details,
        },
        "decision": {
            "independent_acceptance": False,
            "android_latency_verified": False,
            "promotion_eligible": False,
            "interpretation": "depth comparison only; retain one layer as the smaller shadow candidate if the paired outcome is non-inferior, then require fresh writer/formula/device acceptance",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "mini_lm_depth_head_to_head_complete",
        "output": str(args.output.resolve()),
        "arm": args.arm,
        "formula_exact": formula_pair,
        "token_comparison": report["comparison"]["exact_group_token_comparison"],
        "candidate_reachability": candidate_reachability,
        "crohme_rows": 0,
        "product_adopted": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
