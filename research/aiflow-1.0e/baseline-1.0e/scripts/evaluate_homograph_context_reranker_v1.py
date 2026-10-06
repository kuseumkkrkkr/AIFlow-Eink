#!/usr/bin/env python3
"""Evaluate a conservative formula-context reranker over fixed HWR Top-5 candidates.

The trajectory encoder and 372-class head stay frozen.  Direct project ink is
evaluated leave-one-writer-out; CROHME is a noncommercial, formula-disjoint
research diagnostic.  The reranker may reorder candidates inside one visual
family, but cannot add a token or alter stroke ownership.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from audit_replay_protocol_v1 import DEFAULT_CROHME
from character_tensor_v1 import ROOT, _json_lines
from evaluate_48hz_prefix_v1 import (
    DEFAULT_BASE,
    DEFAULT_LOO,
    DEFAULT_PRODUCT,
    INPUT_MODE,
    _crohme_items,
    _direct_items,
    _load_loo_heads,
    _load_model,
    _prefix_tensor,
    _sha256,
)
from train_character_classifier_v1 import apply_input_mode


SCHEMA = "aiflow-homograph-context-reranker-evaluation/v1"
SEED = 20260814
DEFAULT_OUTPUT = ROOT / "artifacts" / "homograph_context_20260814"
DEFAULT_DIRECT_PREDICTIONS = ROOT / "artifacts" / "prefix_48hz_20260814_direct" / "direct_prefix_predictions.jsonl.gz"
DEFAULT_CROHME_PREDICTIONS = ROOT / "artifacts" / "prefix_48hz_20260814_crohme" / "crohme_prefix_predictions.jsonl.gz"

FAMILIES = {
    "vertical_slash": frozenset({"1", "|", "/", r"\mid", "I", "l", "i", r"\prime", r"\backslash", r"\setminus"}),
    "circle": frozenset({"0", "O", "o", r"\mathcal{O}", r"\circ", r"\degree", r"\fullmoon", r"\O"}),
    "cross": frozenset({"x", "X", r"\times", r"\chi", r"\mathcal{X}", r"\mathfrak{X}"}),
}
STRICT_FAMILIES = {
    "vertical_slash": frozenset({"1", "|", "/"}),
    "circle": frozenset({"0", "O", "o"}),
    "cross": frozenset({"x", r"\times"}),
}
VARIANTS = ("emission", "context", "context_geometry")
C_GRID = (None, 0.1, 0.3, 1.0, 3.0)
GEOMETRY_FEATURES = (
    "aspect_log", "path_over_diag", "direction_x", "direction_y", "stroke_count", "point_count_log",
    "center_x", "center_y", "width_rel", "height_rel",
)


def _d_path(path: Path, kind: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{kind} must remain on D:: {resolved}")
    return resolved


def _family(token: str) -> str | None:
    matches = [name for name, values in FAMILIES.items() if token in values]
    if len(matches) > 1:
        raise AssertionError(f"overlapping homograph families: {token}")
    return matches[0] if matches else None


def _role(token: str) -> str:
    if token in {str(value) for value in range(10)}:
        return "digit"
    if token in {"+", "-", "=", "/", r"\times", r"\div", r"\pm", r"\cdot"}:
        return "operator"
    if token in {"(", ")", "[", "]", "{", "}", "|", r"\mid", r"\lceil", r"\rceil", r"\lfloor", r"\rfloor"}:
        return "fence"
    if len(token) == 1 and token.isalpha() or token.startswith((r"\math", r"\chi", r"\alpha", r"\beta", r"\gamma")):
        return "operand"
    if token in {"<S>", "</S>"}:
        return "boundary"
    return "other"


def _fold(formula_id: str, folds: int, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{formula_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _bbox(strokes: list[np.ndarray]) -> dict[str, float]:
    points = np.concatenate(strokes, axis=0)
    left, top = points[:, :2].min(axis=0)
    right, bottom = points[:, :2].max(axis=0)
    width, height = float(right - left), float(bottom - top)
    diagonal = max(math.hypot(width, height), 1e-8)
    path = sum(float(np.linalg.norm(np.diff(stroke[:, :2], axis=0), axis=1).sum()) for stroke in strokes)
    longest = max(strokes, key=lambda stroke: len(stroke))
    direction = longest[-1, :2] - longest[0, :2]
    return {
        "left": float(left), "top": float(top), "right": float(right), "bottom": float(bottom),
        "width": width, "height": height, "cx": float((left + right) / 2), "cy": float((top + bottom) / 2),
        "aspect_log": float(math.log((width + 1e-6) / (height + 1e-6))),
        "path_over_diag": path / diagonal,
        "direction_x": float(direction[0] / diagonal), "direction_y": float(direction[1] / diagonal),
        "stroke_count": float(len(strokes)), "point_count_log": float(math.log1p(len(points))),
    }


def _direct_raw_strokes(rows: list[dict]) -> dict[str, list[np.ndarray]]:
    formulas = {row["sample_id"]: row for row in _json_lines(ROOT / "hf-dataset" / "data" / "formulas_valid.jsonl")}
    by_formula: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_formula[row["formula_id"]].append(row)
    result: dict[str, list[np.ndarray]] = {}
    for annotation in _json_lines(ROOT / "hf-dataset" / "data" / "ownership_train.jsonl"):
        if not annotation.get("accepted"):
            continue
        formula_rows = by_formula[annotation["sample_id"]]
        if len(formula_rows) != len(annotation["groups"]):
            raise ValueError(f"direct geometry ownership mismatch: {annotation['sample_id']}")
        strokes = sorted(formulas[annotation["sample_id"]]["strokes"], key=lambda value: value["order"])
        for row, group in zip(formula_rows, annotation["groups"], strict=True):
            result[row["record_id"]] = [
                np.asarray([[point["x"], point["y"], point["t_ms"]] for point in strokes[int(index)]["points"]], dtype=np.float64)
                for index in sorted(set(group))
            ]
    if set(result) != {row["record_id"] for row in rows}:
        raise ValueError("direct raw geometry coverage mismatch")
    return result


def _geometry(items: list[dict], raw_override: dict[str, list[np.ndarray]] | None = None) -> dict[str, dict[str, float]]:
    stats = {item["record_id"]: _bbox((raw_override or {}).get(item["record_id"], item["strokes"])) for item in items}
    by_formula: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_formula[item["formula_id"]].append(stats[item["record_id"]])
    for item in items:
        value = stats[item["record_id"]]
        formula = by_formula[item["formula_id"]]
        left, top = min(row["left"] for row in formula), min(row["top"] for row in formula)
        right, bottom = max(row["right"] for row in formula), max(row["bottom"] for row in formula)
        width, height = max(right - left, 1e-8), max(bottom - top, 1e-8)
        value.update({
            "center_x": (value["cx"] - left) / width, "center_y": (value["cy"] - top) / height,
            "width_rel": value["width"] / width, "height_rel": value["height"] / height,
        })
    return stats


@torch.inference_mode()
def _score_items(items: list[dict], model, labels: list[str], device: torch.device, heads: dict | None = None, batch_size: int = 512) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        raw = np.stack([_prefix_tensor(item["strokes"]) for item in batch]).astype(np.float32, copy=False)
        embeddings = model.encode(torch.from_numpy(apply_input_mode(raw, INPUT_MODE)).to(device))
        if heads is None:
            logits = model.math_head(embeddings)
        else:
            logits = torch.empty((len(batch), len(labels)), device=device)
            writers = [str(item["writer_group"]) for item in batch]
            for writer in sorted(set(writers)):
                indices = torch.tensor([index for index, value in enumerate(writers) if value == writer], device=device)
                logits[indices] = heads[writer](embeddings[indices])
        probabilities = logits.softmax(dim=1)
        values, indices = probabilities.topk(5, dim=1)
        for item, token_indices, token_probabilities in zip(batch, indices.cpu().tolist(), values.cpu().tolist(), strict=True):
            output[item["record_id"]] = {
                "final_topk": [labels[index] for index in token_indices],
                "final_topk_probabilities": [float(value) for value in token_probabilities],
            }
    return output


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _decorate(
    saved_rows: list[dict], items: list[dict], scores: dict[str, dict],
    geometry: dict[str, dict[str, float]], *, verify_saved_top5: bool = True,
) -> list[dict]:
    item_ids = {item["record_id"] for item in items}
    if item_ids != {row["record_id"] for row in saved_rows}:
        raise ValueError("raw item and saved prediction IDs differ")
    rows = []
    for saved in saved_rows:
        score = scores[saved["record_id"]]
        if verify_saved_top5 and score["final_topk"] != saved["final_topk"]:
            raise ValueError(f"checkpoint replay changed saved Top-5: {saved['record_id']}")
        rows.append({
            key: saved[key] for key in ("record_id", "label", "source", "formula_id", "writer_group") if key in saved
        } | score | {"geometry": geometry[saved["record_id"]]})
    by_formula: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_formula[row["formula_id"]].append(row)
    for sequence in by_formula.values():
        for index, row in enumerate(sequence):
            previous = sequence[index - 1] if index else None
            following = sequence[index + 1] if index + 1 < len(sequence) else None
            row["context"] = {
                "index": index, "length": len(sequence),
                "previous_top1": previous["final_topk"][0] if previous else "<S>",
                "next_top1": following["final_topk"][0] if following else "</S>",
                "previous_dx": row["geometry"]["center_x"] - previous["geometry"]["center_x"] if previous else 0.0,
                "previous_dy": row["geometry"]["center_y"] - previous["geometry"]["center_y"] if previous else 0.0,
                "next_dx": following["geometry"]["center_x"] - row["geometry"]["center_x"] if following else 0.0,
                "next_dy": following["geometry"]["center_y"] - row["geometry"]["center_y"] if following else 0.0,
            }
    return rows


def _options(row: dict) -> list[tuple[str, int, float]]:
    family = _family(row["final_topk"][0])
    if family is None:
        return []
    return [
        (token, rank, row["final_topk_probabilities"][rank])
        for rank, token in enumerate(row["final_topk"])
        if token in FAMILIES[family]
    ]


def _features(row: dict, candidate: str, rank: int, probability: float, variant: str) -> dict[str, str | float | bool]:
    top_probability = row["final_topk_probabilities"][0]
    result: dict[str, str | float | bool] = {
        "family": _family(candidate) or "none", "candidate": candidate, "candidate_role": _role(candidate),
        "rank": float(rank), "is_top1": rank == 0, "probability": probability,
        "log_probability": math.log(max(probability, 1e-12)), "probability_gap": probability - top_probability,
    }
    if variant == "emission":
        return result
    context = row["context"]
    previous, following = context["previous_top1"], context["next_top1"]
    previous_role, following_role = _role(previous), _role(following)
    result.update({
        "previous_top1": previous, "next_top1": following,
        "previous_role": previous_role, "next_role": following_role,
        "previous_candidate": f"{previous}|{candidate}", "candidate_next": f"{candidate}|{following}",
        "role_trigram": f"{previous_role}|{_role(candidate)}|{following_role}",
        "between_digits": previous_role == following_role == "digit",
        "between_operands": previous_role == following_role == "operand",
        "position": context["index"] / max(context["length"] - 1, 1),
        "formula_length_log": math.log1p(context["length"]),
    })
    if variant == "context":
        return result
    # Absolute source coordinates are deliberately excluded; only box/formula-normalized geometry is admissible.
    result.update({key: row["geometry"][key] for key in GEOMETRY_FEATURES})
    result.update({key: context[key] for key in ("previous_dx", "previous_dy", "next_dx", "next_dy")})
    return result


def _fit(rows: list[dict], variant: str, c_value: float) -> tuple[DictVectorizer, LogisticRegression, dict] | None:
    valid = [row for row in rows if len(_options(row)) >= 2 and row["label"] in {token for token, _, _ in _options(row)}]
    counts = Counter(row["label"] for row in valid)
    if len(valid) < 2 or len(counts) < 2:
        return None
    features, targets, weights = [], [], []
    for row in valid:
        weight = len(valid) / (len(counts) * counts[row["label"]])
        for token, rank, probability in _options(row):
            features.append(_features(row, token, rank, probability, variant))
            targets.append(token == row["label"])
            weights.append(weight)
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(features)
    model = LogisticRegression(C=c_value, max_iter=2000, solver="liblinear", random_state=SEED)
    model.fit(matrix, targets, sample_weight=weights)
    return vectorizer, model, {"training_records": len(valid), "truth_labels": dict(sorted(counts.items())), "features": len(vectorizer.feature_names_)}


def _apply(model_bundle, rows: list[dict], variant: str) -> dict[str, str]:
    output = {row["record_id"]: row["final_topk"][0] for row in rows}
    if model_bundle is None:
        return output
    vectorizer, model, _ = model_bundle
    for row in rows:
        options = _options(row)
        if len(options) < 2:
            continue
        matrix = vectorizer.transform([_features(row, *option, variant) for option in options])
        scores = model.predict_proba(matrix)[:, 1]
        output[row["record_id"]] = options[int(scores.argmax())][0]
    return output


def _strict_rows(rows: list[dict]) -> list[dict]:
    strict = set().union(*STRICT_FAMILIES.values())
    return [row for row in rows if row["label"] in strict]


def _selection_score(rows: list[dict], predictions: dict[str, str]) -> tuple[float, float]:
    selected = _strict_rows(rows)
    labels = sorted({row["label"] for row in selected})
    if not selected or not labels:
        return 0.0, 0.0
    micro = sum(predictions[row["record_id"]] == row["label"] for row in selected) / len(selected)
    macro = sum(
        sum(predictions[row["record_id"]] == label for row in selected if row["label"] == label)
        / sum(row["label"] == label for row in selected)
        for label in labels
    ) / len(labels)
    return macro, micro


def _select_c(rows: list[dict], variant: str) -> tuple[float | None, list[dict]]:
    formulas = sorted({row["formula_id"] for row in rows})
    folds = min(3, len(formulas))
    trials = []
    for c_value in C_GRID:
        predictions: dict[str, str] = {}
        for fold in range(folds):
            train = [row for row in rows if _fold(row["formula_id"], folds, f"inner-{variant}") != fold]
            held = [row for row in rows if _fold(row["formula_id"], folds, f"inner-{variant}") == fold]
            bundle = None if c_value is None else _fit(train, variant, c_value)
            predictions.update(_apply(bundle, held, variant))
        macro, micro = _selection_score(rows, predictions)
        trials.append({"c": c_value, "strict_macro": macro, "strict_micro": micro})
    best = max(trials, key=lambda row: (row["strict_macro"], row["strict_micro"], -(row["c"] or 0.0)))
    return best["c"], trials


def _cross_validated(rows: list[dict], variant: str, scope: str) -> tuple[dict[str, str], list[dict]]:
    if scope == "direct":
        held_groups = sorted({row["writer_group"] for row in rows})
        split = lambda row, group: row["writer_group"] == group
    else:
        held_groups = list(range(5))
        split = lambda row, group: _fold(row["formula_id"], 5, "outer-crohme") == group
    predictions, audit = {}, []
    for held_group in held_groups:
        held = [row for row in rows if split(row, held_group)]
        train = [row for row in rows if not split(row, held_group)]
        if {row["formula_id"] for row in train} & {row["formula_id"] for row in held}:
            raise AssertionError("formula leakage across reranker fold")
        c_value, trials = _select_c(train, variant)
        bundle = None if c_value is None else _fit(train, variant, c_value)
        predictions.update(_apply(bundle, held, variant))
        audit.append({
            "held_group": str(held_group), "train_formulas": len({row["formula_id"] for row in train}),
            "held_formulas": len({row["formula_id"] for row in held}), "selected_c": c_value,
            "inner_trials": trials, "fit": bundle[2] if bundle else {"training_records": 0, "features": 0},
        })
    if set(predictions) != {row["record_id"] for row in rows}:
        raise AssertionError("cross-validated prediction coverage mismatch")
    return predictions, audit


def _binomial_sign_test(improved: int, regressed: int) -> float | None:
    total = improved + regressed
    if total == 0:
        return None
    tail = sum(math.comb(total, value) for value in range(min(improved, regressed) + 1)) / (2 ** total)
    return min(1.0, 2 * tail)


def _metrics(rows: list[dict], predictions: dict[str, str]) -> dict:
    baseline = {row["record_id"]: row["final_topk"][0] for row in rows}
    improved = sum(baseline[row["record_id"]] != row["label"] and predictions[row["record_id"]] == row["label"] for row in rows)
    regressed = sum(baseline[row["record_id"]] == row["label"] and predictions[row["record_id"]] != row["label"] for row in rows)
    formulas: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        formulas[row["formula_id"]].append(row)
    by_family, by_label = {}, {}
    for family, labels in STRICT_FAMILIES.items():
        selected = [row for row in rows if row["label"] in labels]
        if selected:
            by_family[family] = {
                "records": len(selected),
                "baseline_top1": sum(baseline[row["record_id"]] == row["label"] for row in selected) / len(selected),
                "reranked_top1": sum(predictions[row["record_id"]] == row["label"] for row in selected) / len(selected),
                "top5_oracle": sum(row["label"] in row["final_topk"] for row in selected) / len(selected),
            }
    for label in sorted(set().union(*STRICT_FAMILIES.values())):
        selected = [row for row in rows if row["label"] == label]
        if selected:
            by_label[label] = {
                "records": len(selected),
                "baseline_hits": sum(baseline[row["record_id"]] == label for row in selected),
                "reranked_hits": sum(predictions[row["record_id"]] == label for row in selected),
                "top5_hits": sum(label in row["final_topk"] for row in selected),
            }
    strict = _strict_rows(rows)
    macro, micro = _selection_score(rows, predictions)
    return {
        "all_records": len(rows),
        "all_top1": sum(predictions[row["record_id"]] == row["label"] for row in rows) / len(rows),
        "baseline_all_top1": sum(baseline[row["record_id"]] == row["label"] for row in rows) / len(rows),
        "strict_records": len(strict), "strict_micro_top1": micro, "strict_macro_top1": macro,
        "eligible_records": sum(len(_options(row)) >= 2 for row in rows),
        "changed": sum(predictions[row["record_id"]] != baseline[row["record_id"]] for row in rows),
        "improved": improved, "regressed": regressed, "paired_sign_test_p": _binomial_sign_test(improved, regressed),
        "formula_exact": sum(all(predictions[row["record_id"]] == row["label"] for row in sequence) for sequence in formulas.values()) / len(formulas),
        "baseline_formula_exact": sum(all(baseline[row["record_id"]] == row["label"] for row in sequence) for sequence in formulas.values()) / len(formulas),
        "by_family": by_family, "by_label": by_label,
    }


def _evaluate_scope(rows: list[dict], scope: str) -> dict:
    baseline = {row["record_id"]: row["final_topk"][0] for row in rows}
    result = {"baseline": _metrics(rows, baseline), "variants": {}}
    for variant in VARIANTS:
        predictions, folds = _cross_validated(rows, variant, scope)
        result["variants"][variant] = {"metrics": _metrics(rows, predictions), "folds": folds}
    emission = result["variants"]["emission"]["metrics"]
    contextual = result["variants"]["context_geometry"]["metrics"]
    result["context_increment_over_emission"] = {
        "strict_micro_top1": contextual["strict_micro_top1"] - emission["strict_micro_top1"],
        "strict_macro_top1": contextual["strict_macro_top1"] - emission["strict_macro_top1"],
        "all_top1": contextual["all_top1"] - emission["all_top1"],
    }
    return result


def _load_or_build(scope: str, cache_path: Path, saved_path: Path, args, device: torch.device) -> tuple[list[dict], dict]:
    if cache_path.is_file():
        source = {"cache_reused": True, "cache_sha256": _sha256(cache_path)}
        if scope == "direct":
            source.update({"checkpoint_sha256": _sha256(args.base_checkpoint), "heads_sha256": _sha256(args.loo_heads)})
        else:
            source["checkpoint_sha256"] = _sha256(args.product_checkpoint)
        return list(_json_lines(cache_path)), source
    saved = list(_json_lines(saved_path))
    if scope == "direct":
        base_path = _d_path(args.base_checkpoint, "base checkpoint")
        model, labels, _ = _load_model(base_path, device)
        heads = _load_loo_heads(_d_path(args.loo_heads, "writer-LOO heads"), base_path, labels, device)
        items = _direct_items()
        scores = _score_items(items, model, labels, device, heads)
        geometry = _geometry(items, _direct_raw_strokes(saved))
        source = {"cache_reused": False, "checkpoint_sha256": _sha256(base_path), "heads_sha256": _sha256(args.loo_heads)}
    else:
        product_path = _d_path(args.product_checkpoint, "product checkpoint")
        model, labels, _ = _load_model(product_path, device)
        items, coverage, _, _ = _crohme_items(args.crohme, set(labels))
        scores = _score_items(items, model, labels, device)
        geometry = _geometry(items)
        source = {"cache_reused": False, "checkpoint_sha256": _sha256(product_path), "coverage": coverage}
    rows = _decorate(
        saved, items, scores, geometry,
        verify_saved_top5=not args.allow_challenger_top5,
    )
    _write_rows(cache_path, rows)
    return rows, source


def _self_test() -> None:
    assert _family("1") == "vertical_slash" and _family(r"\times") == "cross" and _family("+") is None
    assert _fold("formula", 5, "x") == _fold("formula", 5, "x")
    row = {
        "record_id": "r", "label": "1", "formula_id": "f", "final_topk": ["|", "1", "7", "I", "/"],
        "final_topk_probabilities": [0.5, 0.3, 0.1, 0.06, 0.04],
        "geometry": {"center_x": 0.5},
        "context": {"index": 1, "length": 3, "previous_top1": "2", "next_top1": "+", "previous_dx": 0.2, "previous_dy": 0.0, "next_dx": 0.2, "next_dy": 0.0},
    }
    assert [token for token, _, _ in _options(row)] == ["|", "1", "I", "/"]
    fixed = _apply(None, [row], "context")
    assert fixed == {"r": "|"}
    saved = [{
        "record_id": "r", "label": "1", "formula_id": "f",
        "final_topk": ["1", "|"],
    }]
    scores = {
        "r": {"final_topk": ["|", "1"], "final_topk_probabilities": [0.6, 0.4]}
    }
    geometry = {"r": {"center_x": 0.5, "center_y": 0.5}}
    try:
        _decorate(saved, [{"record_id": "r"}], scores, geometry)
    except ValueError:
        pass
    else:
        raise AssertionError("changed challenger Top-5 bypassed the default replay check")
    challenger = _decorate(
        saved, [{"record_id": "r"}], scores, geometry,
        verify_saved_top5=False,
    )
    assert challenger[0]["final_topk"] == ["|", "1"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--loo-heads", type=Path, default=DEFAULT_LOO)
    parser.add_argument("--product-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--direct-predictions", type=Path, default=DEFAULT_DIRECT_PREDICTIONS)
    parser.add_argument("--crohme-predictions", type=Path, default=DEFAULT_CROHME_PREDICTIONS)
    parser.add_argument("--crohme", type=Path, default=DEFAULT_CROHME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--allow-challenger-top5", action="store_true",
        help=(
            "build a new candidate cache from the supplied HWR checkpoint while "
            "using saved rows only for IDs, labels, grouping, and raw geometry"
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test(); print(json.dumps({"self_test": "pass"})); return 0
    output = _d_path(args.output, "output")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "homograph_context_report.json"
    if report_path.exists():
        parser.error(f"refusing to overwrite report: {report_path}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    direct_rows, direct_source = _load_or_build("direct", output / "direct_candidates.jsonl.gz", args.direct_predictions, args, device)
    crohme_rows, crohme_source = _load_or_build("crohme", output / "crohme_candidates.jsonl.gz", args.crohme_predictions, args, device)
    report = {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "frozen Top-5 candidates can be better ranked with neighboring token roles and box geometry",
        "candidate_policy": "reorder only candidates already present inside one declared homograph family",
        "model": {"type": "balanced one-hot logistic candidate ranker", "embedding_model_used": False, "c_grid": C_GRID, "inner_selection": "formula-disjoint strict macro then micro accuracy"},
        "features": {"emission": "candidate identity, rank, frozen-head probability", "context": "emission plus adjacent predicted tokens/roles and sequence position", "context_geometry": "context plus bbox, stroke, direction, and neighbor offsets"},
        "direct": {"scope": "project-owned writer-LOO; product-path evidence", "source": direct_source, "evaluation": _evaluate_scope(direct_rows, "direct")},
        "crohme": {"scope": "CC BY-NC research-only; five formula-disjoint folds; never a product model", "source": crohme_source, "evaluation": _evaluate_scope(crohme_rows, "crohme")},
        "limitations": [
            "The direct set has three writers and no O, o, or vertical-bar truth labels.",
            "This is an exploratory dataset already inspected during model development, not an untouched final acceptance set.",
            "Truth grouping is supplied; grouping and formula decoding are not evaluated here.",
        ],
        "product_adopted": False,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"event": "homograph_context_complete", "output": str(report_path), "product_adopted": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
