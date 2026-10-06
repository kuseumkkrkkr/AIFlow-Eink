#!/usr/bin/env python3
"""Finalize formula characters inside immutable HWR Top-k candidates.

The runtime never receives truth labels and never creates a token, deletes a
glyph, or changes stroke ownership. Load ``OwnedFormulaContextFinalizer`` once
per process, then call ``finalize`` for each candidate batch. The selected r6
context model is followed by conservative semantic guards. Optional formula
sequence and syntax-rescue layers repair only admitted simple numeric syntax
inside Top-5; neither evaluates arithmetic. Arithmetic equation correction is
research-only and must be enabled explicitly because HWR must preserve a
user's wrong answer.

Research training keeps its D-drive storage boundary. This inference entrypoint
accepts ordinary resolved paths so the same frozen checkpoints can run in a
Linux container or behind a Windows UNC path.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path

import torch

from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import DEFAULT_PRODUCT, _sha256
from context_recheck_guard_v1 import apply_context_recheck_guard
from formula_layout_v1 import recontextualize_formula_rows
from formula_script_network_v1 import NeuralScriptPredictor
from formula_sequence_guard_v1 import (
    apply_formula_sequence_guard, validate_configuration,
)
from formula_syntax_rescue_v1 import (
    apply_formula_syntax_rescue,
    validate_configuration as validate_syntax_rescue_configuration,
)
from semantic_equation_guard_v2 import apply_semantic_equation_guard_v2
from semantic_fence_guard_v1 import apply_semantic_fence_guard
from semantic_infix_guard_v1 import apply_semantic_infix_guard
from singleton_shape_rescue_v1 import (
    apply_singleton_shape_rescue,
    validate_configuration as validate_singleton_shape_configuration,
)
import train_masked_context_reranker_v1 as masked
from train_owned_formula_context_v1 import (
    _components,
    decide_owned_formula_rows_supported_exact,
    load_owned_formula_context,
)


DEFAULT_CONTEXT = (
    Path(__file__).resolve().parents[1]
    / "artifacts" / "owned_formula_context_20260822_r6"
    / "owned_formula_context_product.pt"
)
REQUIRED = {
    "record_id", "formula_id", "final_topk", "final_topk_probabilities",
    "geometry",
}
SEQUENCE_CONFIG_SCHEMA = "aiflow-formula-sequence-guard-runtime-config/v1"
SYNTAX_RESCUE_CONFIG_SCHEMAS = frozenset({
    "aiflow-formula-syntax-rescue-runtime-config/v1",
    "aiflow-formula-syntax-rescue-runtime-config/v2",
})
SINGLETON_SHAPE_CONFIG_SCHEMA = "aiflow-singleton-shape-rescue-runtime-config/v1"


def _runtime_rows(
    rows: list[dict], labels: set[str], *, require_context: bool = True,
) -> list[dict]:
    if not rows:
        raise ValueError("context finalizer requires at least one candidate row")
    clean = []
    record_ids = set()
    for source in rows:
        missing = (REQUIRED | ({"context"} if require_context else set())) - set(source)
        if missing:
            raise ValueError(f"candidate row missing fields: {sorted(missing)}")
        row = {key: value for key, value in source.items() if key != "label"}
        record_id = str(row["record_id"])
        if not record_id or record_id in record_ids:
            raise ValueError(f"candidate record_id must be unique: {record_id!r}")
        record_ids.add(record_id)
        candidates = [str(value) for value in row["final_topk"]]
        probabilities = [float(value) for value in row["final_topk_probabilities"]]
        if not 1 <= len(candidates) <= 5 or len(candidates) != len(set(candidates)):
            raise ValueError(f"candidate Top-k is invalid: {record_id}")
        if any(value not in labels for value in candidates):
            raise ValueError(f"candidate token outside frozen HWR vocabulary: {record_id}")
        if len(probabilities) != len(candidates) or any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in probabilities
        ):
            raise ValueError(f"candidate probabilities are invalid: {record_id}")
        if any(left < right for left, right in zip(probabilities, probabilities[1:])):
            raise ValueError(f"candidate probabilities must be descending: {record_id}")
        row["record_id"] = record_id
        row["formula_id"] = str(row["formula_id"])
        if not row["formula_id"]:
            raise ValueError(f"candidate formula_id must be non-empty: {record_id}")
        row["final_topk"] = candidates
        row["final_topk_probabilities"] = probabilities
        clean.append(row)
    if require_context:
        masked._formulae(clean)
    return clean


def _decision_metadata(
    row: dict, hwr: str, context_token: str, equation_token: str,
    fence_token: str, first_final_token: str, pre_sequence_token: str,
    pre_syntax_token: str, pre_singleton_token: str, final_token: str,
    supported_exact: bool = False,
) -> tuple[bool, str, str]:
    context_available = int(row["context"]["length"]) > 1
    if final_token != pre_singleton_token:
        source = "singleton_shape_rescue_v1"
    elif pre_singleton_token != pre_syntax_token:
        source = "formula_syntax_rescue_v1"
    elif pre_syntax_token != pre_sequence_token:
        source = "formula_sequence_guard_v1"
    elif not context_available:
        source = "context_prior_only" if final_token != hwr else "hwr_top1_no_context"
    elif pre_sequence_token != first_final_token:
        source = "context_recheck_guard_v1"
    elif first_final_token != fence_token:
        source = "semantic_infix_guard"
    elif fence_token != equation_token:
        source = "semantic_fence_guard"
    elif equation_token != context_token:
        source = "semantic_equation_guard_v2"
    elif supported_exact:
        source = "owned_supported_exact_context_guard"
    elif context_token != hwr:
        source = "owned_formula_context"
    else:
        source = "owned_context_retained_hwr"
    status = (
        "ambiguous_no_formula_context"
        if not context_available and len(row["final_topk"]) > 1
        else "finalized"
    )
    return context_available, status, source


def _apply_equation_correction(
    rows: list[dict], predictions: dict[str, str], probability_ratio_floor: float,
    enabled: bool,
) -> tuple[dict[str, str], dict]:
    if not enabled:
        return dict(predictions), {
            "enabled": False,
            "policy": "preserve user-authored arithmetic; do not solve equations in HWR",
        }
    output, audit = apply_semantic_equation_guard_v2(
        rows, predictions, probability_ratio_floor
    )
    return output, {"enabled": True, **audit}


class OwnedFormulaContextFinalizer:
    def __init__(
        self, checkpoint: Path, hwr_checkpoint: Path = DEFAULT_PRODUCT,
        *, device: str = "auto", batch_size: int = 128,
        semantic_guards: bool = True, equation_correction: bool = False,
        formula_layout: bool = False,
        script_layout_checkpoint: Path | None = None,
        formula_sequence_config: Path | None = None,
        formula_syntax_rescue_config: Path | None = None,
        singleton_shape_config: Path | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        if equation_correction and not semantic_guards:
            raise ValueError("equation correction requires semantic guards")
        if script_layout_checkpoint is not None and not formula_layout:
            raise ValueError("script layout checkpoint requires formula layout")
        if formula_sequence_config is not None and not formula_layout:
            raise ValueError("formula sequence guard requires formula layout")
        if formula_syntax_rescue_config is not None and not formula_layout:
            raise ValueError("formula syntax rescue requires formula layout")
        resolved_device = (
            "cuda" if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        if resolved_device not in {"cpu", "cuda"}:
            raise ValueError(f"unsupported device: {resolved_device}")
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        checkpoint = Path(checkpoint).expanduser().resolve()
        hwr_checkpoint = Path(hwr_checkpoint).expanduser().resolve()
        self.device = torch.device(resolved_device)
        self.batch_size = batch_size
        self.semantic_guards = semantic_guards
        self.equation_correction = equation_correction
        self.formula_layout = formula_layout
        self.script_layout_predictor = (
            NeuralScriptPredictor(script_layout_checkpoint)
            if script_layout_checkpoint is not None else None
        )
        self.checkpoint_sha256 = _sha256(checkpoint)
        self.hwr_checkpoint_sha256 = _sha256(hwr_checkpoint)
        self.model, self.contract, self.payload = load_owned_formula_context(
            checkpoint, hwr_checkpoint, self.device, require_d_drive=False
        )
        self.labels = set(self.contract["label_to_index"])
        self.sequence_guard_configuration = None
        self.sequence_guard_config_sha256 = None
        if formula_sequence_config is not None:
            sequence_path = Path(formula_sequence_config).expanduser().resolve()
            if not sequence_path.is_file():
                raise FileNotFoundError(f"formula sequence config is missing: {sequence_path}")
            sequence_payload = json.loads(sequence_path.read_text(encoding="utf-8"))
            if (
                sequence_payload.get("schema") != SEQUENCE_CONFIG_SCHEMA
                or sequence_payload.get("checkpoint_sha256") != self.checkpoint_sha256
                or sequence_payload.get("hwr_checkpoint_sha256") != self.hwr_checkpoint_sha256
                or sequence_payload.get("gate", {}).get("runtime_admitted") is not True
            ):
                raise ValueError("formula sequence config contract mismatch")
            self.sequence_guard_configuration = validate_configuration(
                sequence_payload["configuration"]
            )
            self.sequence_guard_config_sha256 = _sha256(sequence_path)
        self.syntax_rescue_configuration = None
        self.syntax_rescue_config_sha256 = None
        if formula_syntax_rescue_config is not None:
            rescue_path = Path(formula_syntax_rescue_config).expanduser().resolve()
            if not rescue_path.is_file():
                raise FileNotFoundError(f"formula syntax rescue config is missing: {rescue_path}")
            rescue_payload = json.loads(rescue_path.read_text(encoding="utf-8"))
            if (
                rescue_payload.get("schema") not in SYNTAX_RESCUE_CONFIG_SCHEMAS
                or rescue_payload.get("checkpoint_sha256") != self.checkpoint_sha256
                or rescue_payload.get("hwr_checkpoint_sha256") != self.hwr_checkpoint_sha256
                or rescue_payload.get("gate", {}).get("runtime_admitted") is not True
            ):
                raise ValueError("formula syntax rescue config contract mismatch")
            self.syntax_rescue_configuration = validate_syntax_rescue_configuration(
                rescue_payload["configuration"]
            )
            self.syntax_rescue_config_sha256 = _sha256(rescue_path)
        self.singleton_shape_configuration = None
        self.singleton_shape_config_sha256 = None
        self.singleton_shape_evidence_sha256 = None
        if singleton_shape_config is not None:
            singleton_path = Path(singleton_shape_config).expanduser().resolve()
            if not singleton_path.is_file():
                raise FileNotFoundError(
                    f"singleton shape rescue config is missing: {singleton_path}"
                )
            singleton_payload = json.loads(singleton_path.read_text(encoding="utf-8"))
            if (
                singleton_payload.get("schema") != SINGLETON_SHAPE_CONFIG_SCHEMA
                or singleton_payload.get("checkpoint_sha256") != self.checkpoint_sha256
                or singleton_payload.get("hwr_checkpoint_sha256") != self.hwr_checkpoint_sha256
                or singleton_payload.get("gate", {}).get("runtime_admitted") is not True
            ):
                raise ValueError("singleton shape rescue config contract mismatch")
            self.singleton_shape_configuration = validate_singleton_shape_configuration(
                singleton_payload["configuration"]
            )
            self.singleton_shape_config_sha256 = _sha256(singleton_path)
            evidence_sha256 = str(singleton_payload.get("direct_auxiliary_sha256", ""))
            if len(evidence_sha256) != 64 or any(
                value not in "0123456789abcdef" for value in evidence_sha256
            ):
                raise ValueError("singleton shape rescue evidence hash is invalid")
            self.singleton_shape_evidence_sha256 = evidence_sha256

    def finalize(
        self, rows: list[dict], singleton_shape_evidence: list[dict] | None = None,
        singleton_shape_evidence_sha256: str | None = None,
    ) -> tuple[list[dict], dict]:
        if (self.singleton_shape_configuration is None) != (
            singleton_shape_evidence is None
        ):
            raise ValueError(
                "singleton shape rescue config and evidence must be supplied together"
            )
        if self.singleton_shape_configuration is not None and (
            singleton_shape_evidence_sha256 != self.singleton_shape_evidence_sha256
        ):
            raise ValueError("singleton shape rescue evidence hash mismatch")
        runtime_rows = _runtime_rows(
            rows, self.labels, require_context=not self.formula_layout,
        )
        if self.formula_layout:
            runtime_rows, layout_audit = recontextualize_formula_rows(
                runtime_rows, script_predictor=self.script_layout_predictor,
            )
            if self.script_layout_predictor is not None:
                layout_audit["script_checkpoint_sha256"] = _sha256(
                    self.script_layout_predictor.checkpoint
                )
            masked._formulae(runtime_rows)
        else:
            layout_audit = {"enabled": False, "policy": "use caller-supplied context order"}
        context_predictions, model_audit = decide_owned_formula_rows_supported_exact(
            self.model, self.contract, self.payload, runtime_rows,
            self.device, self.batch_size,
        )
        supported_exact_records = {
            str(change["record_id"])
            for change in model_audit["supported_exact_context"]["changes"]
        }
        if self.semantic_guards:
            probability_ratio_floor = float(
                self.payload["configuration"]["candidate_probability_ratio_floor"]
            )
            equation_predictions, equation_audit = _apply_equation_correction(
                runtime_rows, context_predictions, probability_ratio_floor,
                self.equation_correction,
            )
            fence_predictions, fence_audit = apply_semantic_fence_guard(
                runtime_rows, equation_predictions
            )
            predictions, infix_audit = apply_semantic_infix_guard(
                runtime_rows, fence_predictions,
                probability_ratio_floor,
            )
            first_predictions = predictions
            recheck_context, recheck_model_audit = (
                decide_owned_formula_rows_supported_exact(
                    self.model, self.contract, self.payload, runtime_rows,
                    self.device, self.batch_size, first_predictions,
                )
            )
            recheck_equation, recheck_equation_audit = (
                _apply_equation_correction(
                    runtime_rows, recheck_context, probability_ratio_floor,
                    self.equation_correction,
                )
            )
            recheck_fence, recheck_fence_audit = apply_semantic_fence_guard(
                runtime_rows, recheck_equation
            )
            recheck_predictions, recheck_infix_audit = apply_semantic_infix_guard(
                runtime_rows, recheck_fence, probability_ratio_floor
            )
            predictions, recheck_audit = apply_context_recheck_guard(
                runtime_rows, first_predictions, recheck_predictions
            )
            semantic_audit = {
                "enabled": True,
                "equation": equation_audit,
                "fence": fence_audit,
                "infix": infix_audit,
                "recheck": {
                    "model": recheck_model_audit,
                    "equation": recheck_equation_audit,
                    "fence": recheck_fence_audit,
                    "infix": recheck_infix_audit,
                    "guard": recheck_audit,
                },
            }
        else:
            equation_predictions = context_predictions
            fence_predictions = context_predictions
            predictions = context_predictions
            first_predictions = predictions
            semantic_audit = {"enabled": False}
        pre_sequence_predictions = predictions
        sequence_components = None
        if self.sequence_guard_configuration is not None:
            sequence_components = _components(
                self.model, self.contract, runtime_rows,
                self.payload["role_grammar"], self.device, self.batch_size,
                pre_sequence_predictions,
            )
            predictions, sequence_audit = apply_formula_sequence_guard(
                runtime_rows, pre_sequence_predictions, sequence_components,
                self.payload["role_grammar"], self.sequence_guard_configuration,
            )
            sequence_audit["config_sha256"] = self.sequence_guard_config_sha256
        else:
            sequence_audit = {
                "enabled": False,
                "policy": "formula sequence correction requires an admitted config",
            }
        pre_syntax_predictions = predictions
        if self.syntax_rescue_configuration is not None:
            syntax_components = (
                sequence_components
                if sequence_components is not None
                and pre_syntax_predictions == pre_sequence_predictions
                else _components(
                    self.model, self.contract, runtime_rows,
                    self.payload["role_grammar"], self.device, self.batch_size,
                    pre_syntax_predictions,
                )
            )
            predictions, syntax_rescue_audit = apply_formula_syntax_rescue(
                runtime_rows, pre_syntax_predictions, syntax_components,
                self.payload["role_grammar"], self.syntax_rescue_configuration,
            )
            syntax_rescue_audit["config_sha256"] = self.syntax_rescue_config_sha256
        else:
            syntax_rescue_audit = {
                "enabled": False,
                "policy": "valid numeric syntax rescue requires an admitted config",
            }
        pre_singleton_predictions = predictions
        if self.singleton_shape_configuration is not None:
            predictions, singleton_shape_audit = apply_singleton_shape_rescue(
                runtime_rows, pre_singleton_predictions,
                singleton_shape_evidence or [], self.singleton_shape_configuration,
            )
            singleton_shape_audit["config_sha256"] = self.singleton_shape_config_sha256
            singleton_shape_audit["evidence_sha256"] = singleton_shape_evidence_sha256
        else:
            singleton_shape_audit = {
                "enabled": False,
                "policy": "singleton shape rescue requires admitted auxiliary evidence",
            }
        output = []
        for row in runtime_rows:
            record_id = str(row["record_id"])
            prediction = str(predictions[record_id])
            if prediction not in row["final_topk"]:
                raise AssertionError("context finalizer invented a candidate")
            context_available, decision_status, decision_source = _decision_metadata(
                row,
                str(row["final_topk"][0]),
                str(context_predictions[record_id]),
                str(equation_predictions[record_id]),
                str(fence_predictions[record_id]),
                str(first_predictions[record_id]),
                str(pre_sequence_predictions[record_id]),
                str(pre_syntax_predictions[record_id]),
                str(pre_singleton_predictions[record_id]),
                prediction,
                record_id in supported_exact_records,
            )
            output.append({
                "schema": (
                    "aiflow-formula-context-finalized/v8"
                    if self.singleton_shape_configuration is not None
                    else (
                        "aiflow-formula-context-finalized/v7"
                        if self.syntax_rescue_configuration is not None
                        else (
                            "aiflow-formula-context-finalized/v6"
                            if self.sequence_guard_configuration is not None
                            else (
                                "aiflow-formula-context-finalized/v5"
                                if self.formula_layout else "aiflow-formula-context-finalized/v4"
                            )
                        )
                    )
                ),
                "record_id": record_id,
                "formula_id": str(row["formula_id"]),
                "context_index": int(row["context"]["index"]),
                "layout_relation_from_previous": row["context"].get(
                    "relation_from_previous"
                ),
                "context_available": context_available,
                "decision_status": decision_status,
                "decision_source": decision_source,
                "hwr_top1": str(row["final_topk"][0]),
                "finalized_top1": prediction,
                "changed": prediction != str(row["final_topk"][0]),
                "final_topk": list(row["final_topk"]),
                "final_topk_probabilities": list(row["final_topk_probabilities"]),
            })
        return output, {
            "context_checkpoint_sha256": self.checkpoint_sha256,
            "hwr_checkpoint_sha256": self.hwr_checkpoint_sha256,
            "pipeline": [
                *(["formula_layout_v1"] if self.formula_layout else []),
                "owned_formula_context_r6",
                "owned_supported_exact_context_guard_v1",
            ] + (
                ["semantic_equation_guard_v2"]
                if self.semantic_guards and self.equation_correction else []
            ) + (
                [
                    "semantic_fence_guard_v1",
                    "semantic_infix_guard_v1",
                    "context_recheck_guard_v1",
                ]
                if self.semantic_guards else []
            ) + (
                ["formula_sequence_guard_v1"]
                if self.sequence_guard_configuration is not None else []
            ) + (
                ["formula_syntax_rescue_v1"]
                if self.syntax_rescue_configuration is not None else []
            ) + (
                ["singleton_shape_rescue_v1"]
                if self.singleton_shape_configuration is not None else []
            ),
            "records": len(output),
            "formulas": len({row["formula_id"] for row in output}),
            "changed": sum(bool(row["changed"]) for row in output),
            "decision_status": dict(Counter(row["decision_status"] for row in output)),
            "decision_source": dict(Counter(row["decision_source"] for row in output)),
            "candidate_preservation_rate": 1.0,
            "new_tokens": 0,
            "deleted_glyphs": 0,
            "grouping_mutations": 0,
            "equation_correction_enabled": self.equation_correction,
            "formula_layout": layout_audit,
            "model": model_audit,
            "semantic_guards": semantic_audit,
            "formula_sequence_guard": sequence_audit,
            "formula_syntax_rescue": syntax_rescue_audit,
            "singleton_shape_rescue": singleton_shape_audit,
        }


def _write(path: Path, rows: list[dict]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _self_test() -> None:
    labels = {"1", "+"}
    rows = [
        {
            "record_id": "a", "formula_id": "f", "label": "not-runtime-input",
            "final_topk": ["1", "+"], "final_topk_probabilities": [0.8, 0.2],
            "context": {"index": 0, "length": 1},
            "geometry": {"center_x": 0.0, "center_y": 0.0, "width_rel": 1.0, "height_rel": 1.0},
        }
    ]
    clean = _runtime_rows(rows, labels)
    assert "label" not in clean[0]
    assert clean[0]["final_topk"] == ["1", "+"]
    assert _decision_metadata(
        clean[0], "1", "+", "+", "+", "+", "+", "+", "+", "+",
    ) == (
        False, "ambiguous_no_formula_context", "context_prior_only",
    )
    contextual = {**clean[0], "context": {"index": 1, "length": 3}}
    assert _decision_metadata(
        contextual, "1", "1", "1", "1", "+", "+", "+", "+", "+",
    ) == (
        True, "finalized", "semantic_infix_guard",
    )
    assert _decision_metadata(
        contextual, "h", "b", "b", "b", "b", "b", "b", "b", "b", True
    ) == (True, "finalized", "owned_supported_exact_context_guard")
    assert _decision_metadata(
        contextual, "x", r"\times", r"\times", r"\times", r"\times",
        "x", "x", "x", "x",
    ) == (True, "finalized", "context_recheck_guard_v1")
    assert _decision_metadata(
        contextual, "b", "b", "b", "b", "b", "b", "6", "6", "6"
    ) == (True, "finalized", "formula_sequence_guard_v1")
    assert _decision_metadata(
        contextual, "4", "4", "4", "4", "4", "4", "4", "+", "+"
    ) == (True, "finalized", "formula_syntax_rescue_v1")
    assert _decision_metadata(
        clean[0], "1", "1", "1", "1", "1", "1", "1", "1", "/",
    ) == (False, "ambiguous_no_formula_context", "singleton_shape_rescue_v1")
    equation_rows = [
        {
            "record_id": f"e{index}", "formula_id": "wrong-answer",
            "final_topk": candidates, "final_topk_probabilities": probabilities,
            "context": {"index": index, "length": 5},
            "geometry": {
                "center_x": float(index), "center_y": 0.5,
                "width_rel": 0.2, "height_rel": 1.0,
            },
        }
        for index, (candidates, probabilities) in enumerate([
            (["1"], [1.0]), (["+"], [1.0]), (["1"], [1.0]),
            (["="], [1.0]), (["3", "2"], [0.6, 0.4]),
        ])
    ]
    wrong_answer = {
        row["record_id"]: row["final_topk"][0] for row in equation_rows
    }
    preserved, disabled = _apply_equation_correction(
        equation_rows, wrong_answer, 0.0, False
    )
    corrected, enabled = _apply_equation_correction(
        equation_rows, wrong_answer, 0.0, True
    )
    assert preserved == wrong_answer and disabled["enabled"] is False
    assert corrected["e4"] == "2" and enabled["enabled"] is True
    try:
        _runtime_rows([{**rows[0], "final_topk": ["1", "x"]}], labels)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-vocabulary candidate was accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--disable-semantic-guards", action="store_true")
    parser.add_argument(
        "--enable-formula-layout", action="store_true",
        help="shadow: derive order and spatial relations from immutable boxes",
    )
    parser.add_argument(
        "--script-layout-checkpoint", type=Path,
        help="optional commercial-safe neural script-relation challenger",
    )
    parser.add_argument(
        "--formula-sequence-config", type=Path,
        help="shadow: admitted candidate-preserving numeric syntax correction",
    )
    parser.add_argument(
        "--formula-syntax-rescue-config", type=Path,
        help="shadow: restore one strongly implied structural token",
    )
    parser.add_argument(
        "--singleton-shape-config", type=Path,
        help="shadow: admit auxiliary HWR evidence for context-free singleton glyphs",
    )
    parser.add_argument(
        "--singleton-shape-evidence", type=Path,
        help="auxiliary HWR candidate rows paired with --singleton-shape-config",
    )
    parser.add_argument(
        "--enable-equation-correction", action="store_true",
        help="research-only: allow exact arithmetic to override HWR candidates",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.enable_equation_correction and args.disable_semantic_guards:
        parser.error("--enable-equation-correction conflicts with --disable-semantic-guards")
    if args.script_layout_checkpoint and not args.enable_formula_layout:
        parser.error("--script-layout-checkpoint requires --enable-formula-layout")
    if args.formula_sequence_config and not args.enable_formula_layout:
        parser.error("--formula-sequence-config requires --enable-formula-layout")
    if args.formula_syntax_rescue_config and not args.enable_formula_layout:
        parser.error("--formula-syntax-rescue-config requires --enable-formula-layout")
    if bool(args.singleton_shape_config) != bool(args.singleton_shape_evidence):
        parser.error(
            "--singleton-shape-config and --singleton-shape-evidence are required together"
        )
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass"}))
        return 0
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --self-test is used")
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"candidate input is missing: {input_path}")
    if output_path.exists():
        parser.error(f"refusing to overwrite finalized output: {output_path}")
    finalizer = OwnedFormulaContextFinalizer(
        args.checkpoint, args.hwr_checkpoint,
        device=args.device, batch_size=args.batch_size,
        semantic_guards=not args.disable_semantic_guards,
        equation_correction=args.enable_equation_correction,
        formula_layout=args.enable_formula_layout,
        script_layout_checkpoint=args.script_layout_checkpoint,
        formula_sequence_config=args.formula_sequence_config,
        formula_syntax_rescue_config=args.formula_syntax_rescue_config,
        singleton_shape_config=args.singleton_shape_config,
    )
    singleton_shape_evidence_path = (
        Path(args.singleton_shape_evidence).expanduser().resolve()
        if args.singleton_shape_evidence is not None else None
    )
    if singleton_shape_evidence_path is not None and not singleton_shape_evidence_path.is_file():
        parser.error(f"singleton shape evidence is missing: {singleton_shape_evidence_path}")
    singleton_shape_evidence = (
        list(_json_lines(singleton_shape_evidence_path))
        if singleton_shape_evidence_path is not None else None
    )
    finalized, audit = finalizer.finalize(
        list(_json_lines(input_path)), singleton_shape_evidence,
        _sha256(singleton_shape_evidence_path)
        if singleton_shape_evidence_path is not None else None,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write(output_path, finalized)
    print(json.dumps({
        "output": str(output_path), "sha256": _sha256(output_path), "audit": audit,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
