#!/usr/bin/env python3
"""Extract frozen UniMERNet-tiny encoder features for the 1.0e shadow trial."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from accuracy_upgrade_contract_v1 import canonical_json_sha256
from train_ocr_decision_adapter_10e import _load_candidates, _load_formula_records, _render_formula  # noqa: E402


DEFAULT_CANDIDATES = ROOT / "artifacts" / "ocr_trocr_hwr95_candidates_20260901_r1" / "candidates.jsonl.gz"
DEFAULT_FORMULAS = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\public-candidate-20260819-r2\data\formulas_valid.jsonl"
)
SCHEMA = "aiflow-1.0e-offline-teacher-features/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--formulas", type=Path, default=DEFAULT_FORMULAS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")

    import argparse as argument_types
    import unimernet.tasks as tasks
    from unimernet.common.config import Config
    from unimernet.processors import load_processor

    rows = _load_candidates(args.candidates, None)
    formulas = _load_formula_records(args.formulas)
    formula_ids = list(dict.fromkeys(str(row["formula_id"]) for row in rows))
    missing = [formula_id for formula_id in formula_ids if formula_id not in formulas]
    if missing:
        raise ValueError(f"missing {len(missing)} formula records")

    config = Config(argument_types.Namespace(cfg_path=str(args.config), options=None))
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    config.config.model.pretrained = str(args.weights.resolve())
    task = tasks.setup_task(config)
    model = task.build_model(config).to(torch.device("cpu")).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    processor_config = config.config.datasets.formula_rec_eval.vis_processor.eval
    processor = load_processor("formula_image_eval", processor_config)
    features: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(formula_ids), args.batch_size):
            selected = formula_ids[start:start + args.batch_size]
            tensor = torch.stack([
                processor(_render_formula(formulas[formula_id]).convert("RGB"))
                for formula_id in selected
            ])
            if tensor.shape[1] == 1:
                tensor = tensor.repeat(1, 3, 1, 1)
            hidden = model.model.model.encoder(pixel_values=tensor, return_dict=True).last_hidden_state
            features.append(hidden.mean(dim=1).float().cpu().numpy())
            done = min(start + args.batch_size, len(formula_ids))
            if done % 20 == 0 or done == len(formula_ids):
                print(json.dumps({"event": "progress", "done": done, "total": len(formula_ids)}), flush=True)
    array = np.concatenate(features, axis=0).astype(np.float32)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "features.npz", formula_ids=np.asarray(formula_ids), features=array)
    manifest = {
        "schema": SCHEMA,
        "status": "shadow_only",
        "teacher": "unimernet-tiny",
        "config": str(args.config),
        "weights": str(args.weights),
        "checkpoint_sha256": {args.weights.name: sha256(args.weights)},
        "formula_count": len(formula_ids),
        "feature_shape": list(array.shape),
        "feature_dtype": str(array.dtype),
        "source_formula_sha256": {key: canonical_json_sha256({"strokes": formulas[key]["strokes"], "canvas": formulas[key].get("canvas", {})}) for key in formula_ids},
        "feature_file_sha256": sha256(args.output / "features.npz"),
        "preprocessing_sha256": sha256(Path(__file__)),
        "raster_source": "project-owned ordered strokes rendered offline",
        "product_runtime_input": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", **manifest}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
