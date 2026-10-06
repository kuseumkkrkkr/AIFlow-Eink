#!/usr/bin/env python3
"""Verify FP32 ONNX Runtime parity for AIFlow's frozen mobile HWR export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_48hz_prefix_v1 import _load_model
from export_mobile_hwr_onnx_v1 import MobileHwrWrapperV1


SCHEMA = "aiflow-mobile-hwr-onnx-parity/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(checkpoint: Path, onnx_path: Path, *, batch: int = 3) -> dict:
    if batch < 1:
        raise ValueError("batch must be positive")
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as error:
        raise RuntimeError("parity verification requires the optional onnxruntime package") from error
    hwr, labels, _report = _load_model(checkpoint, torch.device("cpu"))
    if len(labels) != 372:
        raise ValueError("parity requires a 372-class checkpoint")
    rng = np.random.default_rng(20260914)
    points = rng.normal(size=(batch, 128, 5)).astype(np.float32)
    with torch.inference_mode():
        expected_embedding, expected_logits = MobileHwrWrapperV1(hwr)(torch.from_numpy(points))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual_embedding, actual_logits = session.run(["embedding", "logits"], {"points": points})
    embedding_error = float(np.max(np.abs(actual_embedding - expected_embedding.numpy())))
    logits_error = float(np.max(np.abs(actual_logits - expected_logits.numpy())))
    passed = embedding_error <= 1e-4 and logits_error <= 1e-4
    return {
        "schema": SCHEMA, "checkpoint_sha256": _sha256(checkpoint), "onnx_sha256": _sha256(onnx_path),
        "batch": batch, "embedding_max_abs_error": embedding_error, "logits_max_abs_error": logits_error,
        "threshold": 1e-4, "passed": passed, "crohme_training_or_tuning": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=3)
    args = parser.parse_args()
    result = verify(args.checkpoint.resolve(), args.onnx.resolve(), batch=args.batch)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
