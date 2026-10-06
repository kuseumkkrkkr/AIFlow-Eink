#!/usr/bin/env python3
"""Export the frozen 128x5 online-HWR encoder and 372-class head to ONNX.

This is FP32 parity export only.  INT8 is intentionally a separate challenger
and no checkpoint is modified by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from evaluate_48hz_prefix_v1 import _load_model


SCHEMA = "aiflow-mobile-hwr-onnx/v1"


class MobileHwrWrapperV1(nn.Module):
    def __init__(self, hwr: nn.Module) -> None:
        super().__init__()
        self.hwr = hwr

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.hwr.encode(points)
        return embedding, self.hwr.math_head(embedding)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_hwr(checkpoint: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite ONNX output: {output}")
    try:
        import onnx
    except ModuleNotFoundError as error:
        raise RuntimeError("ONNX export requires the optional 'onnx' package; no model was written") from error
    model, labels, _report = _load_model(checkpoint, torch.device("cpu"))
    if len(labels) != 372:
        raise ValueError("mobile export requires the frozen 372-class head")
    wrapper = MobileHwrWrapperV1(model).eval()
    points = torch.zeros((1, 128, 5), dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, points, output, input_names=["points"], output_names=["embedding", "logits"],
        dynamic_axes={"points": {0: "batch"}, "embedding": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    payload = onnx.load(str(output))
    onnx.checker.check_model(payload)
    return {
        "schema": SCHEMA, "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
        "output": str(output), "output_sha256": _sha256(output), "input": ["N", 128, 5],
        "outputs": {"embedding": ["N", 128], "logits": ["N", 372]}, "precision": "fp32",
        "quantization": "not_performed", "crohme_training_or_tuning": False,
    }


def _self_test() -> None:
    from train_character_classifier_v1 import InkClassifierV1

    wrapper = MobileHwrWrapperV1(InkClassifierV1(372).eval())
    embedding, logits = wrapper(torch.zeros((2, 128, 5), dtype=torch.float32))
    assert embedding.shape == (2, 128) and logits.shape == (2, 372)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass"}))
        return 0
    if args.checkpoint is None or args.output is None:
        parser.error("--checkpoint and --output are required")
    result = export_hwr(args.checkpoint.resolve(), args.output.resolve())
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
