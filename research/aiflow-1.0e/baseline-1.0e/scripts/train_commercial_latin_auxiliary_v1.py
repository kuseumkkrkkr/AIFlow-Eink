#!/usr/bin/env python3
"""Train a writer-disjoint commercial Latin online-ink auxiliary model.

Fit data: UJI train writers, deterministic ISGL fit writers, and the one-writer
UCI corpus.  Selection data: all 20 official UJI test writers plus deterministic
held ISGL writers.  The existing fixed external holdout is excluded from both
fit and selection.  CROHME and MathWriting are not accepted by this command.
"""

from __future__ import annotations

from training_data_guard_v1 import assert_training_entrypoint_arguments_clean
if __name__ == "__main__":
    assert_training_entrypoint_arguments_clean()

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from build_normalized_ink_v1 import _record_id
from character_tensor_v1 import ROOT, _json_lines, tensorize
from train_character_classifier_v1 import (
    InkClassifierV1, _class_balanced_loader, apply_input_mode, train_head,
)
from training_data_guard_v1 import (
    assert_training_path_clean, assert_training_row_clean,
    zero_crohme_training_manifest,
)


SCHEMA = "aiflow-commercial-latin-auxiliary/v1"
SEED = 20260821
EXCLUDED_LABELS = frozenset({"(", ")"})
SOURCE_NAME = {
    "uji": "uji_pen_v2",
    "isgl": "isgl_online",
    "uci": "uci_character_trajectories",
}
DEFAULT_CANONICAL = ROOT / "datasets" / "normalized" / "v1"
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "commercial_latin_auxiliary_20260821_r1_shadow"
)
DEFAULT_UJI = (
    ROOT / "datasets" / "10_approved_external" / "uji_pen_characters_v2"
    / "derived" / "uji_math_curated.jsonl.gz"
)
DEFAULT_ISGL = (
    ROOT / "datasets" / "10_approved_external" / "isgl_online_offline_hwr"
    / "migrated" / "isgl_online.jsonl.gz"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _writer_alias(source: str, writer: str) -> str:
    return hashlib.sha256(f"{source}\0{writer}".encode("utf-8")).hexdigest()[:20]


def _held_isgl_writer(writer: str) -> bool:
    digest = hashlib.sha256(f"isgl-writer-dev-v1\0{writer}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 5 == 0


def _raw_writer_map(uji_path: Path, isgl_path: Path) -> tuple[dict[str, tuple[str, str]], dict]:
    mapping: dict[str, tuple[str, str]] = {}
    uji_writers: dict[str, set[str]] = defaultdict(set)
    for row in _json_lines(uji_path):
        writer = str(row["writer_key"])
        split = str(row["split"])
        uji_writers[split].add(writer)
        key = _record_id(SOURCE_NAME["uji"], str(row["sample_id"]))
        mapping[key] = (_writer_alias("uji", writer), "dev" if split == "test" else "fit")
    if set(uji_writers) != {"train", "test"} or uji_writers["train"] & uji_writers["test"]:
        raise ValueError("UJI official train/test writer separation is invalid")

    isgl_writers: set[str] = set()
    held_isgl: set[str] = set()
    for row in _json_lines(isgl_path):
        writer = str(row["writer_key"])
        isgl_writers.add(writer)
        held = _held_isgl_writer(writer)
        if held:
            held_isgl.add(writer)
        key = _record_id(SOURCE_NAME["isgl"], str(row["sample_id"]))
        mapping[key] = (_writer_alias("isgl", writer), "dev" if held else "fit")
    if not 8 <= len(held_isgl) <= 18:
        raise ValueError(f"unexpected deterministic ISGL held-writer count: {len(held_isgl)}")
    return mapping, {
        "uji_fit_writers": len(uji_writers["train"]),
        "uji_dev_writers": len(uji_writers["test"]),
        "uji_writer_overlap": 0,
        "isgl_writers": len(isgl_writers),
        "isgl_dev_writers": len(held_isgl),
        "isgl_fit_writers": len(isgl_writers - held_isgl),
        "isgl_writer_overlap": 0,
        "isgl_split": "sha256(writer,salt) mod 5 == 0",
    }


def _labels(canonical_root: Path) -> list[str]:
    labels = sorted({
        str(row["label"])
        for row in _json_lines(canonical_root / "uji.jsonl.gz")
    } - EXCLUDED_LABELS)
    if len(labels) != 95 or not {"t", ",", ".", "!"} <= set(labels):
        raise ValueError("unexpected commercial Latin auxiliary vocabulary")
    return labels


def _load_rows(
    canonical_root: Path, uji_path: Path, isgl_path: Path, labels: list[str],
) -> tuple[dict[str, dict], dict]:
    writer_map, writer_audit = _raw_writer_map(uji_path, isgl_path)
    fixed_holdout = set(json.loads(
        (canonical_root / "character_classifier_v1" / "current_external_holdout_ids.json")
        .read_text(encoding="utf-8")
    ))
    label_to_index = {label: index for index, label in enumerate(labels)}
    output = {
        "fit": {"features": [], "labels": [], "metadata": []},
        "dev": {"features": [], "labels": [], "metadata": []},
    }
    excluded_fixed = Counter()
    for source in ("uji", "isgl", "uci"):
        path = canonical_root / f"{source}.jsonl.gz"
        for row in _json_lines(path):
            assert_training_row_clean(row)
            label = str(row["label"])
            if label not in label_to_index:
                continue
            full_record_id = f"{source}:{row['record_id']}"
            if full_record_id in fixed_holdout:
                excluded_fixed[source] += 1
                continue
            if source == "uci":
                writer, split = _writer_alias("uci", "single-writer"), "fit"
            else:
                try:
                    writer, split = writer_map[str(row["record_id"])]
                except KeyError as error:
                    raise ValueError(f"writer provenance missing: {full_record_id}") from error
            features = apply_input_mode(tensorize(row), "uniform-time")
            output[split]["features"].append(features)
            output[split]["labels"].append(label_to_index[label])
            output[split]["metadata"].append({
                "record_id": full_record_id,
                "source": source,
                "writer": writer,
                "label": label,
            })
    fit_writers = {row["writer"] for row in output["fit"]["metadata"]}
    dev_writers = {row["writer"] for row in output["dev"]["metadata"]}
    if fit_writers & dev_writers:
        raise AssertionError("Latin auxiliary fit/dev writer overlap")
    if not output["fit"]["features"] or not output["dev"]["features"]:
        raise ValueError("Latin auxiliary split is empty")
    for split in output.values():
        split["features"] = np.stack(split["features"]).astype(np.float32, copy=False)
        split["labels"] = np.asarray(split["labels"], dtype=np.int64)
    return output, {
        **writer_audit,
        "fit_writers": len(fit_writers),
        "dev_writers": len(dev_writers),
        "writer_overlap": 0,
        "fixed_external_holdout_excluded": dict(sorted(excluded_fixed.items())),
        "fixed_external_holdout_used_for_training": False,
        "fixed_external_holdout_used_for_selection": False,
    }


@torch.inference_mode()
def _predict(
    model: InkClassifierV1, features: np.ndarray, device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    top1 = []
    top5 = []
    model.eval()
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start:start + batch_size]).to(device)
        logits = model(batch, "auxiliary")
        indices = logits.topk(5, dim=1).indices.cpu().numpy()
        top1.append(indices[:, 0])
        top5.append(indices)
    return np.concatenate(top1), np.concatenate(top5)


def _metrics(
    truth: np.ndarray, top1: np.ndarray, top5: np.ndarray,
    metadata: list[dict], labels: list[str],
) -> dict:
    hits1 = top1 == truth
    hits5 = np.any(top5 == truth[:, None], axis=1)
    writers: dict[str, list[int]] = defaultdict(list)
    sources: dict[str, list[int]] = defaultdict(list)
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        writers[str(row["writer"])].append(index)
        sources[str(row["source"])].append(index)
        by_label[str(row["label"])].append(index)

    def subset(indices: list[int]) -> dict:
        values = np.asarray(indices, dtype=np.int64)
        return {
            "records": len(values),
            "top1": float(hits1[values].mean()),
            "top5": float(hits5[values].mean()),
        }

    writer_scores = [subset(indices) for indices in writers.values()]
    return {
        "records": len(truth),
        "top1": float(hits1.mean()),
        "top5": float(hits5.mean()),
        "writer_macro_top1": float(np.mean([row["top1"] for row in writer_scores])),
        "writer_macro_top5": float(np.mean([row["top5"] for row in writer_scores])),
        "worst_writer_top1": min(row["top1"] for row in writer_scores),
        "worst_writer_top5": min(row["top5"] for row in writer_scores),
        "writers": len(writers),
        "by_source": {key: subset(value) for key, value in sorted(sources.items())},
        "by_label": {key: subset(value) for key, value in sorted(by_label.items())},
        "top1_confusions": [
            {"truth": labels[int(left)], "prediction": labels[int(right)], "count": count}
            for (left, right), count in Counter(
                (int(truth[index]), int(top1[index]))
                for index in range(len(truth)) if not hits1[index]
            ).most_common(30)
        ],
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--uji", type=Path, default=DEFAULT_UJI)
    parser.add_argument("--isgl", type=Path, default=DEFAULT_ISGL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_batch_size) < 1 or args.learning_rate <= 0:
        parser.error("invalid training configuration")
    canonical_root = assert_training_path_clean(args.canonical_root, "canonical root")
    uji_path = assert_training_path_clean(args.uji, "UJI source")
    isgl_path = assert_training_path_clean(args.isgl, "ISGL source")
    output = assert_training_path_clean(args.output, "Latin auxiliary output")
    if output.drive.upper() != "D:" or output.exists():
        parser.error("output must be a new directory on D:")
    for path in (uji_path, isgl_path):
        if not path.is_file():
            parser.error(f"missing approved source: {path}")
    resolved_device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if resolved_device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(resolved_device)
    _seed_everything(SEED)
    labels = _labels(canonical_root)
    rows, split_audit = _load_rows(canonical_root, uji_path, isgl_path, labels)
    fit = TensorDataset(
        torch.from_numpy(rows["fit"]["features"]),
        torch.from_numpy(rows["fit"]["labels"]),
    )
    dev_features = rows["dev"]["features"]
    dev_truth = rows["dev"]["labels"]
    fit_loader, loss_weights = _class_balanced_loader(
        fit, rows["fit"]["labels"], args.batch_size, SEED, len(labels),
        device.type == "cuda", "sampler",
    )
    model = InkClassifierV1(1, len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    loss_weights = loss_weights.to(device)
    history = []
    best = None
    best_state = None
    started = time.perf_counter()
    output.mkdir(parents=True)
    for epoch in range(1, args.epochs + 1):
        training = train_head(
            model, optimizer, scaler, fit_loader, loss_weights, "auxiliary",
            device, epoch, max(len(fit_loader), 1),
        )
        top1, top5 = _predict(model, dev_features, device, args.eval_batch_size)
        evaluation = _metrics(
            dev_truth, top1, top5, rows["dev"]["metadata"], labels,
        )
        history.append({"epoch": epoch, "training": training, "writer_disjoint_dev": evaluation})
        key = (
            evaluation["writer_macro_top1"], evaluation["top1"],
            evaluation["writer_macro_top5"], -epoch,
        )
        if best is None or key > best[0]:
            best = (key, epoch, evaluation)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(json.dumps({
            "event": "latin_auxiliary_epoch",
            "epoch": epoch,
            "dev_top1": evaluation["top1"],
            "dev_top5": evaluation["top5"],
            "writer_macro_top1": evaluation["writer_macro_top1"],
            "selected_epoch": best[1],
        }, ensure_ascii=False), flush=True)
    if best is None or best_state is None:
        raise AssertionError("Latin auxiliary selection failed")
    model.load_state_dict(best_state)
    gradient_updates = sum(int(row["training"]["batches"]) for row in history)
    source_counts = Counter(row["source"] for row in rows["fit"]["metadata"])
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "commercial_rights_writer_disjoint_shadow",
        "architecture": {
            "input": [128, 5], "hidden": 128, "transformer_blocks": 4,
            "attention_heads": 4, "latin_auxiliary_classes": len(labels),
            "math_head_classes_unused": 1,
        },
        "input_contract": {
            "mode": "uniform-time",
            "delta_t": "normalized sequence progress",
            "observed": "constant one",
        },
        "vocabulary": labels,
        "data": {
            "fit_records": len(rows["fit"]["metadata"]),
            "dev_records": len(rows["dev"]["metadata"]),
            "fit_sources": dict(sorted(source_counts.items())),
            "selection": split_audit,
            "rights": {
                "uji": "CC BY 4.0",
                "isgl": "CC BY 4.0",
                "uci": "CC BY 4.0; one writer; fit only",
            },
        },
        "optimization": {
            "epochs": args.epochs, "selected_epoch": best[1],
            "learning_rate": args.learning_rate, "optimizer": "AdamW",
            "class_balance": "weighted sampler", "seed": SEED,
        },
        "history": history,
        "selected_writer_disjoint_dev": best[2],
        "training_data_guard": zero_crohme_training_manifest(
            admitted_sources=source_counts, gradient_updates=gradient_updates,
        ),
        "fixed_external_holdout_scored": False,
        "crohme_scored_during_training_or_selection": False,
        "product_adopted": False,
        "use_boundary": (
            "optional formula-complete Latin candidate expert; main 372-class HWR and "
            "context contracts remain unchanged until separate acceptance"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "inputs": {
            "canonical_manifest_sha256": _sha256(canonical_root / "manifest.json"),
            "uji_sha256": _sha256(uji_path),
            "isgl_sha256": _sha256(isgl_path),
        },
    }
    checkpoint = output / "latin_auxiliary_checkpoint.pt"
    torch.save({
        "schema": SCHEMA,
        "state_dict": model.state_dict(),
        "math_labels": ["<unused>"],
        "auxiliary_labels": labels,
        "report": report,
    }, checkpoint)
    (output / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "event": "latin_auxiliary_complete",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "selected_epoch": best[1],
        "dev_top1": best[2]["top1"],
        "dev_top5": best[2]["top5"],
        "crohme_gradient_updates": 0,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
