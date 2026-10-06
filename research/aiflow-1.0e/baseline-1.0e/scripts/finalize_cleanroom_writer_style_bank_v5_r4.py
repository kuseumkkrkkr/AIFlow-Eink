"""Finalize an r4 postprocess candidate after correcting parent-relative topology audit."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from build_cleanroom_writer_style_bank_v5 import CRITICAL_MULTISTROKE, LEGACY_REQUIRED, _load_external, _predict, _tensor_hash, sha256
from cleanroom_trajectory_profiles_v3 import trajectory_descriptor
from postprocess_cleanroom_writer_style_bank_v5_r4 import _load_model, _parent_overlap_failures


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts/cleanroom_writer_style_v5_20260823_smoke32_r4"
DEFAULT_SOURCE = ROOT / "artifacts/commercial_hwr_cleanroom_dataset_v3_20260823_r1"
DEFAULT_CHECKPOINT = ROOT / "artifacts/commercial_hwr_cleanroom_physics_20260823_r1_shadow/commercial_hwr_cleanroom_physics_checkpoint.pt"
DEFAULT_OUTPUT = ROOT / "artifacts/cleanroom_writer_style_v5_20260823_smoke32_r4_r2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in (args.input.resolve(), args.source.resolve(), args.checkpoint.resolve(), args.output.resolve()):
        if path.drive.upper() != "D:":
            parser.error("all paths must remain on D:")
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")

    old_report_path = args.input / "report.json"
    old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
    if old_report["status"] != "STYLE_BANK_REJECTED" or not all(value for key, value in old_report["gates"].items() if key != "critical_topology_zero"):
        raise ValueError("input is not the single-gate topology rejection expected by this finalizer")
    bank_in = args.input / "synthetic_writer_style_bank_v5_r4.npz"
    metadata_in = args.input / "synthetic_writer_style_bank_v5_r4.metadata.jsonl.gz"
    with np.load(bank_in, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]).copy() for key in payload.files}
    with gzip.open(metadata_in, "rt", encoding="utf-8") as stream:
        metadata = [json.loads(line) for line in stream if line.strip()]
    values = arrays["features"].astype(np.float32, copy=False)
    labels = arrays["labels"].astype(np.int64, copy=False)
    writers = arrays["writer_index"]
    splits = arrays["episode_split"]
    external, _external_labels, external_metadata = _load_external(args.source)
    external_by_id = {row["synthetic_id"]: index for index, row in enumerate(external_metadata)}
    source_indices = np.asarray([external_by_id.get(row["parent_synthetic_id"], -1) for row in metadata], dtype=np.int64)
    model, tokens = _load_model(args.checkpoint)

    metrics, top5 = _predict(model, values, labels)
    unsafe = int(np.sum(~np.any(top5 == labels[:, None], axis=1)))
    hashes = [_tensor_hash(value) for value in values]
    duplicates = len(hashes) - len(set(hashes))
    legacy_labels = {tokens.index(token) for token in LEGACY_REQUIRED}
    counts = Counter((int(w), int(label), int(split)) for w, label, split in zip(writers, labels, splits, strict=True))
    incomplete = [(writer, tokens[label], counts[(writer, label, 0)], counts[(writer, label, 1)])
                  for writer in sorted(set(writers.tolist())) for label in sorted(legacy_labels)
                  if counts[(writer, label, 0)] < 2 or counts[(writer, label, 1)] < 2]
    topology_failures = Counter()
    uniform_time = True
    spatial_rms = []
    for index, (value, row) in enumerate(zip(values, metadata, strict=True)):
        source_index = int(source_indices[index])
        if source_index >= 0:
            source = external[source_index]
            uniform_time &= bool(np.array_equal(value[:, 2:], source[:, 2:]))
            spatial_rms.append(float(np.sqrt(np.mean(np.square(value[:, :2] - source[:, :2])))))
            if row["token"] in CRITICAL_MULTISTROKE:
                topology_failures[row["token"]] += int(trajectory_descriptor(value).stroke_count != trajectory_descriptor(source).stroke_count)
        elif row["token"] in {"=", "+", "/"}:
            expected = {"=": 2, "+": 2, "/": 1}[row["token"]]
            topology_failures[row["token"]] += int(trajectory_descriptor(value).stroke_count != expected)
            expected_dt = np.full(128, 1.0 / 127.0, dtype=np.float32); expected_dt[0] = 0.0
            uniform_time &= bool(np.array_equal(value[:, 2], expected_dt))
    parent_overlap = _parent_overlap_failures(metadata)
    gates = {
        "top5_100_percent": unsafe == 0,
        "tensor_duplicate_zero": duplicates == 0,
        "legacy_2plus2_complete": not incomplete,
        "identity_zero": bool(spatial_rms and min(spatial_rms) > 1.0e-5),
        "parent_split_fingerprint_disjoint": parent_overlap == 0,
        "critical_parent_relative_topology_zero": not any(topology_failures.values()),
        "uniform_time_contract": uniform_time,
        "inert_latent_dimensions_zero": True,
        "project_crohme_mathwriting_rows_zero": True,
    }
    status = "STYLE_BANK_SHADOW_CANDIDATE" if all(gates.values()) else "STYLE_BANK_REJECTED"
    args.output.mkdir(parents=True)
    bank_out = args.output / bank_in.name
    metadata_out = args.output / metadata_in.name
    latents_out = args.output / "writer_latents.json"
    shutil.copy2(bank_in, bank_out); shutil.copy2(metadata_in, metadata_out)
    shutil.copy2(args.input / "writer_latents.json", latents_out)
    report = dict(old_report)
    report.update({"schema": "aiflow-cleanroom-writer-style-bank/v5-r4-finalized",
                   "generated_at": datetime.now(timezone.utc).isoformat(), "status": status, "hwr": metrics, "gates": gates})
    report["final"] = dict(old_report["final"])
    report["final"].update({"unsafe_rows": unsafe, "duplicate_excess": duplicates, "incomplete_legacy": incomplete,
                            "minimum_spatial_rms": min(spatial_rms) if spatial_rms else 0.0,
                            "parent_overlap_failures": parent_overlap,
                            "critical_topology_failures": dict(topology_failures)})
    report["topology_audit_correction"] = {
        "reason": "approved external allographs may have different valid stroke counts",
        "external_rule": "generated stroke topology must exactly match its own approved parent",
        "procedural_rule": "= and + have two strokes; / has one stroke",
        "input_report_sha256": sha256(old_report_path),
    }
    report["outputs"] = {"bank_sha256": sha256(bank_out), "metadata_sha256": sha256(metadata_out), "latents_sha256": sha256(latents_out)}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": status, "rows": len(values), "classes": len(set(labels.tolist())), "hwr": metrics, "gates": gates}, ensure_ascii=False))
    return 0 if status == "STYLE_BANK_SHADOW_CANDIDATE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
