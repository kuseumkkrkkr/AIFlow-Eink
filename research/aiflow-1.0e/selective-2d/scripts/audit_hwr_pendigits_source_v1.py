"""Quarantine/audit UCI PenDigits ORIGINAL train strokes; never open test payloads.

The 8-point CSV is used only to cross-check labels, not to synthesize pen paths.
Both Y-axis interpretations are rendered for source QA before admission.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np

from build_normalized_ink_v1 import SourceSample, _canonicalize
from character_tensor_v1 import tensorize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = Path(r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived\hwr-source-audit-20261005\uci-pendigits\uci-pendigits.zip")
EXPECTED_ARCHIVE_SHA256 = "1e02bea023613c2b11c9492f6f34caf975420455934f3527d270cee9a1f03b64"
DEFAULT_OUTPUT = ROOT / "artifacts/hwr_pendigits_source_audit_20261005"
DEFAULT_GZIP = Path(r"C:\Program Files\Git\usr\bin\gzip.exe")
SOURCE_URL = "https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits"
SEGMENT = re.compile(r'^\.SEGMENT\s+DIGIT\s+(\d+)(?:-(\d+))?\s+\?\s+"([0-9])"\s*$')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_train_unipen(text: str) -> list[dict]:
    records, current, active = [], None, False
    stroke_index = 0

    def finish():
        if current is None:
            return
        if active or current.get("writer_group") is None or not current["strokes"] or any(not stroke for stroke in current["strokes"]):
            raise ValueError("incomplete digit, stroke, or comment metadata")
        if current["stroke_ids"] != list(range(current["begin"], current["end"] + 1)):
            raise ValueError("digit segmentation is not an exact ordered stroke cover")
        records.append(current)

    for number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(".SEGMENT"):
            finish()
            match = SEGMENT.fullmatch(line)
            if match is None:
                raise ValueError(f"unsupported segment at line {number}")
            begin, end = int(match[1]), int(match[2] or match[1])
            if begin != stroke_index or end < begin:
                raise ValueError("source stroke ownership is repeated, skipped, or reversed")
            current = {"label": match[3], "begin": begin, "end": end, "strokes": [], "stroke_ids": []}
        elif line.startswith(".COMMENT"):
            parts = line.split()
            if current is None or len(parts) != 4 or parts[1] != current["label"] or "writer_group" in current:
                raise ValueError("digit/comment label mismatch or repeated comment")
            current["writer_group"], current["source_instance"] = int(parts[2]), int(parts[3])
        elif line == ".PEN_DOWN":
            if current is None or active:
                raise ValueError("nested or unowned pen-down")
            active = True
            current["strokes"].append([])
            current["stroke_ids"].append(stroke_index)
            stroke_index += 1
        elif line == ".PEN_UP":
            if not active:
                raise ValueError("pen-up without an active stroke")
            active = False
        elif line.startswith(".DT"):
            if line.split() != [".DT", "100"]:
                raise ValueError("unexpected declared sampling interval")
        elif line.startswith((".INCLUDE", ".LEXICON", ".HIERARCHY")):
            # Metadata only. Never resolve includes, open external paths, or execute directives.
            continue
        elif line.startswith("."):
            raise ValueError(f"unsupported UNIPEN directive at line {number}")
        else:
            if not active or current is None:
                raise ValueError("coordinates outside a pen-down interval")
            fields = line.split()
            if len(fields) != 2:
                raise ValueError("expected original integer X/Y coordinates")
            current["strokes"][-1].append((int(fields[0]), int(fields[1])))
    finish()
    identifiers = [(row["writer_group"], row["source_instance"]) for row in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate source writer/instance keys")
    return records


def _tensor(row: dict, orientation: str) -> np.ndarray:
    from train_character_classifier_v1 import apply_input_mode

    sign = -1 if orientation == "y-up" else 1
    strokes = [[(float(x), float(sign * y), None) for x, y in stroke] for stroke in row["strokes"]]
    canonical = _canonicalize(SourceSample(
        "uci_pendigits", f"train:{row['writer_group']}:{row['source_instance']}", row["label"], "train",
        "quarantine_box_local_digit_pretraining_candidate", strokes,
    ))
    values = apply_input_mode(tensorize(canonical), "uniform-time")
    if values.shape != (128, 5) or not np.isfinite(values).all() or (values[:, :2] < 0.0).any() or (values[:, :2] > 1.0).any():
        raise ValueError("tensor contract failed")
    if int((values[:, 3] > 0.5).sum()) != len(row["strokes"]):
        raise ValueError("tensorization lost a pen-down boundary")
    return values


def _sheet(features_down: np.ndarray, features_up: np.ndarray, rows: list[dict], path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1060, 1240), "white")
    draw = ImageDraw.Draw(image)
    draw.text((12, 8), "UCI original TRAIN only | equal XY scale | native Y vs inverted Y | no model selection", fill="black")
    for digit in range(10):
        indices = [i for i, row in enumerate(rows) if row["label"] == str(digit)][:3]
        for column, index in enumerate(indices):
            for side, features in enumerate((features_down, features_up)):
                left, top = 12 + column * 350 + side * 170, 40 + digit * 118
                draw.text((left, top), f"{digit}: {'native' if side == 0 else 'inverted'} row{index}", fill="black")
                values = features[index]
                starts = list(np.flatnonzero(values[:, 3] > 0.5)) + [128]
                for begin, end in zip(starts, starts[1:]):
                    points = [(left + 25 + float(p[0]) * 84, top + 20 + float(p[1]) * 84) for p in values[begin:end]]
                    if len(points) > 1:
                        draw.line(points, fill="black", width=2)
                    if points:
                        x, y = points[0]
                        draw.ellipse((x-2, y-2, x+2, y+2), fill="red")
    image.save(path)


def _audit(args) -> int:
    if _sha256(args.archive) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("official download archive differs from the sealed source audit")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite source-audit artifacts")
    with zipfile.ZipFile(args.archive) as archive:
        members = [{"name": i.filename, "compressed_bytes": i.compress_size, "bytes": i.file_size} for i in archive.infolist()]
        # Do not run testzip(): that would decompress reserved test payloads too.
        compressed = archive.read("pendigits-orig.tra.Z")
        completed = subprocess.run([str(args.gzip), "-dc"], input=compressed, capture_output=True, check=True, timeout=30)
        raw = completed.stdout.decode("ascii")
        reference_labels = [line.rsplit(",", 1)[-1].strip() for line in archive.read("pendigits.tra").decode("ascii").splitlines() if line.strip()]
        documentation = archive.read("pendigits-orig.names")
    rows = parse_train_unipen(raw)
    if len(rows) != 7494 or [row["label"] for row in rows] != reference_labels:
        raise ValueError("original UNIPEN labels disagree with the official summarized train file")
    if sum(len(row["strokes"]) for row in rows) != 9433:
        raise ValueError("original train stroke count mismatch")
    from audit_hwr_probability_boundary_tube_v1 import _guard_commit
    if _guard_commit("before_pendigits_contract_tensorization") is None:
        return 78
    down = np.stack([_tensor(row, "y-down") for row in rows])
    up = down.copy()
    up[..., 1] = 1.0 - up[..., 1]
    # Independently replay the alternate normalizer on representative raw rows.
    for index in range(0, len(rows), 73):
        if not np.allclose(up[index], _tensor(rows[index], "y-up"), atol=2.0e-7):
            raise AssertionError("Y inversion did not preserve the canonical normalization contract")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"train_features_y_down": down, "train_features_y_up": up,
              "train_digit_labels": np.array([int(row["label"]) for row in rows], dtype=np.int16),
              "train_writer_group_hashes": np.array([hashlib.sha256(f"uci-pendigits:{row['writer_group']}".encode()).hexdigest()[:16] for row in rows], dtype="U16")}
    paths = {}
    for name, array in arrays.items():
        path = args.output_dir / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        paths[name] = {"path": str(path.resolve()), "sha256": _sha256(path), "shape": list(array.shape), "dtype": str(array.dtype)}
    sheet_path = args.output_dir / "train_orientation_qa.png"
    _sheet(down, up, rows, sheet_path)
    report = {
        "schema": "aiflow-hwr-pendigits-source-audit/v1", "status": "quarantined_pending_orientation_qa_and_admission",
        "source_url": SOURCE_URL, "source_license_displayed": "CC BY 4.0", "source_doi": "10.24432/C5MG6K",
        "attribution": "E. Alpaydin and Fevzi. Alimoglu, Pen-Based Recognition of Handwritten Digits, UCI Machine Learning Repository",
        "archive_sha256": _sha256(args.archive), "archive_members_inventory": members,
        "original_train_sha256": hashlib.sha256(completed.stdout).hexdigest(), "original_train_documentation_sha256": hashlib.sha256(documentation).hexdigest(),
        "script_sha256": _sha256(Path(__file__)), "train_rows": len(rows), "train_strokes": 9433,
        "train_points": sum(len(stroke) for row in rows for stroke in row["strokes"]),
        "train_label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "train_comment_writer_groups": len(set(row["writer_group"] for row in rows)),
        "writer_group_basis": "second .COMMENT field; grouping inferred from source structure, not an identity or demographic assertion",
        "raw_segment_exact_cover_passed": True, "labels_match_official_train_csv": True,
        "original_stroke_boundaries_preserved": True, "interpolated_8_point_csv_used_as_ink": False,
        "time_policy": "no per-point timestamps supplied; ordinal order only. .DT 100 is retained as metadata, not invented pen-up timing; model uniform-time contract applies",
        "artifacts": paths, "orientation_qa_image": str(sheet_path.resolve()), "orientation_qa_image_sha256": _sha256(sheet_path),
        "orientation_chosen": None, "admitted_to_training_pipeline": False,
        "test_payloads_opened": [], "test_rows_read": 0, "crohme_rows": 0,
        "student_training_performed": False, "accuracy_evaluation_performed": False, "product_adopted": False,
        "limitations": "digits only; source audit is not fresh writer/device/formula acceptance and not a human ambiguity-boundary annotation",
    }
    path = args.output_dir / "pendigits_source_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "pendigits_train_source_audit_complete", "report": str(path.resolve()), "train_rows": len(rows), "strokes": 9433, "test_rows_read": 0}), flush=True)
    return 0


def _self_test() -> int:
    fixture = '.SEGMENT DIGIT 0-1 ? "2"\n.COMMENT 2 1 1\n.PEN_DOWN\n0 0\n1 1\n.PEN_UP\n.DT 100\n.PEN_DOWN\n2 1\n3 1\n.PEN_UP\n.DT 100\n'
    rows = parse_train_unipen(fixture)
    assert len(rows) == 1 and len(rows[0]["strokes"]) == 2
    assert rows[0]["stroke_ids"] == [0, 1]
    for broken in (fixture.replace("0-1", "1-2"), fixture.replace(".COMMENT 2", ".COMMENT 3"), fixture.replace(".PEN_UP", "", 1)):
        try:
            parse_train_unipen(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed stroke ownership or metadata was admitted")
    print(json.dumps({"self_test": "pass", "exact_cover": True, "raw_label_alignment": True}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("self-test", "audit"), required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--gzip", type=Path, default=DEFAULT_GZIP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    return _self_test() if args.mode == "self-test" else _audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
