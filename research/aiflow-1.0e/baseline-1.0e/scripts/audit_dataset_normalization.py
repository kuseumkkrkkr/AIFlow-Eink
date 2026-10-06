from __future__ import annotations

import gzip
import io
import json
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "normalization_audit_20260811"


def read_jsonl(path: Path, limit: int = 12):
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def strokes(row):
    result = []
    for stroke in row["strokes"]:
        pts = stroke["points"]
        if pts and isinstance(pts[0], dict):
            result.append(np.array([[p["x"], p["y"]] for p in pts], float))
        else:
            result.append(np.array([[p[0], p[1]] for p in pts], float))
    return result


def unit_box(ss):
    allp = np.concatenate(ss)
    lo, hi = allp.min(0), allp.max(0)
    span = np.maximum(hi - lo, 1e-9)
    return [(p - lo) / span for p in ss]


def draw(ax, ss, title, unit=False):
    shown = unit_box(ss) if unit else ss
    for i, p in enumerate(shown):
        ax.plot(p[:, 0], p[:, 1], lw=1.2, marker="." if len(p) < 5 else None, ms=2,
                label=f"stroke {i}" if i < 3 else None)
        ax.scatter(p[0, 0], p[0, 1], s=11, marker="o")
    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()
    ax.tick_params(labelsize=6)
    ax.grid(alpha=.18)


def gallery(name, samples, note):
    fig, axes = plt.subplots(4, 6, figsize=(15, 9), constrained_layout=True)
    for col, (label, ss) in enumerate(samples[:6]):
        draw(axes[0, col], ss, f"native | {label}")
        draw(axes[1, col], ss, "same sample | unit bbox", unit=True)
    for col, (label, ss) in enumerate(samples[6:12]):
        draw(axes[2, col], ss, f"native | {label}")
        draw(axes[3, col], ss, "same sample | unit bbox", unit=True)
    for ax in axes.flat:
        if not ax.has_data(): ax.axis("off")
    fig.suptitle(f"{name}\n{note}", fontsize=12)
    path = OUT / f"{name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    owned = read_jsonl(ROOT / "hf-dataset/data/formulas_valid.jsonl")
    made.append(gallery("01 Project owned", [(r["target_display"], strokes(r)) for r in owned],
                        "stored canvas coordinates; t_ms starts at zero per formula"))
    uji = read_jsonl(ROOT / "datasets/10_approved_external/uji_pen_characters_v2/derived/uji_math_curated.jsonl.gz")
    made.append(gallery("02 UJI derived", [(r["label"], strokes(r)) for r in uji],
                        "derived coordinates; timestamps unavailable"))
    isgl = read_jsonl(ROOT / "datasets/10_approved_external/isgl_online_offline_hwr/migrated/isgl_online.jsonl.gz")
    made.append(gallery("03 ISGL migrated", [(r["label"], strokes(r)) for r in isgl],
                        "1366 x 768 canvas coordinates retained; timestamps unavailable"))
    hwrt = read_jsonl(ROOT / "datasets/10_approved_external/hwrt/derived/validation.jsonl.gz")
    made.append(gallery("04 HWRT curated", [(r["label"], strokes(r)) for r in hwrt],
                        "x/y retained; timestamps shifted to sample-relative zero"))

    zpath = ROOT / "datasets/10_approved_external/uci_character_trajectories/raw/character_trajectories.zip"
    with zipfile.ZipFile(zpath) as z:
        mat = loadmat(io.BytesIO(z.read("mixoutALL_shifted.mat")), squeeze_me=True, struct_as_record=False)
    uci = []
    for i, a in enumerate(mat["mixout"][:12]):
        uci.append((f"sample {i}", [np.asarray(a[:2]).T]))
    made.append(gallery("05 UCI trajectories", uci,
                        "publisher-provided differentiated, smoothed, normalized trajectories; no raw pair"))

    # BDSHWA: inspect aligned CSV directly from the nested archive.
    nested = Path(r"D:\AIFlow-Workspace\Temp\aiflow-normalization-audit-20260811\bdshwa_nested.zip")
    bd = []
    if nested.exists():
        import csv
        with zipfile.ZipFile(nested) as z:
            names = [n for n in z.namelist() if "Preproccessed_Data/aligned_csv/" in n and n.endswith(".csv")][:12]
            for n in names:
                rows = list(csv.DictReader(io.StringIO(z.read(n).decode("utf-8-sig"))))
                by_stroke = {}
                for r in rows:
                    if r.get("pen_down") not in ("1", "1.0", "True", "true"): continue
                    by_stroke.setdefault(r.get("stroke", "0"), []).append([float(r["canvas_x"]), float(r["canvas_y"])])
                ss = [np.asarray(v) for v in by_stroke.values() if v]
                if ss: bd.append((Path(n).stem[-24:], ss))
    made.append(gallery("06 BDSHWA aligned", bd,
                        "hardware raw_x/raw_y mapped to canvas_x/canvas_y; not unit-bbox normalized"))

    summary = {
        "generated": [p.name for p in made],
        "verdicts": {
            "project_owned": "PARTIAL: formula-relative time; spatial canvas coordinates remain",
            "UJI": "SPATIAL_DERIVED: transformed coordinates and baseline context; no time",
            "ISGL": "SCHEMA_ONLY: source canvas coordinates retained; no point time",
            "UCI": "PRENORMALIZED_UNVERIFIABLE: publisher-transformed only, no raw pair",
            "HWRT": "TEMPORAL_ONLY: relative time; x/y retained",
            "BDSHWA": "DEVICE_TO_CANVAS_ONLY: affine canvas mapping, not glyph normalization",
            "CROHME": "EVALUATION_ONLY: raw InkML at rest; median-height normalization in evaluator",
            "HF_reaudit": "UNVERIFIED: parquet reader/license provenance absent from approved pipeline",
        },
    }
    (OUT / "normalization_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
