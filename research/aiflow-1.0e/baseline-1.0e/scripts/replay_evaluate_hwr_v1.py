#!/usr/bin/env python3
"""Prepare point-by-point replay evidence and score supplied HWR predictions.

This runner deliberately has no model import and no training command. A future
predictor writes JSONL predictions, then this script scores the fixed protocols.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Iterator

from character_tensor_v1 import ROOT, _digest, _json_lines, iter_current_external_metadata, iter_direct_ownership_examples, stratified_holdout


DEFAULT_CANONICAL_ROOT = ROOT / "datasets" / "normalized" / "v1"
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_replay_evaluation_v1"


def _raw_strokes(row: dict) -> list[list[list[float]]]:
    return [[[float(point["x"]), float(point["y"])] for point in stroke["points"]] for stroke in sorted(row["strokes"], key=lambda stroke: stroke["order"])]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def write_replay_html(path: Path, title: str, strokes: list[list[list[float]]], detail: str = "") -> None:
    """Write a local canvas replay whose slider advances exactly one source point."""
    safe_title = html.escape(title)
    safe_detail = html.escape(detail)
    payload = json.dumps(strokes, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<meta charset="utf-8">
<title>{safe_title}</title>
<style>body{{font:16px system-ui;margin:24px;max-width:720px}}canvas{{border:1px solid #888;width:100%;height:auto}}input{{width:100%}}button{{margin-right:8px}}</style>
<h1>{safe_title}</h1><p>{safe_detail}</p>
<canvas id="ink" width="640" height="480"></canvas>
<p><button id="play" type="button">play (48 Hz)</button><button id="reset" type="button">reset</button></p>
<input id="step" type="range" min="0" value="0"><output id="count"></output>
<script>
const strokes={payload}; const points=[];
for(let s=0;s<strokes.length;s++) for(let p=0;p<strokes[s].length;p++) points.push([s,p,strokes[s][p]]);
const canvas=document.getElementById('ink'),ctx=canvas.getContext('2d'),step=document.getElementById('step'),count=document.getElementById('count');
step.max=points.length; let timer=null;
const flat=points.map(v=>v[2]), xs=flat.map(v=>v[0]), ys=flat.map(v=>v[1]);
const loX=Math.min(...xs), hiX=Math.max(...xs), loY=Math.min(...ys), hiY=Math.max(...ys), span=Math.max(hiX-loX,hiY-loY,1), pad=32;
function map(p){{return [pad+(p[0]-loX)/span*(canvas.width-2*pad),pad+(p[1]-loY)/span*(canvas.height-2*pad)]}}
function draw(){{const n=Number(step.value);ctx.clearRect(0,0,canvas.width,canvas.height);ctx.lineWidth=2;ctx.strokeStyle='#111';ctx.lineCap='round';for(let i=0;i<n;i++){{const [s,p,point]=points[i],xy=map(point);if(p===0){{ctx.beginPath();ctx.moveTo(...xy)}}else{{ctx.lineTo(...xy);ctx.stroke();ctx.beginPath();ctx.moveTo(...xy)}}ctx.fillStyle='#2563eb';ctx.beginPath();ctx.arc(...xy,2.5,0,Math.PI*2);ctx.fill()}}count.value=`${{n}} / ${{points.length}} points`;}}
step.addEventListener('input',draw);document.getElementById('reset').onclick=()=>{{step.value=0;draw()}};document.getElementById('play').onclick=()=>{{if(timer){{clearInterval(timer);timer=null;return}} timer=setInterval(()=>{{if(Number(step.value)>=points.length){{clearInterval(timer);timer=null}}else{{step.value=Number(step.value)+1;draw()}}}},1000/48)}};draw();
</script>"""
    path.write_text(page, encoding="utf-8", newline="\n")


def _direct_formula_rows(formulas: Path) -> dict[str, dict]:
    return {row["sample_id"]: row for row in _json_lines(formulas)}


def prepare_direct_protocol(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    formulas = ROOT / "hf-dataset" / "data" / "formulas_valid.jsonl"
    ownership = ROOT / "hf-dataset" / "data" / "ownership_train.jsonl"
    source_rows = _direct_formula_rows(formulas)
    ownership_rows = [row for row in _json_lines(ownership) if row.get("accepted")]
    replay_dir = output / "direct_ownership_replays"; replay_dir.mkdir(exist_ok=True)
    ownership_truth = list(iter_direct_ownership_examples(formulas, ownership))
    for annotation in ownership_rows:
        row = source_rows[annotation["sample_id"]]
        detail = "labels: " + " ".join(str(label) for label in annotation["labels"])
        write_replay_html(replay_dir / f"{annotation['sample_id']}.html", annotation["sample_id"], _raw_strokes(row), detail)
    selected = sorted(source_rows.values(), key=lambda row: _digest(["direct-canonical-10", row["sample_id"]]))[:10]
    canonical_dir = output / "direct_canonical_10_replays"; canonical_dir.mkdir(exist_ok=True)
    canonical_truth = []
    for row in selected:
        record_id = str(row["sample_id"])
        canonical_truth.append({"record_id": record_id, "label": str(row["target_display"])})
        write_replay_html(canonical_dir / f"{record_id}.html", record_id, _raw_strokes(row), f"target_display: {row['target_display']}")
    _write_jsonl(output / "direct_ownership_truth.jsonl", ({"record_id": row["record_id"], "label": row["label"]} for row in ownership_truth))
    _write_jsonl(output / "direct_canonical_10_truth.jsonl", canonical_truth)
    result = {"ownership_formulas": len(ownership_rows), "ownership_symbols": len(ownership_truth), "canonical_formula_replays": len(canonical_truth), "canonical_ids": [row["record_id"] for row in canonical_truth]}
    (output / "direct_protocol.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def _local_name(node: ET.Element) -> str:
    return node.tag.rsplit("}", 1)[-1]


def _trace_points(text: str | None) -> list[list[float]]:
    points: list[list[float]] = []
    for piece in (text or "").split(","):
        values = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", piece)
        if len(values) >= 2:
            points.append([float(values[0]), float(values[1])])
    return points


def load_crohme(root: Path) -> tuple[list[dict], list[str]]:
    """Load formula truth while removing exact duplicated InkML formula rows."""
    accepted: list[dict] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for path in sorted(root.rglob("*.inkml")):
        tree = ET.parse(path)
        base = tree.getroot()
        traces = [_trace_points(node.text) for node in base.iter() if _local_name(node) == "trace"]
        traces = [trace for trace in traces if trace]
        truths = [str(node.text or "").strip() for node in base.iter() if _local_name(node) == "annotation" and node.attrib.get("type") == "truth"]
        if traces and truths:
            row = {"record_id": path.relative_to(root).as_posix(), "label": truths[0], "strokes": traces}
            fingerprint = _digest([row["label"], row["strokes"]])
            if fingerprint in seen:
                duplicates.append(row["record_id"])
            else:
                seen.add(fingerprint)
                accepted.append(row)
    return accepted, duplicates


def iter_crohme(root: Path) -> Iterator[dict]:
    yield from load_crohme(root)[0]


def prepare_crohme_protocol(crohme_root: Path, output: Path, limit: int | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows, duplicates = load_crohme(crohme_root)
    rendered = rows if limit is None else rows[:limit]
    replay_dir = output / "crohme_replays"; replay_dir.mkdir(exist_ok=True)
    for row in rendered:
        safe_name = _digest(["crohme", row["record_id"]])[:24]
        write_replay_html(replay_dir / f"{safe_name}.html", row["record_id"], row["strokes"], f"truth: {row['label']}")
    _write_jsonl(output / "crohme_truth.jsonl", ({"record_id": row["record_id"], "label": row["label"]} for row in rows))
    result = {"raw_records": len(rows) + len(duplicates), "records": len(rows), "exact_duplicates_removed": len(duplicates), "duplicate_record_ids": duplicates, "rendered": len(rendered), "metric": "formula_exact"}
    (output / "crohme_protocol.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def prepare_current_holdout(canonical_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    chosen = stratified_holdout(iter_current_external_metadata(canonical_root))
    counts: dict[str, int] = {}
    def rows() -> Iterator[dict]:
        for row in iter_current_external_metadata(canonical_root):
            if row["record_id"] in chosen:
                counts[row["source"]] = counts.get(row["source"], 0) + 1
                yield {key: value for key, value in row.items() if key != "holdout_key"}
    count = _write_jsonl(output / "current_data_10pct_truth.jsonl.gz", rows())
    return {"scope": "approved_external_classifier_sources", "records": count, "by_source": counts, "ratio": 0.10, "minimum_per_source_label": 1, "metric": "top1_top5_by_source", "not_product_evidence": True}


def _predictions(path: Path) -> dict[str, dict]:
    result = {}
    for row in _json_lines(path):
        record_id = str(row.get("record_id", ""))
        if not record_id or record_id in result:
            raise ValueError("prediction record_id must be unique and non-empty")
        result[record_id] = row
    return result


def score_glyphs(truth_path: Path, prediction_path: Path) -> dict:
    return score_glyphs_from_rows(_json_lines(truth_path), _predictions(prediction_path))


def score_formulae(truth_path: Path, prediction_path: Path) -> dict:
    return score_formulae_from_rows(_json_lines(truth_path), _predictions(prediction_path))


def _self_test() -> None:
    assert _trace_points("1 2, 3 4 5") == [[1.0, 2.0], [3.0, 4.0]]
    assert score_glyphs_from_rows([{"record_id": "a", "label": "x"}], {"a": {"topk": ["x"]}})["top1"] == 1.0
    assert score_glyphs_from_rows([{"record_id": "a", "label": "x", "source": "uji"}], {"a": {"topk": ["x"]}})["by_source"]["uji"]["top1"] == 1.0
    assert score_formulae_from_rows([{"record_id": "a", "label": "x"}], {"a": {"prediction": "x"}})["exact"] == 1.0


def score_glyphs_from_rows(truths: Iterable[dict], predictions: dict[str, dict]) -> dict:
    total = top1 = top5 = missing = 0
    source_counts: dict[str, dict[str, int]] = {}
    for truth in truths:
        total += 1; prediction = predictions.get(str(truth["record_id"])); candidates = [str(value) for value in prediction.get("topk", [])] if prediction else []
        source = str(truth.get("source", "all")); bucket = source_counts.setdefault(source, {"records": 0, "missing_predictions": 0, "top1_hits": 0, "top5_hits": 0})
        is_top1 = bool(candidates and candidates[0] == str(truth["label"])); is_top5 = str(truth["label"]) in candidates[:5]
        missing += prediction is None; top1 += is_top1; top5 += is_top5
        bucket["records"] += 1; bucket["missing_predictions"] += prediction is None; bucket["top1_hits"] += is_top1; bucket["top5_hits"] += is_top5
    result = {"metric": "glyph_top1_top5", "records": total, "missing_predictions": missing, "top1": top1 / total if total else None, "top5": top5 / total if total else None}
    if set(source_counts) != {"all"}:
        result["by_source"] = {source: {"records": bucket["records"], "missing_predictions": bucket["missing_predictions"], "top1": bucket["top1_hits"] / bucket["records"], "top5": bucket["top5_hits"] / bucket["records"]} for source, bucket in source_counts.items()}
    return result


def score_formulae_from_rows(truths: Iterable[dict], predictions: dict[str, dict]) -> dict:
    total = exact = missing = 0
    for truth in truths:
        total += 1; prediction = predictions.get(str(truth["record_id"])); missing += prediction is None; exact += bool(prediction and str(prediction.get("prediction", "")) == str(truth["label"]))
    return {"metric": "formula_exact", "records": total, "missing_predictions": missing, "exact": exact / total if total else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--prepare-direct", action="store_true")
    parser.add_argument("--prepare-current-holdout", action="store_true")
    parser.add_argument("--prepare-crohme", type=Path)
    parser.add_argument("--crohme-render-limit", type=int)
    parser.add_argument("--score-glyph", action="store_true")
    parser.add_argument("--score-formula", action="store_true")
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test(); print(json.dumps({"self_test": "pass"})); return 0
    if args.prepare_direct:
        print(json.dumps(prepare_direct_protocol(args.output), ensure_ascii=False, indent=2)); return 0
    if args.prepare_current_holdout:
        print(json.dumps(prepare_current_holdout(args.canonical_root, args.output), ensure_ascii=False, indent=2)); return 0
    if args.prepare_crohme:
        print(json.dumps(prepare_crohme_protocol(args.prepare_crohme, args.output, args.crohme_render_limit), ensure_ascii=False, indent=2)); return 0
    if args.score_glyph == args.score_formula or not args.truth or not args.predictions:
        parser.error("choose one prepare command, or exactly one score command with --truth and --predictions")
    result = score_glyphs(args.truth, args.predictions) if args.score_glyph else score_formulae(args.truth, args.predictions)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
