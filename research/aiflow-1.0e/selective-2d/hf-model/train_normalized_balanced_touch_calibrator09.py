"""Train a glyph-local, class/writer-balanced additive calibrator with '=' support."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from scripts.audit_math_ink_06_case_context import _load_model06
from scripts.train_math_ink_06_behavior_role import _canonical_group06, _formula_box06
from scripts.train_owned_formula_touch_adapter09 import _formula_samples
from scripts.train_pressure_free_touch_adapter09 import _owned_phone


def _public_formula_samples(formulas: Path, ownership: Path) -> list[dict]:
    records = {
        row["sample_id"]: row for row in (
            json.loads(line) for line in formulas.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    }
    output = []
    for annotation in (
        json.loads(line) for line in ownership.read_text(encoding="utf-8").splitlines() if line.strip()
    ):
        if not annotation.get("accepted"):
            continue
        record = records[annotation["sample_id"]]
        strokes = [{
            "order": order, "stroke_id": order,
            "points": [[float(point["x"]), float(point["y"]), float(point["t_ms"])] for point in stroke["points"]],
        } for order, stroke in enumerate(sorted(record["strokes"], key=lambda value: value["order"]))]
        output.append({
            "sample_id": annotation["sample_id"], "writer": annotation["writer_id"], "strokes": strokes,
            "truth_groups": [frozenset(group) for group in annotation["groups"]],
            "truth_labels": [str(label) for label in annotation["labels"]],
        })
    return output


def _materialize_local(samples: list[dict]):
    xs, labels, formula_ids, writers = [], [], [], []
    for sample in samples:
        formula_box = _formula_box06(sample["strokes"])
        for group, label in zip(sample["truth_groups"], sample["truth_labels"], strict=True):
            feature = _canonical_group06(group, sample["strokes"], formula_box)
            # Shape is already bbox-cropped/letterboxed by canonicalize_ink06.
            # Remove formula-relative position from the character path; layout retains it separately.
            feature[:, 10] = 0.0
            feature[:, 11] = 1.0
            feature[:, 12] = 1.0
            feature[:, 13] = 0.5
            feature[:, 14] = 1.0
            xs.append(feature); labels.append(str(label)); formula_ids.append(sample["sample_id"]); writers.append(sample["writer"])
    return torch.stack(xs), labels, formula_ids, writers


class AdditiveObservedHead09(nn.Module):
    def __init__(self, embedding_size: int, observed_existing: list[int]) -> None:
        super().__init__()
        self.observed_existing = observed_existing
        self.correction = nn.Linear(embedding_size, len(observed_existing) + 1)
        nn.init.zeros_(self.correction.weight); nn.init.zeros_(self.correction.bias)

    def forward(self, base_logits: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        correction = self.correction(embedding)
        output = torch.cat((base_logits, correction[:, -1:]), dim=1)
        output[:, self.observed_existing] = output[:, self.observed_existing] + correction[:, :-1]
        return output


def _forward(engine, online, head, features, device):
    rows = []
    engine.model.eval(); online.eval(); head.eval()
    with torch.inference_mode():
        for chunk in features.split(256):
            adapted = online(chunk.to(device))
            embedding = engine.model.encode_trajectory(adapted)
            base = engine.model.exact_head(embedding)
            rows.append(head(base, embedding).cpu())
    return torch.cat(rows)


def _baseline(engine, online, features, device):
    rows = []
    with torch.inference_mode():
        for chunk in features.split(256):
            adapted = online(chunk.to(device)); embedding = engine.model.encode_trajectory(adapted)
            base = engine.model.exact_head(embedding)
            rows.append(torch.cat((base, torch.full((len(base), 1), -1e4, device=device)), dim=1).cpu())
    return torch.cat(rows)


def _metrics(logits, truths, formula_ids, labels):
    index = {label: value for value, label in enumerate(labels)}
    truth = torch.tensor([index[value] for value in truths])
    top = logits.topk(5, dim=1).indices
    top1 = top[:, 0].eq(truth); top5 = top.eq(truth[:, None]).any(dim=1)
    f1, f5 = {}, {}
    for formula, hit1, hit5 in zip(formula_ids, top1.tolist(), top5.tolist(), strict=True):
        f1[formula] = f1.get(formula, True) and hit1; f5[formula] = f5.get(formula, True) and hit5
    eq = torch.tensor([value == "=" for value in truths])
    return {"symbols": len(truths), "formulas": len(f1), "top1": float(top1.float().mean()),
            "top5": float(top5.float().mean()), "formula_exact": sum(f1.values())/len(f1),
            "formula_top5_oracle": sum(f5.values())/len(f5), "equality_samples": int(eq.sum()),
            "equality_top1": float(top1[eq].float().mean()) if eq.any() else None,
            "equality_top5": float(top5[eq].float().mean()) if eq.any() else None}


def _paired(before, after, truths, labels):
    index = {label: value for value, label in enumerate(labels)}
    truth = torch.tensor([index[value] for value in truths])
    b = before.argmax(1).eq(truth); a = after.argmax(1).eq(truth)
    improved = int((~b & a).sum()); regressed = int((b & ~a).sum())
    by_label = {}
    for label in sorted(set(truths)):
        mask = torch.tensor([value == label for value in truths])
        by_label[label] = {"samples": int(mask.sum()), "baseline_correct": int((b&mask).sum()),
                           "candidate_correct": int((a&mask).sum())}
    return {"improved": improved, "regressed": regressed, "unchanged": len(truths)-improved-regressed,
            "by_label": by_label}


def _visualize(features, labels, output: Path) -> None:
    unique = sorted(set(labels)); font = ImageFont.load_default()
    page = Image.new("RGB", (5*180, 5*180), "white"); draw = ImageDraw.Draw(page)
    for slot, label in enumerate(unique):
        tile = np.zeros((128, 128), dtype=np.float32)
        indices = [i for i, value in enumerate(labels) if value == label]
        for i in indices:
            xy = (features[i, :, 2:4].numpy() * 127).round().astype(int).clip(0,127)
            tile[xy[:,1], xy[:,0]] += 1
        tile /= max(float(tile.max()), 1.0)
        image = Image.fromarray(np.uint8(255*(1-tile))).convert("RGB").resize((150,150))
        x, y = (slot%5)*180+15, (slot//5)*180+22
        page.paste(image, (x,y)); draw.text((x,y-16), f"{label}  n={len(indices)}", fill="black", font=font)
    page.save(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--public-formulas", type=Path)
    parser.add_argument("--public-ownership", type=Path)
    parser.add_argument("--adapter", type=Path, required=True); parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--phone-source", type=Path, required=True); parser.add_argument("--phone-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-observed-samples", type=int, default=1)
    args = parser.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine, online = _load_model06(args.base, args.adapter, device)
    for parameter in engine.model.parameters(): parameter.requires_grad_(False)
    for parameter in online.parameters(): parameter.requires_grad_(False)
    if args.public_formulas or args.public_ownership:
        if not args.public_formulas or not args.public_ownership:
            parser.error("--public-formulas and --public-ownership must be provided together")
        train_samples = _public_formula_samples(args.public_formulas, args.public_ownership)
    elif args.annotations:
        train_samples = _formula_samples(args.annotations)
    else:
        parser.error("provide --annotations or the two public dataset paths")
    replay_samples = _owned_phone(args.phone_source, args.phone_annotations)
    train_x, train_y, train_formula, train_writer = _materialize_local(train_samples)
    replay_x, replay_y, replay_formula, _ = _materialize_local(replay_samples)
    labels = tuple(engine.labels) + ("=",); label_index = {label:i for i,label in enumerate(labels)}
    if any(label not in label_index for label in train_y + replay_y):
        raise ValueError("Extended ontology still misses a collected label")
    observed_counts = Counter(train_y)
    observed_existing = sorted({
        label_index[label] for label in train_y
        if label != "=" and observed_counts[label] >= args.min_observed_samples
    })
    head = AdditiveObservedHead09(engine.model.exact_head.in_features, observed_existing).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=8e-4, weight_decay=2e-3)
    label_writers = defaultdict(set); pair_count = Counter(zip(train_y, train_writer, strict=True))
    for label, writer in zip(train_y, train_writer, strict=True): label_writers[label].add(writer)
    weights = torch.tensor([1.0/(len(label_writers[label])*pair_count[(label,writer)])
                            for label,writer in zip(train_y,train_writer,strict=True)], dtype=torch.double)
    target = torch.tensor([label_index[label] for label in train_y], device=device)
    history = []
    for epoch in range(1, args.epochs+1):
        sampler = WeightedRandomSampler(weights, num_samples=len(train_y)*2, replacement=True,
                                        generator=torch.Generator().manual_seed(args.seed*1000+epoch))
        losses=[]; head.train()
        order = torch.tensor(list(sampler))
        for indices in order.split(128):
            features=train_x[indices].to(device); truth=target[indices.to(device)]
            with torch.no_grad(): adapted=online(features); embedding=engine.model.encode_trajectory(adapted); base=engine.model.exact_head(embedding)
            logits=head(base,embedding); correction=head.correction(embedding)
            loss=torch.nn.functional.cross_entropy(logits,truth)+0.005*correction.square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),2.0); optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch":epoch,"loss":sum(losses)/len(losses)})
    args.output.mkdir(parents=True,exist_ok=True)
    baseline=_baseline(engine,online,replay_x,device); candidate=_forward(engine,online,head,replay_x,device)
    baseline_probability = baseline.softmax(1)
    baseline_top = baseline.argmax(1)
    preserve = torch.tensor([
        observed_counts.get(labels[int(index)], 0) < 3 for index in baseline_top
    ]) & baseline_probability.max(1).values.ge(.95)
    guarded = torch.where(preserve[:, None], baseline, candidate)
    train_logits=_forward(engine,online,head,train_x,device)
    _visualize(train_x,train_y,args.output/"normalized_class_means.png")
    checkpoint=args.output/"normalized_balanced_calibrator.pt"
    torch.save({"schema":"aiflow-normalized-balanced-calibrator09/v1","state_dict":deepcopy(head.state_dict()),
                "labels":labels,"observed_existing":observed_existing,"normalization":"glyph_bbox_letterbox_local_context",
                "pressure_used":False,"product_adopted":False,
                "guard_policy":{"baseline_confidence":0.95,"max_training_support":2}},checkpoint)
    report={"experiment":"P-NORMALIZED-BALANCED-TOUCH09-001","generated_at":datetime.now(timezone.utc).isoformat(),
            "seed":args.seed,"device":str(device),"training":{"formulae":len(train_samples),"symbols":len(train_y),
            "labels":len(set(train_y)),"contributors":len(set(train_writer)),"class_writer_balanced":True,
            "normalization":"existing bbox crop/aspect-preserving letterbox + local character context","equality_added":True,
            "min_observed_samples":args.min_observed_samples,"corrected_existing_labels":len(observed_existing)},
            "training_fit":_metrics(train_logits,train_y,train_formula,labels),
            "replay":{"baseline":_metrics(baseline,replay_y,replay_formula,labels),
                      "candidate":_metrics(candidate,replay_y,replay_formula,labels),
                      "guarded_candidate":_metrics(guarded,replay_y,replay_formula,labels),
                      "paired":_paired(baseline,candidate,replay_y,labels),
                      "guarded_paired":_paired(baseline,guarded,replay_y,labels),
                      "guard_preserved_symbols":int(preserve.sum())},"history":history,"adopted":False}
    (args.output/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"training":report["training"],"baseline":report["replay"]["baseline"],
                      "candidate":report["replay"]["candidate"],"paired":{k:v for k,v in report["replay"]["paired"].items() if k!='by_label'}},ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
