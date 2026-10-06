"""Import blind human votes; never substitute teacher confidence for human labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from hwr_boundary_distillation_v1 import DEFAULT_REPORT, load_boundary_candidates

ROOT = Path(__file__).resolve().parents[1]
UNREADABLE = "__unreadable__"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate_votes(reviews: list[dict[str, str]], mapping: list[dict], labels: list[str]) -> dict:
    """Equal votes, including abstention mass; no confidence weighting or sharpening."""
    if len(reviews) < 2 or len(set(labels)) != len(labels):
        raise ValueError("multiple completed reviews and a unique vocabulary are required")
    count = len(mapping)
    identifiers = {item["candidate_id"] for item in mapping}
    if len(identifiers) != count:
        raise ValueError("blind mapping has duplicate candidate IDs")
    rows = [int(item["candidate_row"]) if "candidate_row" in item else
            2 * int(item["cache_row"]) + {"positive": 0, "nonpositive": 1}[item["endpoint"]] for item in mapping]
    if sorted(rows) != list(range(count)):
        raise ValueError("blind mapping must bijectively cover cached candidate rows")
    allowed = set(labels) | {UNREADABLE}
    for review in reviews:
        if set(review) != identifiers or any(label not in allowed for label in review.values()):
            raise ValueError("review is incomplete or has unknown candidates/labels")
    votes = np.zeros((count, len(labels)), dtype=np.int32)
    unreadable = np.zeros(count, dtype=np.int32)
    label_ids = {label: index for index, label in enumerate(labels)}
    for item, row in zip(mapping, rows):
        for review in reviews:
            label = review[item["candidate_id"]]
            if label == UNREADABLE:
                unreadable[row] += 1
            else:
                votes[row, label_ids[label]] += 1
    probabilities = votes.astype(np.float32) / len(reviews)
    unreadable_fraction = unreadable.astype(np.float32) / len(reviews)
    if not np.allclose(probabilities.sum(axis=1) + unreadable_fraction, 1.0):
        raise AssertionError("human-vote probability mass was lost")
    eligible = unreadable == 0
    targets = np.where(eligible[:, None], probabilities, 0.0).astype(np.float32)
    return {"vote_counts": votes, "vote_probabilities": probabilities, "unreadable_fraction": unreadable_fraction,
            "training_targets": targets, "training_eligible": eligible}


def human_vote_kl_loss(student_logits, human_targets, eligible):
    """Unsoftened T=1 human-vote KL; abstention-bearing inputs get no gradient."""
    import torch
    import torch.nn.functional as F

    if student_logits.ndim != 2 or student_logits.shape[1] != 372 or human_targets.shape != student_logits.shape:
        raise ValueError("human vote targets must align with [N,372] model logits")
    if eligible.shape != (len(student_logits),) or eligible.dtype != torch.bool:
        raise ValueError("human-target eligibility must be a boolean mask per row")
    if not torch.isfinite(student_logits).all() or not torch.isfinite(human_targets).all() or (human_targets < 0.0).any():
        raise ValueError("human vote KL needs finite nonnegative targets and finite logits")
    if not eligible.any():
        raise ValueError("no readable human boundary targets are available for training")
    if not torch.allclose(human_targets[eligible].sum(dim=1), torch.ones_like(human_targets[eligible, 0]), atol=1.0e-5, rtol=0.0) or human_targets[~eligible].count_nonzero():
        raise ValueError("eligible targets must sum to one; rejected targets must stay zero")
    return F.kl_div(F.log_softmax(student_logits[eligible].float(), dim=1), human_targets[eligible].detach().float(), reduction="batchmean")


def _read_review(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"candidate_id", "human_label"}.issubset(reader.fieldnames or []):
            raise ValueError("review needs candidate_id and human_label columns")
        result = {}
        for row in reader:
            identifier = (row.get("candidate_id") or "").strip()
            label = (row.get("human_label") or "").strip()
            if not identifier or identifier in result or not label:
                raise ValueError("review contains blank labels, empty IDs, or duplicate IDs")
            result[identifier] = label
    return result


def _build(args) -> int:
    if json.loads(args.report.read_text(encoding="utf-8")).get("schema") in (
        "aiflow-hwr-full-domain-tube-audit/v1", "aiflow-hwr-full-domain-boundary-profile/v1",
    ):
        return _build_full_domain(args)
    report, features, _ = load_boundary_candidates(args.report)
    mapping_path = args.report.parent / "blind_review_mapping.json"
    image_path = args.report.parent / "blind_boundary_candidates.png"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if len(mapping) != len(features):
        raise ValueError("blind mapping and candidate cache disagree")
    if args.review:
        packet = json.loads(args.packet_manifest.read_text(encoding="utf-8"))
        if packet.get("source_report_sha256") != _sha256(args.report) or packet.get("mapping_sha256") != _sha256(mapping_path) or packet.get("review_image_sha256") != _sha256(image_path):
            raise ValueError("candidate report, review image, or mapping changed after the packet was sealed")
    records, reviews, aliases, paths = [], [], set(), set()
    for spec in args.review:
        if "=" not in spec:
            raise ValueError("--review expects ANONYMOUS_ALIAS=CSV_PATH")
        alias, path_text = spec.split("=", 1)
        path = Path(path_text).resolve()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", alias) or alias in aliases or path in paths:
            raise ValueError("use unique anonymous reviewer aliases and separate review files")
        aliases.add(alias)
        paths.add(path)
        reviews.append(_read_review(path))
        records.append({"reviewer_alias": alias, "review_csv_sha256": _sha256(path)})
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite human-target artifacts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "aiflow-hwr-human-boundary-targets/v1", "source_report": str(args.report.resolve()),
        "source_report_sha256": _sha256(args.report), "mapping_sha256": _sha256(mapping_path),
        "review_image": str(image_path.resolve()), "review_image_sha256": _sha256(image_path),
        "teacher_checkpoint_sha256": report["provenance"]["teacher_checkpoint_sha256"],
        "candidate_rows": len(features), "completed_reviews": len(reviews), "required_reviews": args.min_reviewers,
        "reviews": records, "reviewer_independence_verified": False,
        "scope": "pilot reviewer vote frequencies, not calibrated population recognition probabilities",
        "self_reported_confidence_used_as_weight": False, "teacher_probabilities_used_as_labels": False,
        "unreadable_policy": "keep abstention probability mass; exclude any abstention-bearing row from training",
        "heldout_rows_read": 0, "crohme_rows": 0, "student_training_performed": False,
        "accuracy_evaluation_performed": False, "product_adopted": False,
    }
    if len(reviews) < args.min_reviewers:
        result.update(status="needs_completed_human_reviews", human_labels_available=False, numeric_training_targets_written=False)
        exit_code = 2
    else:
        labels = report["adaptive_teacher_boundary_refinement"]["numeric_candidate_cache"]["class_labels"]
        arrays = aggregate_votes(reviews, mapping, labels)
        eligible_count = int(arrays["training_eligible"].sum())
        result.update(status="human_vote_targets_prepared" if eligible_count else "no_readable_human_targets", human_labels_available=True,
                      numeric_training_targets_written=True, class_labels=labels,
                      training_eligible_rows=eligible_count)
        result["artifacts"] = {}
        for name, array in arrays.items():
            path = args.output_dir / f"{name}.npy"
            np.save(path, array, allow_pickle=False)
            result["artifacts"][name] = {"path": str(path.resolve()), "sha256": _sha256(path), "shape": list(array.shape), "dtype": str(array.dtype)}
        exit_code = 0 if eligible_count else 2
    destination = args.output_dir / "human_boundary_targets.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "human_boundary_target_readiness", "status": result["status"], "completed_reviews": len(reviews), "report": str(destination.resolve())}))
    return exit_code


def _build_full_domain(args) -> int:
    """전 범위 검토 패킷의 모든 SHA를 확인하고 독립 표본 투표만 학습 표적으로 변환한다."""
    report = json.loads(args.report.read_text(encoding="utf-8"))
    provenance = report["provenance"]
    if provenance["heldout_rows_read"] or provenance["crohme_rows"] or provenance["official_test_rows_read"] or provenance["real_training_source_ids"] != [0, 1]:
        raise ValueError("full-domain candidates are not admitted REAL TRAIN only")
    if report["frozen_plan_sha256"] != _sha256(args.report.parent / "frozen_plan.json"):
        raise ValueError("full-domain source plan changed")
    for filename, digest in report["review_packet_sha256"].items():
        if Path(filename).name != filename or _sha256(args.report.parent / filename) != digest:
            raise ValueError("sealed full-domain review packet changed")
    cache = report["candidate_cache"]
    labels = cache["class_labels"]
    arrays = {}
    for name, item in cache["artifacts"].items():
        path = args.report.parent / f"{name}.npy"
        if _sha256(path) != item["sha256"]:
            raise ValueError("full-domain candidate array changed")
        arrays[name] = np.load(path, allow_pickle=False)
        if list(arrays[name].shape) != item["shape"] or str(arrays[name].dtype) != item["dtype"] or not np.isfinite(arrays[name]).all():
            raise ValueError("full-domain candidate array contract failed")
    mapping = json.loads((args.report.parent / "blind_review_mapping.json").read_text(encoding="utf-8"))
    if len(labels) != 372 or len(set(labels)) != 372 or arrays["candidate_features"].shape != (len(mapping), 128, 5) or arrays["teacher_logits"].shape != (len(mapping), 372):
        raise ValueError("full-domain vocabulary/row alignment failed")
    if arrays["candidate_features"][..., :2].min() < 0 or arrays["candidate_features"][..., :2].max() > 1:
        raise ValueError("full-domain XY input outside unit square")
    if sorted(int(item["candidate_row"]) for item in mapping) != list(range(len(mapping))):
        raise ValueError("full-domain mapping is not bijective")
    aliases, paths, reviews, records = set(), set(), [], []
    for spec in args.review:
        if "=" not in spec:
            raise ValueError("--review expects ANONYMOUS_ALIAS=CSV_PATH")
        alias, path_text = spec.split("=", 1)
        path = Path(path_text).resolve()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", alias) or alias in aliases or path in paths:
            raise ValueError("use unique reviewer aliases and separate review files")
        aliases.add(alias); paths.add(path)
        reviews.append(_read_review(path))
        records.append(dict(reviewer_alias=alias, review_csv_sha256=_sha256(path)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite human-target artifacts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = dict(schema="aiflow-hwr-full-domain-human-targets/v1", source_report_sha256=_sha256(args.report),
                  class_labels=labels, candidate_rows=len(mapping), completed_reviews=len(reviews), required_reviews=args.min_reviewers,
                  reviews=records, reviewer_independence_verified=False, teacher_probabilities_used_as_labels=False,
                  unreadable_policy="keep abstention mass; reject any abstention-bearing row from training",
                  scope="pilot votes across admitted full-domain candidates, not calibrated population probabilities",
                  heldout_rows_read=0, crohme_rows=0, student_training_performed=False, product_adopted=False)
    if len(reviews) < args.min_reviewers:
        result.update(status="needs_completed_human_reviews", numeric_training_targets_written=False)
        code = 2
    else:
        values = aggregate_votes(reviews, mapping, labels)
        result["artifacts"] = {}
        for name, value in values.items():
            path = args.output_dir / f"{name}.npy"; np.save(path, value, allow_pickle=False)
            result["artifacts"][name] = dict(sha256=_sha256(path), shape=list(value.shape), dtype=str(value.dtype))
        count = int(values["training_eligible"].sum())
        result.update(status="human_vote_targets_prepared" if count else "no_readable_human_targets",
                      numeric_training_targets_written=True, training_eligible_rows=count)
        code = 0 if count else 2
    with (args.output_dir / "human_boundary_targets.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=True, indent=2)
        stream.write("\n")
    print(json.dumps(dict(event="full_domain_human_target_readiness", status=result["status"], completed_reviews=len(reviews))))
    return code


def _self_test() -> int:
    # Toy fixtures only: never emitted as real human annotations or training inputs.
    mapping = [{"candidate_id": "A", "cache_row": 0, "endpoint": "positive"}, {"candidate_id": "B", "cache_row": 0, "endpoint": "nonpositive"}]
    result = aggregate_votes([{"A": "2", "B": "Z"}, {"A": "Z", "B": UNREADABLE}], mapping, ["2", "Z"])
    assert np.array_equal(result["vote_probabilities"], [[0.5, 0.5], [0.0, 0.5]])
    assert np.array_equal(result["unreadable_fraction"], [0.0, 0.5])
    assert np.array_equal(result["training_targets"], [[0.5, 0.5], [0.0, 0.0]])
    assert np.array_equal(result["training_eligible"], [True, False])
    generalized = [{"candidate_id": "A", "candidate_row": 1}, {"candidate_id": "B", "candidate_row": 0}]
    whole = aggregate_votes([{"A": "x", "B": "\\int"}, {"A": "x", "B": "y"}], generalized, ["x", "y", "\\int"])
    assert np.array_equal(whole["vote_probabilities"], [[0., .5, .5], [1., 0., 0.]])
    for bad in ([{"A": "2"}, {"A": "2"}], [{"A": "2", "B": "unknown"}, {"A": "2", "B": "Z"}]):
        try:
            aggregate_votes(bad, mapping, ["2", "Z"])
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete/unknown labels were admitted")
    from audit_hwr_probability_boundary_tube_v1 import _guard_commit
    if _guard_commit("before_human_vote_loss_self_test") is None:
        return 78
    import torch

    logits = torch.zeros(2, 372, requires_grad=True)
    targets = torch.zeros(2, 372)
    targets[0, :2] = 0.5
    eligible = torch.tensor([True, False])
    loss = human_vote_kl_loss(logits, targets, eligible)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad[0].norm() > 0.0
    assert logits.grad[1].count_nonzero() == 0
    print(json.dumps({"self_test": "pass", "abstention_mass_preserved": True, "abstention_gradient_zero": True, "confidence_not_invented": True}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("self-test", "build"), required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--review", action="append", default=[], metavar="ANONYMOUS_ALIAS=CSV_PATH")
    parser.add_argument("--min-reviewers", type=int, default=2, help="minimum for a pilot vote distribution, not statistical acceptance")
    parser.add_argument("--packet-manifest", type=Path, default=ROOT / "artifacts/hwr_human_boundary_readiness_20261005/human_boundary_targets.json", help="sealed report/image/mapping hashes from before human review")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/hwr_human_boundary_readiness_20261005")
    args = parser.parse_args()
    if args.min_reviewers < 2:
        parser.error("at least two completed blind reviews are required for a vote distribution")
    return _self_test() if args.mode == "self-test" else _build(args)


if __name__ == "__main__":
    raise SystemExit(main())
