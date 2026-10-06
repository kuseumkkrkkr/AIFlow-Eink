#!/usr/bin/env python3
"""Validate a newly collected AIFlow 1.0e dataset before any training use."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


APPROVED_ANNOTATIONS = {"independently_reviewed", "approved"}
APPROVED_RIGHTS = {"approved", "commercial_training_and_distribution"}


def _load_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["records"] if isinstance(payload, dict) else payload


def validate_collection(contract: dict, records: list[dict]) -> dict:
    errors: list[str] = []
    required = set(contract["required_fields"]) | {"role", "category"}
    valid_roles = set(contract["roles"])
    valid_categories = set(contract["formula_mix_percent"])
    seen_ids: set[str] = set()
    writers_by_role: dict[str, set[str]] = defaultdict(set)
    prompts_by_role: dict[str, set[str]] = defaultdict(set)
    devices_by_role: dict[str, set[str]] = defaultdict(set)
    sessions_by_writer: dict[tuple[str, str], set[str]] = defaultdict(set)
    role_counts: Counter[str] = Counter()
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for index, row in enumerate(records):
        tag = str(row.get("record_id", f"row[{index}]"))
        missing = sorted(name for name in required if name not in row)
        if missing:
            errors.append(f"{tag}: missing fields {missing}")
            continue
        if tag in seen_ids:
            errors.append(f"{tag}: duplicate record_id")
        seen_ids.add(tag)

        role = str(row["role"])
        category = str(row["category"])
        if role not in valid_roles:
            errors.append(f"{tag}: unknown role {role!r}")
            continue
        if category not in valid_categories:
            errors.append(f"{tag}: unknown category {category!r}")
        role_counts[role] += 1
        category_counts[role][category] += 1
        writer = str(row["writer_id"])
        writers_by_role[role].add(writer)
        prompts_by_role[role].add(str(row["prompt_id"]))
        devices_by_role[role].add(str(row["device_id"]))
        sessions_by_writer[(role, writer)].add(str(row["session_id"]))

        strokes = row["strokes"]
        times = row["stroke_times"]
        if not isinstance(strokes, list) or not strokes:
            errors.append(f"{tag}: strokes must be a non-empty list")
        if not isinstance(times, list) or len(times) != len(strokes):
            errors.append(f"{tag}: stroke_times must align with strokes")
        elif any(len(t) != len(s) for s, t in zip(strokes, times)):
            errors.append(f"{tag}: every point must have a timestamp")

        ownership = row["symbol_ownership"]
        owned = [stroke for symbol in ownership for stroke in symbol.get("stroke_indices", [])]
        expected = list(range(len(strokes))) if isinstance(strokes, list) else []
        if sorted(owned) != expected:
            errors.append(f"{tag}: symbol ownership must cover each stroke exactly once")
        if len(row["symbol_labels"]) != len(ownership):
            errors.append(f"{tag}: symbol_labels and symbol_ownership length differ")
        if row["annotation_status"] not in APPROVED_ANNOTATIONS:
            errors.append(f"{tag}: structure annotation is not independently approved")
        if row["rights_status"] not in APPROVED_RIGHTS:
            errors.append(f"{tag}: commercial rights are not approved")

    if contract["rules"].get("writer_disjoint"):
        _append_role_overlap(errors, "writer", writers_by_role)
    if contract["rules"].get("prompt_disjoint"):
        _append_role_overlap(errors, "prompt", prompts_by_role)
    if contract["rules"].get("device_disjoint_in_sealed_acceptance"):
        sealed = devices_by_role.get("sealed_acceptance", set())
        nonsealed = set().union(*(v for k, v in devices_by_role.items() if k != "sealed_acceptance"))
        overlap = sorted(sealed & nonsealed)
        if overlap:
            errors.append(f"sealed device overlap: {overlap}")

    required_sessions = int(contract["writer_sessions"])
    for (role, writer), sessions in sessions_by_writer.items():
        if len(sessions) != required_sessions:
            errors.append(
                f"writer {writer!r} in {role!r}: expected {required_sessions} sessions, got {len(sessions)}"
            )

    for role, target in contract["roles"].items():
        if role_counts[role] != int(target["formulas"]):
            errors.append(f"role {role!r}: expected {target['formulas']} formulas, got {role_counts[role]}")
        if len(writers_by_role[role]) != int(target["writers"]):
            errors.append(f"role {role!r}: expected {target['writers']} writers, got {len(writers_by_role[role])}")
        for category, percent in contract["formula_mix_percent"].items():
            expected = int(target["formulas"]) * int(percent) / 100
            if category_counts[role][category] != expected:
                errors.append(
                    f"role {role!r} category {category!r}: expected {expected:g}, "
                    f"got {category_counts[role][category]}"
                )

    return {
        "schema": "aiflow-1.0e-collection-validation/v1",
        "status": "passed" if not errors else "failed",
        "record_count": len(records),
        "role_counts": dict(role_counts),
        "writer_counts": {role: len(values) for role, values in writers_by_role.items()},
        "errors": errors,
    }


def _append_role_overlap(errors: list[str], name: str, values_by_role: dict[str, set[str]]) -> None:
    roles = sorted(values_by_role)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            overlap = sorted(values_by_role[left] & values_by_role[right])
            if overlap:
                errors.append(f"{name} overlap between {left!r} and {right!r}: {overlap}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    result = validate_collection(contract, _load_records(args.records))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
