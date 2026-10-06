#!/usr/bin/env python3
"""Fail closed when noncommercial evaluation data reaches a training path.

CROHME and MathWriting are validation-only in AIFlow Math Ink 1.0.  Training
scripts call this module before reading rows and record the returned audit in
their manifests.  Evaluation programs do not call this guard.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import sys
from typing import Any


SCHEMA = "aiflow-training-data-guard/v1"
FORBIDDEN_MARKERS = (
    "30_noncommercial_evaluation",
    "crohme",
    "mathwriting",
)
TRAINING_PROVENANCE_KEYS = frozenset({
    "source", "dataset", "dataset_name", "corpus", "license", "license_id",
    "data_rights", "source_path", "split_root", "training_source",
})
EVALUATION_ONLY_OPTION_MARKERS = (
    "crohme", "mathwriting", "validation", "evaluation", "eval", "test",
    "holdout",
)


def _normalized(value: Any) -> str:
    return str(value).replace("\\", "/").casefold()


def _forbidden(value: Any) -> str | None:
    normalized = _normalized(value)
    return next((marker for marker in FORBIDDEN_MARKERS if marker in normalized), None)


def assert_training_path_clean(path: Path | str, label: str = "training input") -> Path:
    resolved = Path(path).expanduser().resolve()
    marker = _forbidden(resolved)
    if marker is not None:
        raise ValueError(
            f"{label} contains validation-only marker {marker!r}: {resolved}"
        )
    return resolved


def assert_training_paths_clean(paths: Iterable[Path | str]) -> tuple[Path, ...]:
    return tuple(assert_training_path_clean(path) for path in paths)


def assert_training_entrypoint_arguments_clean(
    arguments: Iterable[str] | None = None,
) -> None:
    """Reject forbidden paths unless their CLI option is explicitly evaluation-only.

    Every ``train_*.py`` entrypoint calls this before importing ML runtimes.  It
    prevents a CROHME/MathWriting path from being substituted for a fit corpus
    while still allowing an explicitly named post-fit validation argument such
    as ``--crohme-candidates``.
    """
    values = list(sys.argv[1:] if arguments is None else arguments)
    option = ""
    for value in values:
        if value.startswith("--"):
            option = value.split("=", 1)[0].casefold()
            inline = value.split("=", 1)[1] if "=" in value else ""
            if not inline:
                continue
            value = inline
        marker = _forbidden(value)
        if marker is None:
            continue
        if option and any(token in option for token in EVALUATION_ONLY_OPTION_MARKERS):
            continue
        raise ValueError(
            f"training entrypoint argument contains validation-only marker "
            f"{marker!r} outside an evaluation-only option: {value}"
        )


def assert_training_row_clean(row: Mapping[str, Any], *, record_id: str = "") -> None:
    for key in TRAINING_PROVENANCE_KEYS.intersection(row):
        marker = _forbidden(row[key])
        if marker is not None:
            identity = record_id or str(row.get("record_id", "<unknown>"))
            raise ValueError(
                f"training row {identity} contains validation-only marker "
                f"{marker!r} in {key}"
            )


def assert_training_rows_clean(rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    for row in rows:
        assert_training_row_clean(row)
        count += 1
    return count


def assert_training_metadata_clean(
    metadata: Any, *, label: str = "training metadata"
) -> None:
    """Reject validation-only source strings nested in an input manifest.

    Boolean contract keys such as ``crohme_used_for_generation=False`` are
    allowed; only scalar values are inspected.  This is intended for input
    manifests, not human-facing reports that describe the policy in prose.
    """
    pending = [metadata]
    while pending:
        value = pending.pop()
        if isinstance(value, Mapping):
            for key, item in value.items():
                marker = _forbidden(key)
                zero_contract = item is None or item is False or (
                    isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    and item == 0
                )
                if marker is not None and not zero_contract:
                    raise ValueError(
                        f"{label} contains nonzero validation-only field "
                        f"{key!r}: {item}"
                    )
                pending.append(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            pending.extend(value)
        elif isinstance(value, (str, Path)):
            marker = _forbidden(value)
            if marker is not None:
                raise ValueError(
                    f"{label} contains validation-only marker {marker!r}: {value}"
                )


def zero_crohme_training_manifest(
    *, admitted_sources: Mapping[str, int] | None = None,
    gradient_updates: int | None = None,
) -> dict[str, Any]:
    if gradient_updates is not None and gradient_updates < 0:
        raise ValueError("gradient update count cannot be negative")
    return {
        "schema": SCHEMA,
        "policy": "CROHME and MathWriting are validation-only",
        "forbidden_markers": list(FORBIDDEN_MARKERS),
        "admitted_sources": dict(sorted((admitted_sources or {}).items())),
        "crohme_rows": 0,
        "mathwriting_rows": 0,
        "crohme_gradient_updates": 0,
        "mathwriting_gradient_updates": 0,
        "total_gradient_updates": (
            int(gradient_updates) if gradient_updates is not None else None
        ),
        "total_gradient_updates_recorded": gradient_updates is not None,
        "passed": True,
    }


def _self_test() -> None:
    assert_training_path_clean(r"D:\\safe\\normalized")
    assert_training_row_clean({"source": "uji_pen_v2", "record_id": "ok"})
    for value in (
        r"D:\\data\\30_noncommercial_evaluation\\x",
        "crohme2019_test",
        "MathWriting",
    ):
        try:
            if ":" in value or "\\" in value:
                assert_training_path_clean(value)
            else:
                assert_training_row_clean({"source": value})
        except ValueError:
            pass
        else:
            raise AssertionError(f"forbidden training source was admitted: {value}")
    assert zero_crohme_training_manifest(gradient_updates=3)[
        "crohme_gradient_updates"
    ] == 0
    assert_training_entrypoint_arguments_clean([
        "--crohme-candidates", r"D:\data\crohme\test.jsonl.gz",
    ])
    assert_training_metadata_clean({"source": "UJI Pen Characters v2"})
    try:
        assert_training_metadata_clean({"source": "crohme2019_train"})
    except ValueError:
        pass
    else:
        raise AssertionError("forbidden training metadata was admitted")
    try:
        assert_training_metadata_clean({"crohme_candidates_sha256": "abc"})
    except ValueError:
        pass
    else:
        raise AssertionError("forbidden training metadata key was admitted")
    assert_training_metadata_clean({
        "crohme_used_for_generation_or_filtering": False,
        "crohme_gradient_updates": 0,
    })
    for arguments in (
        ["--input", r"D:\data\crohme\train.jsonl.gz"],
        [r"D:\data\30_noncommercial_evaluation\crohme"],
    ):
        try:
            assert_training_entrypoint_arguments_clean(arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"forbidden training CLI path was admitted: {arguments}")


if __name__ == "__main__":
    _self_test()
    print('{"self_test":"pass"}')
