#!/usr/bin/env python3
"""Compose character-preserving formula function spans after completion.

The glyph records and stroke ownership remain immutable.  A semantic span is
an additional whole-formula view, not a new HWR character or regrouping action.
"""

from __future__ import annotations

from typing import Sequence


SCHEMA = "aiflow-formula-semantic-lexicon/v1"
FUNCTIONS = {
    ("s", "i", "n"): r"\sin",
    ("c", "o", "s"): r"\cos",
    ("t", "a", "n"): r"\tan",
    ("l", "i", "m"): r"\lim",
    ("l", "o", "g"): r"\log",
}


def compose_function_spans(
    tokens: Sequence[str], record_ids: Sequence[str] | None = None,
) -> dict:
    glyph_tokens = [str(value) for value in tokens]
    ids = (
        [str(value) for value in record_ids]
        if record_ids is not None else [str(index) for index in range(len(tokens))]
    )
    if len(ids) != len(glyph_tokens) or len(ids) != len(set(ids)):
        raise ValueError("semantic composition requires unique glyph record IDs")
    semantic_tokens = []
    spans = []
    index = 0
    ordered = sorted(FUNCTIONS.items(), key=lambda item: -len(item[0]))
    while index < len(glyph_tokens):
        match = next((
            (source, target) for source, target in ordered
            if tuple(glyph_tokens[index:index + len(source)]) == source
        ), None)
        if match is None:
            semantic_tokens.append(glyph_tokens[index])
            spans.append({
                "semantic_index": len(semantic_tokens) - 1,
                "token": glyph_tokens[index],
                "glyph_indices": [index],
                "record_ids": [ids[index]],
                "kind": "glyph",
            })
            index += 1
            continue
        source, target = match
        selected = list(range(index, index + len(source)))
        semantic_tokens.append(target)
        spans.append({
            "semantic_index": len(semantic_tokens) - 1,
            "token": target,
            "glyph_indices": selected,
            "record_ids": [ids[value] for value in selected],
            "kind": "function_name_composition",
            "source_tokens": list(source),
        })
        index += len(source)
    covered = sorted(value for span in spans for value in span["glyph_indices"])
    if covered != list(range(len(glyph_tokens))):
        raise AssertionError("semantic composition lost or duplicated a glyph")
    return {
        "schema": SCHEMA,
        "status": "formula_complete_semantic_view",
        "glyph_tokens": glyph_tokens,
        "semantic_tokens": semantic_tokens,
        "semantic_text": " ".join(semantic_tokens),
        "spans": spans,
        "composed_function_spans": sum(
            span["kind"] == "function_name_composition" for span in spans
        ),
        "glyph_count": len(glyph_tokens),
        "covered_glyphs": len(covered),
        "inserted_or_deleted_glyphs": 0,
        "stroke_grouping_mutations": 0,
        "training_performed": False,
    }


def _self_test() -> None:
    result = compose_function_spans(
        ["s", "i", "n", "(", "x", ")", "+", "t"],
        [f"r{index}" for index in range(8)],
    )
    assert result["semantic_tokens"] == [r"\sin", "(", "x", ")", "+", "t"]
    assert result["covered_glyphs"] == 8
    assert result["inserted_or_deleted_glyphs"] == 0


if __name__ == "__main__":
    _self_test()
    print('{"self_test":"pass"}')
