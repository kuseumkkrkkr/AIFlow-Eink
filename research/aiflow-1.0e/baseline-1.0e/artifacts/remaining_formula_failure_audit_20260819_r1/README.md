# Remaining formula failure audit r1

- Re-rendered all 10 failed formulas after the grouping repair and straight-equality shadow layer.
- Inspected whole-formula ownership and all 16 wrong normalized glyphs from raw ordered strokes.
- Found no additional ownership boundary error or missing glyph group.
- Rejected an all-digit-triplet cross-to-plus rule even though it fixes one observed formula, because `247` and `2+7` are both valid user inputs and there is no independent positive/negative acceptance evidence.
- Further safe exact-formula improvement now requires HWR shape data or untouched formula context data; no additional runtime rule was admitted.

The private contact sheet is not approved for public release. It can be regenerated with `scripts/visualize_remaining_formula_failures_v1.py`.
