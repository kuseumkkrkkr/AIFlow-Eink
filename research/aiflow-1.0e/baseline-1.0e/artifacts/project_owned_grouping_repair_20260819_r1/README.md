# Project-owned grouping repair r1

- Source remains private on D:; `r2` was copied append-only to `public-candidate-20260819-r3`.
- Only three audited `ownership_train.jsonl` group boundaries changed: `aiflow_0074`, `aiflow_0163`, and `aiflow_0164`.
- Labels, formulas, strokes, stroke order, and the fixed 95-formula/387-glyph denominator did not change.
- The public dataset validator passed: 164 records, 96 ownership annotations, and zero forbidden privacy keys.
- Frozen-model Top-20 recall moved from 379/387 glyphs and 91/95 formulas to 386/387 and 94/95.
- Replaying the complete existing shadow pipeline moved Top-1 from 362/387 and 81/95 exact formulas to 369/387 and 84/95, with zero regressions.
- This is a data-quality correction on repeatedly observed development formulas, not untouched product acceptance evidence.

The executable repair specification is `grouping_repairs.json`; the generator and fixed-denominator evaluator are `scripts/repair_project_owned_grouping_v1.py` and `scripts/evaluate_project_owned_grouping_repair_v1.py`.
