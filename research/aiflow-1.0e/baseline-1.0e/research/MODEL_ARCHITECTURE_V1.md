# AIFlow Math Ink 1.0 model architecture decision

Status: fixed design decision, 2026-08-12.

## Shared box-local encoder

- Input sequence: exactly 128 resampled points per glyph or group box.
- Point features: `x`, `y`, `delta_t`, `stroke_start`, `observed`.
- Projection: `5 -> 128`, followed by LayerNorm and GELU.
- Encoder: four Transformer blocks, 128 hidden width, four attention heads per block.
- Aggregation: attention pooling produces one 128-dimensional box embedding.

The resampler expands sparse paths and condenses dense paths. It preserves the
ordered stroke representation and does not crop by point index.

## Production mathematical-symbol output

- `math_head`: linear `128 -> 372` logits, then softmax and Top-k candidates.
- Vocabulary: 369 HWRT labels plus `(`, `)`, and `=` from project-owned data.
- Do not allocate an `UNKNOWN` class: no supervised unknown examples exist.
- Abstention is derived from calibrated Top-1 confidence and Top-1/Top-2 margin;
  it does not consume a character-logit slot.

This head is the only output used by the Math Ink product path. Its candidates
remain inputs to the downstream layout and decision layers; it does not own
formula grouping, stroke ownership, spatial relations, or final LaTex output.

## Separate general-text auxiliary output

- `latin_aux_head`: linear `128 -> 97` logits for UJI isolated characters.
- ISGL's 62 labels and UCI's 20 labels are subsets of that 97-label scope.
- This auxiliary objective may pretrain the shared encoder but is not routed into
  the Math Ink production decision path.
- BDSHWA remains excluded from class-head supervision because its labels are
  free-text prompts, not a stable character vocabulary.

The all-source character-label union is 403 labels. It is an audit count, not
the deployed mathematical output dimension.

## Formula-context finalizer

- Input: immutable HWR Top-5 candidates, probabilities, formula order, and seven spatial relations.
- Context encoder: owned 2-layer Transformer, hidden width 128, two attention heads.
- First pass: role/class context score plus HWR shape score, owned role grammar, and deterministic fence/infix guards.
- Product integrity: arithmetic equality solving is disabled by default. It is an explicit research-only suggestion mode because recognition must preserve intentionally wrong answers.
- Second pass: re-read the first finalized formula as MASK context.
- Recheck policy: only retract a first-pass override back to HWR Top-1; never introduce a new candidate choice.
- Contract: candidate preservation 100%, no token creation, deletion, or stroke regrouping.
- Current status: opt-in shadow; commercial accuracy gate remains open until untouched project-owned `|` and `o` formulas pass.
