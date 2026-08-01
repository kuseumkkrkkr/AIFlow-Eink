# AIFlow Math Ink 0.5: geometric gridding method

## Abstract

AIFlow Math Ink 0.5 is a deterministic preprocessing method for partitioning
online handwritten mathematics into image cells suitable for downstream
image-to-LaTeX recognition. It uses stroke trajectories and pen widths rather
than predicted symbols. The method constructs a sparse semantic approximation
through pairwise geometric predicates, augments it with fraction-bar and tall
delimiter bridges, and extracts connected components with disjoint-set union.
It is not a learned model and makes no claim of semantic parsing.

## 1. Input representation

Let a canvas contain ordered strokes

\[
\mathcal{S} = \{s_i\}_{i=1}^{n},
\qquad
s_i = (\mathrm{id}_i, \{(x_{ik},y_{ik})\}_{k=1}^{m_i}, w_i).
\]

Each stroke has a stable identifier, an ordered point sequence, and pen width
\(w_i\). Its axis-aligned bounding box expands the point extrema by
\(w_i/2\). This preserves the visible support of thick strokes.

The parser applies finite caps to stroke count, points per stroke, total points,
coordinate magnitude, and pen width. These caps are part of the runtime safety
contract rather than the mathematical grouping rule.

## 2. Scale estimation

Let \(h_i\) be the height of stroke box \(B_i\). The characteristic scale is

\[
\tilde h = \operatorname{median}_{i}(h_i).
\]

Median height is resistant to a small number of long fraction bars or tall
delimiters. Four distances are then defined:

\[
\tau_b = \max(24, 0.50\tilde h), \quad
\tau_x = \max(72, 1.25\tilde h),
\]

\[
\tau_s = \max(52, 0.80\tilde h), \quad
\tau_a = \max(18, 0.18\tilde h).
\]

The constants are defaults exposed by `GridConfig`; they are not learned
parameters.

## 3. Geometric relation graph

For boxes \(B_i,B_j\), let \(g_x(i,j)\) and \(g_y(i,j)\) be non-negative
axis gaps, and let \(c_y(i)\) be vertical center.

A same-baseline edge is created when

\[
|c_y(i)-c_y(j)| \le \tau_b
\quad\land\quad
g_x(i,j) \le \tau_x.
\]

A local-attachment edge handles detached pieces and scripts:

\[
g_x(i,j)\le\tau_a
\quad\land\quad
g_y(i,j)\le\tau_a
\]

\[
\quad\land\quad
|c_y(i)-c_y(j)|
\le 1.35\max(h_i,h_j)+\tau_a.
\]

These predicates intentionally favor recall of one coherent expression. They
can over-merge dense neighboring work, which is a documented limitation.

## 4. Structural bridges

A box is treated as a horizontal bridge candidate when

\[
\operatorname{width}(B_i)\ge30
\quad\land\quad
\operatorname{width}(B_i)\ge4\operatorname{height}(B_i).
\]

It connects to a nearby box when horizontal overlap is at least 20% of the
smaller width and vertical axis gap is at most \(\tau_s\). This approximates the
role of a fraction bar without recognizing its symbol.

A box is treated as a vertical bridge candidate when

\[
\operatorname{height}(B_i)\ge45
\quad\land\quad
\operatorname{height}(B_i)\ge2.5\operatorname{width}(B_i).
\]

It connects nearby vertically aligned strokes, approximating tall parentheses,
brackets, absolute-value marks, and matrix delimiters.

## 5. Connected components

All accepted relations are union operations over a disjoint-set forest with
path compression and union by size. Each connected component becomes one
formula cell. The cell boundary is the union of member boxes plus configurable
padding.

Cells are deterministically ordered by a quantized vertical band followed by
their left coordinate. This is a stable geometric order, not a full
two-dimensional mathematical parse.

## 6. Rendering contract

For every cell, member strokes are redrawn on a white RGB canvas and cropped to
the padded component boundary. Redrawing suppresses unrelated marks that happen
to occupy the rectangular crop. The returned object is a list of ordinary
`PIL.Image.Image` values and contains no recognizer-specific tensor transform.

This image-level contract is compatible with downstream recognizers that accept
formula image batches, including TexTeller-style pipelines. No recognizer is
included.

## 7. Complexity

The pairwise relation phase is \(O(n^2)\). Disjoint-set operations take
\(O(\alpha(n))\) amortized time, where \(\alpha\) is the inverse Ackermann
function. Rendering cost is proportional to retained stroke points and output
pixels. Bounded input size is therefore required for untrusted server traffic.

## 8. Scientific status and limitations

AIFlow Math Ink 0.5 is an engineering heuristic with explicit assumptions:

- axis-aligned boxes are sufficient local geometry;
- median stroke height approximates device scale;
- long horizontal and vertical strokes are useful structural bridges;
- connected components are adequate formula-cell hypotheses.

The release does not include a learned grouping head, probabilistic confidence,
ground-truth dataset, or benchmark claim. Appropriate evaluation would report
partition exact match, over-merge and under-merge rates, writer/device-disjoint
slices, and end-to-end expression accuracy separately.

