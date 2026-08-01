---
license: apache-2.0
language:
  - en
  - ko
tags:
  - handwriting
  - mathematical-expression-recognition
  - preprocessing
  - stroke-grouping
  - geometry
library_name: aiflow-math-ink
pipeline_tag: image-to-text
---

# AIFlow Math Ink 0.5

AIFlow Math Ink 0.5 is a **model-agnostic geometric gridding layer** for online
handwritten mathematics. It converts ordered pen strokes into spatially coherent
formula cells and renders one clean image per cell.

> This repository contains no OCR model, model weight, tokenizer, training data,
> or generated prediction. It is a deterministic preprocessing layer only.

The output images can be passed to any image-to-LaTeX recognizer. They are
interface-compatible with TexTeller-style image batches, but TexTeller is not
included, redistributed, modified, or required by this repository.

![AIFlow Math Ink 0.5 pipeline](assets/architecture.svg)

## Why this layer exists

An image-to-LaTeX model normally expects one coherent expression image. A pen
canvas may instead contain multiple rows, detached superscripts, fractions, or
matrix delimiters. AIFlow Math Ink 0.5 uses the original stroke geometry to
partition that canvas before recognition.

The layer:

1. validates bounded UTF-8 JSON-like stroke events;
2. computes width-aware stroke bounding boxes;
3. estimates a scale from the median stroke height;
4. creates same-baseline and local-attachment edges;
5. adds structural bridge edges for fraction bars and tall delimiters;
6. extracts connected components with a disjoint-set structure;
7. sorts cells in stable two-dimensional reading order; and
8. re-renders only the member strokes of each cell.

This is a transparent heuristic algorithm, not a trained neural-network layer.

## Installation

```bash
pip install Pillow
```

Clone this repository and add `src` to your Python path, or install it locally:

```bash
pip install -e .
```

## Minimal example

```python
from PIL import Image

from aiflow_math_ink_05 import GridConfig, Stroke, build_formula_grids, render_grid_images

strokes = [
    Stroke(0, ((24, 20), (36, 20)), 3.0),
    Stroke(1, ((10, 42), (60, 42)), 3.0),
    Stroke(2, ((24, 64), (36, 64)), 3.0),
]

grids = build_formula_grids(strokes, GridConfig())
images = render_grid_images(Image.new("RGB", (80, 90), "white"), strokes, grids)

for grid, image in zip(grids, images, strict=True):
    print(grid.stroke_ids)
    image.save(f"cell-{grid.index}.png")
```

For an existing API payload, use `parse_writing_events()` before
`build_formula_grids()`. See [`examples/basic.py`](examples/basic.py).

## Recognizer compatibility

`render_grid_images()` returns ordinary RGB `PIL.Image.Image` objects. A
recognizer can consume the list as a batch without importing this package into
its model implementation. TexTeller compatibility means only this image-level
interface; no TexTeller component is included.

## Method and complexity

The complete method, equations, assumptions, and limitations are documented in
[`docs/METHOD.md`](docs/METHOD.md).

For \(n\) accepted strokes, pair construction is \(O(n^2)\), disjoint-set
operations are effectively near-linear, and rendering is proportional to the
number of retained points plus output pixels. Input caps are mandatory in
server environments; the parser defaults to 2,000 strokes and 200,000 total
points.

## Scope and limitations

- This release performs geometry-based grouping, not symbol recognition.
- It does not infer LaTeX, semantics, or correctness.
- Thresholds are scale-adaptive but remain heuristic.
- Dense overlapping notes and unusual layouts can over-merge.
- The method has not been presented here as a peer-reviewed contribution.
- The repository ships no dataset and makes no benchmark claim.

## License

The original code in this repository is licensed under Apache License 2.0.
Pillow is an install-time dependency and is not vendored. External recognizers
are separate works governed by their own licenses.

