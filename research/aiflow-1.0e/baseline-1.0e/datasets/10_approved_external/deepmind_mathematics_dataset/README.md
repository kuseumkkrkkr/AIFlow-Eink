# DeepMind Mathematics Dataset formula-context derivative

Last verified: 2026-08-20

## Source and rights

- Upstream: `https://github.com/google-deepmind/mathematics_dataset`
- Pinned revision: `427f45075f84b8b9774950196ad63867ca20ffb3`
- Upstream licence: Apache-2.0
- Local upstream checkout: `source/` (ignored from the parent repository)
- The upstream tracked source was not modified. Local Python `__pycache__` files are not source changes.

This source was selected instead of web-scraped formula corpora because its
generator and licence are explicit. OpenWebMath was audited but not admitted:
its ODC-By dataset licence does not replace the licences and terms of the
underlying crawled pages.

## Admitted derivative

`derived/formula_context_v1.jsonl.gz` contains 80,000 unique formula-token
sequences generated from 49 upstream modules. Only expressions represented
exactly by the frozen AIFlow 372-token vocabulary are admitted. Natural-language
fragments and partially representable expressions are rejected rather than
silently rewritten.

The derivative has zero exact token-sequence overlap with the frozen direct
writer evaluation and CROHME diagnostic caches. Its SHA-256 is
`1998779e46e693db2452c2a677dc0d366d4252be49ca27e3974b97df6fcc9bc5`.
See `derived/formula_context_v1_audit.json` for module counts, exclusion counts,
source hashes, and the complete generation contract.

## Permitted role

The derivative may train only the separate formula-context candidate-validity
model. It contains no pen trajectories, raster images, stroke ownership, writer
identity, or grouping labels and therefore cannot train or evaluate the HWR
shape encoder, grouping layer, or streaming input pipeline.

