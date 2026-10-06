# Quarantined CROHME-trained experiment

Status: **INVALID FOR AIFlow 1.0 TRAINING, SELECTION, OR PRODUCT USE**

This directory contains a historical experiment whose context checkpoint was
trained on CROHME train-derived rows. That violates the corrected project
boundary: CROHME and MathWriting are validation-only.

- Do not load `crohme_standard_context_research.pt` in a product or baseline.
- Do not use its epoch, lambda, threshold, or test result for model selection.
- The files remain in place only as an immutable audit record of the mistake.
- Valid HWR baseline: SHA-256
  `04f8608aebcf6c02d45ad6f5735229b9eaa2c4b4e1be0db4793d02273ef2d00e`.
- Valid commercial-rights context baseline: SHA-256
  `ee0033fb42f59b09f3f130300fb5374710cfef010a71c94dbd48e4d1c93dc8b`.

The replacement evaluation path must report `crohme_rows=0` and
`crohme_gradient_updates=0` in its training manifest before CROHME is opened
for a no-gradient validation pass.
