# License review

Review date: 2026-07-25

## Distribution boundary

This repository distributes:

- original AIFlow geometric grouping source;
- original documentation and SVG artwork;
- tests and a small synthetic example.

It does not distribute:

- an OCR model or model architecture implementation;
- model weights, tokenizer files, or inference configuration;
- a dataset or user handwriting;
- source copied from an external recognizer;
- Pillow source or binaries.

## Project license

The distributed work is offered under Apache License 2.0. The repository
includes the full `LICENSE` text and a `NOTICE` file. Apache-2.0 permits
commercial use, modification, and redistribution subject to its notice,
license-copy, and modification-marking conditions. It also includes an express
patent grant and warranty disclaimer.

Primary source:
https://www.apache.org/licenses/LICENSE-2.0

## Runtime dependency

Pillow is declared as an install-time dependency and is not vendored. Pillow is
licensed under the open-source MIT-CMU license. Its license permits use, copy,
modification, and distribution subject to preservation of its copyright and
permission notice.

Primary source:
https://github.com/python-pillow/Pillow/blob/main/LICENSE

## Optional recognizer interoperability

The package returns ordinary RGB Pillow images. Compatibility with an external
recognizer is an interface statement only and does not create a bundled or
derived distribution of that recognizer. Users must obtain and operate any
external recognizer separately under its own license.

TexTeller's official repository currently declares Apache-2.0, but no TexTeller
artifact is included here.

Primary source:
https://github.com/OleehyO/TexTeller

## Conclusion

The repository is suitable for publication as a standalone Apache-2.0 source
release, assuming the uploader is authorized to license the AIFlow source. This
is a technical license review, not jurisdiction-specific legal advice.

