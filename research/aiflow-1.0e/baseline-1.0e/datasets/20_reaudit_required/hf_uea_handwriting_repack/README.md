---
license: cc-by-4.0
task_categories:
- time-series-forecasting
tags:
- classification
- forecasting
- anomaly-detection
- TSLib
- tsfile
- modality:timeseries
pretty_name: Handwriting (TsFile)
size_categories:
- 1M<n<10M
configs:
- config_name: default
  data_files:
  - split: train
    path: Handwriting_train.tsfile
  - split: test
    path: Handwriting_test.tsfile
  - split: train_labels
    path: labels/Handwriting_train_labels.csv
  - split: test_labels
    path: labels/Handwriting_test_labels.csv
modality:
- timeseries
language:
- en
---

# Handwriting (TsFile)

Apache TsFile version of the `Handwriting` UEA classification subset of
[`thuml/Time-Series-Library`](https://huggingface.co/datasets/thuml/Time-Series-Library).

## Overview

Accelerometer traces of writing the 26 letters; classify the letter.

- **Train samples:** 150.
- **Test samples:** 850.
- **Dimensions (channels):** 3.
- **Series length:** 152.
- **Classes:** 26.

Each sample is an independent multivariate series. TRAIN and TEST are stored as
two separate TsFiles (`Handwriting_train.tsfile` / `Handwriting_test.tsfile`).
Class labels are stored as separate CSV sidecar files in `labels/`.

## Schema (TsFile structure)

- **sample_index** (TAG) - one device per sample. Query one sample with
  `WHERE sample_index=0`.
- **Time** (INT64) - within-sample position (0..151), a frame index rather than
  a wall-clock timestamp.
- **dim_0..dim_2** (FIELD, FLOAT) - the 3 channels at each position.
- **Labels** (CSV sidecar) - `labels/Handwriting_train_labels.csv` and
  `labels/Handwriting_test_labels.csv` contain `sample_index,class_label`, one
  row per sample.

Converted from the sktime `.ts` format. No dimensions, time points, samples, or
labels are dropped. Labels are not stored inside the TsFile feature tables.

## Usage

Read the `.tsfile` files with the Apache TsFile Java or Python SDK. Join labels
from the sidecar CSV files by `sample_index` within the matching split.

## Source & license

- Original dataset: https://huggingface.co/datasets/thuml/Time-Series-Library (subset `Handwriting`)
- Author / publisher: thuml (Tsinghua University)
- Paper: https://arxiv.org/abs/2407.13278
- License: CC BY 4.0
