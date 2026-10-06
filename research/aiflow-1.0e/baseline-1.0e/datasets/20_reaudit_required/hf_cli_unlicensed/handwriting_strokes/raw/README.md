---
dataset_info:
  features:
  - name: traveledDistances
    sequence: string
  - name: annotation
    dtype: string
  - name: yCoordinates
    sequence: string
  - name: frameHeight
    dtype: string
  - name: sampleTag
    dtype: string
  - name: timeStamps
    sequence: string
  - name: xCoordinates
    sequence: string
  - name: frameWidth
    dtype: string
  splits:
  - name: train
    num_bytes: 1603549
    num_examples: 920
  - name: validation
    num_bytes: 171243
    num_examples: 100
  - name: test
    num_bytes: 172648
    num_examples: 100
  download_size: 932763
  dataset_size: 1947440
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---
