---
dataset_info:
  features:
  - name: yCoordinates
    sequence: string
  - name: timeStamps
    sequence: string
  - name: frameHeight
    dtype: string
  - name: xCoordinates
    sequence: string
  - name: frameWidth
    dtype: string
  - name: traveledDistances
    sequence: string
  - name: annotation
    dtype: string
  - name: sampleTag
    dtype: string
  splits:
  - name: train
    num_bytes: 589619
    num_examples: 858
  - name: validation
    num_bytes: 16701
    num_examples: 22
  - name: test
    num_bytes: 63109
    num_examples: 157
  download_size: 315892
  dataset_size: 669429
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
