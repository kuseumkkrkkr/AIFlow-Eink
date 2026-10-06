---
dataset_info:
  features:
  - name: annotation
    dtype: string
  - name: frameHeight
    dtype: string
  - name: xCoordinates
    sequence: string
  - name: sampleTag
    dtype: string
  - name: yCoordinates
    sequence: string
  - name: timeStamps
    sequence: string
  - name: traveledDistances
    sequence: string
  - name: frameWidth
    dtype: string
  splits:
  - name: train
    num_bytes: 854129
    num_examples: 894
  - name: validation
    num_bytes: 222562
    num_examples: 310
  - name: test
    num_bytes: 222562
    num_examples: 310
  download_size: 610957
  dataset_size: 1299253
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
