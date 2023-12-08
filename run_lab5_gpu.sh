#!/usr/bin/env bash

set -e
set -u
set -o pipefail

epoch=4

state=gpu_normal

# state=gpu_highload
# Run `glmark2 -b 'shadow:duration=100000'`

# state=gpu_underclock
# Use this: sudo nvidia-smi --lock-gpu-clocks=1000
# And this to reset: sudo nvidia-smi --reset-gpu-clocks


pushd pretrained

python ../lab5.py \
  --data_dir ../data \
  --out_dir ../output_lab5/model1 \
  --model 1 \
  --epoch ${epoch} \
  --gpu \
  --tag ${state}

python ../lab5.py \
  --data_dir ../data \
  --out_dir ../output_lab5/model2 \
  --model 2 \
  --epoch ${epoch} \
  --gpu \
  --tag ${state}

popd
