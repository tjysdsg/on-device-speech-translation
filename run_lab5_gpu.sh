#!/usr/bin/env bash

set -e
set -u
set -o pipefail

epoch=4

state=gpu_normal

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
