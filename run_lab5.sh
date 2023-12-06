#!/usr/bin/env bash

set -e
set -u
set -o pipefail

epoch=4
# state=cpu_normal

state=cpu_highload
# can use `stress` to create high CPU load

# state=cpu_1thread

pushd pretrained

python ../lab5.py \
  --data_dir ../data \
  --out_dir ../output_lab5/model1 \
  --model 1 \
  --epoch ${epoch} \
  --tag ${state}

python ../lab5.py \
  --data_dir ../data \
  --out_dir ../output_lab5/model2 \
  --model 2 \
  --epoch ${epoch} \
  --tag ${state}

popd
