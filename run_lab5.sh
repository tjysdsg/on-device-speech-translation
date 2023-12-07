#!/usr/bin/env bash

set -e
set -u
set -o pipefail

epoch=4

num_threads=-1
state=cpu_normal

# state=cpu_highload
# can use `stress` to create high CPU load

# state=cpu_1thread
# num_threads=1

pushd pretrained

python ../lab5.py \
  --data_dir ../data \
  --out_dir ../output_lab5/model1 \
  --model 1 \
  --epoch ${epoch} \
  --num_threads ${num_threads} \
  --tag ${state}

python ../lab5.py \
  --data_dir ../data \
  --out_dir ../output_lab5/model2 \
  --model 2 \
  --epoch ${epoch} \
  --num_threads ${num_threads} \
  --tag ${state}

popd
