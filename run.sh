#!/usr/bin/env bash

set -e
set -u
set -o pipefail

pushd must_c_test_subset
python ../prepare_data.py --out_dir ../data --sample_rate 16000
popd

pushd pretrained
python ../lab2.py --data_dir ../data --out_dir ../output --num_test_utts 20
popd

python plot_benchmark_results.py --result_dir output --out_dir output/plot_all