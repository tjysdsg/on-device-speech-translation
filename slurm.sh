#!/usr/bin/env bash
#SBATCH --job-name=seemerolling
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-16:1
#SBATCH --time=2:00:00

source $PROJECT/s2st/tools/activate_python.sh

cd pretrained
python ../lab2.py --data_dir ../data --out_dir ../output --num_test_utts 20

