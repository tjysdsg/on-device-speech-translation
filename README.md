CMU 11-767 On-Device Machine Learning. Group Project: Efficient Speech Translation
---
[@tjysdsg](https://github.com/tjysdsg)
[@VincieSlytherin](https://github.com/VincieSlytherin)
[@kurisujhin](https://github.com/kurisujhin)
[@SandyLuXY](https://github.com/SandyLuXY)

# Setup

1. Init all git submodules
2. Enter `pretrained/` and pull git lfs files
   ```bash
   git lfs install
   git lfs pull
   ```
3. Install ESPnet: https://espnet.github.io/espnet/installation.html
4. Prepare data
   ```bash
   cd must_c_test_subset
   python ../prepare_data.py --out_dir ../data --sample_rate 16000
   ```

# Lab2: Baseline

- Pretrained model that's
  used: https://huggingface.co/espnet/brianyan918_mustc-v2_en-de_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp

```bash
cd pretrained
python plot_lab2.py --result_dir output --out_dir output/plot_all
```

# Lab3: Quantization

```bash
lab3.py
plot_all.py
```

# Lab4: Pruning

```bash
lab4.py
lab4_export_onnx.py
lab4_benchmark_onnx.py
```

# Lab5: Energy

1. Export ONNX models w/o other techniques

   ```bash
   cd pretrained/
   python export_onnx.py --pruned --quantized
   ```

2. Run `run_lab5.sh` and `run_lab5_gpu.sh`. Check comments carefully before running.

# Final

Under `pretrained/`:

1. Export ONNX models w/o other techniques

   ```bash
   cd pretrained/
   python ../export_onnx.py --pruned --quantized  # adjust these flags as needed
   ```

2. Run benchmarks

   ```bash
   cd pretrained/
   python ../final_benchmark.py --data_dir ../data --out_dir ../output
   ```

# Ablation study

`output_onnx/` is constructed from results of `final_benchmark.py`, containing ablation study of ONNX, pruning, and
quantization.
Run `plot_all.py` on this folder to generate the plot.

# See also

- ESPnet: https://github.com/espnet/espnet
- ESPnet documentation: https://espnet.github.io/espnet/
- https://github.com/mjpost/sacrebleu
- Source code for speech translation inference: https://github.com/espnet/espnet/blob/master/espnet2/bin/st_inference.py
- Source code for speech translation model: https://github.com/espnet/espnet/blob/master/espnet2/st/espnet_model.py

