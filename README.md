# Lab2 Baseline Benchmarking

- Pretrained model that's
  used: https://huggingface.co/espnet/brianyan918_mustc-v2_en-de_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp

```bash
./run.sh
```

## Notes

- Decreasing model size without re-training causes auto-regressive decoder to output long and repeated garbage,
  thus leading to higher average latency.
- There's an update to the python environment between Lab2 and Lab3, and I backed up results of lab2
  in [output_lab2](output_lab2).
- `qint8` quantization: 200MB -> 70.5MB

# See also

- ESPnet: https://github.com/espnet/espnet
- ESPnet documentation: https://espnet.github.io/espnet/
- https://github.com/mjpost/sacrebleu
- Source code for speech translation inference: https://github.com/espnet/espnet/blob/master/espnet2/bin/st_inference.py
- Source code for speech translation model: https://github.com/espnet/espnet/blob/master/espnet2/st/espnet_model.py

# Lab3

# Lab4

# Lab5

# Final

Under `pretrained/`:

1. Export ONNX models w/o other techniques

   ```bash
   python export_onnx.py --pruned --quantized
   ```

2. Run benchmarks

   ```bash
   python final_benchmark.py --data_dir ../data --out_dir ../output
   ```