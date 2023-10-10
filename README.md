# Lab2 Baseline Benchmarking

- Pretrained model that's
  used: https://huggingface.co/espnet/brianyan918_mustc-v2_en-de_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp

```bash
./run.sh
```

## Notes

- Decreasing model size without re-training causes auto-regressive decoder to output long and repeated garbage,
  thus leading to higher average latency.

# See also

- ESPnet: https://github.com/espnet/espnet
- ESPnet documentation: https://espnet.github.io/espnet/
- https://github.com/mjpost/sacrebleu
- Source code for speech translation inference: https://github.com/espnet/espnet/blob/master/espnet2/bin/st_inference.py
- Source code for speech translation model: https://github.com/espnet/espnet/blob/master/espnet2/st/espnet_model.py

# TODO

- [ ] Some FLOPs are missing