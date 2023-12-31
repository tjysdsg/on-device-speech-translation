import os
from lab2 import LabExpRunner, read_data, MODEL_CONFIG_MODIFIERS, INPUT_SIZE_MODIFIERS


def main():
    # Must be running in the directory that contains 'exp' and 'data' directories
    if not os.path.isdir('exp') or not os.path.isdir('data'):
        raise RuntimeError("Please run this script in `pretrained` directory")

    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    runner = LabExpRunner()  # <---

    # ==== Load Original Pre-trained Model ====
    pretrained_model = "exp/st_train_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp/valid.acc.ave_10best.pth"
    pretrained_config = "exp/st_train_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp/config.yaml"

    # Original model without quantization
    pipeline = runner.create_inference_pipeline(pretrained_model, pretrained_config, quantized=False)

    # Quantized
    p = runner.create_inference_pipeline(
        pretrained_model, pretrained_config,
        quantized=True, quantize_dtype=args.quantize_dtype,
    )
    # To view quantized model's size:
    # torch.save(p.st_model.state_dict(), 'shit.pth')

    # Load test data (520 utterances out of MUST_C_v2 TST-COMMON subset)
    utt2wav, utt2text = read_data(args.data_dir)

    # TODO: FX Graph Mode doesn't work cuz ESPnet is not symbolically traceable
    #   # Prepare calibration data
    #   assert args.num_calibration_utts + args.num_test_utts <= len(utt2wav)
    #   ptq_calibration_data = list(utt2wav.items())
    #   ptq_calibration_data = [speech for utt, speech in ptq_calibration_data][-args.num_calibration_utts:]
    #   from st_inference import FxPTQFactory
    #   ptq_factory = FxPTQFactory(pipeline)
    #   p = ptq_factory.create_static_ptq(ptq_calibration_data)

    # Results of the original pretrained model
    runner.run_benchmark(
        p, 'original', args.out_dir, utt2wav, utt2text, num_utts=args.num_test_utts, calculate_flops=False
    )

    # Change model size and run benchmarks
    for m in INPUT_SIZE_MODIFIERS:  # MODEL_CONFIG_MODIFIERS:
        p = runner.resize_model(
            pipeline.st_model, pretrained_config, m,
            quantized=True, quantize_dtype=args.quantize_dtype,
        )
        runner.run_benchmark(
            p, m.__name__, args.out_dir, utt2wav, utt2text, num_utts=args.num_test_utts, calculate_flops=False
        )


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--num_test_utts', type=int, default=100)
    parser.add_argument('--num_calibration_utts', type=int, default=100)
    parser.add_argument('--quantize_dtype', type=str, choices=['qint8', 'float16'], default='qint8')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
