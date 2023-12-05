import os
from lab2 import LabExpRunner, read_data, PRETRAINED_MODEL, PRETRAINED_CONFIG
from onnx_utils.st_model import Speech2Text
from benchmarking import test_st_model
from lab4_benchmark_onnx import PROBLEM_MATIC_UTTS


def model1(utt2wav, utt2text, epoch):
    runner = LabExpRunner()
    pipeline = runner.create_inference_pipeline(PRETRAINED_MODEL, PRETRAINED_CONFIG)

    for i in range(epoch):
        runner.run_benchmark1(
            pipeline, utt2wav, utt2text, calculate_flops=False, num_utts=100,
        )


def model2(utt2wav, utt2text, epoch):
    p = Speech2Text(tag_name='q1_p1', use_quantized=True)

    for i in range(epoch):
        test_st_model(p, utt2wav, utt2text, 100)


def main():
    # Must be running in the directory that contains 'exp' and 'data' directories
    if not os.path.isdir('exp') or not os.path.isdir('data'):
        raise RuntimeError("Please run this script in `pretrained` directory")

    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load test data (520 utterances out of MUST_C_v2 TST-COMMON subset)
    utt2wav, utt2text = read_data(args.data_dir)
    for utt in PROBLEM_MATIC_UTTS:
        utt2wav.pop(utt)

    model1(utt2wav, utt2text, args.epoch)
    # model2(utt2wav, utt2text)


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
