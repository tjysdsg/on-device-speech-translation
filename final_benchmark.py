import os
import torch
from lab2 import LabExpRunner, read_data, PRETRAINED_MODEL, PRETRAINED_CONFIG
from lab4 import optimal_config as optimal_pruning
from onnx_utils.st_model import Speech2Text
from benchmarking import test_st_model
from lab4_benchmark_onnx import PROBLEM_MATIC_UTTS

ONNX_MODEL_TAGS = [
    'q0_p0',
    'q1_p0',
    'q0_p1',
    'q1_p1',
]


def prune_quantize_no_onnx(utt2wav, utt2text, num_test_utts):
    runner = LabExpRunner()
    p = runner.create_inference_pipeline(PRETRAINED_MODEL, PRETRAINED_CONFIG, quantized=False)

    _, func = optimal_pruning()
    func(p.st_model)

    pruned_path = "pruned.pth"
    torch.save(p.st_model.state_dict(), pruned_path)

    p = runner.create_inference_pipeline(pruned_path, PRETRAINED_CONFIG, quantized=True)

    import gc
    gc.collect()

    result = runner.run_benchmark1(
        p, utt2wav, utt2text, num_utts=num_test_utts, calculate_flops=False
    )
    return result


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

    # Quantization + Pruning
    result = prune_quantize_no_onnx(utt2wav, utt2text, args.num_test_utts)
    LabExpRunner.save_exp_statistics(result, os.path.join(out_dir, f'q1_p1_no_onnx.json'))

    # Same but with ONNX
    for tag in ONNX_MODEL_TAGS:
        quantize = 'q1' in tag
        p = Speech2Text(tag_name=tag, use_quantized=quantize)

        import gc
        gc.collect()

        result = test_st_model(p, utt2wav, utt2text, args.num_test_utts)
        LabExpRunner.save_exp_statistics(result, os.path.join(out_dir, f'{tag}.json'))


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--num_test_utts', type=int, default=100)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
