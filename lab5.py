import os
import torch
from lab2 import LabExpRunner, read_data, PRETRAINED_MODEL, PRETRAINED_CONFIG
from onnx_utils.st_model import Speech2Text
from benchmarking import test_st_model
from lab4_benchmark_onnx import PROBLEM_MATIC_UTTS
from codecarbon import EmissionsTracker
from codecarbon.core import cpu, gpu


def model1(utt2wav, utt2text, epoch: int, num_cpu_threads: int, cuda: bool):
    if num_cpu_threads > 0:
        torch.set_num_threads(num_cpu_threads)
        print(f'torch num_threads set to: {num_cpu_threads}')

    device = 'cpu'
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available.')
        device = "cuda"

    runner = LabExpRunner(device=device)
    pipeline = runner.create_inference_pipeline(PRETRAINED_MODEL, PRETRAINED_CONFIG)

    for i in range(epoch):
        runner.run_benchmark1(
            pipeline, utt2wav, utt2text, calculate_flops=False, num_utts=100,
        )


def model2(utt2wav, utt2text, epoch: int, num_cpu_threads: int, cuda: bool):
    # TODO: cuda

    p = Speech2Text(
        tag_name='q1_p1',
        use_quantized=True,
        session_option_dict=dict(
            intra_op_num_threads=num_cpu_threads,
            inter_op_num_threads=num_cpu_threads,
        ),
    )

    for i in range(epoch):
        test_st_model(p, utt2wav, utt2text, 100)


def main():
    # Must be running in the directory that contains 'exp' and 'data' directories
    if not os.path.isdir('exp') or not os.path.isdir('data'):
        raise RuntimeError("Please run this script in `pretrained` directory")

    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    num_cpu_threads = args.num_threads
    if num_cpu_threads > 0:
        if args.gpu:
            print('Ignoring --num_threads since GPU is being used')
            num_cpu_threads = 0
    else:
        num_cpu_threads = 0

    if args.gpu and not torch.cuda.is_available():
        raise RuntimeError('Requested GPU inference but CUDA is not available.')

    # Load test data (520 utterances out of MUST_C_v2 TST-COMMON subset)
    utt2wav, utt2text = read_data(args.data_dir)
    for utt in PROBLEM_MATIC_UTTS:
        utt2wav.pop(utt)

    # Check if we have the tools to get meaningful power consumption readings
    if args.gpu:
        if not gpu.is_gpu_details_available():
            raise RuntimeError("GPU details are not available")
    elif not cpu.is_powergadget_available() and not cpu.is_rapl_available():
        raise RuntimeError(
            "Neither PowerGadget nor RAPL is available, cannot get meaning full results.\n"
            "Might need to run this if on linux: `sudo chmod -R a+r /sys/class/powercap/intel-rapl`"
        )

    # Run one of the models and track energy
    with EmissionsTracker(output_file=f'{args.tag}.csv', output_dir=out_dir):
        if args.model == 1:
            model1(utt2wav, utt2text, args.epoch, num_cpu_threads=num_cpu_threads, cuda=args.gpu)
        elif args.model == 2:
            model2(utt2wav, utt2text, args.epoch, num_cpu_threads=num_cpu_threads, cuda=args.gpu)
        else:
            raise ValueError("Invalid model number")


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--model', type=int, required=True)
    parser.add_argument('--tag', type=str, required=True, help='Tag used to name the output file.')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--num_threads', type=int, default=-1,
                        help='Use this to limit the number of threads used for computation.')
    return parser.parse_args()


if __name__ == '__main__':
    main()
