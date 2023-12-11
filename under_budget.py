import os
import torch
from lab2 import LabExpRunner, read_data, PRETRAINED_MODEL, PRETRAINED_CONFIG
from onnx_utils.st_model import Speech2Text
from lab4_benchmark_onnx import PROBLEM_MATIC_UTTS
from codecarbon import EmissionsTracker
from codecarbon.core import cpu, gpu
import json


class UnderBudgetExp:
    def __init__(self, max_watt_hour: float, check_power_interval=1):
        self.check_power_interval = check_power_interval
        self.max_watt_hour = max_watt_hour
        self.should_stop = False
        self.using_gpu = False

        self.reset()

    def check_max_power(self):
        if self.using_gpu:
            watt_hour = self.tracker._total_gpu_energy.kWh * 1000
        else:
            watt_hour = self.tracker._total_cpu_energy.kWh * 1000

        print(f'Current energy consumption: {watt_hour} Wh')
        self.should_stop = watt_hour >= self.max_watt_hour

    def model1(self, speech, num_cpu_threads: int, cuda: bool):
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

        while not self.should_stop:
            pipeline(speech)
            self.check_max_power()
            self.iterations += 1

    def model2(self, speech, num_cpu_threads: int, cuda: bool):
        execution_provider = 'CPUExecutionProvider'
        if cuda:
            execution_provider = 'CUDAExecutionProvider'

        p = Speech2Text(
            tag_name='q1_p1',
            use_quantized=True,
            providers=[execution_provider],
            session_option_dict=dict(
                intra_op_num_threads=num_cpu_threads,
                inter_op_num_threads=num_cpu_threads,
            ),
        )

        while not self.should_stop:
            p(speech)
            self.check_max_power()
            self.iterations += 1

    def __call__(
            self,
            model: int,
            speech,
            num_cpu_threads: int,
            cuda: bool,
    ):
        self.using_gpu = cuda

        self.tracker.start()

        if model == 1:
            self.model1(speech, num_cpu_threads=num_cpu_threads, cuda=cuda)
        elif model == 2:
            self.model2(speech, num_cpu_threads=num_cpu_threads, cuda=cuda)
        else:
            raise ValueError("Invalid model number")

        self.tracker.stop()
        return self.iterations

    def reset(self):
        self.iterations = 0
        self.tracker = EmissionsTracker(save_to_file=False, measure_power_secs=self.check_power_interval)
        self.should_stop = False


def main():
    # Must be running in the directory that contains 'exp' and 'data' directories
    if not os.path.isdir('exp') or not os.path.isdir('data'):
        raise RuntimeError("Please run this script in `pretrained` directory")

    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    num_cpu_threads = args.num_threads
    if num_cpu_threads < 0:
        num_cpu_threads = 0

    assert torch.cuda.is_available()

    # Load test data (520 utterances out of MUST_C_v2 TST-COMMON subset)
    utt2wav, utt2text = read_data(args.data_dir)
    for utt in PROBLEM_MATIC_UTTS:
        utt2wav.pop(utt)
    test_data = list(utt2wav.items())
    speech = test_data[0][1]

    # Check if we have the tools to get meaningful power consumption readings
    assert gpu.is_gpu_details_available()
    assert cpu.is_powergadget_available() or cpu.is_rapl_available()

    # Run one of the models and track energy
    exp = UnderBudgetExp(10)

    result = {}
    for model in [1, 2]:
        for cuda in [True, False]:
            name = f'model{model}_'
            name += 'gpu' if cuda else 'cpu'

            n = exp(
                model,
                speech,
                num_cpu_threads,
                cuda,
            )

            print(f'Number of iterations of {name}: {n}')
            result[name] = n

            exp.reset()

    with open(os.path.join(out_dir, 'under_10Wh.json'), 'w') as f:
        json.dump(result, f)


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--num_threads', type=int, default=-1,
                        help='Use this to limit the number of threads used for computation.')
    return parser.parse_args()


if __name__ == '__main__':
    main()
