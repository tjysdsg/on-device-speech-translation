from benchmarking import BenchmarkResult
import os
import json
from utils import plot_benchmark_result


def main():
    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    flop = {}
    latency = {}
    bleu = {}
    for path in os.listdir(args.result_dir):
        if not path.endswith('.json'):
            continue

        tag = path.split('.')[0]
        with open(os.path.join(args.result_dir, path), 'r', encoding='utf-8') as f:
            r = json.load(f)
            result = BenchmarkResult(**r)

            flop[tag] = result.flop
            latency[tag] = result.latency
            bleu[tag] = result.bleu

    # print(flop, latency, bleu)
    plot_benchmark_result(
        list(bleu.values()),
        list(latency.values()),
        list(flop.values()),
        out_dir,
    )


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
