import os
import json
import os
import matplotlib.pyplot as plt
from typing import Dict, List
from benchmarking import BenchmarkResult


def _plot(ax, x: list, y: list, annotation: List[str], label: str = None):
    ax.scatter(x, y, marker='^', label=label)

    for i, j, a in zip(x, y, annotation):
        ax.annotate(f'{a}', (i, j), fontsize=9)


def plot_benchmark_result(results: Dict[str, BenchmarkResult], out_dir: str):
    labels = []
    flop = []
    latency = []
    bleu = []

    for tag, r in results.items():
        labels.append(tag)
        flop.append(r.flop)
        latency.append(r.latency)
        bleu.append(r.bleu)

    # FLOP - BLEU
    ax = plt.subplot()
    _plot(ax, flop, bleu, labels)
    ax.set_xlabel('FLOPs')
    ax.set_ylabel('BLEU score')
    ax.set_title('BLEU score vs FLOPs')
    plt.savefig(os.path.join(out_dir, 'BLEU_vs_FLOP.png'))
    plt.close('all')

    # LATENCY - BLEU
    ax = plt.subplot()
    _plot(ax, latency, bleu, labels)
    ax.set_xlabel('latency (s)')
    ax.set_ylabel('BLEU score')
    ax.set_title('BLEU score vs latency')
    plt.savefig(os.path.join(out_dir, 'BLEU_vs_latency.png'))
    plt.close('all')

    # FLOP - LATENCY
    ax = plt.subplot()
    _plot(ax, flop, latency, labels)
    ax.set_xlabel('FLOPs')
    ax.set_ylabel('latency (s)')
    ax.set_title('latency (s) vs FLOPs')
    plt.savefig(os.path.join(out_dir, 'latency_vs_FLOP.png'))
    plt.close('all')


def main():
    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for path in os.listdir(args.result_dir):
        if not path.endswith('.json'):
            continue

        tag = path.split('.')[0]
        with open(os.path.join(args.result_dir, path), 'r', encoding='utf-8') as f:
            r = json.load(f)
            results[tag] = BenchmarkResult(**r)

    plot_benchmark_result(results, out_dir)


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
