import os
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import json
import os
import matplotlib.pyplot as plt
from typing import Dict, List
from benchmarking import BenchmarkResult

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
mpl.rc("savefig", dpi=300)

import seaborn as sns

sns.set_theme()  # sns.set_style('whitegrid')


def _plot(ax, x: list, y: list, annotation: List[str] = None, label: str = None):
    ax.scatter(x, y, marker='^', label=label)

    if annotation is not None:
        for i, j, a in zip(x, y, annotation):
            ax.annotate(f'{a}', (i, j), fontsize=7)


def plot_benchmark_result(results: Dict[str, Dict[str, BenchmarkResult]], out_dir: str, annotate=True, fit_curve=False):
    ax = plt.subplot()

    all_latency = []
    all_bleu = []

    # LATENCY - BLEU
    for exp_set, R in results.items():
        latency = []
        bleu = []
        annotations = []

        for tag, r in R.items():
            latency.append(r.latency)
            bleu.append(r.bleu)
            annotations.append(tag)

        if not annotate:
            annotations = None
        _plot(ax, latency, bleu, annotations, label=exp_set)

        all_latency += latency
        all_bleu += bleu

    ax.set_xlabel('latency (s)')
    ax.set_ylabel('BLEU score')
    ax.set_title('BLEU score vs latency')

    if fit_curve:
        poly = Polynomial.fit(all_latency, all_bleu, 2)
        x = np.linspace(min(all_latency), max(all_latency), 1000)
        y = poly(x)

        ax.plot(x, y, color='r', alpha=0.5, label='Polynomial Fit')

    plt.legend()
    plt.savefig(os.path.join(out_dir, 'BLEU_vs_latency.png'))
    plt.close('all')


def main():
    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for subfolder in os.listdir(args.result_dir):
        sf_path = os.path.join(args.result_dir, subfolder)
        if not os.path.isdir(sf_path):
            continue

        results.setdefault(subfolder, dict())
        for file in os.listdir(sf_path):
            tag = file.split('.')[0]
            with open(os.path.join(sf_path, file), 'r', encoding='utf-8') as f:
                r = json.load(f)
                results[subfolder][tag] = BenchmarkResult(**r)

    plot_benchmark_result(results, out_dir, annotate=not args.no_annotation, fit_curve=args.fit_curve)


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--fit_curve', '-f', action='store_true')
    parser.add_argument('--no_annotation', '-na', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
