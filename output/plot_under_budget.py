import matplotlib.pyplot as plt
from typing import Dict, List
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
mpl.rc("savefig", dpi=300)

import seaborn as sns

sns.set_theme()  # sns.set_style('whitegrid')

n_iters_10Wh = {
    "original_gpu": 1967,
    "original_cpu": 1787,
    "final_gpu": 2570,
    "final_cpu": 4057
}
BLEU = {
    "original_gpu": 56.6,
    "original_cpu": 56.6,
    "final_gpu": 42.7,
    "final_cpu": 42.7
}


def _plot(ax, x: list, y: list, annotation: List[str] = None, label: str = None):
    ax.scatter(x, y, marker='^', label=label)

    if annotation is not None:
        for i, j, a in zip(x, y, annotation):
            ax.annotate(f'{a}', (i, j), fontsize=7)


def main():
    ax = plt.subplot()

    # LATENCY - BLEU
    for tag in n_iters_10Wh.keys():
        ax.scatter([n_iters_10Wh[tag]], [BLEU[tag]], marker='^', label=tag)

    ax.set_xlabel('Number of iterations ran under 10Wh')
    ax.set_ylabel('BLEU score')

    plt.legend()
    plt.savefig('under_10Wh.png')
    plt.close('all')


if __name__ == '__main__':
    main()
