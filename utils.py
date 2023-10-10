import yaml
import json
import os
import matplotlib.pyplot as plt


def modify_model_config(config_path: str, new_config_path: str, modifier):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config = modifier(config)

    with open(new_config_path, "w", encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)


def copy_state_dict(dst_state, src_state):
    """Copy weights to the new model and save to file
    This works as long as the new model has fewer/equal number of layers or smaller/same layer sizes
    """

    ret = {}
    for key, value in src_state.items():
        if key in dst_state and (dst_state[key].size() == src_state[key].size()):
            ret[key] = value
        elif key not in dst_state:
            print(f"Skipping `{key}` from pretrained dict because of it's not found in target dict")
        else:
            dst_size = dst_state[key].shape
            src_size = src_state[key].shape
            n = len(dst_size)
            assert n == len(src_size)

            idx = []
            for ss, ds in zip(src_size, dst_size):
                assert ss >= ds, f'{src_size} => {dst_size} for layer {key}'
                idx.append(slice(ds))

            print(f"Changing layer size of `{key}` from {src_size} to {dst_size}")
            ret[key] = src_state[key][idx]
    return ret


def plot_benchmark_result(bleu: list, latency: list, flop: list, out_dir: str):
    plt.plot(flop, bleu)
    plt.xlabel('FLOPs')
    plt.ylabel('BLEU score')
    plt.title('BLEU score vs FLOPs')
    plt.savefig(os.path.join(out_dir, 'BLEU_vs_FLOP.png'))
    plt.close('all')

    plt.plot(latency, bleu)
    plt.xlabel('latency (s)')
    plt.ylabel('BLEU score')
    plt.title('BLEU score vs latency')
    plt.savefig(os.path.join(out_dir, 'BLEU_vs_latency.png'))
    plt.close('all')

    plt.plot(flop, latency)
    plt.xlabel('FLOPs')
    plt.ylabel('latency (s)')
    plt.title('latency (s) vs FLOPs')
    plt.savefig(os.path.join(out_dir, 'latency_vs_FLOP.png'))
    plt.close('all')