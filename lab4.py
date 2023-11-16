import os
from lab2 import LabExpRunner, read_data
from torch.nn.utils import prune, parameters_to_vector
import torch.nn as nn
from typing import List, Tuple, Callable
from espnet2.st.espnet_model import ESPnetSTModel
from utils import model_size_in_bytes, sparse_model_size_in_bytes

PruneParamsType = List[Tuple[nn.Module, str]]


# =============== Module filter functions ===============
def subsampling(model: ESPnetSTModel) -> PruneParamsType:
    return [
        (model.encoder.embed.conv[0], 'weight'),
        (model.encoder.embed.conv[2], 'weight'),
        (model.encoder.embed.out[0], 'weight'),
    ]


def encoder(model: ESPnetSTModel) -> PruneParamsType:
    ret = []
    for enc in model.encoder.encoders:
        ret += [
            # attention
            (enc.self_attn.linear_q, 'weight'),
            (enc.self_attn.linear_k, 'weight'),
            (enc.self_attn.linear_v, 'weight'),
            (enc.self_attn.linear_out, 'weight'),
            (enc.self_attn.linear_pos, 'weight'),

            # feed forward
            (enc.feed_forward.w_1, 'weight'),
            (enc.feed_forward.w_2, 'weight'),
            (enc.feed_forward_macaron.w_1, 'weight'),
            (enc.feed_forward_macaron.w_2, 'weight'),

            # conv
            (enc.conv_module.pointwise_conv1, 'weight'),
            (enc.conv_module.pointwise_conv2, 'weight'),
            (enc.conv_module.depthwise_conv, 'weight'),
        ]

    return ret + subsampling(model)


# =======================================================


def l1_unstructured(
        param_filter: Callable[[ESPnetSTModel], PruneParamsType],
        amount=0.33
) -> Tuple[str, Callable[[ESPnetSTModel], PruneParamsType]]:
    def func(model: ESPnetSTModel):
        params = param_filter(model)
        prune.global_unstructured(
            params,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )

        # Make pruning permanent
        # print(model.state_dict().keys())
        for module, name in params:
            prune.remove(module, name)
        # print(model.state_dict().keys())

        return params

    return f'l1_u_{param_filter.__name__}_{amount}', func


PRUNING_METHODS = [
    l1_unstructured(encoder, amount=0.33),
]


def main():
    # Must be running in the directory that contains 'exp' and 'data' directories
    if not os.path.isdir('exp') or not os.path.isdir('data'):
        raise RuntimeError("Please run this script in `pretrained` directory")

    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    runner = LabExpRunner()  # <---

    pretrained_model = "exp/st_train_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp/valid.acc.ave_10best.pth"
    pretrained_config = "exp/st_train_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp/config.yaml"

    # Load test data (520 utterances out of MUST_C_v2 TST-COMMON subset)
    utt2wav, utt2text = read_data(args.data_dir)

    # Results of the original pretrained model
    orig_p = runner.create_inference_pipeline(pretrained_model, pretrained_config, quantized=False)
    # runner.run_benchmark(
    #     orig_p, 'original', args.out_dir, utt2wav, utt2text, num_utts=args.num_test_utts, calculate_flops=False
    # )
    orig_size = model_size_in_bytes(orig_p.st_model)

    # Pruning and run benchmarks
    for name, func in PRUNING_METHODS:
        # Create a new copy each time
        p = runner.create_inference_pipeline(pretrained_model, pretrained_config, quantized=False)

        print(f'\nBenchmarking {name}...')
        pruned_modules = func(p.st_model)

        runner.run_benchmark(
            p, name, args.out_dir, utt2wav, utt2text, num_utts=args.num_test_utts, calculate_flops=False
        )

        # must be after benchmarking since we change some tensors to sparse
        new_size = sparse_model_size_in_bytes(p.st_model, [p[0] for p in pruned_modules])
        print(f'Model size: {new_size / orig_size:.2f} ({new_size}/{orig_size})')


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--num_test_utts', type=int, default=100)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
