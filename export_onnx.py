import os
from lab2 import LabExpRunner
from pathlib import Path
from lab4 import optimal_config as optimal_pruning
from onnx_utils.export_st import STModelExport


def main():
    # Must be running in the directory that contains 'exp' and 'data' directories
    if not os.path.isdir('exp') or not os.path.isdir('data'):
        raise RuntimeError("Please run this script in `pretrained` directory")

    args = get_args()

    runner = LabExpRunner()

    pretrained_model = "exp/st_train_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp/valid.acc.ave_10best.pth"
    pretrained_config = "exp/st_train_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp/config.yaml"
    p = runner.create_inference_pipeline(pretrained_model, pretrained_config, quantized=False)

    # Prune
    if args.pruned:
        _, func = optimal_pruning()
        func(p.st_model)

    tag = f'q{int(args.quantized)}_p{int(args.pruned)}'
    print(f"Exporting ONNX model with tag: {tag}")

    m = STModelExport()
    m.export(
        p,
        tag_name=tag,
        quantize=args.quantized,  # Quantize
        optimize=False,
    )


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Export ONNX model to `~.cache/espnet_onnx/{tag}/`.'
                                        'Can load the model using the same tag name.')
    parser.add_argument('--quantized', '-q', action='store_true')
    parser.add_argument('--pruned', '-p', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
