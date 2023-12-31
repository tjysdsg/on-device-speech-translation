import os
from lab2 import LabExpRunner
from pathlib import Path
from lab4 import optimal_config
from onnx_utils.export_st import STModelExport


def main():
    # Must be running in the directory that contains 'exp' and 'data' directories
    if not os.path.isdir('exp') or not os.path.isdir('data'):
        raise RuntimeError("Please run this script in `pretrained` directory")

    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    runner = LabExpRunner()  # <---

    # Prune
    pretrained_model = "exp/st_train_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp/valid.acc.ave_10best.pth"
    pretrained_config = "exp/st_train_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp/config.yaml"
    p = runner.create_inference_pipeline(pretrained_model, pretrained_config, quantized=False)

    _, func = optimal_config()
    func(p.st_model)

    m = STModelExport(
        # cache_dir=out_dir,  # FIXME: https://github.com/espnet/espnet_onnx/issues/101
    )
    m.export(
        p,
        tag_name='pruned_onnx',
        quantize=False,
        optimize=False,
    )

    # res_dir = Path.home() / ".cache" / "espnet_onnx"
    # shutil.copytree(res_dir / 'pruned_onnx', Path(out_dir) / 'pruned_onnx')


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Export ONNX model to `~.cache/espnet_onnx/{tag}/`.'
                                        'Can load the model using the same tag name.')
    parser.add_argument('--out_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
