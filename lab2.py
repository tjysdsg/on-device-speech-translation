from benchmarking import init_inference_pipeline, test_st_model, BenchmarkResult
import os
import json
import pickle
import torch
from dataclasses import asdict
from utils import modify_model_config, copy_state_dict
from typing import Callable


def save_exp_statistics(result: BenchmarkResult, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)


def read_data(data_dir):
    with open(os.path.join(data_dir, 'utt2wav.pkl'), 'rb') as f:
        utt2wav = pickle.load(f)

    with open(os.path.join(data_dir, 'utt2text.json'), 'r', encoding='utf-8') as f:
        utt2text = json.load(f)

    return utt2wav, utt2text


def main():
    # Must be running in the directory that contains 'exp' and 'data' directories
    if not os.path.isdir('exp') or not os.path.isdir('data'):
        raise RuntimeError("Please run this script in `pretrained` directory")

    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    """Load Original Pre-trained Model"""
    pretrained_model = "exp/st_train_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp/valid.acc.ave_10best.pth"
    pretrained_config = "exp/st_train_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp/config.yaml"
    pipeline = init_inference_pipeline(
        pretrained_model, pretrained_config,
        # Check decode settings from:
        #    https://github.com/espnet/espnet/blob/master/egs2/must_c_v2/st1/conf/tuning/decode_st_conformer.yaml
    )
    # print(pipeline.st_model)

    # Load test data (520 utterances out of MUST_C_v2 TST-COMMON subset)
    utt2wav, utt2text = read_data(args.data_dir)

    # Results of the original pretrained model
    result = test_st_model(pipeline, utt2wav, utt2text, num_utts=args.num_test_utts)
    save_exp_statistics(result, os.path.join(out_dir, 'original.json'))

    # Change model size and run benchmarks
    def decoder1(config: dict):
        config['decoder_conf']['num_blocks'] = 4
        return config

    def decoder2(config: dict):
        config['decoder_conf']['attention_heads'] = 2
        config['decoder_conf']['linear_units'] = 1024
        return config

    def encoder1(config: dict):
        config['encoder_conf']['num_blocks'] = 8
        return config

    def encoder2(config: dict):  # actually decreases decoder width as well
        config['encoder_conf']['output_size'] = 128
        config['encoder_conf']['attention_heads'] = 2
        config['encoder_conf']['linear_units'] = 1024
        return config

    benchmark_modified_model(args, 'encoder1', pipeline.st_model, pretrained_config, utt2wav, utt2text, encoder1)
    benchmark_modified_model(args, 'encoder2', pipeline.st_model, pretrained_config, utt2wav, utt2text, encoder2)
    benchmark_modified_model(args, 'decoder1', pipeline.st_model, pretrained_config, utt2wav, utt2text, decoder1)
    benchmark_modified_model(args, 'decoder2', pipeline.st_model, pretrained_config, utt2wav, utt2text, decoder2)


def benchmark_modified_model(
        args,
        tag: str,
        orig_model: torch.nn.Module,
        pretrained_config: str,
        utt2wav: dict,
        utt2text: dict,
        model_config_modifier: Callable[[dict], dict],
):
    """
    Change the model size and run benchmarks

    1. Load config.yaml, change some settings, and save it to a new file
    3. Initialize an empty ST model using the new yaml file
    4. Copy (part of) original model weights to this new model and save it to a new checkpoint
    """

    new_config_file = "new_config.yaml"
    modify_model_config(pretrained_config, new_config_file, modifier=model_config_modifier)

    # Build an empty ST model using the new config
    from espnet2.tasks.st import STTask
    new_model, _ = STTask.build_model_from_file(
        new_config_file, device='cpu'
    )
    # print(new_model)

    new_state_dict = copy_state_dict(new_model.state_dict(), orig_model.state_dict())
    torch.save(new_state_dict, 'new_model.pth')

    # Construct a new inference pipeline using the new weights
    pipeline = init_inference_pipeline(
        'new_model.pth', new_config_file,
    )
    # print(pipeline.st_model)

    # Run the new model
    result = test_st_model(pipeline, utt2wav, utt2text, num_utts=args.num_test_utts)
    save_exp_statistics(result, os.path.join(args.out_dir, f'{tag}.json'))


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--num_test_utts', type=int, default=20)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
