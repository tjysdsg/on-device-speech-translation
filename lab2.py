from benchmarking import test_st_model, BenchmarkResult, calc_flops
import os
import json
import pickle
import torch
from dataclasses import asdict
from utils import modify_model_config, copy_state_dict
from typing import Callable, Literal, Optional, List
# from espnet2.bin.st_inference import Speech2Text
from st_inference import Speech2Text


def save_exp_statistics(result: BenchmarkResult, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)


def read_data(data_dir):
    with open(os.path.join(data_dir, 'utt2wav.pkl'), 'rb') as f:
        utt2wav = pickle.load(f)

    with open(os.path.join(data_dir, 'utt2text.json'), 'r', encoding='utf-8') as f:
        utt2text = json.load(f)

    return utt2wav, utt2text


# =========== Model config modifiers ===========

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


def input_size1(config: dict):
    config['frontend_conf']['hop_length'] = 200
    return config


def input_size2(config: dict):
    config['frontend_conf']['hop_length'] = 250
    return config


def input_size3(config: dict):
    config['frontend_conf']['hop_length'] = 300
    return config


MODEL_CONFIG_MODIFIERS = [encoder1, encoder2, decoder1, decoder2, input_size1, input_size2, input_size3]
INPUT_SIZE_MODIFIERS = [input_size1, input_size2, input_size3]


# ==============================================


class LabExpRunner:
    def __init__(self, device='cpu'):
        self.device = device

    def create_inference_pipeline(
            self,
            model_file: str,
            config_file: str,
            beam_size: int = 10,
            quantized: bool = False,
            **kwargs,
    ):
        # Check decode settings from:
        #    https://github.com/espnet/espnet/blob/master/egs2/must_c_v2/st1/conf/tuning/decode_st_conformer.yaml
        quantize_modules = None
        quantize_dtype = None
        if quantized:
            quantize_modules = ['Linear']
            quantize_dtype = 'qint8'

        pipeline = Speech2Text(
            st_model_file=model_file,
            st_train_config=config_file,
            beam_size=beam_size,
            device=self.device,
            quantized=quantized,
            quantize_modules=quantize_modules,
            quantize_dtype=quantize_dtype,
            **kwargs,
        )
        # print(pipeline.st_model)
        # from torchinfo import summary
        # print(summary(pipeline.st_model))
        return pipeline

    def resize_model(
            self,
            orig_model: torch.nn.Module,
            pretrained_config: str,
            model_config_modifier: Callable[[dict], dict],
            quantized: bool = False,
    ) -> Speech2Text:
        """
        1. Load config.yaml, change some settings, and save it to a new file
        3. Initialize an empty ST model using the new yaml file
        4. Copy (part of) original model weights to this new model and save it to a new checkpoint
        """

        new_config_file = "new_config.yaml"
        modify_model_config(pretrained_config, new_config_file, modifier=model_config_modifier)

        # Build an empty ST model using the new config
        from espnet2.tasks.st import STTask
        new_model, _ = STTask.build_model_from_file(
            new_config_file, device=self.device
        )

        new_state_dict = copy_state_dict(new_model.state_dict(), orig_model.state_dict())
        torch.save(new_state_dict, 'new_model.pth')

        # Construct a new inference pipeline using the new weights
        pipeline = self.create_inference_pipeline(
            'new_model.pth', new_config_file, quantized=quantized
        )
        return pipeline

    def run_benchmark(
            self,
            pipeline: Speech2Text,
            tag: str,
            out_dir: str,
            utt2wav: dict,
            utt2text: dict,
            calculate_flops=True,
            num_utts=20,
    ):
        # Run the new model
        result = test_st_model(pipeline, utt2wav, utt2text, num_utts)
        if calculate_flops:
            result.flop = calc_flops(pipeline, utt2wav, num_utts)
        save_exp_statistics(result, os.path.join(out_dir, f'{tag}.json'))


def main():
    # Must be running in the directory that contains 'exp' and 'data' directories
    if not os.path.isdir('exp') or not os.path.isdir('data'):
        raise RuntimeError("Please run this script in `pretrained` directory")

    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    runner = LabExpRunner()

    # ==== Load Original Pre-trained Model ====
    pretrained_model = "exp/st_train_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp/valid.acc.ave_10best.pth"
    pretrained_config = "exp/st_train_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp/config.yaml"
    pipeline = runner.create_inference_pipeline(pretrained_model, pretrained_config)

    # Load test data (520 utterances out of MUST_C_v2 TST-COMMON subset)
    utt2wav, utt2text = read_data(args.data_dir)

    # Results of the original pretrained model
    runner.run_benchmark(
        pipeline, 'original', args.out_dir, utt2wav, utt2text, num_utts=args.num_test_utts, calculate_flops=False
    )

    # Change model size and run benchmarks
    for m in MODEL_CONFIG_MODIFIERS:
        p = runner.resize_model(pipeline.st_model, pretrained_config, m)
        runner.run_benchmark(
            p, m.__name__, args.out_dir, utt2wav, utt2text, num_utts=args.num_test_utts, calculate_flops=False
        )


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--num_test_utts', type=int, default=20)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
