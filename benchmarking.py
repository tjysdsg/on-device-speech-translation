from kaldiio import ReadHelper
import time
import torch
import string
from espnet2.bin.st_inference import Speech2Text
import os
import json
from typing import List
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from sacrebleu.metrics import BLEU
import yaml
from dataclasses import dataclass, asdict


def init_inference_pipeline(
        model_file: str,
        config_file: str,
        beam_size=10,
        **kwargs,
):
    # a wrapper responsible for calling the actual pytorch model
    return Speech2Text(
        st_model_file=model_file,
        st_train_config=config_file,
        beam_size=beam_size,
        device="cpu",
        **kwargs,
    )


@dataclass
class BenchmarkResult:
    flop: float = 0
    latency: float = 0
    bleu: float = 0


def test_st_model(
        st_pipeline: Speech2Text,
        utt2wav: dict,
        utt2text: dict,
        num_utts=-1,  # number of utterances used for testing if positive
) -> BenchmarkResult:
    bleu = BLEU()

    test_data = list(utt2wav.items())
    if num_utts > 0:
        test_data = test_data[:num_utts]

    preds = []
    refs = []
    total_time = 0
    for i, (utt, speech) in enumerate(test_data):
        # ====================
        start = time.perf_counter()

        text, tokens, _, _ = st_pipeline(speech)[0]

        end = time.perf_counter()
        if i > 0:  # skip the first warm-up sample
            total_time += end - start
        # ====================

        # FIXME: somehow this model's tokenizer
        #   cannot convert tokens back to text properly
        text = ''.join(tokens).replace('▁', ' ')

        preds.append(text)
        tgt_text = utt2text[utt]['tgt_text']
        refs.append([tgt_text])

        print(f"Reference source text: {utt2text[utt]['src_text']}")
        print(f"Translation results: {text}")

        print(f"Reference target text: {tgt_text}")
        print(f"Sentence BLEU Score: {bleu.sentence_score(text, [tgt_text])}")
        print("-" * 50)

    res = bleu.corpus_score(preds, refs)
    print(f'BLEU score of {len(test_data)} utterances is: {res}')

    avg_latency = total_time / (len(test_data) - 1)
    print(f'Average inference latency is: {avg_latency}s')

    return BenchmarkResult(latency=avg_latency, bleu=res.score)


def plot(path):
    assert (os.path.exists(path))
    fig_path_root = path[:-5]
    with open(path, 'r') as f:
        stat = json.load(f)
        assert (len(stat['score']) == len(stat['latency']))
        assert (len(stat['score']) == len(stat['flop']))
    plt.plot(stat['flop'], stat['score'])
    plt.xlabel('FLOPs')
    plt.ylabel('BLEU score')
    plt.title('BLEU score vs FLOPs')
    plt.savefig(fig_path_root + '_BLUE_vs_FLOP.png')
    plt.show()

    plt.plot(stat['latency'], stat['score'])
    plt.xlabel('latency (s)')
    plt.ylabel('BLEU score')
    plt.title('BLEU score vs latency')
    plt.savefig(fig_path_root + '_BLUE_vs_latency.png')
    plt.show()

    plt.plot(stat['flop'], stat['latency'])
    plt.xlabel('FLOPs')
    plt.ylabel('latency (s)')
    plt.title('latency (s) vs FLOPs')
    plt.savefig(fig_path_root + '_latency_vs_FLOP.png')
    plt.show()
