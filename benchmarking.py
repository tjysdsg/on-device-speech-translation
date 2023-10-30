import time
from sacrebleu.metrics import BLEU
from dataclasses import dataclass
from espnet2.bin.st_inference import Speech2Text


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


def calc_flops(
        st_pipeline: Speech2Text,
        utt2wav,
        num_utts=-1,  # number of utterances used for testing if positive
):
    from deepspeed.profiling.flops_profiler import FlopsProfiler

    test_data = list(utt2wav.items())
    if num_utts > 0:
        test_data = test_data[:num_utts]

    prof = FlopsProfiler(st_pipeline.st_model)
    prof.start_profile()
    for i, (utt, speech) in enumerate(test_data):
        st_pipeline(speech)
    prof.stop_profile()

    return int(prof.get_total_flops() / len(test_data))
