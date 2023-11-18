import numpy as np
import onnxruntime as ort
import copy
import time
import torch
import string
from st import Speech2Text
from espnet2.asr.transducer.beam_search_transducer import Hypothesis as TransHypothesis
from espnet.nets.beam_search import BeamSearch, Hypothesis


def run_onnx_model(pipeline_original, input_node_name0, input_node_name1, model, ouput_node_name, speech):
    speech = speech.unsqueeze(0)
    length = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
    a, b = pipeline_original.st_model._extract_feats(speech, length)
    x = model.run(output_names=ouput_node_name,
                  input_feed={input_node_name0: a.numpy(), input_node_name1: b.numpy()})
    nbest_hyps = pipeline.beam_search(
        x=torch.tensor(x[0]), maxlenratio=pipeline.maxlenratio, minlenratio=pipeline.minlenratio
    )
    results = []
    for hyp in nbest_hyps:
        assert isinstance(hyp, (Hypothesis, TransHypothesis)), type(hyp)

        # remove sos/eos and get results
        last_pos = None if pipeline_original.st_model.st_use_transducer_decoder else -1
        if isinstance(hyp.yseq, list):
            token_int = hyp.yseq[1:last_pos]
        else:
            token_int = hyp.yseq[1:last_pos].tolist()

        # remove blank symbol id, which is assumed to be 0
        token_int = list(filter(lambda x: x != 0, token_int))

        # Change integer-ids to tokens
        token = pipeline_original.converter.ids2tokens(token_int)

        if pipeline_original.tokenizer is not None:
            text = pipeline_original.tokenizer.tokens2text(token)
        else:
            text = None
        results.append((text, token, token_int, hyp))
    return results[0]


pretrained_model = "exp/st_train_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp/valid.acc.ave_10best.pth"
pretrained_config = "exp/st_train_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp/config.yaml"

pipeline = init_inference_pipeline(
    pretrained_model, pretrained_config,
)

print(pipeline.st_model)

pipeline_bak = copy.deepcopy(pipeline)

onnx_name = 'tmp.onnx'
speech = torch.randn([32000])
input_node_name0, input_node_name1, model, ouput_node_name = get_onnx(speech, pipeline, pipeline_bak, onnx_name)
speech = torch.randn([32000])
results = run_onnx_model(pipeline, input_node_name0, input_node_name1, model, ouput_node_name, speech)
