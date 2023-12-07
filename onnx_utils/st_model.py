import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import onnxruntime as ort

import numpy as np
from typeguard import check_argument_types

from onnx_utils.abs_st_model import AbsSTModel
from espnet_onnx.asr.beam_search.hyps import Hypothesis


class Speech2Text(AbsSTModel):
    def __init__(
            self,
            tag_name: str = None,
            model_dir: Union[Path, str] = None,
            providers: List[str] = ["CPUExecutionProvider"],
            use_quantized: bool = False,
            session_option_dict: Optional[dict] = None
    ):
        assert check_argument_types()
        self._check_argument(tag_name, model_dir)
        self._load_config()

        # check onnxruntime version and providers
        self._check_ort_version(providers)

        # check if model is exported for streaming.
        if self.config.encoder.enc_type == "ContextualXformerEncoder":
            raise RuntimeError(
                "Onnx model is built for streaming. Use StreamingSpeech2Text instead."
            )

        # check quantize and optimize model
        self._check_flags(use_quantized)
        self._build_model(providers, use_quantized)

        self.start_idx = 1
        self.last_idx = -1

        # Setting onnxruntime inference session options
        encoder_opt = self.encoder.encoder.get_session_options()
        decoder_opt = self.decoder.decoder.get_session_options()
        for k, v in session_option_dict.items():
            print(f'Setting onnx inference session option `{k}` to `{v}`')
            setattr(encoder_opt, k, v)
            setattr(decoder_opt, k, v)

    def __call__(
            self, speech: np.ndarray
    ) -> List[Tuple[Optional[str], List[str], List[int], Union[Hypothesis],]]:
        """Inference
        Args:
            data: Input speech data
        Returns:
            text, token, token_int, hyp
        """
        assert check_argument_types()

        # check dtype
        if speech.dtype != np.float32:
            speech = speech.astype(np.float32)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech[np.newaxis, :]
        # lengths: (1,)
        lengths = np.array([speech.shape[1]]).astype(np.int64)

        # b. Forward Encoder
        enc, _ = self.encoder(speech=speech, speech_length=lengths)
        if isinstance(enc, tuple):
            enc = enc[0]
        assert len(enc) == 1, len(enc)

        nbest_hyps = self.beam_search(enc[0])[:1]

        results = []
        for hyp in nbest_hyps:
            # remove sos/eos and get results
            if self.last_idx is not None:
                token_int = list(hyp.yseq[self.start_idx: self.last_idx])
            else:
                token_int = list(hyp.yseq[self.start_idx:])

            # remove blank symbol id, which is assumed to be 0
            token_int = list([int(i) for i in filter(lambda x: x != 0, token_int)])

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

        return results
