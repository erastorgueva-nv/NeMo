# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import torch
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.inference.utils.text_segment import Word

_whisper_normalizer = EnglishTextNormalizer()


def clean_pred_text(text: str) -> str:
    """Clean prediction text for fair WER comparison.

    First strips model-specific tokens (turn markers, timestamps, pad tokens)
    that the Whisper normalizer doesn't know about, then applies
    ``EnglishTextNormalizer`` — the same normalizer used by the offline eval
    metrics in ``speechlm2.parts.metrics.wer``.
    """
    if not text:
        return ""
    # Strip model-specific tokens
    text = text.lstrip('^')
    text = re.sub(r'</?s>', '', text)
    text = re.sub(r'<\$[\d.]+\$>', '', text)
    text = re.sub(r'<\|[\d.]+\|>', '', text)
    text = re.sub(r'<SPECIAL_12>', '', text)
    # Normalize with Whisper's EnglishTextNormalizer (same as offline eval)
    return _whisper_normalizer(text)


class PipelineOutput:
    """
    Class to store the output of the S2S pipeline.
    """

    def __init__(
        self,
        texts: list[str] | None = None,
        words: list[list[Word]] | None = None,
        asr_texts: list[str] | None = None,
        texts_with_timestamps: list[str] | None = None,
        asr_texts_with_timestamps: list[str] | None = None,
        raw_texts: list[str] | None = None,
        raw_asr_texts: list[str] | None = None,
        token_texts: list[torch.Tensor | None] | None = None,
        token_asr_texts: list[torch.Tensor | None] | None = None,
        token_function_texts: list[torch.Tensor | None] | None = None,
        token_lengths: list[int | None] | None = None,
        audio_filepaths: list[str | None] | None = None,
        debug_data: list[list] | None = None,
    ):
        if texts is None and words is None:
            raise ValueError("At least one of the 'texts' or 'words' should be provided.")
        self.texts = texts
        self.words = words
        self.asr_texts = asr_texts
        self.texts_with_timestamps = texts_with_timestamps
        self.asr_texts_with_timestamps = asr_texts_with_timestamps
        self.raw_texts = raw_texts
        self.raw_asr_texts = raw_asr_texts
        self.token_texts = token_texts
        self.token_asr_texts = token_asr_texts
        self.token_function_texts = token_function_texts
        self.token_lengths = token_lengths
        self.audio_filepaths = audio_filepaths
        self.debug_data = debug_data
