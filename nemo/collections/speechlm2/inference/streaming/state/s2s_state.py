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

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from nemo.collections.asr.inference.utils.text_segment import Word
from nemo.utils import logging


@dataclass
class S2SStreamingState:
    """Pipeline-level output accumulator for a single S2S stream.

    Collects generated audio samples, text strings, and word timings
    produced by each inference step.  Kept alive by the pipeline's
    ``_state_pool`` until ``close_session()`` so the final
    :class:`PipelineOutput` can be assembled.

    This is *not* the model-level decode state (KV caches, token
    workspaces) -- that is :class:`StreamingDecodeState` in
    ``model_wrappers/decode_state.py``.
    """

    # Required init metadata
    device: torch.device
    dtype: torch.dtype
    output_sample_rate: int

    # Growing audio buffer — shape (1, T), appended each step
    audio_buffer: torch.Tensor = field(init=False)

    # Accumulated agent response text (built incrementally per step)
    output_text_str: str = ""
    # Accumulated ASR (user) text
    output_asr_text_str: str = ""
    # Word-level timings for the agent response
    output_words: List[Word] = field(default_factory=list)

    # Snapshots of full token-ID tensors, saved from StreamingDecodeState
    # before the decode context is destroyed at end-of-stream.
    # Used for post-hoc tokens_to_str conversion.
    final_gen_text: Optional[torch.Tensor] = None
    final_gen_asr_text: Optional[torch.Tensor] = None
    final_gen_function_text: Optional[torch.Tensor] = None
    final_total_frames: int = 0

    def __post_init__(self) -> None:
        # Depends on self.device and self.dtype, so can't be a field default.
        self.audio_buffer = torch.empty((1, 0), device=self.device, dtype=self.dtype)

    def reset(self) -> None:
        """Reset all accumulated outputs to initial state."""
        self.audio_buffer = torch.empty((1, 0), device=self.device, dtype=self.dtype)
        self.output_text_str = ""
        self.output_asr_text_str = ""
        self.output_words.clear()
        self.final_gen_text = None
        self.final_gen_asr_text = None
        self.final_gen_function_text = None
        self.final_total_frames = 0

    def append_step_output(
        self,
        audio: torch.Tensor,
        text: str | None = None,
        asr_text: str | None = None,
    ) -> None:
        """Append generated audio and optional text from one inference step."""
        if audio is None:
            return
        if not isinstance(audio, torch.Tensor):
            raise TypeError("audio must be a torch.Tensor")

        append_tensor = audio
        if append_tensor.dim() > 1:
            append_tensor = append_tensor.reshape(1, -1)
        elif append_tensor.dim() == 1:
            append_tensor = append_tensor.unsqueeze(0)
        prior_samples = int(self.audio_buffer.shape[-1])
        appended_samples = int(append_tensor.shape[-1])
        self.audio_buffer = torch.cat(
            [self.audio_buffer, append_tensor.to(self.device, dtype=self.dtype)], dim=-1
        )

        if isinstance(text, str) and text:
            self.output_text_str += text
            if appended_samples > 0 and self.output_sample_rate > 0:
                start_t = float(prior_samples) / float(self.output_sample_rate)
                end_t = float(prior_samples + appended_samples) / float(self.output_sample_rate)
                self.output_words.append(Word(text=text, start=start_t, end=end_t, conf=1.0))

        if isinstance(asr_text, str) and asr_text:
            self.output_asr_text_str += asr_text

    def save_token_tensors(
        self,
        gen_text: torch.Tensor,
        gen_asr_text: torch.Tensor,
        total_frames: int,
        gen_function_text: Optional[torch.Tensor] = None,
    ) -> None:
        """Snapshot the full token-ID tensors from the decode context before it is destroyed."""
        self.final_gen_text = gen_text[:, :total_frames].clone().cpu()
        self.final_gen_asr_text = gen_asr_text[:, :total_frames].clone().cpu()
        self.final_total_frames = total_frames
        self.final_gen_function_text = (
            gen_function_text[:, :total_frames].clone().cpu()
            if gen_function_text is not None else None
        )

    def get_token_tensors(self) -> Optional[tuple]:
        """Return (gen_text, gen_asr_text, total_frames, gen_function_text) or None if not saved."""
        if self.final_gen_text is None:
            return None
        return self.final_gen_text, self.final_gen_asr_text, self.final_total_frames, self.final_gen_function_text

    def clear_audio_buffer(self) -> None:
        """Clear the audio buffer (e.g. after sending audio to a client)."""
        self.audio_buffer = torch.empty((1, 0), device=self.device, dtype=self.dtype)
