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

"""State and result types for streaming S2S inference.

These dataclasses define the *model wrapper's* interface contract:

* :class:`StreamingDecodeState` — mutable per-stream decode state
  (KV caches, token workspaces, perception/codec caches).  Created by
  the wrapper, mutated in-place by ``infer_one_step``, and held between
  steps by the pipeline's context manager.

* :class:`InferenceStepResult` — immutable per-step outputs returned
  by ``infer_one_step`` (predicted tokens, text strings, audio).

Defined here (in ``model_wrappers/``) because the wrapper is the
component that knows what state it needs.  The context manager and
pipeline import from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from nemo.collections.speechlm2.inference.model_wrappers.perception_cache import PerceptionCacheState


@dataclass
class StreamingDecodeState:
    """Per-stream model-level decode state for streaming S2S inference.

    Holds KV caches, token workspaces, perception cache, and codec state
    that persist across inference steps within a single stream.
    """

    frame_idx: int
    gen_text: torch.Tensor
    gen_asr_text: torch.Tensor
    gen_function_text: torch.Tensor | None
    input_embeds_history: list[torch.Tensor]
    llm_cache: Any  # DynamicCache or HybridMambaAttentionDynamicCache
    tts_past_key_values: Any
    tts_code: torch.Tensor | None
    subword_mask: torch.Tensor | None
    perception_cache: "PerceptionCacheState" | None = None
    tts_codec_cache: Any = None
    llm_cache_position_offset: int = 0


@dataclass
class InferenceStepResult:
    """Output from a single ``infer_one_step`` call.

    State mutations (caches, token workspaces, frame_idx) are applied
    in-place on :class:`StreamingDecodeState`.  This dataclass carries
    only the per-step *outputs* needed by the pipeline.
    """

    predicted_text_tokens: torch.Tensor
    asr_predicted_text_tokens: torch.Tensor
    predicted_text_strs: list[str]
    asr_predicted_text_strs: list[str]
    decoded_audio: torch.Tensor | None = None
    function_predicted_text_tokens: torch.Tensor | None = None
    debug: dict | None = None
