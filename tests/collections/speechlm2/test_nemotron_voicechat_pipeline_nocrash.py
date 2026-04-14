# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""No-crash pipeline tests for NemotronVoiceChat streaming inference.

Exercises ``StreamingS2SPipeline.run()`` with a tiny random-weight model
under various config combinations.  Each test verifies only that the
pipeline completes without raising — no output quality checks.

Run from the NeMo repo root::

    CUDA_VISIBLE_DEVICES=0 pytest tests/collections/speechlm2/test_nemotron_voicechat_pipeline_nocrash.py -v -s
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.speechlm2.inference.factory.s2s_pipeline_builder import S2SPipelineBuilder
from nemo.collections.speechlm2.inference.pipelines.streaming_s2s_pipeline import StreamingS2SPipeline

_CONF_YAML = os.path.join(
    os.path.dirname(__file__),
    "../../../examples/speechlm2/nemo_inference_pipelines/conf/s2s_streaming.yaml",
)
_MOCK_SYSTEM_PROMPT = "This is a mock prompt for the test"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_no_crash_pipeline(
    model_path: str,
    audio_path: str,
    output_dir: str,
    *,
    s2s_overrides: dict[str, Any] | None = None,
    streaming_overrides: dict[str, Any] | None = None,
) -> StreamingS2SPipeline:
    """Build a :class:`StreamingS2SPipeline` with custom overrides for no-crash testing."""
    cfg = OmegaConf.load(_CONF_YAML)

    s2s_cfg: dict[str, Any] = {
        "model_path": model_path,
        "engine_type": "native",
        "compute_dtype": "float32",
        "deterministic": False,
        "decode_audio": False,
        "use_perception_cache": False,
        "use_perception_cudagraph": False,
        "use_llm_cache": False,
        "system_prompt": None,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
    }
    streaming_cfg: dict[str, Any] = {
        "chunk_size_in_secs": 0.08,
        "buffer_size_in_secs": 71 * 0.08,
    }

    if s2s_overrides:
        s2s_cfg.update(s2s_overrides)
    if streaming_overrides:
        streaming_cfg.update(streaming_overrides)

    overrides = {
        "audio_file": audio_path,
        "output_dir": output_dir,
        "s2s": s2s_cfg,
        "streaming": streaming_cfg,
    }
    cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return S2SPipelineBuilder.build_pipeline(cfg)


# ---------------------------------------------------------------------------
# Parametrized configs
# ---------------------------------------------------------------------------

# Text-only configs (decode_audio=False): minimal STT-path smoke checks.
# Most config variations are folded into the audio tests below.
_TEXT_CONFIGS = [
    pytest.param({}, {}, id="baseline"),
    pytest.param(
        {"use_llm_cache": True, "use_perception_cache": True},
        {},
        id="both_caches",
    ),
]

# Audio configs (decode_audio=True): exercises the full STT + TTS pipeline.
_AUDIO_CONFIGS = [
    pytest.param({}, {}, id="baseline"),
    pytest.param(
        {"use_llm_cache": True, "use_perception_cache": True, "system_prompt": _MOCK_SYSTEM_PROMPT},
        {"chunk_size_in_secs": 0.24},
        id="both_caches_prompt_multiframe",
    ),
    pytest.param(
        {"use_llm_cache": True, "top_p": 0.9, "temperature": 0.7, "repetition_penalty": 1.1},
        {},
        id="sampling",
    ),
    pytest.param(
        {"use_tts_subword_cache": True, "use_tts_torch_compile": True},
        {},
        id="tts_optimizations",
    ),
    pytest.param(
        {"deterministic": True, "temperature": 0.0},
        {},
        id="deterministic",
    ),
    pytest.param(
        {"profile_timing": True},
        {},
        id="profile_timing",
    ),
]

# ---------------------------------------------------------------------------
# Tests (tiny_model_artifacts fixture is provided by conftest.py)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("s2s_overrides,streaming_overrides", _TEXT_CONFIGS)
def test_pipeline_no_crash_tiny_model(tiny_model_artifacts, s2s_overrides, streaming_overrides):
    """Run the streaming pipeline with various configs and verify it doesn't crash."""
    model_dir, audio_path, _ = tiny_model_artifacts
    output_dir = tempfile.mkdtemp(prefix="no-crash-text-")

    pipeline = _build_no_crash_pipeline(
        model_dir, audio_path, output_dir,
        s2s_overrides=s2s_overrides, streaming_overrides=streaming_overrides,
    )
    result = pipeline.run([audio_path])
    assert result is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("s2s_overrides,streaming_overrides", _AUDIO_CONFIGS)
def test_pipeline_no_crash_tiny_model_decode_audio(tiny_model_artifacts, s2s_overrides, streaming_overrides):
    """Run the streaming pipeline with decode_audio=True and verify it doesn't crash."""
    model_dir, audio_path, speaker_ref_path = tiny_model_artifacts
    output_dir = tempfile.mkdtemp(prefix="no-crash-audio-")

    audio_overrides = {"decode_audio": True, "speaker_reference": speaker_ref_path}
    audio_overrides.update(s2s_overrides)

    pipeline = _build_no_crash_pipeline(
        model_dir, audio_path, output_dir,
        s2s_overrides=audio_overrides, streaming_overrides=streaming_overrides,
    )
    result = pipeline.run([audio_path])
    assert result is not None
