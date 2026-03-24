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

"""Offline vs. incremental inference parity tests for NemotronVoiceChat.

``test_parity_tiny_model`` checks offline-vs-incremental parity using a
tiny model with random weights (no checkpoint needed, requires only a GPU).

``test_parity_real_checkpoint`` does the same on a real exported checkpoint
and is skipped unless
the following environment variables point to a real exported checkpoint::

    PARITY_CHECKPOINT_PATH=/path/to/exported/checkpoint
    PARITY_AUDIO_PATH=/path/to/test.wav
    PARITY_SPEAKER_NAME=<name>          # optional

Run from the NeMo repo root (use ``-s`` to see live progress)::

    # unit tests only
    CUDA_VISIBLE_DEVICES=0 pytest tests/collections/speechlm2/test_offline_incremental_parity.py -v -s

    # include integration test
    PARITY_CHECKPOINT_PATH=... PARITY_AUDIO_PATH=... \\
        CUDA_VISIBLE_DEVICES=0 pytest tests/collections/speechlm2/test_offline_incremental_parity.py -v -s
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

from nemo.utils import logging

from nemo.collections.speechlm2.inference.factory.s2s_pipeline_builder import S2SPipelineBuilder
from nemo.collections.speechlm2.inference.model_wrappers.nemotron_voicechat_inference_wrapper import (
    FRAME_SIZE_SAMPLES,
    SAMPLE_RATE,
)
from nemo.collections.speechlm2.inference.pipelines.streaming_s2s_pipeline import StreamingS2SPipeline
from nemo.collections.speechlm2.inference.streaming.framing.s2s_request_options import S2SRequestOptions
from nemo.collections.speechlm2.models import NemotronVoiceChat

# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def compare_logits(
    a: torch.Tensor,
    b: torch.Tensor,
) -> dict[str, Any]:
    """Prefix-aware comparison of two ``(B, T, V)`` logit tensors."""
    a = a.detach().cpu().float()
    b = b.detach().cpu().float()
    prefix_len = min(a.shape[1], b.shape[1])
    if prefix_len == 0:
        return {"prefix_len": 0, "match": True, "max_abs_diff": 0.0, "mean_abs_diff": 0.0}
    ap, bp = a[:, :prefix_len], b[:, :prefix_len]
    diff = (ap - bp).abs()
    reduce_dims = tuple(i for i in range(diff.dim()) if i != 1)
    per_step_max = diff.amax(dim=reduce_dims) if reduce_dims else diff
    first_diff_step = None
    nonzero = (per_step_max > 0).nonzero(as_tuple=False)
    if nonzero.numel() > 0:
        first_diff_step = int(nonzero[0].item())
    return {
        "prefix_len": prefix_len,
        "match": bool(torch.equal(ap, bp)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "first_diff_step": first_diff_step,
    }


def compare_tokens(
    a: torch.Tensor | None,
    b: torch.Tensor | None,
) -> dict[str, Any]:
    """Compare two ``(B, T)`` token tensors with prefix-aware logic."""
    if a is None or b is None:
        return {"match": None, "note": "one or both tensors missing"}
    a, b = a.detach().cpu(), b.detach().cpu()
    prefix_len = min(a.shape[-1], b.shape[-1])
    if prefix_len == 0:
        return {"prefix_len": 0, "match": True}
    ap, bp = a[..., :prefix_len], b[..., :prefix_len]
    match = bool(torch.equal(ap, bp))
    first_diff_index = None
    if not match:
        diffs = (ap != bp).flatten().nonzero(as_tuple=False)
        if diffs.numel() > 0:
            first_diff_index = int(diffs[0].item())
    return {"prefix_len": prefix_len, "match": match, "first_diff_index": first_diff_index}


def _merge_incremental_debug_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-step debug dicts from the pipeline into a single dict."""
    if not steps:
        return {}
    all_text_logits = [s["text_logits"] for s in steps if s.get("text_logits") is not None]
    all_asr_logits = [s["asr_logits"] for s in steps if s.get("asr_logits") is not None]
    return {
        "text_logits": torch.cat(all_text_logits, dim=1) if all_text_logits else None,
        "asr_logits": torch.cat(all_asr_logits, dim=1) if all_asr_logits else None,
    }


def run_parity_check(
    pipeline: StreamingS2SPipeline,
    audio_path: str,
    *,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Run offline and pipeline-based incremental inference, return comparison.

    The offline path calls :meth:`NemotronVoiceChat.offline_inference`
    (the same code path used by ``nemotron_voicechat_eval.py``).
    The incremental path runs through :meth:`StreamingS2SPipeline.run`,
    which is the real production code path (buffering, framing, prefill).

    Only STT-level tokens and logits are compared; TTS is irrelevant for
    the core parity invariant.
    """
    import librosa

    wrapper = pipeline.s2s_model
    t0 = time.time()
    logging.info("=" * 60)
    logging.info("PARITY CHECK  --  offline vs. incremental (pipeline)")
    logging.info("=" * 60)
    logging.info(f"Audio : {audio_path}")
    logging.info(f"Prompt: {system_prompt or '(none)'}")

    logging.info("Loading audio ...")
    audio_np, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    total_frames = math.ceil(len(audio_np) / FRAME_SIZE_SAMPLES)
    padded_len = total_frames * FRAME_SIZE_SAMPLES
    audio = torch.tensor(audio_np, device=wrapper.device, dtype=wrapper.dtype).unsqueeze(0)
    if audio.shape[1] < padded_len:
        audio = torch.nn.functional.pad(audio, (0, padded_len - audio.shape[1]))
    audio_lens = torch.tensor([audio.shape[1]], device=wrapper.device, dtype=torch.long)
    logging.info(f"  {len(audio_np)} samples (padded to {audio.shape[1]}), {total_frames} frames, {len(audio_np)/SAMPLE_RATE:.2f}s")

    # -- prompt tokens for offline --
    prompt_tokens = prompt_token_lens = None
    if system_prompt:
        tok = wrapper.tokenizer
        ids = [tok.bos_id] + tok.text_to_ids(system_prompt) + [tok.eos_id]
        prompt_tokens = torch.tensor(ids, device=wrapper.device, dtype=torch.long).unsqueeze(0)
        prompt_token_lens = torch.tensor([len(ids)], device=wrapper.device, dtype=torch.long)

    # -- ensure speaker info for offline_inference's mandatory TTS init --
    if wrapper.speaker_name is not None:
        OmegaConf.update(wrapper.model.cfg, "inference_speaker_name", wrapper.speaker_name, force_add=True)
    elif wrapper.speaker_reference:
        OmegaConf.update(wrapper.model.cfg, "inference_speaker_reference", wrapper.speaker_reference, force_add=True)
    speaker_kw: dict[str, Any] = {}
    cfg = wrapper.model.cfg
    if not cfg.get("inference_speaker_name", None) and not cfg.get("inference_speaker_reference", None):
        speaker_kw["speaker_audio"] = torch.randn(1, 22050, device=wrapper.device)
        speaker_kw["speaker_audio_lens"] = torch.tensor([22050], device=wrapper.device, dtype=torch.long)

    # ---- Offline path (eval.py code path) ----
    logging.info("Running offline_inference (with return_logits=True) ...")
    t_offline = time.time()
    offline = wrapper.model.offline_inference(
        input_signal=audio,
        input_signal_lens=audio_lens,
        prompt_tokens=prompt_tokens,
        prompt_token_lens=prompt_token_lens,
        decode_audio=False,
        return_logits=True,
        **speaker_kw,
    )
    logging.info(f"  offline_inference done in {time.time() - t_offline:.2f}s")

    # ---- Incremental path (pipeline.run) ----
    logging.info("Running incremental inference (pipeline.run) ...")
    t_incr = time.time()
    pipeline.collect_debug = True
    pipeline_output = pipeline.run(
        [audio_path],
        options=[S2SRequestOptions(system_prompt=system_prompt)],
    )
    logging.info(f"  pipeline.run done in {time.time() - t_incr:.2f}s")

    # Extract tokens and logits from pipeline output
    inc_tokens = pipeline_output.token_texts[0] if pipeline_output.token_texts else None
    inc_asr_tokens = pipeline_output.token_asr_texts[0] if pipeline_output.token_asr_texts else None

    incremental_debug = {}
    if pipeline_output.debug_data and pipeline_output.debug_data[0]:
        incremental_debug = _merge_incremental_debug_steps(pipeline_output.debug_data[0])

    # ---- Build comparison report ----
    logging.info("Comparing outputs ...")
    report: dict[str, Any] = {
        "audio_path": audio_path,
        "total_frames": total_frames,
        "system_prompt": system_prompt,
        "offline_text": offline.get("text", [""])[0],
        "token_comparison": compare_tokens(offline.get("tokens_text"), inc_tokens),
        "asr_token_comparison": compare_tokens(offline.get("tokens_text_src"), inc_asr_tokens),
    }

    off_tl = offline.get("text_logits")
    inc_tl = incremental_debug.get("text_logits")
    if off_tl is not None and inc_tl is not None:
        report["text_logit_comparison"] = compare_logits(off_tl, inc_tl)

    off_al = offline.get("asr_logits")
    inc_al = incremental_debug.get("asr_logits")
    if off_al is not None and inc_al is not None:
        report["asr_logit_comparison"] = compare_logits(off_al, inc_al)

    # ---- Summary ----
    elapsed = time.time() - t0
    logging.info("-" * 60)
    logging.info("PARITY CHECK SUMMARY")
    logging.info("-" * 60)
    tc = report.get("token_comparison", {})
    ac = report.get("asr_token_comparison", {})
    logging.info(f"  Text tokens match : {tc.get('match')}  (prefix_len={tc.get('prefix_len')})")
    logging.info(f"  ASR tokens match  : {ac.get('match')}  (prefix_len={ac.get('prefix_len')})")
    for tag, key in [("Text logits", "text_logit_comparison"), ("ASR logits", "asr_logit_comparison")]:
        lc = report.get(key, {})
        if lc:
            logging.info(
                f"  {tag:16s}: match={lc.get('match')}, "
                f"max_abs_diff={lc.get('max_abs_diff', 0):.6e}, "
                f"mean_abs_diff={lc.get('mean_abs_diff', 0):.6e}"
            )
    logging.info(f"  Total time: {elapsed:.2f}s")
    logging.info("=" * 60)

    return report


def assert_parity(
    report: dict[str, Any],
    *,
    strict: bool = True,
    atol: float = 0.0,
) -> None:
    """Raise :class:`AssertionError` if parity checks in *report* fail.

    Args:
        report: Dict returned by :func:`run_parity_check`.
        strict: When ``True``, also assert logit-level match.
        atol: Absolute tolerance for logit comparison (only used when *strict*).
    """
    failures: list[str] = []

    tc = report.get("token_comparison", {})
    if tc.get("match") is False:
        failures.append(f"text tokens diverge at index {tc.get('first_diff_index')}")

    ac = report.get("asr_token_comparison", {})
    if ac.get("match") is False:
        failures.append(f"ASR tokens diverge at index {ac.get('first_diff_index')}")

    if strict:
        for key in ("text_logit_comparison", "asr_logit_comparison"):
            lc = report.get(key, {})
            if lc and lc.get("match") is False:
                max_diff = lc.get("max_abs_diff", float("inf"))
                if max_diff > atol:
                    failures.append(
                        f"{key}: max_abs_diff={max_diff:.6e} > atol={atol:.6e}, "
                        f"first diff at step {lc.get('first_diff_step')}"
                    )

    if failures:
        detail = json.dumps(
            {k: v for k, v in report.items() if k != "debug"},
            indent=2,
            default=str,
        )
        raise AssertionError(
            "Offline/incremental parity failed:\n  - "
            + "\n  - ".join(failures)
            + f"\n\nFull report:\n{detail}"
        )


# ---------------------------------------------------------------------------
# Tiny-model configuration (derived from test_nemotron_voicechat.py)
# ---------------------------------------------------------------------------

_pretrained_llm = "TinyLlama/TinyLlama_v1.1"
if os.path.exists("/home/TestData/speechlm/pretrained_models"):
    _pretrained_llm = "/home/TestData/speechlm/pretrained_models/TinyLlama--TinyLlama_v1.1"


def _tiny_voicechat_config() -> dict:
    """Return a minimal NemotronVoiceChat config with random weights."""
    return {
        "model": {
            "scoring_asr": "stt_en_fastconformer_transducer_large",
            "stt": {
                "model": {
                    "pretrained_llm": _pretrained_llm,
                    "pretrained_weights": False,
                    "predict_user_text": True,
                    "audio_loss_weight": 1,
                    "text_loss_weight": 3,
                    "source_sample_rate": 16000,
                    "validation_save_path": "/tmp/test_parity_stt_logs",
                    "perception": {
                        "_target_": "nemo.collections.speechlm2.modules.perception.AudioPerceptionModule",
                        "preprocessor": {
                            "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                            "features": 80,
                        },
                        "encoder": {
                            "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                            "feat_in": 80,
                            "d_model": 512,
                            "n_heads": 8,
                            "n_layers": 1,
                            "subsampling_factor": 8,
                        },
                        "modality_adapter": {
                            "_target_": "nemo.collections.speechlm2.modules.perception.IdentityConnector",
                            "d_model": 512,
                        },
                        "output_dim": 2048,
                    },
                    "optimizer": {"_target_": "torch.optim.AdamW"},
                },
                "data": {"source_sample_rate": 16000},
                "exp_manager": {"explicit_log_dir": "/tmp/test_parity_stt_logs"},
            },
            "speech_generation": {
                "model": {
                    "pretrained_lm_name": _pretrained_llm,
                    "pretrained_ae_dir": None,
                    "pretrained_tts_model": None,
                    "scoring_asr": "stt_en_fastconformer_transducer_large",
                    "freeze_params": [r"^audio_codec\..+$", r"^embed_tokens\..+$"],
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "pad_token": "<SPECIAL_12>",
                    "audio_codec_run_dtype": "float32",
                    "prevent_freeze_params": [],
                    "audio_save_path": "",
                    "inference_guidance_scale": 0.5,
                    "inference_noise_scale": 0.8,
                    "inference_top_p_or_k": 0.8,
                    "inference_guidance_enabled": False,
                    "subword_mask_exactly_as_eartts": False,
                    "context_hidden_mask_exactly_as_eartts": False,
                    "optimizer": {
                        "_target_": "torch.optim.AdamW",
                        "lr": 4e-5,
                        "betas": [0.9, 0.98],
                        "weight_decay": 0,
                        "foreach": True,
                    },
                    "lr_scheduler": {
                        "_target_": "nemo.core.optim.lr_scheduler.InverseSquareRootAnnealing",
                        "warmup_steps": 2500,
                        "min_lr": 1e-6,
                        "max_steps": 100_000_000,
                    },
                    "codec_config": {
                        "latent_size": 512,
                        "n_fft": 16,
                        "hop_length": 4,
                        "base_hidden_size": 384,
                        "channel_mult": [1, 2, 4],
                        "rates": [7, 7, 9],
                        "num_blocks": 3,
                        "kernel_size": 7,
                        "groups": 1,
                        "codebook_size": 1024,
                        "num_quantizers": 31,
                        "wav_to_token_ratio": 1764,
                    },
                    "tts_config": {
                        "use_gated_fusion_for_text_audio": True,
                        "disable_eos_prediction": True,
                        "use_bos_eos_emb": True,
                        "use_subword_flag_emb": True,
                        "num_delay_speech_tokens": 2,
                        "backbone_type": "gemma3_text",
                        "backbone_model_class": None,
                        "backbone_config_class": None,
                        "backbone_config": {
                            "hidden_size": 1152,
                            "intermediate_size": 4608,
                            "num_hidden_layers": 1,
                            "num_attention_heads": 16,
                            "num_key_value_heads": 16,
                            "head_dim": 72,
                            "attention_dropout": 0.1,
                            "use_cache": False,
                        },
                        "latent_size": 512,
                        "codebook_size": 1024,
                        "num_quantizers": 31,
                        "context_hidden_size": None,
                        "cas_config": {
                            "backbone_type": "t5gemma",
                            "backbone_model_class": None,
                            "backbone_config_class": None,
                            "backbone_config": {
                                "is_encoder_decoder": False,
                                "encoder": {
                                    "hidden_size": 1152,
                                    "intermediate_size": 4608,
                                    "num_hidden_layers": 1,
                                    "num_attention_heads": 16,
                                    "num_key_value_heads": 16,
                                    "head_dim": 72,
                                    "use_cache": False,
                                    "attention_dropout": 0.1,
                                },
                            },
                        },
                        "mog_head_config": {
                            "intermediate_size": 4608,
                            "num_layers": 3,
                            "low_rank": 64,
                            "num_predictions": 1024,
                            "min_log_std": -4.0,
                            "eps": 1e-6,
                        },
                        "p_uncond": 0.1,
                        "label_smoothing": 0.01,
                        "max_training_rate": 0.8,
                        "quantizer_dropout": 0.5,
                        "random_target_masking": False,
                        "exponent": 3.0,
                    },
                },
                "data": {
                    "add_text_bos_and_eos_in_each_turn": True,
                    "add_audio_prompt": True,
                    "audio_prompt_duration": 3.0,
                    "frame_length": 0.08,
                    "source_sample_rate": 16000,
                    "target_sample_rate": 22050,
                },
                "exp_manager": {"explicit_log_dir": "/tmp/test_parity_tts_logs"},
            },
        },
        "data": {
            "frame_length": 0.08,
            "source_sample_rate": 16000,
            "target_sample_rate": 22050,
            "input_roles": ["user", "User"],
            "output_roles": ["agent", "Assistant", "assistant", "Agent"],
        },
        "exp_manager": {"explicit_log_dir": "/tmp/test_parity_logs"},
    }


def _build_parity_pipeline(
    model_path: str,
    audio_path: str,
    output_dir: str,
    *,
    speaker_name: str | None = None,
) -> StreamingS2SPipeline:
    """Build a :class:`StreamingS2SPipeline` configured for strict parity testing."""
    import librosa

    audio_np, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    total_frames = math.ceil(len(audio_np) / FRAME_SIZE_SAMPLES)
    chunk_secs = total_frames * FRAME_SIZE_SAMPLES / SAMPLE_RATE

    speaker_kw = {}
    if speaker_name:
        speaker_kw["speaker_name"] = speaker_name

    pipeline_cfg = OmegaConf.create(
        {
            "output_dir": output_dir,
            "s2s": {
                "model_path": model_path,
                **speaker_kw,
                "compute_dtype": "float32",
                "engine_type": "native",
                "deterministic": True,
                "use_perception_cache": False,
                "use_perception_cudagraph": False,
                "use_llm_cache": False,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "temperature": 0.0,
                "decode_audio": False,
            },
            "streaming": {
                "input_sample_rate": SAMPLE_RATE,
                "output_sample_rate": 22050,
                "batch_size": 1,
                "att_context_size": [70, 0],
                "chunk_size_in_secs": chunk_secs,
                "buffer_size_in_secs": max(71 * 0.08, chunk_secs),
                "request_type": "frame",
                "max_len": 8192,
            },
        }
    )
    return S2SPipelineBuilder.build_pipeline(pipeline_cfg)


# ---------------------------------------------------------------------------
# Parity test -- tiny model (no real checkpoint needed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_parity_tiny_model(tmp_path):
    """Offline/incremental parity with a tiny random-weight model.

    Saves the model as an HF checkpoint, then loads it through the real
    ``S2SPipelineBuilder`` so the test exercises the same code path as
    ``test_parity_real_checkpoint``.
    """
    import json as _json

    audio_path = str(tmp_path / "test_audio.wav")
    sf.write(audio_path, np.random.RandomState(42).randn(16000).astype(np.float32), 16000)

    cfg = _tiny_voicechat_config()
    model = NemotronVoiceChat(cfg)
    model.to("cuda")
    model.eval()

    model_dir = str(tmp_path / "model")
    model.save_pretrained(model_dir)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        _json.dump(cfg, f)
    del model
    torch.cuda.empty_cache()

    pipeline = _build_parity_pipeline(model_dir, audio_path, str(tmp_path / "output"))
    report = run_parity_check(pipeline, audio_path)
    assert_parity(report, strict=True, atol=0.0)


# ---------------------------------------------------------------------------
# Integration test -- real checkpoint (skipped when env vars are not set)
# ---------------------------------------------------------------------------


def _real_checkpoint_available() -> bool:
    path = os.environ.get("PARITY_CHECKPOINT_PATH", "")
    audio = os.environ.get("PARITY_AUDIO_PATH", "")
    return bool(path) and os.path.isdir(path) and bool(audio) and os.path.isfile(audio)


@pytest.mark.skipif(not _real_checkpoint_available(), reason="set PARITY_CHECKPOINT_PATH and PARITY_AUDIO_PATH")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_parity_real_checkpoint():
    """Parity check using a real exported checkpoint.

    Configure via environment variables::

        PARITY_CHECKPOINT_PATH=/path/to/exported/checkpoint
        PARITY_AUDIO_PATH=/path/to/test.wav
        PARITY_SPEAKER_NAME=<name>          # optional
    """
    import tempfile

    ckpt = os.environ["PARITY_CHECKPOINT_PATH"]
    audio = os.environ["PARITY_AUDIO_PATH"]
    speaker = os.environ.get("PARITY_SPEAKER_NAME")

    pipeline = _build_parity_pipeline(
        ckpt, audio, tempfile.mkdtemp(prefix="parity-"), speaker_name=speaker,
    )
    report = run_parity_check(pipeline, audio)
    assert_parity(report, strict=True, atol=0.0)
