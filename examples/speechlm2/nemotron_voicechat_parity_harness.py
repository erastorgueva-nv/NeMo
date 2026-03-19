from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import librosa
import soundfile as sf
import torch
from omegaconf import MISSING, OmegaConf

from nemo.collections.speechlm2.inference.factory.s2s_pipeline_builder import S2SPipelineBuilder
from nemo.collections.speechlm2.inference.model_wrappers.nemotron_voicechat_inference_wrapper import (
    FRAME_SIZE_SAMPLES,
    NemotronVoicechatInferenceWrapper,
)
from nemo.collections.speechlm2.inference.streaming.framing.s2s_request_options import S2SRequestOptions
def _bool_arg(parser: argparse.ArgumentParser, name: str, help_text: str) -> None:
    parser.add_argument(name, action=argparse.BooleanOptionalAction, default=None, help=help_text)


def _default_s2s_streaming_config_path() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "examples" / "speechlm2" / "nemo_inference_pipelines" / "conf" / "s2s_streaming.yaml")


def _load_s2s_inference_config(config_path: str | None = None):
    path = config_path or _default_s2s_streaming_config_path()
    cfg = OmegaConf.load(path)
    for key, value in {
        "audio_file": "",
        "output_dir": "./generated",
        "s2s.model_path": None,
        "s2s.llm_checkpoint_path": None,
        "s2s.decode_audio": True,
        "s2s.engine_type": "native",
        "s2s.system_prompt": None,
        "streaming.chunk_size_in_secs": FRAME_SIZE_SAMPLES / 16000.0,
        "streaming.buffer_size_in_secs": 71 * (FRAME_SIZE_SAMPLES / 16000.0),
    }.items():
        if OmegaConf.select(cfg, key, default=MISSING) is MISSING:
            OmegaConf.update(cfg, key, value, force_add=True)
    return cfg


def _apply_inference_overrides(cfg, overrides: dict[str, Any]):
    for key, value in overrides.items():
        if value is not None:
            OmegaConf.update(cfg, key, value, force_add=True)
    return cfg


def _load_audio_tensor(audio_path: str, sample_rate: int, device: torch.device, dtype: torch.dtype):
    audio_np, _ = librosa.load(audio_path, sr=sample_rate)
    audio = torch.tensor(audio_np, device=device, dtype=dtype).unsqueeze(0)
    audio_lens = torch.tensor([audio.shape[1]], device=device, dtype=torch.long)
    return audio, audio_lens


def _build_prompt_token_ids(tokenizer, system_prompt: str | None) -> list[int]:
    if not system_prompt or not system_prompt.strip():
        return []
    return [tokenizer.bos_id] + tokenizer.text_to_ids(system_prompt) + [tokenizer.eos_id]


def _resolve_num_frames_per_chunk(args, total_frames: int) -> int:
    if args.num_frames_per_chunk is not None:
        value = int(args.num_frames_per_chunk)
    elif args.chunk_size_in_secs is not None:
        value = int(round(float(args.chunk_size_in_secs) / (FRAME_SIZE_SAMPLES / 16000.0)))
    else:
        value = total_frames

    if value < 1:
        raise ValueError(f"num_frames_per_chunk must be >= 1, got {value}")
    return value


def _apply_deterministic_runtime_settings(enabled: bool) -> None:
    if not enabled:
        return
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    torch.use_deterministic_algorithms(True, warn_only=False)



def _compute_min_buffer_frames(wrapper, num_frames_per_chunk: int) -> int:
    att_context_size = wrapper.model.stt_model.perception.encoder._cfg.att_context_size
    if wrapper.use_perception_cache:
        return num_frames_per_chunk * (att_context_size[1] + 1) + 2
    return att_context_size[0] + att_context_size[1] + 1


def _compute_min_buffer_frames_from_cfg(cfg, num_frames_per_chunk: int) -> int:
    att_context_size = cfg.streaming.get("att_context_size", [70, 0])
    if cfg.s2s.get("use_perception_cache", False):
        return num_frames_per_chunk * (att_context_size[1] + 1) + 2
    return att_context_size[0] + att_context_size[1] + 1


def _first_diff(a: torch.Tensor, b: torch.Tensor) -> int | None:
    a = a.detach().cpu()
    b = b.detach().cpu()
    if a.shape != b.shape:
        return 0
    diff = (a != b).flatten()
    if not diff.any():
        return None
    return int(diff.nonzero(as_tuple=False)[0].item())


def _prefix_compare(a: torch.Tensor, b: torch.Tensor) -> tuple[int | None, bool | None, int | None]:
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return None, None, None
    a = a.detach().cpu()
    b = b.detach().cpu()
    if a.dim() != b.dim():
        return None, None, None

    prefix_len = min(a.shape[-1], b.shape[-1])
    if prefix_len == 0:
        return 0, True, None

    a_prefix = a[..., :prefix_len]
    b_prefix = b[..., :prefix_len]
    if torch.equal(a_prefix, b_prefix):
        return prefix_len, True, None

    diff = (a_prefix != b_prefix).flatten()
    first_diff = int(diff.nonzero(as_tuple=False)[0].item())
    return prefix_len, False, first_diff


def _prefix_tensor_diff(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any] | None:
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return None
    a = a.detach().cpu()
    b = b.detach().cpu()
    if a.dim() != b.dim():
        return None
    prefix_len = min(a.shape[1], b.shape[1]) if a.dim() >= 2 else min(a.shape[0], b.shape[0])
    if prefix_len <= 0:
        return {"prefix_len": 0, "match": True, "max_abs_diff": 0.0, "mean_abs_diff": 0.0}
    if a.dim() == 2:
        a_prefix = a[:, :prefix_len]
        b_prefix = b[:, :prefix_len]
    else:
        a_prefix = a[:, :prefix_len, ...]
        b_prefix = b[:, :prefix_len, ...]
    diff = (a_prefix - b_prefix).abs()
    reduce_dims = tuple(i for i in range(diff.dim()) if i != 1)
    if reduce_dims:
        per_step_max = diff.amax(dim=reduce_dims)
    else:
        per_step_max = diff
    first_step_diff_index = None
    differing_steps = (per_step_max > 0).nonzero(as_tuple=False)
    if differing_steps.numel() > 0:
        first_step_diff_index = int(differing_steps[0].item())
    return {
        "prefix_len": prefix_len,
        "match": bool(torch.equal(a_prefix, b_prefix)),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "first_step_diff_index": first_step_diff_index,
    }


def _dtype_name(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return str(value.dtype)
    if isinstance(value, torch.dtype):
        return str(value)
    return str(value)


def _module_param_dtype(module) -> str | None:
    if module is None:
        return None
    try:
        return str(next(module.parameters()).dtype)
    except StopIteration:
        return None
    except Exception:
        return None


def _collect_model_dtypes(wrapper: NemotronVoicechatInferenceWrapper) -> dict[str, Any]:
    stt_model = wrapper.model.stt_model
    return {
        "wrapper_dtype": _dtype_name(wrapper.dtype),
        "llm_dtype": _module_param_dtype(getattr(stt_model, "llm", None)),
        "lm_head_dtype": _module_param_dtype(getattr(stt_model, "lm_head", None)),
        "asr_head_dtype": _module_param_dtype(getattr(stt_model, "asr_head", None)),
        "embed_tokens_dtype": _module_param_dtype(getattr(stt_model, "embed_tokens", None)),
        "embed_asr_tokens_dtype": _module_param_dtype(getattr(stt_model, "embed_asr_tokens", None)),
        "perception_dtype": _module_param_dtype(getattr(stt_model, "perception", None)),
        "tts_dtype": _module_param_dtype(getattr(wrapper.model, "tts_model", None)),
    }


def _tensor_summary_diff(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any] | None:
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return None
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "match": False}
    diff = (a - b).abs()
    return {
        "shape": list(a.shape),
        "match": bool(torch.equal(a, b)),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def _step_component_diagnostics(
    wrapper: NemotronVoicechatInferenceWrapper,
    offline_debug: dict[str, Any],
    incremental_debug: dict[str, Any],
) -> dict[str, Any] | None:
    input_embed_diff = _prefix_tensor_diff(offline_debug.get("input_embeds"), incremental_debug.get("input_embeds"))
    if input_embed_diff is None:
        return {"status": "missing_input_embed_diff"}
    step_idx = input_embed_diff.get("first_step_diff_index")
    if step_idx is None:
        return {"status": "no_input_embed_drift"}
    if step_idx == 0:
        return {"first_step_diff_index": 0, "note": "Drift starts at step 0; component breakdown not specialized."}

    offline_source = offline_debug.get("source_encoded")
    incremental_source = incremental_debug.get("source_encoded")
    selected_indices = incremental_debug.get("selected_frame_indices") or []
    offline_tokens = offline_debug.get("gen_text")
    offline_asr = offline_debug.get("gen_asr")
    incremental_tokens = incremental_debug.get("gen_text")
    incremental_asr = incremental_debug.get("gen_asr")
    required = {
        "offline_source_encoded": offline_source,
        "incremental_source_encoded": incremental_source,
        "offline_gen_text": offline_tokens,
        "offline_gen_asr": offline_asr,
        "incremental_gen_text": incremental_tokens,
        "incremental_gen_asr": incremental_asr,
    }
    missing = [name for name, value in required.items() if not isinstance(value, torch.Tensor)]
    if missing:
        return {
            "status": "missing_tensors",
            "first_step_diff_index": step_idx,
            "missing": missing,
        }
    if step_idx >= len(selected_indices):
        return {
            "status": "selected_index_out_of_range",
            "first_step_diff_index": step_idx,
            "selected_frame_indices_len": len(selected_indices),
        }

    stt_model = wrapper.model.stt_model
    source_frame_offline = offline_source[:, step_idx : step_idx + 1, :]
    source_frame_incremental = incremental_source[:, selected_indices[step_idx] : selected_indices[step_idx] + 1, :]
    prev_offline_text = offline_tokens[:, step_idx - 1]
    prev_incremental_text = incremental_tokens[:, step_idx - 1]
    prev_offline_asr = offline_asr[:, step_idx - 1]
    prev_incremental_asr = incremental_asr[:, step_idx - 1]

    offline_text_emb = stt_model.embed_tokens(prev_offline_text.to(wrapper.device)).detach().cpu()
    incremental_text_emb = stt_model.embed_tokens(prev_incremental_text.to(wrapper.device)).detach().cpu()
    offline_asr_emb = stt_model.embed_asr_tokens(prev_offline_asr.to(wrapper.device)).detach().cpu()
    incremental_asr_emb = stt_model.embed_asr_tokens(prev_incremental_asr.to(wrapper.device)).detach().cpu()

    text_weight = stt_model.cfg.get("duplex_text_channel_weight", 1.0)
    asr_weight = stt_model.cfg.get("duplex_asr_text_weight", 1.0)
    offline_last_emb = offline_text_emb * text_weight + offline_asr_emb * asr_weight
    incremental_last_emb = incremental_text_emb * text_weight + incremental_asr_emb * asr_weight

    offline_input = offline_debug["input_embeds"][:, step_idx : step_idx + 1, :]
    incremental_input = incremental_debug["input_embeds"][:, step_idx : step_idx + 1, :]

    offline_style = source_frame_incremental.detach().cpu().clone()
    offline_style += incremental_last_emb.unsqueeze(1)
    incremental_style = source_frame_incremental.detach().cpu().clone()
    incremental_style += (incremental_text_emb * text_weight).unsqueeze(1)
    incremental_style += (incremental_asr_emb * asr_weight).unsqueeze(1)

    return {
        "status": "ok",
        "first_step_diff_index": step_idx,
        "selected_frame_index": selected_indices[step_idx],
        "prev_text_token_equal": bool(torch.equal(prev_offline_text.cpu(), prev_incremental_text.cpu())),
        "prev_asr_token_equal": bool(torch.equal(prev_offline_asr.cpu(), prev_incremental_asr.cpu())),
        "source_frame_diff": _tensor_summary_diff(source_frame_offline.cpu(), source_frame_incremental.cpu()),
        "text_embedding_diff": _tensor_summary_diff(offline_text_emb, incremental_text_emb),
        "asr_embedding_diff": _tensor_summary_diff(offline_asr_emb, incremental_asr_emb),
        "last_emb_diff": _tensor_summary_diff(offline_last_emb, incremental_last_emb),
        "offline_input_vs_incremental_input": _tensor_summary_diff(offline_input.cpu(), incremental_input.cpu()),
        "offline_input_vs_offline_style_rebuild": _tensor_summary_diff(offline_input.cpu(), offline_style),
        "incremental_input_vs_offline_style_rebuild": _tensor_summary_diff(incremental_input.cpu(), offline_style),
        "offline_input_vs_incremental_style_rebuild": _tensor_summary_diff(offline_input.cpu(), incremental_style),
        "incremental_input_vs_incremental_style_rebuild": _tensor_summary_diff(incremental_input.cpu(), incremental_style),
    }


def _compare_debug_outputs(offline_debug: dict[str, Any] | None, incremental_debug: dict[str, Any] | None) -> dict[str, Any] | None:
    if offline_debug is None or incremental_debug is None:
        return None

    offline_encoder = offline_debug.get("source_encoded")
    incremental_encoder = incremental_debug.get("source_encoded")
    selected_indices = incremental_debug.get("selected_frame_indices") or []
    selected_incremental = None
    selected_prefix = None
    if isinstance(incremental_encoder, torch.Tensor) and selected_indices:
        selected_incremental = incremental_encoder[:, selected_indices, :]
    if isinstance(offline_encoder, torch.Tensor) and selected_incremental is not None:
        prefix_len = min(offline_encoder.shape[1], selected_incremental.shape[1])
        selected_prefix = _prefix_tensor_diff(
            offline_encoder[:, :prefix_len, :],
            selected_incremental[:, :prefix_len, :],
        )

    report = {
        "offline_tensor_dtypes": {
            "source_encoded": _dtype_name(offline_debug.get("source_encoded")),
            "input_embeds": _dtype_name(offline_debug.get("input_embeds")),
            "text_logits": _dtype_name(offline_debug.get("text_logits")),
            "asr_logits": _dtype_name(offline_debug.get("asr_logits")),
        },
        "incremental_tensor_dtypes": {
            "source_encoded": _dtype_name(incremental_debug.get("source_encoded")),
            "input_embeds": _dtype_name(incremental_debug.get("input_embeds")),
            "text_logits": _dtype_name(incremental_debug.get("text_logits")),
            "asr_logits": _dtype_name(incremental_debug.get("asr_logits")),
        },
        "offline_source_encoded_shape": list(offline_encoder.shape) if isinstance(offline_encoder, torch.Tensor) else None,
        "incremental_source_encoded_shape": list(incremental_encoder.shape) if isinstance(incremental_encoder, torch.Tensor) else None,
        "incremental_selected_frame_indices": selected_indices,
        "selected_encoder_prefix": selected_prefix,
        "offline_input_embeds_shape": list(offline_debug["input_embeds"].shape)
        if isinstance(offline_debug.get("input_embeds"), torch.Tensor)
        else None,
        "incremental_input_embeds_shape": list(incremental_debug["input_embeds"].shape)
        if isinstance(incremental_debug.get("input_embeds"), torch.Tensor)
        else None,
        "input_embeds_prefix": _prefix_tensor_diff(offline_debug.get("input_embeds"), incremental_debug.get("input_embeds")),
        "offline_text_logits_shape": list(offline_debug["text_logits"].shape)
        if isinstance(offline_debug.get("text_logits"), torch.Tensor)
        else None,
        "incremental_text_logits_shape": list(incremental_debug["text_logits"].shape)
        if isinstance(incremental_debug.get("text_logits"), torch.Tensor)
        else None,
        "text_logits_prefix": _prefix_tensor_diff(offline_debug.get("text_logits"), incremental_debug.get("text_logits")),
        "offline_asr_logits_shape": list(offline_debug["asr_logits"].shape)
        if isinstance(offline_debug.get("asr_logits"), torch.Tensor)
        else None,
        "incremental_asr_logits_shape": list(incremental_debug["asr_logits"].shape)
        if isinstance(incremental_debug.get("asr_logits"), torch.Tensor)
        else None,
        "asr_logits_prefix": _prefix_tensor_diff(offline_debug.get("asr_logits"), incremental_debug.get("asr_logits")),
    }
    report["step_component_diagnostics"] = None
    return report


def _compare_outputs(offline: dict[str, Any], incremental: dict[str, Any]) -> dict[str, Any]:
    offline_tokens = offline.get("tokens_text")
    incremental_tokens = incremental.get("tokens_text")
    offline_asr = offline.get("tokens_text_src")
    incremental_asr = incremental.get("asr_tokens")
    token_prefix_len, token_prefix_match, token_prefix_first_diff = _prefix_compare(offline_tokens, incremental_tokens)
    asr_prefix_len, asr_prefix_match, asr_prefix_first_diff = _prefix_compare(offline_asr, incremental_asr)

    token_match = (
        isinstance(offline_tokens, torch.Tensor)
        and isinstance(incremental_tokens, torch.Tensor)
        and offline_tokens.shape == incremental_tokens.shape
        and torch.equal(offline_tokens.detach().cpu(), incremental_tokens.detach().cpu())
    )
    asr_token_match = None
    if isinstance(offline_asr, torch.Tensor) and isinstance(incremental_asr, torch.Tensor):
        asr_token_match = offline_asr.shape == incremental_asr.shape and torch.equal(
            offline_asr.detach().cpu(), incremental_asr.detach().cpu()
        )

    offline_audio_len = offline.get("audio_len")
    incremental_audio = incremental.get("audio")
    audio_sample_count_equal = None
    if offline_audio_len is not None and incremental_audio is not None:
        expected = int(offline_audio_len[0].item())
        got = int(incremental_audio.shape[-1])
        audio_sample_count_equal = expected == got

    report = {
        "offline_text": offline.get("text", [""])[0],
        "incremental_text": incremental.get("text", [""])[0],
        "text_equal": offline.get("text", [""])[0] == incremental.get("text", [""])[0],
        "offline_asr_text": (offline.get("src_text") or [""])[0] if offline.get("src_text") is not None else None,
        "incremental_asr_text": (incremental.get("asr_text") or [""])[0] if incremental.get("asr_text") is not None else None,
        "asr_text_equal": (
            offline.get("src_text") is not None
            and incremental.get("asr_text") is not None
            and offline["src_text"][0] == incremental["asr_text"][0]
        ),
        "offline_token_shape": list(offline_tokens.shape) if isinstance(offline_tokens, torch.Tensor) else None,
        "incremental_token_shape": list(incremental_tokens.shape) if isinstance(incremental_tokens, torch.Tensor) else None,
        "token_match": token_match,
        "token_first_diff_index": _first_diff(offline_tokens, incremental_tokens)
        if isinstance(offline_tokens, torch.Tensor) and isinstance(incremental_tokens, torch.Tensor)
        else None,
        "token_prefix_len": token_prefix_len,
        "token_prefix_match": token_prefix_match,
        "token_prefix_first_diff_index": token_prefix_first_diff,
        "offline_asr_token_shape": list(offline_asr.shape) if isinstance(offline_asr, torch.Tensor) else None,
        "incremental_asr_token_shape": list(incremental_asr.shape) if isinstance(incremental_asr, torch.Tensor) else None,
        "asr_token_match": asr_token_match,
        "asr_token_first_diff_index": _first_diff(offline_asr, incremental_asr)
        if isinstance(offline_asr, torch.Tensor) and isinstance(incremental_asr, torch.Tensor)
        else None,
        "asr_token_prefix_len": asr_prefix_len,
        "asr_token_prefix_match": asr_prefix_match,
        "asr_token_prefix_first_diff_index": asr_prefix_first_diff,
        "audio_sample_count_equal": audio_sample_count_equal,
    }
    return report


def _merge_incremental_debug_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-step debug dicts from the pipeline into a single dict matching offline debug format."""
    if not steps:
        return {}
    all_source_encoded = [s["source_encoded"] for s in steps if s.get("source_encoded") is not None]
    all_input_embeds = [s["input_embeds"] for s in steps if s.get("input_embeds") is not None]
    all_text_logits = [s["text_logits"] for s in steps if s.get("text_logits") is not None]
    all_asr_logits = [s["asr_logits"] for s in steps if s.get("asr_logits") is not None]
    all_gen_text = [s["gen_text"] for s in steps if s.get("gen_text") is not None]
    all_gen_asr = [s["gen_asr"] for s in steps if s.get("gen_asr") is not None]
    selected_frame_indices = []
    for s in steps:
        selected_frame_indices.extend(s.get("selected_frame_indices", []))
    return {
        "source_encoded": all_source_encoded[-1] if all_source_encoded else None,
        "input_embeds": torch.cat(all_input_embeds, dim=1) if all_input_embeds else None,
        "gen_text": all_gen_text[-1] if all_gen_text else None,
        "gen_asr": all_gen_asr[-1] if all_gen_asr else None,
        "text_logits": torch.cat(all_text_logits, dim=1) if all_text_logits else None,
        "asr_logits": torch.cat(all_asr_logits, dim=1) if all_asr_logits else None,
        "selected_frame_indices": selected_frame_indices,
    }


def _collect_offline_debug(
    wrapper: NemotronVoicechatInferenceWrapper,
    audio: torch.Tensor,
    audio_lens: torch.Tensor,
    prompt_tokens: torch.Tensor | None,
    prompt_token_lens: torch.Tensor | None,
) -> dict[str, Any]:
    buffer_len = audio_lens.to(device=wrapper.device, dtype=torch.long)
    source_encoded, _, _ = wrapper.model.stt_model.perception(
        input_signal=audio,
        input_signal_length=buffer_len,
        return_encoder_emb=True,
    )
    source_encoded = source_encoded.to(wrapper.dtype)

    inference_state = wrapper.model.stt_model.streaming_inference._init_inference(
        audio,
        audio_lens,
        0,
        prompt_tokens,
        prompt_token_lens,
    )
    ans, inference_state = wrapper.model.stt_model.streaming_inference._step_zero(inference_state)
    text_logits = [ans["text_logits"][:, -1].detach().cpu()]
    asr_logits = [ans["asr_logits"][:, -1].detach().cpu()] if "asr_logits" in ans else []
    T = inference_state["T"]
    for t in range(1, T):
        ans = wrapper.model.stt_model.streaming_inference._step_inference(t, inference_state, ans)
        text_logits.append(ans["text_logits"][:, -1].detach().cpu())
        if "asr_logits" in ans:
            asr_logits.append(ans["asr_logits"][:, -1].detach().cpu())

    return {
        "source_encoded": source_encoded.detach().cpu(),
        "input_embeds": inference_state["input_embeds"].detach().cpu(),
        "gen_text": inference_state["gen_text"].detach().cpu(),
        "gen_asr": inference_state["gen_asr"].detach().cpu() if inference_state.get("gen_asr") is not None else None,
        "text_logits": torch.stack(text_logits, dim=1),
        "asr_logits": torch.stack(asr_logits, dim=1) if asr_logits else None,
    }


def run_parity_harness(args) -> dict[str, Any]:
    inference_cfg = _load_s2s_inference_config(args.config_path)

    if args.strict_runtime_parity and args.tts_system_prompt:
        raise ValueError(
            "Strict offline/incremental parity does not currently support `tts_system_prompt`, "
            "because offline_inference has no equivalent string prompt API for TTS conditioning."
        )

    overrides = {
        "s2s.model_path": args.model_path,
        "s2s.llm_checkpoint_path": args.llm_checkpoint_path,
        "s2s.speaker_reference": args.speaker_reference,
        "s2s.speaker_name": args.speaker_name,
        "s2s.compute_dtype": args.compute_dtype,
        "s2s.decode_audio": args.decode_audio,
        "s2s.system_prompt": args.system_prompt,
        "s2s.tts_system_prompt": args.tts_system_prompt,
        "s2s.engine_type": args.engine_type,
        "s2s.use_perception_cache": args.use_perception_cache,
        "s2s.use_perception_cudagraph": args.use_perception_cudagraph,
        "s2s.use_llm_cache": args.use_llm_cache,
        "s2s.use_codec_cache": args.use_codec_cache,
        "s2s.deterministic": args.deterministic,
        "s2s.top_p": args.top_p,
        "s2s.repetition_penalty": args.repetition_penalty,
        "s2s.temperature": args.temperature,
    }

    if args.strict_runtime_parity:
        strict_defaults = {
            "s2s.engine_type": "native",
            "s2s.compute_dtype": "float32",
            "s2s.use_perception_cache": False,
            "s2s.use_perception_cudagraph": False,
            "s2s.use_llm_cache": False,
            "s2s.use_codec_cache": False,
            "s2s.deterministic": True,
            "s2s.top_p": 1.0,
            "s2s.repetition_penalty": 1.0,
            "s2s.temperature": 0.0,
        }
        for key, value in strict_defaults.items():
            overrides[key] = value if overrides.get(key) is None else overrides[key]

    inference_cfg = _apply_inference_overrides(inference_cfg, overrides)
    _apply_deterministic_runtime_settings(bool(inference_cfg.s2s.get("deterministic", False)))

    input_sample_rate = int(inference_cfg.streaming.get("input_sample_rate", 16000))
    audio_np, _ = librosa.load(args.audio_path, sr=input_sample_rate)
    total_samples = len(audio_np)
    total_frames = int(math.ceil(total_samples / FRAME_SIZE_SAMPLES))
    num_frames_per_chunk = _resolve_num_frames_per_chunk(args, total_frames)
    chunk_size_in_secs = num_frames_per_chunk * (FRAME_SIZE_SAMPLES / float(input_sample_rate))
    buffer_size_frames = max(num_frames_per_chunk, _compute_min_buffer_frames_from_cfg(inference_cfg, num_frames_per_chunk))

    with tempfile.TemporaryDirectory(prefix="voicechat-parity-") as tmpdir:
        inference_cfg = _apply_inference_overrides(
            inference_cfg,
            {
                "output_dir": tmpdir,
                "streaming.chunk_size_in_secs": chunk_size_in_secs,
                "streaming.buffer_size_in_secs": buffer_size_frames * (FRAME_SIZE_SAMPLES / float(input_sample_rate)),
            },
        )
        pipeline = S2SPipelineBuilder.build_pipeline(inference_cfg)
        do_collect_debug = args.collect_debug if args.collect_debug is not None else bool(args.strict_runtime_parity)
        pipeline.collect_debug = do_collect_debug
        wrapper = pipeline.s2s_model

        audio, audio_lens = _load_audio_tensor(
            args.audio_path,
            sample_rate=wrapper.model.source_sample_rate,
            device=wrapper.device,
            dtype=wrapper.dtype,
        )

        prompt_tokens = None
        prompt_token_lens = None
        if inference_cfg.s2s.get("system_prompt"):
            prompt_token_ids = _build_prompt_token_ids(wrapper.tokenizer, inference_cfg.s2s.system_prompt)
            prompt_tokens = torch.tensor(prompt_token_ids, device=wrapper.device, dtype=torch.long).unsqueeze(0)
            prompt_token_lens = torch.tensor([len(prompt_token_ids)], device=wrapper.device, dtype=torch.long)

        if wrapper.speaker_name is not None:
            OmegaConf.update(wrapper.model.cfg, "inference_speaker_name", wrapper.speaker_name, force_add=True)
        elif wrapper.speaker_reference:
            OmegaConf.update(wrapper.model.cfg, "inference_speaker_reference", wrapper.speaker_reference, force_add=True)

        offline = wrapper.model.offline_inference(
            input_signal=audio,
            input_signal_lens=audio_lens,
            prompt_tokens=prompt_tokens,
            prompt_token_lens=prompt_token_lens,
            decode_audio=bool(inference_cfg.s2s.get("decode_audio", True)),
        )
        offline_debug = _collect_offline_debug(
            wrapper,
            audio=audio,
            audio_lens=audio_lens,
            prompt_tokens=prompt_tokens,
            prompt_token_lens=prompt_token_lens,
        )
        pipeline_output = pipeline.run(
            [args.audio_path],
            options=[S2SRequestOptions(system_prompt=inference_cfg.s2s.get("system_prompt"))],
        )

        incremental_audio = None
        incremental_audio_path = None
        audio_sample_count_equal = None
        if getattr(pipeline_output, "audio_filepaths", None):
            incremental_audio_path = pipeline_output.audio_filepaths[0]
        if incremental_audio_path:
            incremental_audio, _ = sf.read(incremental_audio_path)
            incremental_audio = torch.tensor(incremental_audio).reshape(1, -1)
        incremental = {
            "text": [pipeline_output.texts_with_timestamps[0] if pipeline_output.texts_with_timestamps else pipeline_output.texts[0]],
            "asr_text": [pipeline_output.asr_texts_with_timestamps[0] if pipeline_output.asr_texts_with_timestamps else pipeline_output.asr_texts[0]],
            "tokens_text": pipeline_output.token_texts[0] if pipeline_output.token_texts else None,
            "asr_tokens": pipeline_output.token_asr_texts[0] if pipeline_output.token_asr_texts else None,
            "audio": incremental_audio,
        }

        incremental_debug = None
        if pipeline_output.debug_data and pipeline_output.debug_data[0]:
            incremental_debug = _merge_incremental_debug_steps(pipeline_output.debug_data[0])

    debug_comparison = _compare_debug_outputs(offline_debug, incremental_debug) if incremental_debug else None

    report = {
        "audio_path": args.audio_path,
        "total_samples": int(total_samples),
        "total_frames": total_frames,
        "num_frames_per_chunk": num_frames_per_chunk,
        "buffer_size_frames": buffer_size_frames,
        "strict_runtime_parity": bool(args.strict_runtime_parity),
        "engine_type": inference_cfg.s2s.get("engine_type"),
        "use_perception_cache": bool(inference_cfg.s2s.get("use_perception_cache", False)),
        "use_llm_cache": bool(inference_cfg.s2s.get("use_llm_cache", False)),
        "use_codec_cache": bool(inference_cfg.s2s.get("use_codec_cache", False)),
        "deterministic": bool(inference_cfg.s2s.get("deterministic", False)),
        "model_dtypes": _collect_model_dtypes(wrapper),
        "comparison": _compare_outputs(offline, incremental),
        "debug_comparison": debug_comparison,
        "debug": {
            "incremental_mode": "pipeline",
            "incremental_audio_filepath": incremental_audio_path,
            "offline_debug": offline_debug,
            "incremental_debug": incremental_debug,
        },
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable_report = {k: v for k, v in report.items() if k != "debug"}
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)

    if args.strict_runtime_parity:
        comparison = report["comparison"]
        failed = []
        if not comparison["text_equal"]:
            failed.append("text")
        if comparison["token_match"] is False:
            failed.append("tokens")
        if comparison["asr_token_match"] is False:
            failed.append("asr_tokens")
        if comparison["audio_sample_count_equal"] is False:
            failed.append("audio_length")
        if failed:
            raise AssertionError(
                "Offline/incremental parity failed for: "
                + ", ".join(failed)
                + f". Report: {json.dumps({k: v for k, v in report.items() if k != 'debug'}, ensure_ascii=False)}"
            )

    return report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Nemotron VoiceChat offline inference against incremental decoding with one full-audio chunk."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to S2S/TTS checkpoint directory.")
    parser.add_argument("--llm_checkpoint_path", type=str, required=True, help="Path to LLM/perception checkpoint directory.")
    parser.add_argument("--audio_path", type=str, required=True, help="Audio file to compare across both paths.")
    parser.add_argument("--speaker_reference", type=str, default=None, help="Speaker reference audio path.")
    parser.add_argument("--speaker_name", type=str, default=None, help="Registered speaker name.")
    parser.add_argument("--config_path", type=str, default=None, help="Optional path to s2s_streaming.yaml.")
    parser.add_argument("--system_prompt", type=str, default=None, help="Optional system prompt.")
    parser.add_argument("--tts_system_prompt", type=str, default=None, help="Optional TTS system prompt.")
    parser.add_argument(
        "--num_frames_per_chunk",
        type=int,
        default=None,
        help="Override incremental chunk size in 80ms frames. If unset, defaults to full audio length.",
    )
    parser.add_argument(
        "--chunk_size_in_secs",
        type=float,
        default=None,
        help="Override incremental chunk size in seconds. If set, converted to 80ms frames. If unset, defaults to full audio length.",
    )
    parser.add_argument("--engine_type", type=str, default=None, help="Override engine type.")
    parser.add_argument("--compute_dtype", type=str, default=None, help="Override compute dtype (for example: float32, bfloat16).")
    _bool_arg(parser, "--decode_audio", "Whether to decode waveform outputs.")
    _bool_arg(parser, "--use_perception_cache", "Override perception cache usage.")
    _bool_arg(parser, "--use_perception_cudagraph", "Override perception CUDA-graph usage.")
    _bool_arg(parser, "--use_llm_cache", "Override LLM cache usage.")
    _bool_arg(parser, "--use_codec_cache", "Override codec cache usage.")
    _bool_arg(parser, "--deterministic", "Override deterministic mode.")
    parser.add_argument("--top_p", type=float, default=None, help="Override top-p.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Override repetition penalty.")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--strict_runtime_parity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled, force a strict native/deterministic parity profile and raise if text/token/audio-length "
            "comparisons differ."
        ),
    )
    _bool_arg(
        parser,
        "--collect_debug",
        "Collect per-step encoder outputs and logits for comparison. "
        "Stores tensors on CPU each step; disable for long audio to avoid OOM.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    report = run_parity_harness(args)
    printable = {k: v for k, v in report.items() if k != "debug"}
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
