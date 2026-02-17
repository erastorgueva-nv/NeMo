#!/usr/bin/env python3
"""Simple S2S inference using vLLM for FastConformer, Nemotron LLM, and EarTTS."""

import argparse
import asyncio
import gc
import json
import time
from collections import defaultdict

import librosa
import numpy as np
import soundfile as sf
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

SAMPLE_RATE = 16000
FRAME_SIZE_SAMPLES = 1280  # 80ms
TTS_SAMPLE_RATE = 22050


def print_gpu_memory(label: str, device_id: int = 0):
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(device_id)
        used = (total - free) / 1024**3
        total_gb = total / 1024**3
        print(f"[GPU] {label}: {used:.2f} / {total_gb:.2f} GiB used")


async def create_fastconformer_engine(model_path: str, gpu_mem: float, max_model_len: int = 1024):
    engine_args = AsyncEngineArgs(
        model=model_path,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        gpu_memory_utilization=gpu_mem,
        block_size=128,
        skip_tokenizer_init=True,
        enable_prefix_caching=False,
        dtype="float32",
        compilation_config={"cudagraph_mode": "FULL"},
    )
    engine = AsyncLLM.from_vllm_config(engine_args.create_engine_config())
    sampling = SamplingParams(max_tokens=max_model_len, skip_sampling=True)
    return engine, sampling


async def create_llm_engine(model_path: str, gpu_mem: float, max_model_len: int = 1024):
    engine_args = AsyncEngineArgs(
        model=model_path,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        gpu_memory_utilization=gpu_mem,
        trust_remote_code=True,
        mamba_ssm_cache_dtype="float32",
        dtype="bfloat16",
        skip_tokenizer_init=False,
        enable_prefix_caching=False,
    )
    engine = AsyncLLM.from_vllm_config(engine_args.create_engine_config())
    sampling = SamplingParams(max_tokens=max_model_len, skip_sampling=True, ignore_eos=True)
    return engine, sampling


async def create_eartts_engine(model_path: str, gpu_mem: float, max_model_len: int = 1024):
    import os
    from vllm.attention.selector import _cached_get_attn_backend

    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
    _cached_get_attn_backend.cache_clear()

    engine_args = AsyncEngineArgs(
        model=model_path,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        gpu_memory_utilization=gpu_mem,
        trust_remote_code=True,
        dtype="float32",
        skip_tokenizer_init=True,
        enable_prefix_caching=False,
    )
    engine = AsyncLLM.from_vllm_config(engine_args.create_engine_config())
    sampling = SamplingParams(max_tokens=max_model_len, skip_sampling=True, ignore_eos=True, guidance_scale=0.5)

    os.environ.pop("VLLM_ATTENTION_BACKEND", None)
    _cached_get_attn_backend.cache_clear()
    return engine, sampling


def load_embeddings(s2s_ckpt_path: str):
    """Load embed_tokens and embed_asr_tokens from the converted LLM checkpoint (kept on CPU)."""
    weights = load_file(f"{s2s_ckpt_path}/model.safetensors")

    def make_embedding(w):
        emb = torch.nn.Embedding(w.shape[0], w.shape[1])
        emb.weight = torch.nn.Parameter(w)
        return emb.eval().to(torch.bfloat16).cpu()

    embed_tokens = make_embedding(weights["stt_model.embed_tokens.weight"])
    embed_asr_tokens = make_embedding(weights["stt_model.embed_asr_tokens.weight"])
    del weights
    gc.collect()
    return embed_tokens, embed_asr_tokens


def load_codec(s2s_ckpt_path: str, device: str = "cuda"):
    """Load only the audio codec from s2s checkpoint."""
    from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS
    from nemo.collections.speechlm2.parts.precision import fp32_precision

    with open(f"{s2s_ckpt_path}/config.json", "r") as f:
        config_dict = json.load(f)["model"]["speech_generation"]
    cfg = DictConfig(config_dict)
    cfg.model.tts_config.use_unshifthed_prompt = True
    cfg.data.add_audio_prompt_after_description = True
    cfg.model.subword_mask_exactly_as_eartts = False
    cfg.model.context_hidden_mask_exactly_as_eartts = False
    cfg.model.tts_config.disable_eos_prediction = True
    cfg.model.inference_force_speech_silence_on_eos = True
    cfg.model.use_word_sep_tokenizer = False
    cfg.model.num_delay_speech_tokens = 0
    cfg.data.source_sample_rate = 22050
    cfg.data.target_sample_rate = 22050
    cfg.model.pretrained_model = None

    # Compatibility fix: remove 'pretrained_tokenizer_name' from cas_config
    # (the new codebase's CharAwareSubwordEncoder no longer accepts this parameter;
    #  NemotronVoiceChat.__init__ handles this, but we bypass it here)
    _pretrained_tokenizer_name = None
    if hasattr(cfg.model, "tts_config") and hasattr(cfg.model.tts_config, "cas_config"):
        _pretrained_tokenizer_name = cfg.model.tts_config.cas_config.get("pretrained_tokenizer_name", None)
        if _pretrained_tokenizer_name is not None:
            del cfg.model.tts_config.cas_config.pretrained_tokenizer_name

    with fp32_precision():
        model = DuplexEARTTS(OmegaConf.to_container(cfg, resolve=True))

    weights = load_file(f"{s2s_ckpt_path}/model.safetensors")
    tts_weights = {k.replace("tts_model.", ""): v
                   for k, v in weights.items() if k.startswith("tts_model.")}
    model.load_state_dict(tts_weights, strict=False)

    audio_codec = model.audio_codec
    silence_tokens = model.codec_silence_tokens.clone()
    target_fps = model.target_fps

    del model.tts_model
    if hasattr(model, 'embed_tokens'):
        del model.embed_tokens
    del model
    gc.collect()
    torch.cuda.empty_cache()

    audio_codec.eval().to(device)
    return audio_codec, silence_tokens.to(device), target_fps


def preemphasis(audio: torch.Tensor, coef: float = 0.97) -> torch.Tensor:
    return torch.cat([audio[0:1], audio[1:] - coef * audio[:-1]], dim=0)


# --- Warmup / prefill methods ---

async def warmup_fastconformer(engine, sampling, request_id):
    audio_chunk = torch.randn(1, FRAME_SIZE_SAMPLES) * 0.001
    inputs = {"prompt_token_ids": [0], "custom_inputs": {"audio": audio_chunk}}
    it = engine.generate(inputs, sampling_params=sampling, request_id=request_id)
    await it.__anext__()
    for _ in range(7):
        await engine.append_request(request_id=request_id, custom_inputs={"audio": audio_chunk})
        await it.__anext__()
    return it


async def warmup_llm(engine, sampling, request_id, prefill_data):
    prompt_len = prefill_data.shape[0]
    inputs = {
        "prompt_token_ids": [0] * prompt_len,
        "custom_inputs": {"combined_embeds": prefill_data.cpu().to(torch.bfloat16)},
    }
    it = engine.generate(inputs, sampling_params=sampling, request_id=request_id)
    await it.__anext__()
    return it


async def warmup_eartts(engine, sampling, request_id, prompt_data):
    prompt_len = prompt_data["prompt_acoustic_tokens"].shape[0]
    inputs = {
        "prompt_token_ids": [0] * prompt_len,
        "custom_inputs": {
            "acoustic_tokens": prompt_data["prompt_acoustic_tokens"].cpu().to(torch.int32),
            "text_tokens": prompt_data["prompt_subword_ids"].cpu().to(torch.int32),
            "text_mask": prompt_data["prompt_subword_mask"].cpu().to(torch.float32),
            "bos_mask": prompt_data["prompt_bos_mask"].cpu().to(torch.float32),
        },
    }
    it = engine.generate(inputs, sampling_params=sampling, request_id=request_id)
    await it.__anext__()
    return it


async def run_inference(
    audio_path: str,
    tts_prompt_data_path: str,
    llm_prompt_data_path: str,
    output_path: str,
    fastconformer_path: str,
    llm_path: str,
    eartts_path: str,
    s2s_ckpt_path: str,
    gpu_mems: list,
    max_model_len: int = 1024,
):
    device = "cuda"
    timings = defaultdict(list)

    # Load prompt data
    tts_prompt_data = torch.load(tts_prompt_data_path)
    llm_prompt_data = torch.load(llm_prompt_data_path)
    prev_acoustic_tokens = tts_prompt_data["first_acoustic_tokens"]

    # Load audio
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    audio = torch.from_numpy(audio).float()
    audio = preemphasis(audio)
    num_frames = audio.shape[0] // FRAME_SIZE_SAMPLES
    print(f"Audio: {audio_path}, frames: {num_frames}")

    # Load engines
    print("Loading FastConformer engine...")
    fc_engine, fc_sampling = await create_fastconformer_engine(fastconformer_path, gpu_mems[0], max_model_len)
    print_gpu_memory("After FastConformer")

    print("Loading LLM engine...")
    llm_engine, llm_sampling = await create_llm_engine(llm_path, gpu_mems[1], max_model_len)
    print_gpu_memory("After LLM")

    print("Loading EarTTS engine...")
    tts_engine, tts_sampling = await create_eartts_engine(eartts_path, gpu_mems[2], max_model_len)
    print_gpu_memory("After EarTTS")

    # Load embedding layers for preparing LLM combined embeddings
    print("Loading embedding layers...")
    embed_tokens, embed_asr_tokens = load_embeddings(llm_path)

    # Load codec
    print("Loading codec...")
    codec, silence_tokens, target_fps = load_codec(s2s_ckpt_path, device)
    print_gpu_memory("After codec")

    codec_token_history_size = 60
    samples_per_frame = int(TTS_SAMPLE_RATE / target_fps)

    # Init state
    pad_id = 12  # <SPECIAL_12>
    prev_text_token = pad_id
    prev_asr_token = pad_id
    audio_toks_buffer = silence_tokens.view(1, 1, -1).expand(1, codec_token_history_size, -1).clone()

    gen_text_tokens = []
    gen_asr_tokens = []
    gen_acoustic_tokens = []
    audio_segments = []

    fc_rid = "fc_0"
    llm_rid = "llm_0"
    tts_rid = "tts_0"

    # Warmup all models
    print("Warming up FastConformer...")
    t0 = time.time()
    fc_iter = await warmup_fastconformer(fc_engine, fc_sampling, fc_rid)
    print(f"  done in {(time.time() - t0) * 1000:.0f} ms")

    print("Warming up LLM...")
    t0 = time.time()
    llm_iter = await warmup_llm(llm_engine, llm_sampling, llm_rid, llm_prompt_data)
    print(f"  done in {(time.time() - t0) * 1000:.0f} ms")

    print("Warming up EarTTS...")
    t0 = time.time()
    tts_iter = await warmup_eartts(tts_engine, tts_sampling, tts_rid, tts_prompt_data)
    print(f"  done in {(time.time() - t0) * 1000:.0f} ms")

    print("Starting inference...")
    for frame_idx in range(num_frames):
        t_frame_start = time.time()

        # --- FastConformer ---
        t0 = time.time()
        audio_chunk = audio[frame_idx * FRAME_SIZE_SAMPLES : (frame_idx + 1) * FRAME_SIZE_SAMPLES].view(1, FRAME_SIZE_SAMPLES)
        await fc_engine.append_request(request_id=fc_rid, custom_inputs={"audio": audio_chunk})
        fc_output = await fc_iter.__anext__()
        acoustic_emb = fc_output.outputs[0].custom_outputs["acoustic_emb"].bfloat16()  # [1, H]
        timings["fastconformer"].append(time.time() - t0)

        # --- LLM (combined embeddings + external sampling) ---
        t0 = time.time()
        text_emb = embed_tokens(torch.tensor([prev_text_token]))
        asr_emb = embed_asr_tokens(torch.tensor([prev_asr_token]))
        combined_emb = acoustic_emb.cpu() + text_emb + asr_emb

        await llm_engine.append_request(
            request_id=llm_rid,
            custom_inputs={"combined_embeds": combined_emb},
        )
        llm_output = await llm_iter.__anext__()

        text_logits = llm_output.outputs[0].custom_outputs["text_logits"][-1]  # [V]
        text_token = text_logits.argmax().item()
        asr_token = llm_output.outputs[0].custom_outputs["asr_tokens"][-1].item()
        timings["llm"].append(time.time() - t0)

        gen_text_tokens.append(text_token)
        gen_asr_tokens.append(asr_token)
        prev_text_token = text_token
        prev_asr_token = asr_token

        # --- EarTTS ---
        t0 = time.time()
        await tts_engine.append_request(
            request_id=tts_rid,
            custom_inputs={
                "acoustic_tokens": prev_acoustic_tokens,
                "text_tokens": torch.tensor([text_token], dtype=torch.int32),
                "text_mask": torch.ones(1, dtype=torch.float32),
                "bos_mask": torch.zeros(1, dtype=torch.float32),
            },
        )
        tts_output = await tts_iter.__anext__()
        acoustic_tokens = tts_output.outputs[0].custom_outputs["acoustic_tokens"][-1:]  # [1, 31]
        timings["eartts"].append(time.time() - t0)

        gen_acoustic_tokens.append(acoustic_tokens.clone())
        prev_acoustic_tokens = gen_acoustic_tokens[-1]

        # --- Codec decode (sliding window) ---
        t0 = time.time()
        audio_toks_buffer = torch.cat([audio_toks_buffer[:, 1:, :], acoustic_tokens.unsqueeze(0).to(device)], dim=1)
        buffer_for_decode = audio_toks_buffer.clone()
        buffer_for_decode[buffer_for_decode == 1024] = 0
        buffer_len = torch.tensor([codec_token_history_size], dtype=torch.long, device=device)
        with torch.no_grad():
            decoded_audio, _ = codec.decode(buffer_for_decode, buffer_len)
        audio_segments.append(decoded_audio[:, :, -samples_per_frame:])
        timings["codec"].append(time.time() - t0)

        timings["frame"].append(time.time() - t_frame_start)
        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx}/{num_frames}", flush=True)

    # Concatenate and save
    print("Concatenating audio segments...")
    audio_out = torch.cat(audio_segments, dim=-1)
    audio_out_np = audio_out.squeeze().cpu().numpy()

    audio_in, _ = librosa.load(audio_path, sr=TTS_SAMPLE_RATE)
    max_len = max(len(audio_in), len(audio_out_np))
    stereo = np.stack([
        np.pad(audio_in, (0, max_len - len(audio_in))),
        np.pad(audio_out_np, (0, max_len - len(audio_out_np))),
    ], axis=1)
    sf.write(output_path, stereo, TTS_SAMPLE_RATE)
    print(f"Saved: {output_path}")

    # Detokenize
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    asr_text = tokenizer.decode(gen_asr_tokens).replace('Ġ', ' ').replace('<SPECIAL_12>', '')
    gen_text = tokenizer.decode(gen_text_tokens).replace('Ġ', ' ').replace('<SPECIAL_12>', '')
    print(f"ASR text: {asr_text}")
    print(f"Generated text: {gen_text}")

    # Timing stats
    print("\n=== Timing Stats (ms) ===")
    for name, times in timings.items():
        if times:
            times_ms = [t * 1000 for t in times]
            mean_ms = np.mean(times_ms[10:]) if len(times_ms) > 10 else np.mean(times_ms)
            first5 = [f"{t:.2f}" for t in times_ms[:5]]
            last5 = [f"{t:.2f}" for t in times_ms[-5:]]
            print(f"{name}: [{', '.join(first5)}] ... [{', '.join(last5)}], mean: {mean_ms:.2f} ms (n={len(times_ms)})")

    # Cleanup
    await fc_engine.abort(fc_rid)
    await llm_engine.abort(llm_rid)
    await tts_engine.abort(tts_rid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Input audio file")
    parser.add_argument("--tts-prompt-data", required=True, help="TTS prompt data file (.pt)")
    parser.add_argument("--llm-prompt-data", required=True, help="LLM prefill data file (.pt)")
    parser.add_argument("--output", required=True, help="Output audio file")
    parser.add_argument("--fastconformer", required=True, help="FastConformer vLLM checkpoint")
    parser.add_argument("--llm", required=True, help="Nemotron LLM vLLM checkpoint")
    parser.add_argument("--eartts", required=True, help="EarTTS vLLM checkpoint")
    parser.add_argument("--s2s-ckpt", required=True, help="S2S checkpoint (for codec)")
    parser.add_argument("--gpu-mems", nargs=3, type=float, default=[0.3, 0.4, 0.2],
                        help="GPU memory utilization for [fastconformer, llm, eartts]")
    parser.add_argument("--max-model-len", type=int, default=280,
                        help="Max model length for all vLLM engines")
    args = parser.parse_args()
    asyncio.run(run_inference(
        audio_path=args.audio,
        tts_prompt_data_path=args.tts_prompt_data,
        llm_prompt_data_path=args.llm_prompt_data,
        output_path=args.output,
        fastconformer_path=args.fastconformer,
        llm_path=args.llm,
        eartts_path=args.eartts,
        s2s_ckpt_path=args.s2s_ckpt,
        gpu_mems=args.gpu_mems,
        max_model_len=args.max_model_len,
    ))


if __name__ == "__main__":
    main()
