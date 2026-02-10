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
MAX_NUM_BATCHED_TOKENS = 768


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
    )
    vllm_config = engine_args.create_engine_config()
    engine = AsyncLLM.from_vllm_config(vllm_config)
    sampling_params = SamplingParams(max_tokens=max_model_len, skip_sampling=True)
    return engine, sampling_params


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
    vllm_config = engine_args.create_engine_config()
    engine = AsyncLLM.from_vllm_config(vllm_config)
    sampling_params = SamplingParams(max_tokens=max_model_len, temperature=0.0, top_p=0.9, ignore_eos=True)
    return engine, sampling_params


async def create_eartts_engine(model_path: str, gpu_mem: float, max_model_len: int = 1024):
    import os
    from vllm.attention.selector import _cached_get_attn_backend

    # Force TRITON_ATTN backend for EarTTS specifically
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
    vllm_config = engine_args.create_engine_config()
    engine = AsyncLLM.from_vllm_config(vllm_config)
    sampling_params = SamplingParams(max_tokens=max_model_len, skip_sampling=True, ignore_eos=True, guidance_scale=0.5)

    # Restore: unset env var so it doesn't affect other engines
    os.environ.pop("VLLM_ATTENTION_BACKEND", None)
    _cached_get_attn_backend.cache_clear()

    return engine, sampling_params


def load_codec_only(s2s_ckpt_path: str, device: str = "cuda"):
    """Load only the audio codec from s2s checkpoint."""
    from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS
    from nemo.collections.speechlm2.parts.precision import fp32_precision
    
    config_path = f"{s2s_ckpt_path}/config.json"
    with open(config_path, "r") as f:
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
    
    with fp32_precision():
        model = DuplexEARTTS(OmegaConf.to_container(cfg, resolve=True))
    
    # Load TTS weights (includes codec)
    weights = load_file(f"{s2s_ckpt_path}/model.safetensors")
    tts_weights = {k.replace("tts_model.", ""): v 
                   for k, v in weights.items() if k.startswith("tts_model.")}
    model.load_state_dict(tts_weights, strict=False)
    
    # Extract codec and silence tokens before deleting model parts
    audio_codec = model.audio_codec
    silence_tokens = model.codec_silence_tokens.clone()
    target_fps = model.target_fps  # frames per second for TTS
    
    # Delete everything except codec
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


async def run_inference(
    audio_path: str,
    tts_prompt_data_path: str,
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
    
    # Load EarTTS prompt data (prefill inputs from model_factory.py)
    prompt_data = torch.load(tts_prompt_data_path)
    prompt_acoustic_tokens = prompt_data["prompt_acoustic_tokens"]
    prompt_subword_ids = prompt_data["prompt_subword_ids"]
    prompt_subword_mask = prompt_data["prompt_subword_mask"]
    prompt_bos_mask = prompt_data["prompt_bos_mask"]
    prev_acoustic_tokens = prompt_data["first_acoustic_tokens"]
    prompt_len = prompt_acoustic_tokens.shape[0]
    
    # Load audio
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    audio = torch.from_numpy(audio).float()
    audio = preemphasis(audio)
    num_frames = audio.shape[0] // FRAME_SIZE_SAMPLES
    print(f"Audio: {audio_path}, frames: {num_frames}")
    
    # Load FastConformer engine
    print("Loading FastConformer engine...")
    fc_engine, fc_sampling = await create_fastconformer_engine(fastconformer_path, gpu_mems[0], max_model_len)
    print_gpu_memory("After FastConformer")
    
    # Load LLM engine
    print("Loading LLM engine...")
    llm_engine, llm_sampling = await create_llm_engine(llm_path, gpu_mems[1], max_model_len)
    print_gpu_memory("After LLM")
    
    # Load EarTTS engine
    print("Loading EarTTS engine...")
    tts_engine, tts_sampling = await create_eartts_engine(eartts_path, gpu_mems[2], max_model_len)
    print_gpu_memory("After EarTTS")
    
    # Load codec
    print("Loading codec...")
    codec, silence_tokens, target_fps = load_codec_only(s2s_ckpt_path, device)
    print_gpu_memory("After codec")
    
    # Codec buffer parameters (matching inference_streaming_realtime.py)
    codec_token_history_size = 60  # sliding window of 60 frames
    samples_per_frame = int(TTS_SAMPLE_RATE / target_fps)  # samples per 80ms frame
    
    # Init state
    pad_id = 12  # <SPECIAL_12>
    prev_text_token = pad_id
    prev_asr_token = pad_id
    # Use last token from prompt as initial prev_acoustic_tokens (not silence!)
    # This matches inference_streaming_realtime.py: code = self.first_tts_code_input.detach().clone()
    
    # Initialize codec token buffer with silence (sliding window)
    # Shape: [1, codec_token_history_size, 31]
    audio_toks_buffer = silence_tokens.view(1, 1, -1).expand(1, codec_token_history_size, -1).clone()
    
    gen_text_tokens = []
    gen_asr_tokens = []
    gen_acoustic_tokens = []
    audio_segments = []  # For per-chunk decoded audio
    
    fc_request_id = "fc_0"
    llm_request_id = "llm_0"
    tts_request_id = "tts_0"
    
    print("Starting inference...")
    fc_iter = None
    llm_iter = None
    
    # TTS prefill before loop (only needs prompt data from npz)
    t0 = time.time()
    tts_inputs = {
        "prompt_token_ids": [0] * prompt_len,
        "custom_inputs": {
            "acoustic_tokens": prompt_acoustic_tokens,
            "text_tokens": prompt_subword_ids,
            "text_mask": prompt_subword_mask,
            "bos_mask": prompt_bos_mask,
        }
    }
    tts_iter = tts_engine.generate(tts_inputs, sampling_params=tts_sampling, request_id=tts_request_id)
    tts_output = await tts_iter.__anext__()
    timings["eartts_prefill"].append(time.time() - t0)
    print(f"TTS prefill done in {(time.time() - t0) * 1000:.2f} ms")

    # feed some silence as audio
    audio_chunk = torch.randn(1, FRAME_SIZE_SAMPLES) * 0.001
    fc_inputs = {
        "prompt_token_ids": [0],
        "custom_inputs": {"audio": audio_chunk}
    }
    t0 = time.time()
    fc_iter = fc_engine.generate(fc_inputs, sampling_params=fc_sampling, request_id=fc_request_id)
    await fc_iter.__anext__()
    timings["fastconformer_prefill"].append(time.time() - t0)
    for _ in range(7):  # somewhat affects perceived latency of LLM 
        t0 = time.time()
        await fc_engine.append_request(request_id=fc_request_id, custom_inputs={"audio": audio_chunk})
        await fc_iter.__anext__()
        timings["fastconformer_prefill"].append(time.time() - t0)
    
    for frame_idx in range(num_frames):
        # FastConformer step
        t_frame_start = time.time()
        t0 = time.time()
        audio_chunk = audio[frame_idx * FRAME_SIZE_SAMPLES : (frame_idx + 1) * FRAME_SIZE_SAMPLES]
        audio_chunk = audio_chunk.view(1, FRAME_SIZE_SAMPLES)
        
        if fc_iter is None:
            fc_inputs = {
                "prompt_token_ids": [0],
                "custom_inputs": {"audio": audio_chunk}
            }
            fc_iter = fc_engine.generate(fc_inputs, sampling_params=fc_sampling, request_id=fc_request_id)
        else:
            await fc_engine.append_request(request_id=fc_request_id, custom_inputs={"audio": audio_chunk})
        
        fc_output = await fc_iter.__anext__()
        acoustic_emb = fc_output.outputs[0].custom_outputs["acoustic_emb"].bfloat16()  # 1 x H, bf16 for LLM
        timings["fastconformer"].append(time.time() - t0)
        
        # LLM step
        t0 = time.time()
        asr_token_ids = torch.tensor([prev_asr_token], dtype=torch.int32)
        
        if llm_iter is None:
            llm_inputs = {
                "prompt_token_ids": [prev_text_token],
                "custom_inputs": {
                    "acoustic_embeds": acoustic_emb,
                    "asr_token_ids": asr_token_ids,
                }
            }
            llm_iter = llm_engine.generate(llm_inputs, sampling_params=llm_sampling, request_id=llm_request_id)
        else:
            await llm_engine.append_request(
                request_id=llm_request_id,
                custom_inputs={
                    "acoustic_embeds": acoustic_emb,
                    "asr_token_ids": asr_token_ids,
                }
            )
        
        llm_output = await llm_iter.__anext__()
        text_token = llm_output.outputs[0].token_ids[-1]
        asr_token = llm_output.outputs[0].custom_outputs["asr_tokens"][-1].item()
        timings["llm"].append(time.time() - t0)
        
        gen_text_tokens.append(text_token)
        gen_asr_tokens.append(asr_token)
        prev_text_token = text_token
        prev_asr_token = asr_token
        
        # EarTTS step (prefill already done before loop)
        t0 = time.time()
        current_subword_id = torch.tensor([text_token], dtype=torch.int32)
        subword_mask_step = torch.ones(1, dtype=torch.float32)
        bos_mask_step = torch.zeros(1, dtype=torch.float32)
        
        await tts_engine.append_request(
            request_id=tts_request_id,
            custom_inputs={
                "acoustic_tokens": prev_acoustic_tokens,
                "text_tokens": current_subword_id,
                "text_mask": subword_mask_step,
                "bos_mask": bos_mask_step,
            }
        )
        
        tts_output = await tts_iter.__anext__()
        acoustic_tokens = tts_output.outputs[0].custom_outputs["acoustic_tokens"][-1:]  # 1 x 31
        timings["eartts"].append(time.time() - t0)
        
        gen_acoustic_tokens.append(acoustic_tokens.clone())
        prev_acoustic_tokens = gen_acoustic_tokens[-1]
        
        # Per-chunk codec decoding (sliding window approach)
        # Update buffer: slide left by 1, append new token
        t0 = time.time()
        audio_toks_buffer = torch.cat([audio_toks_buffer[:, 1:, :], acoustic_tokens.unsqueeze(0).to(device)], dim=1)
        
        # Replace control codes (1024) with 0 before decoding
        buffer_for_decode = audio_toks_buffer.clone()
        buffer_for_decode[buffer_for_decode == 1024] = 0
        
        # Codec expects [B, T, C] where C=31 codebooks - no transpose needed
        buffer_len = torch.tensor([codec_token_history_size], dtype=torch.long, device=device)
        
        with torch.no_grad():
            decoded_audio, _ = codec.decode(buffer_for_decode, buffer_len)
        
        # Extract only the newest samples (last frame's worth)
        decoded_audio_new = decoded_audio[:, :, -samples_per_frame:]
        audio_segments.append(decoded_audio_new)
        timings["codec"].append(time.time() - t0)
        timings["frame"].append(time.time() - t_frame_start)
        
        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx}/{num_frames}", flush=True)

    # Concatenate all per-chunk decoded audio
    print("Concatenating audio segments...")
    audio_out = torch.cat(audio_segments, dim=-1)
    
    # Save output
    audio_out_np = audio_out.squeeze().cpu().numpy()
    
    # Combine input and output as stereo
    audio_in, _ = librosa.load(audio_path, sr=TTS_SAMPLE_RATE)
    max_len = max(len(audio_in), len(audio_out_np))
    audio_in_padded = np.pad(audio_in, (0, max_len - len(audio_in)))
    audio_out_padded = np.pad(audio_out_np, (0, max_len - len(audio_out_np)))
    stereo = np.stack([audio_in_padded, audio_out_padded], axis=1)
    sf.write(output_path, stereo, TTS_SAMPLE_RATE)
    print(f"Saved: {output_path}\n")
    print(f"Recognized ASR tokens: {gen_asr_tokens}")
    print(f"Generated text tokens: {gen_text_tokens}")
    
    # Detokenize using HuggingFace tokenizer (runs on CPU)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    asr_text = tokenizer.decode(gen_asr_tokens).replace('Ġ', ' ').replace('<SPECIAL_12>', '')
    gen_text = tokenizer.decode(gen_text_tokens).replace('Ġ', ' ').replace('<SPECIAL_12>', '')
    print(f"ASR text: {asr_text}")
    print(f"Generated text: {gen_text}")
    
    # Print timing stats
    print("\n=== Timing Stats (ms) ===")
    for name, times in timings.items():
        if times:
            times = [t * 1000 for t in times]
            mean_ms = np.mean(times[10:])
            first5 = [f"{t:.2f}" for t in times[:5]]
            last5 = [f"{t:.2f}" for t in times[-5:]]
            print(f"{name}: [{', '.join(first5)}] ... [{', '.join(last5)}], mean: {mean_ms:.2f} ms (n={len(times)})")
    
    # Cleanup
    await fc_engine.abort(fc_request_id)
    await llm_engine.abort(llm_request_id)
    await tts_engine.abort(tts_request_id)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Input audio file")
    parser.add_argument("--tts-prompt-data", required=True, help="TTS prompt data file")
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
