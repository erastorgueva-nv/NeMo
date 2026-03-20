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

import gc
import os
import time
import types
from typing import Optional, Tuple

import torch
import torchaudio
from omegaconf import OmegaConf, DictConfig

from nemo.utils import logging
from transformers import DynamicCache

from nemo.collections.speechlm2.models.nemotron_voicechat import NemotronVoiceChat
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.speechlm2.modules.ear_tts_vae_codec import CausalConv1dCache
from nemo.collections.speechlm2.inference.model_wrappers.model_factory import create_model
from nemo.collections.speechlm2.inference.model_wrappers.perception_cache import (
    PerceptionCacheState,
    PerceptionCacheManager,
)


# --- Configuration ---
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Streaming Parameters ---
SAMPLE_RATE = 16000
FRAME_SIZE_SEC = 0.08  # 80ms per frame
FRAME_SIZE_SAMPLES = int(SAMPLE_RATE * FRAME_SIZE_SEC)  # 1280 samples

TTS_SAMPLE_RATE = 22050


# Default hyper-parameters that can be overridden via `model_cfg`
DEFAULT_BUFFER_SIZE_FRAMES = 71
DEFAULT_NUM_FRAMES_PER_CHUNK = 1
# Only used when use_codec_cache=False (sliding-window fallback).
# Ignored when the codec streaming cache is enabled.
DEFAULT_CODEC_TOKEN_HISTORY_SIZE = 600


class NemotronVoicechatInferenceWrapper:
    """
    Inference wrapper for NemotronVoiceChat models.
    Uses a sliding window buffer and processes audio frame by frame.
    """

    def __init__(self, model_cfg: DictConfig):
        """
        Initialize the model for realtime streaming inference.

        Args:
            model_cfg (DictConfig): Configuration describing the model paths and inference parameters.
        """
        if model_cfg is None:
            raise ValueError("model_cfg must be provided")
        if not isinstance(model_cfg, DictConfig):
            model_cfg = OmegaConf.create(model_cfg)

        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")

        self._deterministic = bool(model_cfg.get("deterministic", False))
        if self._deterministic:
            engine_type = model_cfg.get("engine_type", "native")
            if "vllm" in engine_type.lower():
                raise ValueError(
                    "`deterministic` is not compatible with vLLM engines because vLLM uses custom "
                    "CUDA kernels (PagedAttention, FlashAttention) that do not support deterministic mode. "
                    f"Got engine_type='{engine_type}'. Use engine_type='native' for deterministic inference."
                )

            # Required by torch.use_deterministic_algorithms for cuBLAS reproducibility
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.use_deterministic_algorithms(True, warn_only=False)

            logging.info("Deterministic mode ENABLED")
            logging.info(f"  CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
            logging.info(f"  flash_sdp enabled: {torch.backends.cuda.flash_sdp_enabled()}")
            logging.info(f"  mem_efficient_sdp enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
            logging.info(
                "  NOTE: deterministic mode uses different CUDA kernels (e.g. math SDPA instead of "
                "FlashAttention), so results may differ slightly from non-deterministic mode. "
                "Inference will also be slower."
            )

        self.model_cfg = model_cfg

        self.model_path = model_cfg.get("model_path")
        if not self.model_path:
            raise ValueError("`model_cfg.model_path` must be provided.")

        self.decode_audio = bool(model_cfg.get("decode_audio", True))
        # Number of past codec tokens kept in the sliding-window decode buffer.
        # Only used when use_codec_cache=False (the fallback path). When the
        # codec cache is enabled, context is maintained incrementally inside
        # CausalConv1dCache and this value is ignored.
        self.codec_token_history_size = int(
            model_cfg.get("codec_token_history_size", DEFAULT_CODEC_TOKEN_HISTORY_SIZE)
        )

        self.speaker_reference = model_cfg.get("speaker_reference")
        self.speaker_name = model_cfg.get("speaker_name", None)
        if self.decode_audio and not self.speaker_reference and not self.speaker_name:
            raise ValueError("`model_cfg.speaker_reference` or `model_cfg.speaker_name` must be provided when decode_audio is enabled.")

        self.tts_system_prompt = model_cfg.get("tts_system_prompt", None)
        logging.info(f"TTS system prompt: {self.tts_system_prompt}")

        compute_dtype = model_cfg.get("compute_dtype", "bfloat16")
        self.dtype = self._resolve_dtype(compute_dtype)

        self.device = self._resolve_device(
            device=model_cfg.get("device"),
            device_id=model_cfg.get("device_id"),
        )

        logging.info("=" * 70)
        logging.info("INITIALIZING REALTIME STREAMING INFERENCE")
        logging.info("=" * 70)
        logging.info(f"Frame size: {FRAME_SIZE_SEC}s ({FRAME_SIZE_SAMPLES} samples @ {SAMPLE_RATE}Hz)")
        logging.info(f"Device: {self.device}")
        logging.info(f"Compute dtype: {self.dtype}")
        logging.info(f"Decode audio: {self.decode_audio}")
        logging.info(f"Engine type: {model_cfg.get('engine_type', 'native')}")
        logging.info(f"Sampling - top_p: {model_cfg.get('top_p', 1.0)}, repetition_penalty: {model_cfg.get('repetition_penalty', 1.0)}, temperature: {model_cfg.get('temperature', 1.0)}")
        logging.info(f"Float32 matmul precision: {torch.get_float32_matmul_precision()}")
        logging.info("=" * 70)

        # Cached TTS helpers populated during initialization/warmup
        self.first_context_subword_id = None
        self.generation_config = None
        self.first_tts_code_input = None
        self.first_tts_past_key_values_input = None

        self.model = None
        self.model_llm_interface = None
        self.tokenizer = None

        # vLLM configuration
        self.engine_type = model_cfg.get("engine_type", "native")
        self.use_vllm_llm = "vllm_llm" in self.engine_type.lower()
        self.use_vllm_eartts = "vllm_eartts" in self.engine_type.lower()
        self.vllm_llm_config = model_cfg.get("vllm_llm_config", None)
        self.vllm_tts_config = model_cfg.get("vllm_tts_config", None)
        self.request_id = "streaming_request_0"  # For vLLM streaming

        # Sampling parameters
        self.top_p = float(model_cfg.get("top_p", 1.0))
        self.repetition_penalty = float(model_cfg.get("repetition_penalty", 1.0))
        self.temperature = float(model_cfg.get("temperature", 1.0))

        # Codec streaming cache: decode only new tokens each step using the
        # codec's CausalConv1dCache, which maintains ConvNeXt and ISTFT state
        # across calls for sample-continuous audio. When enabled, the
        # codec_token_history_size parameter and audio_toks_buffer are unused.
        # When disabled, falls back to the sliding-window decode that re-decodes
        # codec_token_history_size tokens each step and extracts the tail.
        self.use_codec_cache = bool(model_cfg.get("use_codec_cache", True))
        if self.use_codec_cache and self.decode_audio:
            configured_history = model_cfg.get("codec_token_history_size", None)
            if configured_history is not None:
                logging.info(
                    f"use_codec_cache is enabled — codec_token_history_size ({configured_history}) "
                    f"will be ignored (context is maintained incrementally by the codec cache)."
                )

        # LLM KV cache: when enabled, uses DynamicCache (standard) or
        # HybridMambaAttentionDynamicCache (Nemotron) for incremental decoding.
        # When disabled, falls back to full-history reprocessing each step.
        self.use_llm_cache = bool(model_cfg.get("use_llm_cache", True))

        # Perception cache configuration
        self.use_perception_cache = bool(model_cfg.get("use_perception_cache", False))
        use_perception_cudagraph = bool(model_cfg.get("use_perception_cudagraph", False))
        if use_perception_cudagraph and not self.use_perception_cache:
            raise ValueError(
                "use_perception_cudagraph requires use_perception_cache to be enabled. "
                "Please also set use_perception_cache=True."
            )
        self.perception_cache_mgr: Optional[PerceptionCacheManager] = None
        self._use_perception_cudagraph = use_perception_cudagraph

        self._initialize_model()

        logging.info("NemotronVoicechatInferenceWrapper initialized successfully.")

    @staticmethod
    def _resolve_dtype(compute_dtype):
        if isinstance(compute_dtype, torch.dtype):
            return compute_dtype
        if compute_dtype is None:
            return torch.bfloat16
        if isinstance(compute_dtype, str):
            key = compute_dtype.lower()
            mapping = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "full": torch.float32,
            }
            if key in mapping:
                return mapping[key]
        raise ValueError(f"Unsupported compute_dtype: {compute_dtype}")

    @staticmethod
    def _resolve_device(device=None, device_id=None):
        if isinstance(device, torch.device):
            resolved_device = device
        else:
            if device is None:
                resolved_device = DEFAULT_DEVICE
            else:
                device_str = str(device)
                base = device_str
                if device_id is not None and device_str.startswith("cuda") and ":" not in device_str:
                    base = f"{device_str}:{device_id}"
                resolved_device = torch.device(base)
        return resolved_device

    def _samples_per_audio_output_frame(self):
        rate = getattr(self, "target_sample_rate", None)
        if rate is None:
            cfg_rate = None
            try:
                cfg_rate = self.model_cfg.get("tts_sample_rate", None)
            except Exception:
                cfg_rate = None
            if cfg_rate is None:
                try:
                    cfg_rate = self.model_cfg.get("output_sample_rate", None)
                except Exception:
                    cfg_rate = None
            if cfg_rate is not None:
                rate = float(cfg_rate)
        if rate is None:
            rate = TTS_SAMPLE_RATE
        samples = int(float(rate) * FRAME_SIZE_SEC)
        return samples

    def _initialize_model(self):
        """Initialize the NemotronVoiceChat model from an HF checkpoint."""
        logging.info("Initializing model structure...")
        start_model_init = time.time()

        self.model = NemotronVoiceChat.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        logging.info(f"NemotronVoiceChat initialized in {time.time() - start_model_init:.1f}s")

        if self.use_vllm_eartts:
            # Use object.__setattr__ to bypass PyTorch's module registration
            # since VllmEARTTSModel is not a torch.nn.Module
            del self.model.tts_model.tts_model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            object.__setattr__(
                self.model.tts_model,
                'tts_model',
                create_model(
                    model=self.model_path,
                    engine_type="vllm_eartts",
                    vllm_config=self.vllm_tts_config)
            )
            from nemo.collections.speechlm2.inference.vllm.vllm_patch import patched_infer_codes_one_step
            self.model.tts_model.infer_codes_one_step = types.MethodType(patched_infer_codes_one_step, self.model.tts_model)

        # If using vLLM for LLM, delete native LLM BEFORE moving to device to save memory
        if self.use_vllm_llm:
            logging.info("\nDeleting native LLM before GPU transfer (will use vLLM instead)...")
            if hasattr(self.model.stt_model, 'llm') and self.model.stt_model.llm is not None:
                # Delete all submodules of LLM to free memory
                for name, child in list(self.model.stt_model.llm.named_children()):
                    delattr(self.model.stt_model.llm, name)
                del self.model.stt_model.llm
                self.model.stt_model.llm = None
            gc.collect()
            torch.cuda.empty_cache()
            logging.info("  Native LLM deleted")

        # Setup model
        self.model.to(self.device)
        self.model.eval()

        # Convert some S2S components to the configured dtype
        logging.info(f"Converting some S2S components to {self.dtype} (keeping perception & TTS in float32)...")
        self.model.stt_model.llm = self.model.stt_model.llm.to(self.dtype)
        self.model.stt_model.lm_head = self.model.stt_model.lm_head.to(self.dtype)
        self.model.stt_model.embed_tokens = self.model.stt_model.embed_tokens.to(self.dtype)
        self.model.stt_model.asr_head = self.model.stt_model.asr_head.to(self.dtype)
        self.model.stt_model.embed_asr_tokens = self.model.stt_model.embed_asr_tokens.to(self.dtype)
        if self.model.stt_model.function_head is not None:
            self.model.stt_model.function_head = self.model.stt_model.function_head.to(self.dtype)
            logging.info("function_head converted to %s", self.dtype)

        # torch.compile for native TTS backbone
        use_tts_torch_compile = bool(self.model_cfg.get("use_tts_torch_compile", False))
        if use_tts_torch_compile and not self.use_vllm_eartts and hasattr(self.model, 'tts_model'):
            tts_backbone = getattr(self.model.tts_model, 'tts_model', None)
            if tts_backbone is not None and hasattr(tts_backbone, 'backbone'):
                logging.info("Compiling TTS backbone with torch.compile(mode='default')...")
                tts_backbone.backbone = torch.compile(tts_backbone.backbone, mode="default")
                logging.info("  TTS backbone compiled")

        # Inject TTS speedup flags into the TTS model config so ear_tts_model.py can read them
        tts_inner = getattr(self.model.tts_model, 'tts_model', None) if hasattr(self.model, 'tts_model') else None
        if tts_inner is not None and hasattr(tts_inner, 'config'):
            if bool(self.model_cfg.get("use_tts_subword_cache", False)):
                OmegaConf.update(tts_inner.config, "use_tts_subword_cache", True)
                logging.info("TTS speedup enabled: use_tts_subword_cache")
                if hasattr(tts_inner, 'embed_subword') and tts_inner.embed_subword is not None and hasattr(tts_inner.embed_subword, 'use_tts_subword_cache'):
                    tts_inner.embed_subword.use_tts_subword_cache = True

        self.tokenizer = self.model.stt_model.tokenizer

        # Allow overrides from wrapper config into the model config (e.g. logit boosts).
        _BOOST_KEYS = (
            "inference_pad_boost",
            "inference_bos_boost",
            "inference_eos_boost",
            "inference_user_pad_boost",
            "inference_user_bos_boost",
            "inference_user_eos_boost",
        )
        for key in _BOOST_KEYS:
            val = self.model_cfg.get(key, None)
            if val is not None:
                OmegaConf.update(self.model.stt_model.cfg, key, val)
        boost_values = {k: self.model.stt_model.cfg.get(k, None) for k in _BOOST_KEYS}
        logging.info(f"Inference logit boosts: {boost_values}")

        # Wrap model with appropriate interface (Native or vLLM)
        if self.use_vllm_llm:
            logging.info("\nWrapping model with VllmLLMModel interface...")
            if self.vllm_llm_config is None:
                raise ValueError("vllm_llm_config must be provided when engine_type contains 'vllm_llm'")

            # LLM already deleted above, just ensure cleanup
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Set logit boosts as env vars BEFORE creating the vLLM engine,
            # so they are inherited by the forked worker process.  The modified
            # nemotron_h.py reads VLLM_ASR_BOOST_<token_id> and
            # VLLM_TEXT_BOOST_<token_id> in __init__.
            stt = self.model.stt_model
            asr_boost_map = {
                "inference_user_pad_boost": stt.text_pad_id,
                "inference_user_bos_boost": stt.text_bos_id,
                "inference_user_eos_boost": stt.text_eos_id,
            }
            for cfg_key, token_id in asr_boost_map.items():
                val = self.model_cfg.get(cfg_key, None)
                if val is not None and float(val) != 0.0:
                    env_key = f"VLLM_ASR_BOOST_{token_id}"
                    os.environ[env_key] = str(float(val))
                    logging.info(f"Set env {env_key}={val} (from {cfg_key})")

            text_boost_map = {
                "inference_pad_boost": stt.text_pad_id,
                "inference_bos_boost": stt.text_bos_id,
                "inference_eos_boost": stt.text_eos_id,
            }
            for cfg_key, token_id in text_boost_map.items():
                val = self.model_cfg.get(cfg_key, None)
                if val is not None and float(val) != 0.0:
                    env_key = f"VLLM_TEXT_BOOST_{token_id}"
                    os.environ[env_key] = str(float(val))
                    logging.info(f"Set env {env_key}={val} (from {cfg_key})")

            self.model_llm_interface = create_model(
                model=self.model_path,
                engine_type="vllm_llm",
                vllm_config=self.vllm_llm_config,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
            )

            logging.info("VllmLLMModel interface created")
        else:
            logging.info("\nWrapping model with NativeModel interface...")
            self.model_llm_interface = create_model(
                model=self.model,
                engine_type="native",
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
            )
            logging.info("NativeModel interface created")

        # Get TTS info
        if hasattr(self.model, 'tts_model'):
            self.target_fps = self.model.tts_model.target_fps
            self.target_sample_rate = self.model.tts_model.target_sample_rate
            logging.info(f"\nTTS model initialized: target_fps={self.target_fps}, sample_rate={self.target_sample_rate}")
            if self.decode_audio:
                self._prepare_tts_initial_state()
        else:
            logging.warning("Warning: TTS model not found in the model")

        # Setup perception cache if enabled
        if self.use_perception_cache:
            self.perception_cache_mgr = PerceptionCacheManager(
                model=self.model,
                device=self.device,
                dtype=self.dtype,
                use_cudagraph=self._use_perception_cudagraph,
            )
            if not self.perception_cache_mgr.setup():
                self.use_perception_cache = False
                self.perception_cache_mgr = None

    def _get_bos_embedding(self):
        """Get beginning of sequence embedding."""
        text_bos = torch.full((1,), fill_value=self.model.stt_model.text_pad_id, device=self.device)
        input_embeds = self.model.stt_model.embed_tokens(text_bos)
        return input_embeds.to(dtype=self.dtype)

    def _get_asr_bos_embedding(self) -> torch.Tensor:
        """Get ASR BOS embedding for AR decoding."""
        text_bos = torch.full((1,), fill_value=self.model.stt_model.text_pad_id, device=self.device)
        input_embeds = self.model.stt_model.embed_asr_tokens(text_bos)
        return input_embeds.to(dtype=self.dtype)

    def _prepare_system_prompt_embeddings(
        self,
        system_prompt: str,
    ) -> Tuple[Optional[torch.Tensor], int]:
        if not system_prompt or not system_prompt.strip():
            return None, 0

        prompt_token_ids = self._build_prompt_token_ids(system_prompt)
        prompt_tokens = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        prompt_embedded = self.model.stt_model.embed_tokens(prompt_tokens).to(dtype=self.dtype)
        prompt_len = prompt_tokens.shape[1]

        pad_id = self.model.stt_model.text_pad_id
        pad_token = torch.full((1,), fill_value=pad_id, device=self.device, dtype=torch.long)
        pad_emb = self.model.stt_model.embed_tokens(pad_token).to(dtype=self.dtype)
        pad_asr_emb = self.model.stt_model.embed_asr_tokens(pad_token).to(dtype=self.dtype)

        has_fc = self.model.stt_model.function_head is not None
        if prompt_len > 1:
            prompt_embedded[:, 1:, :] += pad_emb
            prompt_embedded[:, 1:, :] += pad_asr_emb
            if has_fc:
                prompt_embedded[:, 1:, :] += pad_emb

        bos_emb = self._get_bos_embedding()
        asr_bos_emb = self._get_asr_bos_embedding()
        prompt_embedded[:, 0, :] += bos_emb.squeeze(0)
        prompt_embedded[:, 0, :] += asr_bos_emb.squeeze(0)
        if has_fc:
            prompt_embedded[:, 0, :] += pad_emb.squeeze(0)

        return prompt_embedded, prompt_len

    def _clone_cache(self, cache):
        """Deep clone cache structures to ensure complete isolation between streams."""
        if cache is None:
            return None
        if isinstance(cache, torch.Tensor):
            return cache.detach().clone()
        if isinstance(cache, (list, tuple)):
            return type(cache)(self._clone_cache(x) for x in cache)
        if isinstance(cache, dict):
            return {k: self._clone_cache(v) for k, v in cache.items()}
        if hasattr(cache, '__dict__'):
            import copy
            return copy.deepcopy(cache)
        return cache

    def _build_prompt_token_ids(self, system_prompt: str | None) -> list[int]:
        if not system_prompt or not system_prompt.strip():
            return []
        return [self.tokenizer.bos_id] + self.tokenizer.text_to_ids(system_prompt) + [self.tokenizer.eos_id]

    def _create_generation_workspace(self, max_len: int):
        stt_model = self.model.stt_model
        gen_text = torch.full((1, max_len), stt_model.text_pad_id, device=self.device, dtype=torch.long)
        gen_asr_text = torch.full((1, max_len), stt_model.text_pad_id, device=self.device, dtype=torch.long)
        gen_function_text = None
        if getattr(stt_model, "function_head", None) is not None:
            gen_function_text = torch.full((1, max_len), stt_model.text_pad_id, device=self.device, dtype=torch.long)
        return gen_text, gen_asr_text, gen_function_text

    def _create_llm_cache(self):
        if not self.use_llm_cache or self.use_vllm_llm:
            return None
        pretrained_llm = str(self.model.stt_model.cfg.get("pretrained_llm", ""))
        if "Nemotron" in pretrained_llm:
            return self.model.stt_model._create_nemotron_cache(batch_size=1)
        return DynamicCache()

    def _create_codec_state(self, max_len: int):
        if not self.decode_audio or not hasattr(self.model, "tts_model"):
            return None, None, None

        audio_toks_buffer = None
        codec_cache = None
        if self.use_codec_cache:
            codec_cache = CausalConv1dCache()
        elif self.codec_token_history_size > 0:
            silence_tokens = self.model.tts_model.codec_silence_tokens.detach().clone()
            audio_toks_buffer = silence_tokens.view(1, 1, -1).expand(
                1, self.codec_token_history_size, -1
            ).contiguous().to(self.device)

        subword_mask = torch.ones((1, max_len), device=self.device, dtype=torch.bool)
        return audio_toks_buffer, subword_mask, codec_cache

    def _prepare_tts_initial_state(self):
        if not self.decode_audio:
            return
        if not hasattr(self.model, 'tts_model'):
            return

        logging.info("Preparing TTS warmup state...")

        if self.speaker_name is not None:
            logging.info(f"Using registered speaker name: {self.speaker_name}")
            speaker_audio = None
            speaker_audio_lens = None
        else:
            with fp32_precision():
                speaker_audio, speaker_sr = torchaudio.load(self.speaker_reference)
                speaker_audio = resample(speaker_audio, speaker_sr, self.model.tts_model.target_sample_rate)
            speaker_audio = speaker_audio.to(self.device)
            speaker_audio_lens = torch.tensor([speaker_audio.size(1)], device=self.device).long()

        self.model.tts_model.set_init_inputs(
            speaker_audio=speaker_audio,
            speaker_audio_lens=speaker_audio_lens,
            system_prompt=self.tts_system_prompt,
            speaker_name=self.speaker_name,
        )
        init_inputs = self.model.tts_model.get_init_inputs(B=1)

        self.generation_config = self.model.tts_model._get_generation_config(guidance_enabled=True)
        init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": True})

        with torch.no_grad():
            if self.use_vllm_eartts:
                self.tts_prompt_token_ids = init_inputs["subword_ids"].squeeze().cpu().numpy().tolist()
                self.tts_init_inputs = init_inputs
                outputs = self.model.tts_model.tts_model(
                    self.tts_init_inputs,
                    request_id="tts_system_prompt_prefill_request",
                    prompt_token_ids=self.tts_prompt_token_ids
                )
                # abort this request
                self.model.tts_model.tts_model.abort_request("tts_system_prompt_prefill_request")
            else:
                outputs = self.model.tts_model.tts_model(**init_inputs)

            code = init_inputs["code"][:, -1:]

        self.first_context_subword_id = init_inputs["subword_ids"][:, -1].unsqueeze(-1)
        self.first_tts_code_input = code.detach().clone()
        self.first_tts_past_key_values_input = self._clone_cache(outputs.past_key_values)

        logging.info("TTS warmup state prepared")

    def create_decode_state(self, max_len: int):
        gen_text, gen_asr_text, gen_function_text = self._create_generation_workspace(max_len)
        llm_cache = self._create_llm_cache()
        audio_toks_buffer, subword_mask, codec_cache = self._create_codec_state(max_len)
        perception_cache = None
        if self.use_perception_cache and self.perception_cache_mgr is not None:
            perception_cache = self.perception_cache_mgr.get_initial_state(batch_size=1)

        past_key_values = None
        code = None
        if self.decode_audio and self.first_tts_code_input is not None:
            past_key_values = self._clone_cache(self.first_tts_past_key_values_input)
            code = self.first_tts_code_input.detach().clone()

        return {
            "frame_idx": 0,
            "gen_text": gen_text,
            "gen_asr_text": gen_asr_text,
            "gen_function_text": gen_function_text,
            "audio_toks_buffer": audio_toks_buffer,
            "input_embeds_history": [],
            "dynamic_cache": llm_cache,
            "past_key_values": past_key_values,
            "code": code,
            "subword_mask": subword_mask,
            "perception_cache": perception_cache,
            "codec_cache": codec_cache,
            "cache_position_offset": 0,
        }

    def infer_one_step(self,
                       audio_input,
                       num_frames_per_chunk,
                       frame_idx,
                       gen_text,
                       audio_toks_buffer,
                       input_embeds_history,
                       dynamic_cache,
                       past_key_values=None,
                       code=None,
                       subword_mask=None,
                       gen_asr_text=None,
                       gen_function_text=None,
                       request_id: Optional[str] = None,
                       perception_cache: Optional[PerceptionCacheState] = None,
                       has_prompt: bool = False,
                       codec_cache=None,
                       cache_position_offset: int = 0,
                       return_debug: bool = False):

        # Set up effective request ID for vLLM streaming
        effective_request_id = request_id or self.request_id

        start_time_one_step = time.time()
        use_cache = dynamic_cache is not None
        batch_size = gen_text.shape[0]

        predicted_tokens = torch.empty((batch_size, num_frames_per_chunk), dtype=gen_text.dtype, device=gen_text.device)
        asr_predicted_tokens = torch.empty((batch_size, num_frames_per_chunk), dtype=gen_text.dtype, device=gen_text.device)
        function_predicted_tokens = torch.empty((batch_size, num_frames_per_chunk), dtype=gen_text.dtype, device=gen_text.device)
        debug_text_logits = []
        debug_asr_logits = []
        debug_input_embeds = []
        selected_frame_indices = []

        # Do "perception" step outside the for-loop
        start_perception = time.time()

        if self.use_perception_cache and perception_cache is not None and perception_cache.is_initialized():
            # Cache-aware perception
            source_encoded, perception_cache = self.perception_cache_mgr.step(
                audio_input=audio_input,
                frame_idx=frame_idx,
                num_frames_per_chunk=num_frames_per_chunk,
                perception_cache=perception_cache,
            )
        else:
            # Standard perception (full buffer processing)
            buffer_len = torch.tensor([audio_input.shape[1]], dtype=torch.long, device=self.device)
            source_encoded, _, _ = self.model.stt_model.perception(
                input_signal=audio_input,
                input_signal_length=buffer_len,
                return_encoder_emb=True,
            )

        torch.cuda.synchronize()
        time_perception = time.time() - start_perception
        logging.info(f"Time taken for perception: {time_perception:.3f}s")
        source_encoded = source_encoded.to(self.dtype)
        total_encoded_frames = source_encoded.shape[1]

        # Determine embedding position based on whether we're using cache
        if self.use_perception_cache and perception_cache is not None and perception_cache.is_initialized():
            # With cache: we get exactly num_frames_per_chunk output frames
            # Use all of them directly
            embedding_position = 0
            newest_frame_index = total_encoded_frames - 1
            base_frame_index = 0
        else:
            # Without cache: Use the second-to-last encoded frame (-2) as the "newest" frame embedding.
            # This is because the model's expects the chunk sizes to be size 10ms, 80ms, 80ms, 80ms, ....,
            # but we pass in always 80ms, 80ms, 80ms....
            # e.g.
            # (1) if we pass in just one 80ms chunk -> the model treats it as 10ms, then 70ms with 10ms silence padding at the end.
            # (2) if we pass 80ms, 80ms -> the model treats it as 10ms, 80ms, 70ms with 10ms silence padding at the end.
            # => we do not want to use the final embedding due to containing silence padding. We want to use the second-to-last embedding.
            embedding_position = -2
            newest_frame_index = total_encoded_frames + embedding_position
            base_frame_index = newest_frame_index - (num_frames_per_chunk - 1)
            base_frame_index = max(base_frame_index, 0)

        new_input_embeds = []
        new_codes_for_decode = []
        for frame_offset in range(num_frames_per_chunk):
            current_frame_idx = frame_idx + frame_offset
            current_frame_index = base_frame_index + frame_offset
            current_frame_index = min(current_frame_index, total_encoded_frames - 1)
            selected_frame_indices.append(current_frame_index)
            current_frame_embedding = source_encoded[:, current_frame_index:current_frame_index + 1, :]

            current_input_emb = current_frame_embedding.clone()
            current_input_emb *= self.model.stt_model.cfg.get("duplex_user_channel_weight", 1.0)

            has_fc = gen_function_text is not None

            if current_frame_idx == 0 and not has_prompt:
                # Only add BOS if there's no prompt (BOS is already in prompt's position 0)
                current_input_emb += self._get_bos_embedding() * self.model.stt_model.cfg.get(
                    "duplex_text_channel_weight", 1.0
                )
                current_input_emb += self._get_asr_bos_embedding() * self.model.stt_model.cfg.get(
                    "duplex_asr_text_weight", 1.0
                )
                if has_fc:
                    pad_id = self.model.stt_model.text_pad_id
                    fc_pad_token = torch.full((1,), fill_value=pad_id, device=self.device, dtype=torch.long)
                    current_input_emb += self.model.stt_model.embed_tokens(fc_pad_token).to(dtype=self.dtype)
            elif current_frame_idx == 0 and has_prompt:
                # With prompt: first audio frame uses pad embedding (like offline_inference)
                # gen_text[:, -1] from prompt positions is pad_id
                pad_id = self.model.stt_model.text_pad_id
                pad_token = torch.full((1,), fill_value=pad_id, device=self.device, dtype=torch.long)
                pad_emb = self.model.stt_model.embed_tokens(pad_token).to(dtype=self.dtype)
                pad_asr_emb = self.model.stt_model.embed_asr_tokens(pad_token).to(dtype=self.dtype)
                current_input_emb += pad_emb
                current_input_emb += pad_asr_emb
                if has_fc:
                    current_input_emb += self.model.stt_model.embed_tokens(pad_token).to(dtype=self.dtype)
            else:
                # t > 0: add embeddings from model's own predictions at t-1
                last_token_emb = self.model.stt_model.embed_tokens(
                    gen_text[:, current_frame_idx - 1]
                ) * self.model.stt_model.cfg.get("duplex_text_channel_weight", 1.0)
                last_asr_token_emb = self.model.stt_model.embed_asr_tokens(
                    gen_asr_text[:, current_frame_idx - 1]
                ) * self.model.stt_model.cfg.get("duplex_asr_text_weight", 1.0)
                current_input_emb += last_token_emb + last_asr_token_emb
                if has_fc:
                    last_fc_token_emb = self.model.stt_model.embed_tokens(gen_function_text[:, current_frame_idx - 1])
                    current_input_emb += last_fc_token_emb.to(dtype=self.dtype)
            if return_debug:
                debug_input_embeds.append(current_input_emb.detach().cpu())

            start_stt_model = time.time()

            if use_cache or self.use_vllm_llm:
                if self.use_vllm_llm:
                    # vLLM requires request_id
                    ans = self.model_llm_interface(
                        current_input_emb,
                        request_id=effective_request_id,
                        generated_tokens=gen_text,
                        current_step=current_frame_idx
                    )
                else:
                    cache_pos = torch.tensor(
                        [cache_position_offset + frame_offset], device=self.device
                    )
                    ans = self.model_llm_interface(
                        current_input_emb,
                        cache=dynamic_cache,
                        cache_position=cache_pos,
                        generated_tokens=gen_text,
                        current_step=current_frame_idx,
                        return_logits=return_debug,
                    )
                dynamic_cache = ans["cache"]
            else:
                new_input_embeds.append(current_input_emb)
                full_input_embeds = torch.cat(input_embeds_history + new_input_embeds, dim=1)
                ans = self.model_llm_interface(
                    full_input_embeds,
                    cache=None,
                    generated_tokens=gen_text,
                    current_step=current_frame_idx,
                    return_logits=return_debug,
                )

            torch.cuda.synchronize()
            time_stt_model = time.time() - start_stt_model
            logging.info(f"Time taken for stt_model: {time_stt_model:.3f}s")

            predicted_token = ans["predicted_token"]
            asr_predicted_token = ans["asr_predicted_token"]
            if return_debug and "text_logits" in ans:
                debug_text_logits.append(ans["text_logits"][:, -1].detach().cpu())
            if return_debug and "asr_logits" in ans and ans["asr_logits"] is not None:
                debug_asr_logits.append(ans["asr_logits"][:, -1].detach().cpu())

            gen_text[:, current_frame_idx] = predicted_token
            predicted_tokens[:, frame_offset] = predicted_token

            gen_asr_text[:, current_frame_idx] = asr_predicted_token
            asr_predicted_tokens[:, frame_offset] = asr_predicted_token

            if "function_predicted_token" in ans:
                function_predicted_tokens[:, frame_offset] = ans["function_predicted_token"]
                if gen_function_text is not None:
                    gen_function_text[:, current_frame_idx] = ans["function_predicted_token"]

            # Apply forced turn taking based on ASR results
            self._maybe_apply_forced_turn_taking(current_frame_idx, gen_text, gen_asr_text)
            # Update predicted_tokens with any changes made by forced turn taking
            predicted_tokens[:, frame_offset] = gen_text[:, current_frame_idx]

            if self.decode_audio:
                current_subword_id = gen_text[:, current_frame_idx].unsqueeze(-1)

                # do one step inference on Duplex TTS model
                if current_frame_idx == 0:
                    if self.first_context_subword_id is None:
                        raise RuntimeError("first_context_subword_id is not initialized. Ensure TTS warmup ran successfully.")
                    prev_subword_id = self.first_context_subword_id
                else:
                    prev_subword_id = gen_text[:, current_frame_idx-1].unsqueeze(-1)

                # create subword_mask
                current_subword_mask = subword_mask[:, current_frame_idx].unsqueeze(-1)

                if self.generation_config is None:
                    raise RuntimeError("generation_config is not initialized. Ensure TTS warmup ran successfully.")

                start_tts_model = time.time()
                inputs = {
                    "current_subword_id": current_subword_id,
                    "prev_subword_id": prev_subword_id,
                    "current_subword_mask": current_subword_mask,
                    "prev_audio_tokens": code,
                    "past_key_values": past_key_values,
                    "guidance_enabled": True,
                    "generation_config": self.generation_config,
                    "ignore_eos_flag_stop": True,
                }
                if self.use_vllm_eartts:
                    inputs["request_id"] = effective_request_id

                code, past_key_values = self.model.tts_model.infer_codes_one_step(
                        **inputs
                )

                torch.cuda.synchronize()
                time_tts_model = time.time() - start_tts_model
                logging.info(f"Time taken for tts_model: {time_tts_model:.3f}s")

                new_codes_for_decode.append(code.clone())
                # Update sliding-window buffer (only needed for fallback decode when codec_cache is off)
                if audio_toks_buffer is not None:
                    audio_toks_buffer = torch.cat([audio_toks_buffer[:, 1:], code], dim=1)

                # now that we've saved audio_toks_buffer for audio decoding purposes,
                # we can potentially overwrite the audio token with silence tokens (for feeding to the audio token predictor)
                if self.model.cfg.get('inference_force_speech_silence_on_eos', None):
                    silence_codes = self.model.tts_model.codec_silence_tokens.view(1, 1, -1).expand(code.shape)
                    code = torch.where(
                        current_subword_id.unsqueeze(-1) == self.model.tts_model.text_eos_id,
                        silence_codes,
                        code,
                    )

        # exit for-loop & do audio decoding non-autoregressively (if decode_audio is True)
        if self.decode_audio:
            samples_per_audio_output_frame = self._samples_per_audio_output_frame()
            logging.debug(f"\nDecoding audio for {frame_idx}-th frame  ({num_frames_per_chunk=})")

            start_time_decode = time.time()
            with fp32_precision(), torch.no_grad():
                if codec_cache is not None and new_codes_for_decode:
                    # Incremental decode: feed only the num_frames_per_chunk new tokens
                    # to the codec. CausalConv1dCache maintains all necessary ConvNeXt
                    # and ISTFT overlap state from prior calls, so no history buffer
                    # is needed — this replaces the sliding-window approach entirely.
                    new_codes_tensor = torch.cat(new_codes_for_decode, dim=1)
                    if hasattr(self.model.tts_model, '_control_codes'):
                        from nemo.collections.speechlm2.models.duplex_ear_tts import replace_control_speech_codes
                        new_codes_tensor = replace_control_speech_codes(
                            new_codes_tensor,
                            self.model.tts_model._control_codes,
                            getattr(self.model.tts_model, 'codec_silence_tokens', None),
                        )
                    new_code_len = torch.tensor(
                        [new_codes_tensor.shape[1]], dtype=torch.long, device=self.device
                    )
                    decoded_audio_new, _ = self.model.tts_model.audio_codec.decode(
                        new_codes_tensor, new_code_len, cache=codec_cache,
                    )
                    logging.debug(f"   Incremental decode: {new_codes_tensor.shape[1]} new tokens -> {decoded_audio_new.shape}")
                else:
                    # Fallback: full-buffer sliding-window decode (original behavior)
                    len_audio_toks_buffer = torch.tensor(
                        [self.codec_token_history_size], dtype=torch.long, device=self.device
                    )
                    decoded_audio, decoded_audio_len = self.model.tts_model.audio_codec.decode(
                        audio_toks_buffer, len_audio_toks_buffer,
                    )
                    decoded_audio_new = decoded_audio[:, :, -samples_per_audio_output_frame * num_frames_per_chunk:]
                    logging.debug(f"   Sliding-window decode: extracted {decoded_audio_new.shape} from {decoded_audio.shape}")

            torch.cuda.synchronize()
            time_audio_codec = time.time() - start_time_decode
            logging.info(f"Time taken for audio_codec: {time_audio_codec:.3f}s")

        else:
            audio_toks_buffer = None
            decoded_audio_new = None
            time_tts_model = 0
            time_audio_codec = 0

        # Convert new text tokens to string via tokens_to_text (convert_tokens_to_string)
        # so byte-level BPE is decoded properly (e.g. "Ã©" → "é") and leading spaces
        # from Ġ-prefixed tokens are preserved for correct concatenation of incremental
        # chunks: " Musée" + " National" → " Musée National".
        # NOTE: multi-byte UTF-8 characters whose BPE tokens span two frames will show
        # as replacement chars (�) because each frame is decoded independently. A proper
        # fix would require an incremental UTF-8 decoder that buffers incomplete trailing
        # bytes across frames.
        predicted_text_strs = []
        for predicted_tok_ids_b in predicted_tokens:
            predicted_tok_ids_b = predicted_tok_ids_b.tolist()
            predicted_toks_b = self.tokenizer.ids_to_tokens(predicted_tok_ids_b)
            predicted_toks_b = [tok for tok in predicted_toks_b if tok != '<SPECIAL_12>']
            predicted_text_strs.append(self.tokenizer.tokens_to_text(predicted_toks_b))

        # convert new ASR tokens to string
        asr_predicted_text_strs = []
        for asr_predicted_tok_ids_b in asr_predicted_tokens:
            asr_predicted_tok_ids_b = asr_predicted_tok_ids_b.tolist()
            asr_predicted_toks_b = self.tokenizer.ids_to_tokens(asr_predicted_tok_ids_b)
            asr_predicted_toks_b = [tok for tok in asr_predicted_toks_b if tok != '<SPECIAL_12>']
            asr_predicted_text_strs.append(self.tokenizer.tokens_to_text(asr_predicted_toks_b))

        logging.info(f'frame {frame_idx}: USER\'s asr_predicted_text_strs: {asr_predicted_text_strs}')
        logging.info(f'frame {frame_idx}: --------------------------------AGENT\'s predicted_text_strs: {predicted_text_strs}')

        torch.cuda.synchronize()

        time_for_one_step = time.time() - start_time_one_step
        logging.info(f'frame {frame_idx}: Time taken for one step: {time_for_one_step:.3f}s')

        result = {
            'predicted_text_tokens': predicted_tokens,
            'asr_predicted_text_tokens': asr_predicted_tokens,
            'audio_toks_buffer': audio_toks_buffer,
            'decoded_audio_new': decoded_audio_new,
            'predicted_text_strs': predicted_text_strs,
            'asr_predicted_text_strs': asr_predicted_text_strs,
            'input_embeds_history': input_embeds_history + new_input_embeds if not use_cache else input_embeds_history,
            'dynamic_cache': dynamic_cache if use_cache else None,
            'past_key_values': past_key_values,
            'code': code,
            'perception_cache': perception_cache,
            'codec_cache': codec_cache,
            'cache_position_offset': cache_position_offset + num_frames_per_chunk if use_cache else cache_position_offset,
        }
        if self.model.stt_model.function_head is not None:
            result['function_predicted_text_tokens'] = function_predicted_tokens
        if return_debug:
            result["debug"] = {
                "source_encoded": source_encoded.detach().cpu(),
                "selected_frame_indices": selected_frame_indices,
                "input_embeds": torch.cat(debug_input_embeds, dim=1) if debug_input_embeds else None,
                "gen_text": gen_text.detach().cpu(),
                "gen_asr": gen_asr_text.detach().cpu() if gen_asr_text is not None else None,
                "text_logits": torch.stack(debug_text_logits, dim=1) if debug_text_logits else None,
                "asr_logits": torch.stack(debug_asr_logits, dim=1) if debug_asr_logits else None,
            }
        return result

    def abort_request(self, request_id: Optional[str]) -> bool:
        """
        Abort an in-flight vLLM streaming request if the backend supports it.
        """
        if not request_id:
            return False

        success = False

        # Abort LLM if applicable
        if self.use_vllm_llm:
            abort_fn = getattr(self.model_llm_interface, "abort_request", None)
            if callable(abort_fn):
                try:
                    if abort_fn(request_id):
                        success = True
                    logging.info(f"Aborted LLM request {request_id} successfully.")
                except Exception as exc:
                    logging.warning(f"Failed to abort LLM request {request_id}: {exc}")

        # Abort EarTTS if applicable
        if self.use_vllm_eartts:
            abort_fn = getattr(self.model.tts_model.tts_model, "abort_request", None)
            if callable(abort_fn):
                try:
                    if abort_fn(request_id):
                        success = True
                    logging.info(f"Aborted EarTTS request {request_id} successfully.")
                except Exception as exc:
                    logging.warning(f"Failed to abort EarTTS request {request_id}: {exc}")

        return success


    def _maybe_apply_forced_turn_taking(self, t, gen_text, gen_asr):
        """Apply forced turn-taking rules based on ASR channel tokens."""
        if not self.model_cfg.get("force_turn_taking", False):
            return

        threshold = self.model_cfg.get("force_turn_taking_threshold", 40)
        pad_window_steps = self.model_cfg.get("force_turn_taking_pad_window", 25)

        B = gen_text.size(0)

        for batch_idx in range(B):
            lookback_start = max(0, t - threshold)
            agent_text_window = gen_text[batch_idx, lookback_start:t]
            current_asr_token = gen_asr[batch_idx, t]

            # ASR EOS or ~1 sec of pad tokens → insert agent BOS if not present in window
            # Skip if we don't have enough tokens at the beginning
            if t < pad_window_steps:
                continue

            pad_lookback_start = t - pad_window_steps
            asr_recent_tokens = gen_asr[batch_idx, pad_lookback_start:t]
            has_pad_window = (asr_recent_tokens == self.model.stt_model.text_pad_id).all() if len(asr_recent_tokens) > 0 else False

            # Require that the pad window starts after a non-pad token
            if has_pad_window and pad_lookback_start > 0:
                token_before_window = gen_asr[batch_idx, pad_lookback_start - 1]
                has_pad_window = (token_before_window != self.model.stt_model.text_pad_id) and (token_before_window != self.model.stt_model.text_bos_id)
            elif has_pad_window and pad_lookback_start == 0:
                # If the pad window starts at position 0, it doesn't meet the requirement
                has_pad_window = False

            if has_pad_window:
                if not (agent_text_window == self.model.stt_model.text_bos_id).any():
                    gen_text[batch_idx, t] = self.model.stt_model.text_bos_id
                    logging.info(f"Forced turn-taking at frame {t}: inserted agent BOS (reason: pad window)")

            # ASR BOS → insert agent EOS if not present in window
            elif current_asr_token == self.model.stt_model.text_bos_id:
                if not (agent_text_window == self.model.stt_model.text_eos_id).any():
                    gen_text[batch_idx, t] = self.model.stt_model.text_eos_id
                    logging.info(f"Forced turn-taking at frame {t}: inserted agent EOS (reason: user started speaking)")

