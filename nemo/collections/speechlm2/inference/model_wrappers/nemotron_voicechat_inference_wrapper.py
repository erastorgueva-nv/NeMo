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

import torch
import yaml
from omegaconf import OmegaConf, DictConfig
import numpy as np
import librosa
import time
from transformers import DynamicCache
import re
import os
import sys
import argparse
import torchaudio
import functools
from dataclasses import dataclass
from typing import Optional, Tuple
from nemo.utils import logging
from jiwer import wer

import gc
import types


# NOTE: sys.path is configured via PYTHONPATH in the shell script (run_streaming_realtime_json_mode.sh)
# Do NOT hardcode paths here as they may conflict with the correct paths set in the shell script

# Set environment variables (use existing env vars if set, otherwise use defaults)
_default_cache = "/tmp/cache"
os.environ.setdefault("HF_HOME", _default_cache)
os.environ.setdefault("TORCH_HOME", _default_cache)
os.environ.setdefault("NEMO_CACHE_DIR", _default_cache)
os.environ.setdefault("NEMO_NLP_TMP", os.path.join(_default_cache, "nemo_nlp_tmp"))

from nemo.collections.speechlm2.models.nemotron_voicechat import NemotronVoiceChat

from nemo.collections.speechlm2.models.duplex_s2s_model import tokens_to_str
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.speechlm2.inference.model_wrappers.model_factory import create_model


def tokens_to_str_raw(tokens: torch.Tensor, lengths: torch.Tensor, tokenizer, pad_id: int) -> list:
    """
    Convert token IDs to text strings, preserving ALL special tokens including <SPECIAL_12> (pad token).

    Unlike tokens_to_str, this function uses ids_to_tokens which preserves special tokens,
    and does NOT filter out any tokens (including pad tokens like <SPECIAL_12>).

    Args:
        tokens: Token IDs tensor (B, T)
        lengths: Length of each sequence (B,)
        tokenizer: Tokenizer for decoding
        pad_id: Pad token ID (not used for filtering in raw mode, kept for API compatibility)

    Returns:
        List of decoded text strings with ALL special tokens preserved (including <SPECIAL_12>)
    """
    ans = []
    for hyp_ids, hyp_len in zip(tokens.cpu(), lengths.cpu()):
        hyp_ids = hyp_ids[:hyp_len]
        # Do NOT filter out any tokens - keep everything including pad tokens (<SPECIAL_12>)
        hyp_ids_list = hyp_ids.tolist()

        # Use ids_to_tokens which preserves special tokens like <SPECIAL_12>
        toks = tokenizer.ids_to_tokens(hyp_ids_list)

        # Only replace 'Ġ' with space for proper word boundaries, keep all special tokens
        toks = [tok.replace('Ġ', ' ') for tok in toks]

        ans.append("".join(toks))
    return ans


def clean_pred_text(text: str) -> str:
    """Clean prediction text by removing special markers, timestamps, punctuation, and lowercasing."""
    if not text:
        return ""
    text = text.lstrip('^')
    text = re.sub(r'<\$[\d.]+\$>', '', text)
    text = re.sub(r'<\|[\d.]+\|>', '', text)
    text = re.sub(r'<SPECIAL_12>', '', text)
    text = text.replace('Ġ', ' ')
    # Lowercase and remove punctuation for fair WER comparison
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return ' '.join(text.split())

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
DEFAULT_CODEC_TOKEN_HISTORY_SIZE = 600


@dataclass
class PerceptionCacheState:
    """Cache state for streaming perception inference.

    Holds the cache tensors for the ASR encoder used in the perception module.
    This enables cache-aware streaming inference without needing the full audio buffer.
    """
    cache_last_channel: Optional[torch.Tensor] = None
    cache_last_time: Optional[torch.Tensor] = None
    cache_last_channel_len: Optional[torch.Tensor] = None

    def is_initialized(self) -> bool:
        """Check if the cache has been initialized."""
        return self.cache_last_channel is not None


@dataclass
class PerceptionCUDAGraphState:
    """State for CUDA graph-accelerated perception encoder.
    
    Holds separate graphs for first chunk (different size) and subsequent chunks.
    Also holds static buffers for inputs/outputs to enable graph replay.
    """
    # CUDA graphs
    graph_first: Optional[torch.cuda.CUDAGraph] = None
    graph_subsequent: Optional[torch.cuda.CUDAGraph] = None
    
    # Static input buffers (for copying data before graph replay)
    static_mel_first: Optional[torch.Tensor] = None
    static_mel_subsequent: Optional[torch.Tensor] = None
    static_mel_len_first: Optional[torch.Tensor] = None
    static_mel_len_subsequent: Optional[torch.Tensor] = None
    
    # Static cache input buffers
    static_cache_channel_in: Optional[torch.Tensor] = None
    static_cache_time_in: Optional[torch.Tensor] = None
    static_cache_channel_len_in: Optional[torch.Tensor] = None
    
    # Static output buffers (results are written here during replay)
    static_encoded_first: Optional[torch.Tensor] = None
    static_encoded_subsequent: Optional[torch.Tensor] = None
    static_encoded_len_first: Optional[torch.Tensor] = None
    static_encoded_len_subsequent: Optional[torch.Tensor] = None
    
    # Static cache output buffers - SEPARATE for first and subsequent graphs
    # (each graph writes to its own output tensors during replay)
    static_cache_channel_out_first: Optional[torch.Tensor] = None
    static_cache_time_out_first: Optional[torch.Tensor] = None
    static_cache_channel_len_out_first: Optional[torch.Tensor] = None
    static_cache_channel_out_subsequent: Optional[torch.Tensor] = None
    static_cache_time_out_subsequent: Optional[torch.Tensor] = None
    static_cache_channel_len_out_subsequent: Optional[torch.Tensor] = None
    
    def is_captured(self) -> bool:
        """Check if graphs have been captured."""
        return self.graph_first is not None and self.graph_subsequent is not None


class NemotronVoicechatInferenceWrapper:
    """
    Inference wrapper for NemotronVoiceChat models.
    Uses a sliding window buffer and processes audio frame by frame.
    """

    def __init__(self, model_cfg: DictConfig):
        """
        Initialize the model for realtime streaming inference.

        Args:
            model_cfg (DictConfig): Configuration describing the model paths and runtime parameters.
        """
        if model_cfg is None:
            raise ValueError("model_cfg must be provided")
        if not isinstance(model_cfg, DictConfig):
            model_cfg = OmegaConf.create(model_cfg)


        logging.info(f"pythonpath: {sys.path}")


        logging.info(f"before setting - torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}")
        logging.info(f"before setting - torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}")
        logging.info(f"before setting - torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}")

        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")

        logging.info(f"after setting - torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}")
        logging.info(f"after setting - torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}")
        logging.info(f"after setting - torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}")


        self.model_cfg = model_cfg

        self.model_path = model_cfg.get("model_path")
        if not self.model_path:
            raise ValueError("`model_cfg.model_path` must be provided.")

        self.llm_checkpoint_path = model_cfg.get("llm_checkpoint_path")
        if not self.llm_checkpoint_path:
            raise ValueError("`model_cfg.llm_checkpoint_path` must be provided.")

        self.decode_audio = bool(model_cfg.get("decode_audio", True))
        self.codec_token_history_size = int(
            model_cfg.get("codec_token_history_size", DEFAULT_CODEC_TOKEN_HISTORY_SIZE)
        )

        self.speaker_reference = model_cfg.get("speaker_reference")
        if self.decode_audio and not self.speaker_reference:
            raise ValueError("`model_cfg.speaker_reference` must be provided when decode_audio is enabled.")

        self.tts_system_prompt = model_cfg.get("tts_system_prompt", None)
        logging.info(f"TTS system prompt: {self.tts_system_prompt}")

        compute_dtype = model_cfg.get("compute_dtype", "bfloat16")
        self.dtype = self._resolve_dtype(compute_dtype)

        self.device = self._resolve_device(
            device=model_cfg.get("device"),
            device_id=model_cfg.get("device_id"),
        )

        #logging.setLevel(logging.DEBUG)

        logging.info("=" * 70)
        logging.info("INITIALIZING REALTIME STREAMING INFERENCE")
        logging.info("=" * 70)
        logging.info(f"Frame size: {FRAME_SIZE_SEC}s ({FRAME_SIZE_SAMPLES} samples @ {SAMPLE_RATE}Hz)")
        logging.info(f"Device: {self.device}")
        logging.info(f"Compute dtype: {self.dtype}")
        logging.info(f"Decode audio: {self.decode_audio}")
        logging.info(f"Engine type: {model_cfg.get('engine_type', 'native')}")
        logging.info(f"Sampling - top_p: {model_cfg.get('top_p', 1.0)}, repetition_penalty: {model_cfg.get('repetition_penalty', 1.0)}")
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

        # Perception cache configuration
        self.use_perception_cache = bool(model_cfg.get("use_perception_cache", False))
        self.perception_streaming_cfg = None  # Will be populated after model init if cache is used
        self.perception_preprocessor = None  # Separate preprocessor for cache-aware streaming
        
        # CUDA graph configuration for perception encoder
        self.use_perception_cudagraph = bool(model_cfg.get("use_perception_cudagraph", False))
        self.perception_cudagraph_state: Optional[PerceptionCUDAGraphState] = None
        
        # CUDA graphs require perception cache to be enabled
        if self.use_perception_cudagraph and not self.use_perception_cache:
            raise ValueError(
                "use_perception_cudagraph requires use_perception_cache to be enabled. "
                "Please also set use_perception_cache=True."
            )

        self._initialize_model()


        logging.info(f"\n✅ NemotronVoicechatInferenceWrapper initialized successfully.")

        logging.info(f"{self.model.stt_model.perception.encoder._cfg = }")
        logging.info(f"{self.model.stt_model.perception.encoder.streaming_cfg = }")

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

    def _load_and_merge_configs(self):
        """Load and merge configurations from both nano and eartts checkpoints."""
        logging.info("\n📋 Loading and merging configurations...")

        # Load nano's config (for LLM, perception)
        nano_config_file = os.path.join(self.llm_checkpoint_path, "config.json")
        logging.info(f"  Loading nano config: {nano_config_file}")
        with open(nano_config_file, 'r') as f:
            import json
            nano_cfg_dict = json.load(f)
        nano_cfg = DictConfig(nano_cfg_dict)

        # Load eartts's config (for TTS)
        eartts_config_file = os.path.join(self.model_path, "config.json")
        logging.info(f"  Loading eartts config: {eartts_config_file}")
        with open(eartts_config_file, 'r') as f:
            eartts_cfg_dict = json.load(f)
        eartts_cfg = DictConfig(eartts_cfg_dict)

        # Start with nano's config as base
        merged_cfg = nano_cfg

        # Override TTS-related parts with eartts's config
        logging.info("  Merging: Using nano's config for LLM/perception, eartts's for TTS")
        if 'model' in eartts_cfg and 'speech_generation' in eartts_cfg.model:
            merged_cfg.model.speech_generation = eartts_cfg.model.speech_generation
            logging.info("    ✓ TTS config from eartts")

        # Set speaker reference
        if 'model' not in merged_cfg:
            merged_cfg.model = {}
        merged_cfg.model.inference_speaker_reference = self.speaker_reference

        # Ensure data section has correct sample rates
        if 'data' not in merged_cfg:
            merged_cfg.data = eartts_cfg.data

        logging.info(f"  Final config:")
        logging.info(f"    - pretrained_llm: {merged_cfg.model.stt.model.pretrained_llm}")
        logging.info(f"    - perception.d_model: {merged_cfg.model.stt.model.perception.modality_adapter.d_model}")
        logging.info(f"    - speech_generation: {'present' if 'speech_generation' in merged_cfg.model else 'missing'}")

        return merged_cfg

    def _initialize_model(self):
        """Initialize the NemotronVoiceChat with hybrid loading."""
        from safetensors.torch import load_file
        from nemo.collections.speechlm2.parts.pretrained import set_model_dict_for_partial_init

        logging.info("\n🚀 Initializing model with hybrid loading strategy...")


        # Step 1: Load and merge configs
        cfg = self._load_and_merge_configs()

        # Step 2: DO NOT set pretrained_s2s_model - we'll load weights manually
        cfg.model.stt.model.pretrained_s2s_model = None
        cfg.model.speech_generation.model.pretrained_model = None

        # Convert to dict for model initialization
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        # Step 3: Initialize model structure
        logging.info("\n🏗️  Initializing model structure...")
        start_DuplexS2S_init = time.time()
        self.model = NemotronVoiceChat(cfg_dict)
        logging.info(f"🕒 Time taken to initialize NemotronVoiceChat: {time.time() - start_DuplexS2S_init} seconds")
        logging.info("  ✓ Model structure initialized")

        # Step 4: Load nano's checkpoint (LLM + perception)
        if self.llm_checkpoint_path is not None:
            logging.info(f"\n📦 Loading LLM + perception:")
            logging.info(f"  Path: {self.llm_checkpoint_path}")

            nano_state_dict = load_file(os.path.join(self.llm_checkpoint_path, "model.safetensors"))

            # Filter to non-TTS weights
            tts_keys = ['tts_model.', 'speech_generation.']

            # If using vLLM for LLM, also exclude LLM weights to save memory
            # vLLM will load its own copy of the LLM
            if self.use_vllm_llm:
                llm_keys = ['stt_model.llm.']
                exclude_keys = tts_keys + llm_keys
                logging.info(f"  Using vLLM - excluding LLM weights from nano checkpoint")
            else:
                exclude_keys = tts_keys

            nano_filtered = {k: v for k, v in nano_state_dict.items()
                           if not any(k.startswith(prefix) for prefix in exclude_keys)}

            logging.info(f"  Loading {len(nano_filtered)} parameters (excluded: {exclude_keys})...")

            # Free the full state dict immediately to save CPU memory
            del nano_state_dict
            gc.collect()

            nano_filtered = set_model_dict_for_partial_init(nano_filtered, self.model.state_dict())
            missing, unexpected = self.model.load_state_dict(nano_filtered, strict=False)

            # Free filtered dict
            del nano_filtered
            gc.collect()

            missing_non_excluded = [k for k in missing if not any(k.startswith(prefix) for prefix in exclude_keys)]
            unexpected_non_excluded = [k for k in unexpected if not any(k.startswith(prefix) for prefix in exclude_keys)]

            if missing_non_excluded:
                logging.info(f"  ⚠️  {len(missing_non_excluded)} keys missing (might be OK)")
            if unexpected_non_excluded:
                logging.info(f"  ⚠️  {len(unexpected_non_excluded)} unexpected keys")

        # Step 5: Load eartts's checkpoint (TTS only)
        if self.model_path is not None:
            logging.info(f"\n📦 Loading TTS checkpoint:")
            logging.info(f"  Path: {self.model_path}")

            eartts_state_dict = load_file(os.path.join(self.model_path, "model.safetensors"))

            # Filter to only TTS weights
            tts_keys_filter = ['tts_model.']
            eartts_tts_only = {k: v for k, v in eartts_state_dict.items()
                                 if any(k.startswith(prefix) for prefix in tts_keys_filter)}

            logging.info(f"  Loading {len(eartts_tts_only)} TTS parameters...")

            start_tts_load_state_dict = time.time()
            missing, unexpected = self.model.load_state_dict(eartts_tts_only, strict=False)
            logging.info(f"🕒 Time taken to load TTS state dict: {time.time() - start_tts_load_state_dict} seconds")

            missing_tts = [k for k in missing if any(k.startswith(prefix) for prefix in tts_keys_filter)]
            unexpected_tts = [k for k in unexpected if any(k.startswith(prefix) for prefix in tts_keys_filter)]

            if missing_tts:
                logging.info(f"  ⚠️  {len(missing_tts)} TTS keys missing")
            if unexpected_tts:
                logging.info(f"  ⚠️  {len(unexpected_tts)} unexpected TTS keys")

            if self.use_vllm_eartts:
                # gonna convert and load vllm eartts engine
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

            logging.info(f"  ✓ eartts checkpoint loaded (TTS only)")

        logging.info("\n✅ Hybrid loading completed!")

        # If using vLLM for LLM, delete native LLM BEFORE moving to device to save memory
        if self.use_vllm_llm:
            logging.info("\n🔧 Deleting native LLM before GPU transfer (will use vLLM instead)...")
            if hasattr(self.model.stt_model, 'llm') and self.model.stt_model.llm is not None:
                # Delete all submodules of LLM to free memory
                for name, child in list(self.model.stt_model.llm.named_children()):
                    delattr(self.model.stt_model.llm, name)
                del self.model.stt_model.llm
                self.model.stt_model.llm = None
            gc.collect()
            torch.cuda.empty_cache()
            logging.info("  ✓ Native LLM deleted")

        # Setup model
        self.model.to(self.device)
        self.model.eval()

        # Convert only the S2S components to the configured dtype, not the TTS model
        logging.info(f"Converting S2S components to {self.dtype} (keeping TTS in float32)...")
        if self.model.stt_model.llm is not None:
            self.model.stt_model.llm = self.model.stt_model.llm.to(self.dtype)
        self.model.stt_model.lm_head = self.model.stt_model.lm_head.to(self.dtype)
        self.model.stt_model.embed_tokens = self.model.stt_model.embed_tokens.to(self.dtype)
        self.model.stt_model.asr_head = self.model.stt_model.asr_head.to(self.dtype)
        self.model.stt_model.embed_asr_tokens = self.model.stt_model.embed_asr_tokens.to(self.dtype)
        #self.model.stt_model.perception = self.model.stt_model.perception.to(self.dtype)
        logging.info("✓ S2S components converted, TTS kept in float32")
        logging.info("new update, perception also is kept in float32")

        # commenting this out to avoid error when try vllm tts
        # and anyway - when sticking to "native", saw no difference in output
        # with and without this call
        #self.model.on_train_epoch_start()
        self.tokenizer = self.model.stt_model.tokenizer

        # Print inference boost values
        logging.info(f"inference_eos_boost: {self.model.stt_model.cfg.get('inference_eos_boost', None)}")
        logging.info(f"inference_bos_boost: {self.model.stt_model.cfg.get('inference_bos_boost', None)}")
        logging.info(f"inference_pad_boost: {self.model.stt_model.cfg.get('inference_pad_boost', None)}")

        # Wrap model with appropriate interface (Native or vLLM)
        if self.use_vllm_llm:
            logging.info("\n🔧 Wrapping model with VllmLLMModel interface...")
            if self.vllm_llm_config is None:
                raise ValueError("vllm_llm_config must be provided when engine_type contains'vllm_llm'")

            # LLM already deleted above, just ensure cleanup
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            self.model_llm_interface = create_model(
                model=self.model_path,
                engine_type="vllm_llm",
                vllm_config=self.vllm_llm_config,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty
            )

            logging.info("✓ VllmLLMModel interface created")
        else:
            logging.info("\n🔧 Wrapping model with NativeModel interface...")
            self.model_llm_interface = create_model(
                model=self.model,
                engine_type="native",
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty
            )
            logging.info("✓ NativeModel interface created")

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
            self._setup_perception_cache()

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
        """
        Prepare system prompt embeddings consistent with offline_inference.

        In offline_inference, prompt embeddings are structured as:
        - Position 0: prompt_token_emb + bos_emb + asr_bos
        - Position t > 0: prompt_token_emb + pad_emb + pad_asr

        Args:
            system_prompt: The system prompt text

        Returns:
            Tuple of (prompt_embedded [1, prompt_len, H], prompt_length)
            Returns (None, 0) if system_prompt is empty
        """

        if not system_prompt or not system_prompt.strip():
            return None, 0

        logging.info(f"\n📝 Preparing system prompt: {system_prompt[:100]}...")

        # Step 1: Tokenize the prompt
        # Format: [bos] + text_tokens + [eos] (consistent with collate_system_prompt)
        prompt_token_ids = (
            [self.tokenizer.bos_id] +
            self.tokenizer.text_to_ids(system_prompt) +
            [self.tokenizer.eos_id]
        )
        prompt_tokens = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, prompt_len]
        prompt_len = prompt_tokens.shape[1]

        logging.info(f"   Prompt length: {prompt_len} tokens")

        # Step 2: Embed the prompt tokens (this acts as the "audio channel" for prompt positions)
        prompt_embedded = self.model.stt_model.embed_tokens(prompt_tokens)  # [1, prompt_len, H]
        prompt_embedded = prompt_embedded.to(dtype=self.dtype)

        # Step 3: Add pad embeddings for text and ASR channels (for positions t > 0)
        # In offline_inference, prompt positions use gen_text[:, t-1] = pad_id
        pad_id = self.model.stt_model.text_pad_id
        pad_token = torch.full((1,), fill_value=pad_id, device=self.device, dtype=torch.long)
        pad_emb = self.model.stt_model.embed_tokens(pad_token).to(dtype=self.dtype)  # [1, H]
        pad_asr_emb = self.model.stt_model.embed_asr_tokens(pad_token).to(dtype=self.dtype)  # [1, H]

        # For positions t > 0, add pad embeddings (simulating gen_text[:, t-1] = pad_id)
        if prompt_len > 1:
            prompt_embedded[:, 1:, :] += pad_emb
            prompt_embedded[:, 1:, :] += pad_asr_emb

        # Step 4: For position 0, add BOS embeddings
        bos_emb = self._get_bos_embedding()  # [1, H]
        asr_bos_emb = self._get_asr_bos_embedding()  # [1, H]
        prompt_embedded[:, 0, :] += bos_emb.squeeze(0)
        prompt_embedded[:, 0, :] += asr_bos_emb.squeeze(0)

        logging.info(f"   ✅ System prompt embeddings prepared: shape {prompt_embedded.shape}")

        return prompt_embedded, prompt_len

    def _setup_perception_cache(self):
        """Setup cache-aware streaming for the perception encoder."""
        import copy
        from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder

        perception = self.model.stt_model.perception
        encoder = perception.encoder

        # Check if encoder supports streaming
        if not isinstance(encoder, StreamingEncoder):
            logging.warning("Perception encoder does not support streaming. Disabling perception cache.")
            self.use_perception_cache = False
            return

        # Setup streaming params if not already done
        if encoder.streaming_cfg is None:
            encoder.setup_streaming_params()

        self.perception_streaming_cfg = encoder.streaming_cfg

        # Create a separate preprocessor for cache-aware streaming (no dither, no normalization padding)
        cfg = copy.deepcopy(perception.cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0

        self.perception_preprocessor = perception.from_config_dict(cfg.preprocessor)
        self.perception_preprocessor.to(self.device)

        # Get subsampling factor and input features
        self.perception_subsampling_factor = encoder.subsampling_factor
        self.perception_input_features = encoder._feat_in

        # Get sampling frames for checking minimum chunk size
        if hasattr(encoder, "pre_encode") and hasattr(encoder.pre_encode, "get_sampling_frames"):
            self.perception_sampling_frames = encoder.pre_encode.get_sampling_frames()
        else:
            self.perception_sampling_frames = None

        logging.info(f"✅ Perception cache setup complete:")
        logging.info(f"   Streaming config: chunk_size={self.perception_streaming_cfg.chunk_size}, "
                     f"shift_size={self.perception_streaming_cfg.shift_size}")
        logging.info(f"   Pre-encode cache size: {self.perception_streaming_cfg.pre_encode_cache_size}")
        logging.info(f"   Subsampling factor: {self.perception_subsampling_factor}")
        
        # Setup CUDA graphs if enabled
        if self.use_perception_cudagraph:
            logging.info(f"   Setting up CUDA graphs for perception encoder...")
            self._capture_perception_cudagraphs()
            logging.info(f"   ✓ CUDA graphs captured")

    def _capture_perception_cudagraphs(self):
        """Capture CUDA graphs for perception encoder with both chunk sizes.

        Note: "chunk" in the streaming encoder config (chunk_size, shift_size, etc.)
        follows NeMo's cache-aware streaming encoder API and is measured in
        mel-spectrogram time-steps, not audio samples or seconds.
        """
        encoder = self.model.stt_model.perception.encoder
        perception = self.model.stt_model.perception
        streaming_cfg = self.perception_streaming_cfg
        
        # Get chunk sizes
        if isinstance(streaming_cfg.chunk_size, list):
            chunk_size_first = streaming_cfg.chunk_size[0]
            chunk_size_subsequent = streaming_cfg.chunk_size[1]
        else:
            chunk_size_first = streaming_cfg.chunk_size
            chunk_size_subsequent = streaming_cfg.chunk_size
            
        if isinstance(streaming_cfg.pre_encode_cache_size, list):
            pre_encode_cache_first = streaming_cfg.pre_encode_cache_size[0]
            pre_encode_cache_subsequent = streaming_cfg.pre_encode_cache_size[1]
        else:
            pre_encode_cache_first = streaming_cfg.pre_encode_cache_size
            pre_encode_cache_subsequent = streaming_cfg.pre_encode_cache_size
        
        # Total mel lengths for each chunk type
        mel_len_first = chunk_size_first + pre_encode_cache_first
        mel_len_subsequent = chunk_size_subsequent + pre_encode_cache_subsequent
        
        logging.info(f"   CUDA graph mel lengths: first={mel_len_first}, subsequent={mel_len_subsequent}")
        
        # Get initial cache state for warmup
        cache_last_channel, cache_last_time, cache_last_channel_len = encoder.get_initial_cache_state(
            batch_size=1
        )

        
        # Create static buffers (use float32 since perception stays in float32)
        state = PerceptionCUDAGraphState()
        
        # Static mel input buffers
        state.static_mel_first = torch.zeros(
            (1, self.perception_input_features, mel_len_first),
            dtype=torch.float32, device=self.device
        )
        state.static_mel_subsequent = torch.zeros(
            (1, self.perception_input_features, mel_len_subsequent),
            dtype=torch.float32, device=self.device
        )
        state.static_mel_len_first = torch.tensor([mel_len_first], dtype=torch.long, device=self.device)
        state.static_mel_len_subsequent = torch.tensor([mel_len_subsequent], dtype=torch.long, device=self.device)
        
        # Static cache input buffers (clone from initial state)
        if cache_last_channel is not None:
            state.static_cache_channel_in = cache_last_channel.clone()
        if cache_last_time is not None:
            state.static_cache_time_in = cache_last_time.clone()
        if cache_last_channel_len is not None:
            state.static_cache_channel_len_in = cache_last_channel_len.clone()
        
        # Warmup runs (required before graph capture)
        logging.info(f"   Warming up encoder for CUDA graph capture...")
        for _ in range(3):
            with torch.no_grad():
                # Warmup first chunk
                _ = encoder.cache_aware_stream_step(
                    processed_signal=state.static_mel_first,
                    processed_signal_length=state.static_mel_len_first,
                    cache_last_channel=state.static_cache_channel_in.clone() if state.static_cache_channel_in is not None else None,
                    cache_last_time=state.static_cache_time_in.clone() if state.static_cache_time_in is not None else None,
                    cache_last_channel_len=state.static_cache_channel_len_in.clone() if state.static_cache_channel_len_in is not None else None,
                    keep_all_outputs=True,
                    drop_extra_pre_encoded=0,
                )
                # Warmup subsequent chunk
                _ = encoder.cache_aware_stream_step(
                    processed_signal=state.static_mel_subsequent,
                    processed_signal_length=state.static_mel_len_subsequent,
                    cache_last_channel=state.static_cache_channel_in.clone() if state.static_cache_channel_in is not None else None,
                    cache_last_time=state.static_cache_time_in.clone() if state.static_cache_time_in is not None else None,
                    cache_last_channel_len=state.static_cache_channel_len_in.clone() if state.static_cache_channel_len_in is not None else None,
                    keep_all_outputs=True,
                    drop_extra_pre_encoded=streaming_cfg.drop_extra_pre_encoded,
                )
        torch.cuda.synchronize()
        
        # Capture graph for FIRST chunk
        logging.info(f"   Capturing CUDA graph for first chunk (mel_len={mel_len_first})...")
        state.graph_first = torch.cuda.CUDAGraph()
        
        # Reset cache to initial state before capture
        if state.static_cache_channel_in is not None:
            state.static_cache_channel_in.copy_(cache_last_channel)
        if state.static_cache_time_in is not None:
            state.static_cache_time_in.copy_(cache_last_time)
        if state.static_cache_channel_len_in is not None:
            state.static_cache_channel_len_in.copy_(cache_last_channel_len)
        
        with torch.cuda.graph(state.graph_first):
            (
                encoded_first,
                encoded_len_first,
                cache_channel_out_first,
                cache_time_out_first,
                cache_channel_len_out_first,
            ) = encoder.cache_aware_stream_step(
                processed_signal=state.static_mel_first,
                processed_signal_length=state.static_mel_len_first,
                cache_last_channel=state.static_cache_channel_in,
                cache_last_time=state.static_cache_time_in,
                cache_last_channel_len=state.static_cache_channel_len_in,
                keep_all_outputs=True,
                drop_extra_pre_encoded=0,
            )
            # Apply modality adapter and projection inside graph
            encoded_adapted_first, _ = perception.modality_adapter(audio_signal=encoded_first, length=encoded_len_first)
            encoded_chunk_first = perception.proj(encoded_adapted_first.transpose(1, 2))
        
        # Store output buffer references for first graph
        state.static_encoded_first = encoded_chunk_first
        state.static_encoded_len_first = encoded_len_first
        state.static_cache_channel_out_first = cache_channel_out_first
        state.static_cache_time_out_first = cache_time_out_first
        state.static_cache_channel_len_out_first = cache_channel_len_out_first
        
        # Capture graph for SUBSEQUENT chunks
        logging.info(f"   Capturing CUDA graph for subsequent chunks (mel_len={mel_len_subsequent})...")
        state.graph_subsequent = torch.cuda.CUDAGraph()
        
        # Reset cache to initial state before capture (for consistent graph)
        if state.static_cache_channel_in is not None:
            state.static_cache_channel_in.copy_(cache_last_channel)
        if state.static_cache_time_in is not None:
            state.static_cache_time_in.copy_(cache_last_time)
        if state.static_cache_channel_len_in is not None:
            state.static_cache_channel_len_in.copy_(cache_last_channel_len)
        
        with torch.cuda.graph(state.graph_subsequent):
            (
                encoded_subsequent,
                encoded_len_subsequent,
                cache_channel_out_subsequent,
                cache_time_out_subsequent,
                cache_channel_len_out_subsequent,
            ) = encoder.cache_aware_stream_step(
                processed_signal=state.static_mel_subsequent,
                processed_signal_length=state.static_mel_len_subsequent,
                cache_last_channel=state.static_cache_channel_in,
                cache_last_time=state.static_cache_time_in,
                cache_last_channel_len=state.static_cache_channel_len_in,
                keep_all_outputs=True,
                drop_extra_pre_encoded=streaming_cfg.drop_extra_pre_encoded,
            )
            # Apply modality adapter and projection inside graph
            encoded_adapted_subsequent, _ = perception.modality_adapter(audio_signal=encoded_subsequent, length=encoded_len_subsequent)
            encoded_chunk_subsequent = perception.proj(encoded_adapted_subsequent.transpose(1, 2))
        
        # Store output buffer references for subsequent graph
        state.static_encoded_subsequent = encoded_chunk_subsequent
        state.static_encoded_len_subsequent = encoded_len_subsequent
        state.static_cache_channel_out_subsequent = cache_channel_out_subsequent
        state.static_cache_time_out_subsequent = cache_time_out_subsequent
        state.static_cache_channel_len_out_subsequent = cache_channel_len_out_subsequent
        
        self.perception_cudagraph_state = state
        logging.info(f"   ✓ CUDA graphs captured successfully")

    def _get_initial_perception_cache_state(self, batch_size: int = 1) -> PerceptionCacheState:
        """Get initial cache state for perception encoder."""
        if not self.use_perception_cache:
            return PerceptionCacheState()

        encoder = self.model.stt_model.perception.encoder
        cache_last_channel, cache_last_time, cache_last_channel_len = encoder.get_initial_cache_state(
            batch_size=batch_size
        )
        
        
        return PerceptionCacheState(
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )

    def _cache_aware_perception_step(
        self,
        audio_input: torch.Tensor,
        frame_idx: int,
        num_frames_per_chunk: int,
        perception_cache: PerceptionCacheState,
    ) -> Tuple[torch.Tensor, PerceptionCacheState]:
        """
        Perform cache-aware perception encoding for streaming inference.

        Note: "chunk" in this method (chunk_size, mel_chunk, etc.) follows NeMo's
        cache-aware streaming encoder API and is measured in mel-spectrogram time-steps,
        not audio samples or seconds.

        This method computes the full mel spectrogram from the audio buffer, then slices
        it appropriately based on the frame index. It supports processing multiple
        "base steps" in a single call, where each base step processes (lookahead + 1) frames.

        Processing logic per sub-step:
        - First sub-step (sub_frame_idx == 0): take first chunk_size_first columns, 
          prepend zeros for pre_encode_cache
        - Subsequent sub-steps (sub_frame_idx > 0): take chunk_size columns starting from
          (shift_size_first + (step_number-1)*shift_size), prepend pre_encode_cache_size 
          columns from mel spec

        The method loops over sub-steps, running the encoder for each and concatenating
        the outputs. This allows num_frames_per_chunk to be a multiple of (lookahead + 1).

        Args:
            audio_input: Audio buffer tensor [B, T] (full buffer with all samples)
            frame_idx: Current frame index in the stream
            num_frames_per_chunk: Number of 80ms frames to process. Must be a multiple
                of (lookahead + 1), i.e., encoder._cfg.att_context_size[1] + 1
            perception_cache: Current cache state containing encoder caches

        Returns:
            Tuple of (encoded_output [B, T_out, D], updated_perception_cache)
            where T_out = num_frames_per_chunk (one output frame per input frame)
        """
        perception = self.model.stt_model.perception
        encoder = perception.encoder
        streaming_cfg = self.perception_streaming_cfg

        # Extract mel spectrogram from the full audio buffer
        audio_len = torch.tensor([audio_input.shape[1]], dtype=torch.long, device=self.device)
        _t_start_preprocessor = time.time()
        processed_signal, _ = self.perception_preprocessor(
            input_signal=audio_input,
            length=audio_len,
        ) # returns processed_signal, processed_signal_length
        # processed_signal shape: [B, n_mels, T_mel]
        logging.info(f"preprocessor time: {time.time() - _t_start_preprocessor:.3f}s")


        # Get streaming config values
        if isinstance(streaming_cfg.chunk_size, list):
            chunk_size_first = streaming_cfg.chunk_size[0]
            chunk_size = streaming_cfg.chunk_size[1]
        else:
            chunk_size_first = streaming_cfg.chunk_size
            chunk_size = streaming_cfg.chunk_size

        if isinstance(streaming_cfg.shift_size, list):
            shift_size_first = streaming_cfg.shift_size[0]
            shift_size = streaming_cfg.shift_size[1]
        else:
            shift_size_first = streaming_cfg.shift_size
            shift_size = streaming_cfg.shift_size

        if isinstance(streaming_cfg.pre_encode_cache_size, list):
            pre_encode_cache_size_first = streaming_cfg.pre_encode_cache_size[0]
            pre_encode_cache_size = streaming_cfg.pre_encode_cache_size[1]
        else:
            pre_encode_cache_size_first = streaming_cfg.pre_encode_cache_size
            pre_encode_cache_size = streaming_cfg.pre_encode_cache_size

        # Initialize current cache state
        cache_last_channel = perception_cache.cache_last_channel
        cache_last_time = perception_cache.cache_last_time
        cache_last_channel_len = perception_cache.cache_last_channel_len

        # num_frames_per_chunk must be a multiple of (lookahead + 1)
        # Each "base step" processes (lookahead + 1) frames
        base_step_size = encoder._cfg.att_context_size[1] + 1
        if num_frames_per_chunk % base_step_size != 0:
            raise ValueError(
                f"num_frames_per_chunk must be a multiple of (lookahead + 1) = {base_step_size}. "
                f"Got num_frames_per_chunk={num_frames_per_chunk}"
            )
        num_sub_steps = num_frames_per_chunk // base_step_size

        # Run the encoder with cache (using CUDA graphs if available)
        start_time = time.time()
        
        # Collect encoded chunks from all sub-steps
        encoded_chunks = []
        
        for sub_step in range(num_sub_steps):
            sub_step_start_time = time.time()
            
            # Compute the effective frame index for this sub-step
            sub_frame_idx = frame_idx + (sub_step * base_step_size)
            is_first_sub_step = (sub_frame_idx == 0)

            if is_first_sub_step:
                # First frame: take first chunk_size_first columns from mel spectrogram
                cur_chunk_size = chunk_size_first
                cur_pre_encode_cache_size = pre_encode_cache_size_first
                drop_extra_pre_encoded = 0

                # Slice the main chunk: [0 : chunk_size_first]
                mel_chunk = processed_signal[:, :, :cur_chunk_size]

                # Prepend zeros for pre_encode_cache on the left
                if cur_pre_encode_cache_size > 0:
                    zeros_pad = torch.zeros(
                        (processed_signal.size(0), self.perception_input_features, cur_pre_encode_cache_size),
                        device=self.device,
                        dtype=processed_signal.dtype,
                    )
                    mel_chunk = torch.cat([zeros_pad, mel_chunk], dim=-1)
            else:
                # N-th sub-step (sub_frame_idx > 0):
                # Compute step_number based on sub_frame_idx and base_step_size
                cur_chunk_size = chunk_size
                cur_pre_encode_cache_size = pre_encode_cache_size
                drop_extra_pre_encoded = streaming_cfg.drop_extra_pre_encoded

                mel_T = processed_signal.shape[-1]

                # Calculate start position for the main chunk
                # step_number counts how many base_step_size chunks have been processed before this one
                step_number = sub_frame_idx // base_step_size
                chunk_start = shift_size_first + (step_number - 1) * shift_size
                chunk_end = chunk_start + cur_chunk_size

                # Check if we've exceeded the mel spectrogram length (buffer is full and sliding)
                # When buffer is full, we take from the end but shifted back by (chunk_size - shift_size_first)
                # to account for the different sizing of the first chunk vs subsequent chunks
                offset = chunk_size - shift_size_first
                if chunk_end > mel_T - offset:
                    # Buffer is full - take from the end of the mel spectrogram, staggered by sub_step
                    # sub_step 0 should be furthest from end (oldest of the new chunks)
                    # last sub_step should be closest to end (newest chunk)
                    sub_steps_remaining = num_sub_steps - 1 - sub_step
                    chunk_end = mel_T - offset - sub_steps_remaining * shift_size
                    chunk_start = chunk_end - cur_chunk_size

                # Get the main chunk
                main_chunk = processed_signal[:, :, chunk_start:chunk_end]

                # Get the pre-encode cache (columns before chunk_start)
                cache_start = max(0, chunk_start - cur_pre_encode_cache_size)
                cache_mel = processed_signal[:, :, cache_start:chunk_start]

                # Pad with zeros on the left if we don't have enough cache
                if cache_mel.shape[-1] < cur_pre_encode_cache_size:
                    zeros_pad = torch.zeros(
                        (cache_mel.size(0), cache_mel.size(1), cur_pre_encode_cache_size - cache_mel.shape[-1]),
                        device=self.device,
                        dtype=cache_mel.dtype,
                    )
                    cache_mel = torch.cat([zeros_pad, cache_mel], dim=-1)

                # Combine: pre_encode_cache + main chunk
                mel_chunk = torch.cat([cache_mel, main_chunk], dim=-1)

            chunk_lengths = torch.tensor([mel_chunk.shape[-1]], dtype=torch.long, device=self.device)
            
            if self.use_perception_cudagraph and self.perception_cudagraph_state is not None and self.perception_cudagraph_state.is_captured():
                # Use CUDA graph path
                graph_state = self.perception_cudagraph_state
                
                # Copy inputs to static buffers
                if is_first_sub_step:
                    graph_state.static_mel_first.copy_(mel_chunk)
                else:
                    graph_state.static_mel_subsequent.copy_(mel_chunk)
                
                # Copy cache inputs
                if graph_state.static_cache_channel_in is not None and cache_last_channel is not None:
                    graph_state.static_cache_channel_in.copy_(cache_last_channel)
                if graph_state.static_cache_time_in is not None and cache_last_time is not None:
                    graph_state.static_cache_time_in.copy_(cache_last_time)
                if graph_state.static_cache_channel_len_in is not None and cache_last_channel_len is not None:
                    graph_state.static_cache_channel_len_in.copy_(cache_last_channel_len)
                
                # Replay the appropriate graph and read from the corresponding output buffers
                if is_first_sub_step:
                    graph_state.graph_first.replay()
                    encoded_chunk = graph_state.static_encoded_first.clone()
                    # Read cache from FIRST graph's output buffers
                    cache_last_channel = graph_state.static_cache_channel_out_first.clone() if graph_state.static_cache_channel_out_first is not None else None
                    cache_last_time = graph_state.static_cache_time_out_first.clone() if graph_state.static_cache_time_out_first is not None else None
                    cache_last_channel_len = graph_state.static_cache_channel_len_out_first.clone() if graph_state.static_cache_channel_len_out_first is not None else None
                else:
                    graph_state.graph_subsequent.replay()
                    encoded_chunk = graph_state.static_encoded_subsequent.clone()
                    # Read cache from SUBSEQUENT graph's output buffers
                    cache_last_channel = graph_state.static_cache_channel_out_subsequent.clone() if graph_state.static_cache_channel_out_subsequent is not None else None
                    cache_last_time = graph_state.static_cache_time_out_subsequent.clone() if graph_state.static_cache_time_out_subsequent is not None else None
                    cache_last_channel_len = graph_state.static_cache_channel_len_out_subsequent.clone() if graph_state.static_cache_channel_len_out_subsequent is not None else None
                
            else:
                # Standard path (no CUDA graphs)
                (
                    encoded,
                    encoded_len,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                ) = encoder.cache_aware_stream_step(
                    processed_signal=mel_chunk,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=True,
                    drop_extra_pre_encoded=drop_extra_pre_encoded,
                )

                # Apply modality adapter
                modality_adapter = perception.modality_adapter
                encoded_adapted, _ = modality_adapter(audio_signal=encoded, length=encoded_len)
                
                # Apply projection: encoded_adapted is [B, C, T], proj expects [B, T, C]
                encoded_chunk = perception.proj(encoded_adapted.transpose(1, 2))  # [B, T, D]
            
            torch.cuda.synchronize()
            logging.info(f"  Sub-step {sub_step}/{num_sub_steps} (sub_frame_idx={sub_frame_idx}, first={is_first_sub_step}): {time.time() - sub_step_start_time:.4f}s")
            encoded_chunks.append(encoded_chunk)
        
        # Concatenate all encoded chunks along the time dimension
        if len(encoded_chunks) > 1:
            encoded_chunk = torch.cat(encoded_chunks, dim=1)  # [B, num_sub_steps * T_per_step, D]
        else:
            encoded_chunk = encoded_chunks[0]
        
        torch.cuda.synchronize()
        logging.info(f"Time taken for encoder ({num_sub_steps} sub-steps): {time.time() - start_time}")

        # Update cache state (no mel_buffer needed - we recompute each time)
        new_perception_cache = PerceptionCacheState(
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )

        return encoded_chunk, new_perception_cache

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
        # Handle complex objects (e.g., DynamicCache with __dict__ attributes)
        # Use deepcopy to ensure complete isolation between streams
        if hasattr(cache, '__dict__'):
            import copy
            return copy.deepcopy(cache)
        return cache

    def _prepare_tts_initial_state(self):
        if not self.decode_audio:
            return
        if not hasattr(self.model, 'tts_model'):
            return

        logging.info("\n🎯 Preparing TTS warmup state...")

        with fp32_precision():
            speaker_audio, speaker_sr = torchaudio.load(self.speaker_reference)
            speaker_audio = resample(speaker_audio, speaker_sr, self.model.tts_model.target_sample_rate)

        speaker_audio = speaker_audio.to(self.device)
        speaker_audio_lens = torch.tensor([speaker_audio.size(1)], device=self.device).long()

        #  init tts_model
        self.model.tts_model.set_init_inputs(
            speaker_audio=speaker_audio,
            speaker_audio_lens=speaker_audio_lens,
            system_prompt=self.tts_system_prompt,
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
            # code, _, _ = self.model.tts_model.tts_model.generate_step(
            #     outputs.hidden_states[:, -1:], **self.generation_config
            # )

        self.first_context_subword_id = init_inputs["subword_ids"][:, -1].unsqueeze(-1)
        self.first_tts_code_input = code.detach().clone()
        self.first_tts_past_key_values_input = self._clone_cache(outputs.past_key_values)


        logging.info("✅ TTS warmup state prepared")

    def _update_audio_buffer(self, audio_buffer, buffer_fill_level, new_audio, buffer_size_samples):
        """
        Append incoming samples to the sliding-window buffer and produce the view used for inference.

        Parameters:
            audio_buffer (torch.Tensor): Tensor of shape `[1, buffer_size_samples]` holding the latest audio samples.
            buffer_fill_level (int): Number of valid samples currently stored in `audio_buffer`.
            new_audio (torch.Tensor): Incoming samples of shape `[1, slice_n_samples]` for the current step.
            buffer_size_samples (int): Total capacity of the buffer in samples.

        Returns:
            Tuple[torch.Tensor, int, torch.Tensor]:
                - Updated `audio_buffer` containing the newest samples (always capped to `buffer_size_samples`).
                - Updated `buffer_fill_level`, reflecting how many contiguous samples are valid.
                - `current_buffer`, a view over the valid portion of the buffer used for the model input.

        Notes:
            `audio_buffer` always retains the last `buffer_size_samples` samples even when overfilled,
            whereas `current_buffer` may be shorter during the initial warm-up phase when the buffer
            is not yet full.
        """
        if new_audio.shape[1] == 0:
            current_buffer = audio_buffer[:, :buffer_fill_level]
            return audio_buffer, buffer_fill_level, current_buffer

        remaining = new_audio

        if buffer_fill_level < buffer_size_samples and remaining.shape[1] > 0:
            warmup_take = min(buffer_size_samples - buffer_fill_level, remaining.shape[1])
            if warmup_take > 0:
                audio_buffer[:, buffer_fill_level:buffer_fill_level + warmup_take] = remaining[:, :warmup_take]
                buffer_fill_level += warmup_take
                remaining = remaining[:, warmup_take:]

        if remaining.shape[1] > 0:
            if remaining.shape[1] >= buffer_size_samples:
                audio_buffer = remaining[:, -buffer_size_samples:]
            else:
                audio_buffer = torch.cat([
                    audio_buffer[:, remaining.shape[1]:],
                    remaining
                ], dim=1)
            buffer_fill_level = buffer_size_samples
        current_buffer = audio_buffer if buffer_fill_level == buffer_size_samples else audio_buffer[:, :buffer_fill_level]
        return audio_buffer, buffer_fill_level, current_buffer

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
                       request_id: Optional[str] = None,
                       perception_cache: Optional[PerceptionCacheState] = None,
                       source_encoded_dict: Optional[dict] = None,
                       has_prompt: bool = False):

        # Set up effective request ID for vLLM streaming
        effective_request_id = request_id or self.request_id

        start_time_one_step = time.time()
        use_cache = dynamic_cache is not None
        batch_size = gen_text.shape[0]

        predicted_tokens = torch.empty((batch_size, num_frames_per_chunk), dtype=gen_text.dtype, device=gen_text.device)
        asr_predicted_tokens = torch.empty((batch_size, num_frames_per_chunk), dtype=gen_text.dtype, device=gen_text.device)

        # Do "perception" step outside the for-loop
        start_perception = time.time()

        if self.use_perception_cache and perception_cache is not None and perception_cache.is_initialized():
            # Cache-aware perception
            source_encoded, perception_cache = self._cache_aware_perception_step(
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

        # Save source_encoded if dict is provided
        if source_encoded_dict is not None:
            end_frame_idx = frame_idx + num_frames_per_chunk - 1
            key = f"{frame_idx}_{end_frame_idx}"
            source_encoded_dict[key] = source_encoded.detach().cpu().clone()

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
        for frame_offset in range(num_frames_per_chunk):
            current_frame_idx = frame_idx + frame_offset
            current_frame_index = base_frame_index + frame_offset
            current_frame_index = min(current_frame_index, total_encoded_frames - 1)
            current_frame_embedding = source_encoded[:, current_frame_index:current_frame_index + 1, :]

            current_input_emb = current_frame_embedding.clone()

            if current_frame_idx == 0 and not has_prompt:
                # Only add BOS if there's no prompt (BOS is already in prompt's position 0)
                current_input_emb += self._get_bos_embedding()
                current_input_emb += self._get_asr_bos_embedding()
            elif current_frame_idx == 0 and has_prompt:
                # With prompt: first audio frame uses pad embedding (like offline_inference)
                # gen_text[:, -1] from prompt positions is pad_id
                pad_id = self.model.stt_model.text_pad_id
                pad_token = torch.full((1,), fill_value=pad_id, device=self.device, dtype=torch.long)
                pad_emb = self.model.stt_model.embed_tokens(pad_token).to(dtype=self.dtype)
                pad_asr_emb = self.model.stt_model.embed_asr_tokens(pad_token).to(dtype=self.dtype)
                current_input_emb += pad_emb
                current_input_emb += pad_asr_emb
            else:
                # t > 0: add embeddings from model's own predictions at t-1
                last_token_emb = self.model.stt_model.embed_tokens(gen_text[:, current_frame_idx - 1])
                current_input_emb += last_token_emb
                last_asr_token_emb = self.model.stt_model.embed_asr_tokens(gen_asr_text[:, current_frame_idx - 1])
                current_input_emb += last_asr_token_emb

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
                    ans = self.model_llm_interface(
                        current_input_emb,
                        cache=dynamic_cache,
                        generated_tokens=gen_text,
                        current_step=current_frame_idx
                    )
                dynamic_cache = ans["cache"]
            else:
                new_input_embeds.append(current_input_emb)
                full_input_embeds = torch.cat(input_embeds_history + new_input_embeds, dim=1)
                ans = self.model_llm_interface(
                    full_input_embeds,
                    cache=None,
                    generated_tokens=gen_text,
                    current_step=current_frame_idx
                )

            torch.cuda.synchronize()
            time_stt_model = time.time() - start_stt_model
            logging.info(f"Time taken for stt_model: {time_stt_model:.3f}s")

            predicted_token = ans["predicted_token"]
            asr_predicted_token = ans["asr_predicted_token"]

            gen_text[:, current_frame_idx] = predicted_token
            predicted_tokens[:, frame_offset] = predicted_token

            gen_asr_text[:, current_frame_idx] = asr_predicted_token
            asr_predicted_tokens[:, frame_offset] = asr_predicted_token

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

                # update audio_toks_buffer with new code
                # * we do this still inside the for-loop, and update one frame at a time
                # * note: audio_toks_buffer will be fed to the audio decoder
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
            logging.debug(f"\n🔊 Decoding audio for {frame_idx}-th frame  ({num_frames_per_chunk=})")

            len_audio_toks_buffer = torch.tensor([self.codec_token_history_size], dtype=torch.long, device=self.device)

            start_time_decode = time.time()
            with fp32_precision(), torch.no_grad():
                decoded_audio, decoded_audio_len = self.model.tts_model.audio_codec.decode(
                    audio_toks_buffer,
                    len_audio_toks_buffer
                )

            torch.cuda.synchronize()
            time_audio_codec = time.time() - start_time_decode
            logging.info(f"Time taken for audio_codec: {time_audio_codec:.3f}s")

            logging.debug(f"   Decoded Audio full shape: {decoded_audio.shape}")
            logging.debug(f"   Decoded Audio length: {decoded_audio_len}")
            logging.debug(f"   samples_per_audio_output_frame={samples_per_audio_output_frame}")
            logging.debug(f"   num_frames_per_chunk={num_frames_per_chunk}")
            logging.debug(f"   Expected new samples to extract: {samples_per_audio_output_frame * num_frames_per_chunk}")

            decoded_audio_new = decoded_audio[:, :, -samples_per_audio_output_frame * num_frames_per_chunk:]
            logging.debug(f"   Extracted decoded_audio_new shape: {decoded_audio_new.shape}")

        else:
            audio_toks_buffer = None
            decoded_audio_new = None
            time_tts_model = 0
            time_audio_codec = 0

        # convert new text tokens to string that we can show to the user
        predicted_text_strs = []
        # loop over batch dimension
        for predicted_tok_ids_b in predicted_tokens:
            predicted_tok_ids_b = predicted_tok_ids_b.tolist()
            predicted_toks_b = self.tokenizer.ids_to_tokens(predicted_tok_ids_b)

            # TODO: make more robust to tokenizer changes
            # replace "Ġ" with " " to restore proper word boundaries
            # replace '<SPECIAL_12>' with ""
            predicted_toks_b = [tok.replace('<SPECIAL_12>', "").replace('Ġ', ' ') for tok in predicted_toks_b]

            predicted_text_strs.append("".join(predicted_toks_b))

        # convert new ASR tokens to string
        asr_predicted_text_strs = []
        for asr_predicted_tok_ids_b in asr_predicted_tokens:
            asr_predicted_tok_ids_b = asr_predicted_tok_ids_b.tolist()
            asr_predicted_toks_b = self.tokenizer.ids_to_tokens(asr_predicted_tok_ids_b)

            # TODO: make more robust to tokenizer changes
            # replace "Ġ" with " " to restore proper word boundaries
            # replace '<SPECIAL_12>' with ""
            asr_predicted_toks_b = [tok.replace('<SPECIAL_12>', "").replace('Ġ', ' ') for tok in asr_predicted_toks_b]

            asr_predicted_text_strs.append("".join(asr_predicted_toks_b))

        logging.info(f'frame {frame_idx}: USER\'s asr_predicted_text_strs: {asr_predicted_text_strs}')
        logging.info(f'frame {frame_idx}: --------------------------------AGENT\'s predicted_text_strs: {predicted_text_strs}')

        torch.cuda.synchronize()

        time_for_one_step = time.time() - start_time_one_step
        logging.info(f'frame {frame_idx}: Time taken for one step: {time_for_one_step:.3f}s')

        return {
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
            'source_encoded_dict': source_encoded_dict,
        }

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
                    logging.info(f"🎤→🤖 Forced turn-taking at frame {t}: inserted agent BOS (reason: pad window)")

            # ASR BOS → insert agent EOS if not present in window
            elif current_asr_token == self.model.stt_model.user_bos_id:
                if not (agent_text_window == self.model.stt_model.text_eos_id).any():
                    gen_text[batch_idx, t] = self.model.stt_model.text_eos_id
                    logging.info(f"🤖→🎤 Forced turn-taking at frame {t}: inserted agent EOS (reason: user started speaking)")

    @torch.no_grad()
    def inference_realtime_streaming(self, audio_path: str, num_frames_per_chunk: int = None, request_id: Optional[str] = None, pad_audio_to_sec: Optional[float] = None, audio_id: Optional[str] = None, system_prompt: Optional[str] = None):
        """
        Perform realtime streaming inference simulating microphone capture.

        Args:
            audio_path: Path to input audio file (simulates microphone input)
            num_frames_per_chunk: Number of frames to process per inference step (default: 1)
            request_id: Optional request ID for vLLM streaming
            pad_audio_to_sec: Optional duration to pad audio to (in seconds)
            audio_id: Optional audio ID for saving source_encoded tensors
            system_prompt: Optional system prompt to provide context to the model

        Returns:
            Dictionary with 'text', 'tokens_text', 'tokens_audio', 'audio', 'audio_len', 'system_prompt'
        """
        # Use provided value or default
        if num_frames_per_chunk is None:
            num_frames_per_chunk = DEFAULT_NUM_FRAMES_PER_CHUNK
        if num_frames_per_chunk < 1:
            raise ValueError("num_frames_per_chunk must be at least 1")
        start_time = time.time()

        logging.info("\n" + "=" * 70)
        logging.info("STARTING REALTIME STREAMING INFERENCE")
        logging.info("=" * 70)

        # Set up request ID for vLLM streaming
        stream_request_id = request_id or self.request_id

        buffer_size_frames = int(self.model_cfg.get("buffer_size_frames", DEFAULT_BUFFER_SIZE_FRAMES))
        buffer_size_samples = buffer_size_frames * FRAME_SIZE_SAMPLES
        if num_frames_per_chunk > buffer_size_frames:
            raise ValueError(
                f"num_frames_per_chunk ({num_frames_per_chunk}) must be "
                f"less than or equal to buffer_size_frames ({buffer_size_frames})."
            )

        att_context_size = self.model.stt_model.perception.encoder._cfg.att_context_size
        if self.use_perception_cache:
            min_buffer = num_frames_per_chunk * (att_context_size[1] + 1) + 2
            reason = (
                f"must be >= num_frames_per_chunk * (att_context_size[1] + 1) + 2 = "
                f"{num_frames_per_chunk} * ({att_context_size[1]} + 1) + 2 = {min_buffer} "
                f"when using perception cache (+2 to minimize windowing artifacts)"
            )
        else:
            min_buffer = att_context_size[0] + att_context_size[1] + 1
            reason = (
                f"must be >= att_context_size[0] + att_context_size[1] + 1 = "
                f"{att_context_size[0]} + {att_context_size[1]} + 1 = {min_buffer} "
                f"without perception cache"
            )
        if buffer_size_frames < min_buffer:
            raise ValueError(
                f"buffer_size_frames ({buffer_size_frames}) is too small: {reason}."
            )
        if self.decode_audio and num_frames_per_chunk > self.codec_token_history_size:
            raise ValueError(
                f"num_frames_per_chunk ({num_frames_per_chunk}) must be "
                f"<= codec_token_history_size ({self.codec_token_history_size}) when decode_audio=True. "
                f"Either reduce num_frames_per_chunk or increase codec_token_history_size."
            )
        logging.info(f"Buffer size: {buffer_size_frames} frames ({buffer_size_frames * FRAME_SIZE_SEC}s)")
        logging.info(f"Frames per inference step: {num_frames_per_chunk}")

        # Load audio file (simulating microphone stream)
        logging.info(f"\n📁 Loading audio file: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        total_samples = len(audio_signal)
        total_duration = total_samples / SAMPLE_RATE

        logging.info(f"   Total duration: {total_duration:.2f}s")
        logging.info(f"   Total samples: {total_samples}")

        # Optionally pad audio to a specific duration
        if pad_audio_to_sec is not None and pad_audio_to_sec > total_duration:
            target_samples = int(pad_audio_to_sec * SAMPLE_RATE)
            audio_signal = np.pad(audio_signal, (0, target_samples - total_samples), mode='constant')
            total_samples = len(audio_signal)
            logging.info(f"   Padded to {pad_audio_to_sec:.2f}s ({total_samples} samples)")

        # derive num_inference_steps
        total_frames_maybe = int(np.ceil(total_samples / FRAME_SIZE_SAMPLES)) # "maybe" because we might need to add padding
        num_inference_steps = (total_frames_maybe // num_frames_per_chunk)
        if total_frames_maybe % num_frames_per_chunk != 0:
            num_inference_steps += 1
        total_frames = num_inference_steps * num_frames_per_chunk

        # pad audio signal so that it is divisible by num_inference_steps
        padded_total_samples = num_inference_steps * num_frames_per_chunk * FRAME_SIZE_SAMPLES
        if padded_total_samples > total_samples:
            audio_signal = np.pad(audio_signal, (0, padded_total_samples - total_samples), mode='constant')
            logging.info(f"   Padded to: {padded_total_samples} samples")
        logging.info(f" {num_frames_per_chunk=} => {total_frames=}, {num_inference_steps=}")

        # convert audio signal to tensor
        audio_signal_tensor = torch.tensor(audio_signal, dtype=self.dtype, device=self.device).unsqueeze(0)

        # Check if Nemotron (no cache support)
        use_cache = 'Nemotron' not in self.model.stt_model.cfg.pretrained_llm
        logging.info(f"\n⚙️  Model: {self.model.stt_model.cfg.pretrained_llm}")
        logging.info(f"   Use cache: {use_cache}")

        # Initialize buffer and state
        audio_buffer = torch.zeros(1, buffer_size_samples, dtype=self.dtype, device=self.device)
        buffer_fill_level = 0  # How many samples currently in buffer

        # Initialize LLM cache
        if use_cache:
            llm_cache = DynamicCache()
        else:
            llm_cache = None
            input_embeds_history = []  # For no-cache mode

        # Process system prompt if provided (before streaming audio)
        prompt_embedded = None
        prompt_len = 0
        
        if system_prompt:
            start_get_prompt_embeddings = time.time()
            prompt_embedded, prompt_len = self._prepare_system_prompt_embeddings(system_prompt)
            logging.info(f"Time taken to get prompt embeddings: {time.time() - start_get_prompt_embeddings:.3f}s")
            if prompt_embedded is not None and "vllm" in self.engine_type.lower():
                # Prepare token IDs for the prompt
                prompt_token_ids = (
                    [self.tokenizer.bos_id] +
                    self.tokenizer.text_to_ids(system_prompt) +
                    [self.tokenizer.eos_id]
                )

                # For vLLM mode: use efficient BATCH prefill (~20x faster than sequential)
                logging.info(f"   Batch prefilling {prompt_len} prompt embeddings...")
                start_batch_prefill = time.time()
                with torch.no_grad():
                    success = self.model_llm_interface(
                        prompt_embedded,
                        request_id=stream_request_id,
                        decode_steps=0,
                        prompt_token_ids=prompt_token_ids,
                    )
                logging.info(f"Time taken to batch prefill stt model: {time.time() - start_batch_prefill:.3f}s")
                if success:
                    logging.info(f" System prompt prefilled ({prompt_len} tokens)")
                else:
                    raise RuntimeError("vLLM batch prefill for system prompt failed.")
            elif prompt_embedded is not None and not use_cache:
                # For no-cache mode (Nemotron): add prompt embeddings to history
                # Split into individual frames for consistent processing
                for t in range(prompt_len):
                    input_embeds_history.append(prompt_embedded[:, t:t+1, :])
                logging.info(f"   Added {prompt_len} prompt embeddings to input_embeds_history")
            elif prompt_embedded is not None and use_cache:
                # For cache mode: process prompt through LLM to update cache
                with torch.no_grad():
                    ans = self.model.stt_model(prompt_embedded, cache=llm_cache)
                    llm_cache = ans.get("cache", llm_cache)
                logging.info(f"   ✅ System prompt processed, cache updated")

        # Initialize TTS
        code = None
        past_key_values = None
        subword_mask = None
        if self.decode_audio and hasattr(self.model, 'tts_model'):

            # init audio toks buffer with codec_token_history_size number of silence tokens
            # shape of audio_toks_buffer is: [1, codec_token_history_size, num_codes] (batch size = 1)
            audio_toks_buffer = self.model.tts_model.codec_silence_tokens.view(1, 1, -1).expand(
                -1, self.codec_token_history_size, -1
            ).to(self.device)

            if (
                self.first_context_subword_id is None
                or self.generation_config is None
                or self.first_tts_code_input is None
                or self.first_tts_past_key_values_input is None
            ) and not self.use_vllm_eartts:
                raise RuntimeError("TTS warmup state was not prepared during initialization.")

            if not self.use_vllm_eartts:
                past_key_values = self._clone_cache(self.first_tts_past_key_values_input)
                code = self.first_tts_code_input.detach().clone()
            else:
                start_batch_prefill = time.time()
                logging.info(f"   Batch prefilling TTS model with speaker embedding...")
                # use speaker embedding to prefill EarTTS's vLLM
                tts_result = self.model.tts_model.tts_model(
                    self.tts_init_inputs,
                    request_id=stream_request_id,
                    prompt_token_ids=self.tts_prompt_token_ids
                )
                #code, past_key_values = tts_result['codes'], tts_result['past_key_values']
                code = self.first_tts_code_input.detach().clone()
                past_key_values = None
                logging.info(f"Time taken to batch prefill tts model: {time.time() - start_batch_prefill:.3f}s")
                # Initialize subword_mask for vLLM path as well
            subword_mask = torch.ones(1, total_frames, device=self.device, dtype=torch.bool)
            logging.info(f"✅ TTS initialized")

        # Initialize perception cache if enabled
        perception_cache = None
        if self.use_perception_cache:
            perception_cache = self._get_initial_perception_cache_state(batch_size=1)
            logging.info(f"✅ Perception cache initialized")

        gen_text = torch.full((1, total_frames), self.model.stt_model.text_pad_id, device=self.device, dtype=torch.long)
        gen_asr_text = torch.full((1, total_frames), self.model.stt_model.text_pad_id, device=self.device, dtype=torch.long)

        # initialize list to which we will append generated audio segments
        audio_segments = []

        # Initialize dict to store source_encoded tensors if directory is specified
        # path_to_enc_save is a directory; each audio's encodings are saved as {audio_id}_source_encoded.pt
        path_to_enc_save = self.model_cfg.get("path_to_enc_save", None)
        source_encoded_dict = {} if path_to_enc_save else None

        # Derive audio_id from audio_path if not provided
        if audio_id is None:
            audio_id = os.path.splitext(os.path.basename(audio_path))[0]

        logging.info("\n" + "=" * 70)
        logging.info("🎤 STARTING FRAME-BY-FRAME PROCESSING")
        logging.info("=" * 70)

        # frame_idx corresponds to index of the first frame passed to infer_one_step
        # (we need this distinction in the case that num_frames_per_chunk > 1)
        frame_idx = 0
        while frame_idx < total_frames:
            slice_start = frame_idx * FRAME_SIZE_SAMPLES
            slice_n_samples = num_frames_per_chunk * FRAME_SIZE_SAMPLES
            slice_end = slice_start + slice_n_samples
            new_audio = audio_signal_tensor[:, slice_start:slice_end]

            audio_buffer, buffer_fill_level, current_buffer = self._update_audio_buffer(
                audio_buffer, buffer_fill_level, new_audio, buffer_size_samples
            )

            result = self.infer_one_step(
                audio_input=current_buffer,
                num_frames_per_chunk=num_frames_per_chunk,
                frame_idx=frame_idx,
                gen_text=gen_text,
                audio_toks_buffer=audio_toks_buffer if self.decode_audio else None,
                input_embeds_history=input_embeds_history if not use_cache else [],
                dynamic_cache=llm_cache if use_cache else None,
                past_key_values=past_key_values if self.decode_audio else None,
                code=code if self.decode_audio else None,
                subword_mask=subword_mask if self.decode_audio else None,
                gen_asr_text=gen_asr_text,
                request_id=stream_request_id,
                perception_cache=perception_cache,
                source_encoded_dict=source_encoded_dict,
                has_prompt=(prompt_len > 0),
            )

            # handle results from infer_one_step
            input_embeds_history = result['input_embeds_history']
            llm_cache = result['dynamic_cache']
            if self.use_perception_cache:
                perception_cache = result.get('perception_cache', perception_cache)
            if source_encoded_dict is not None:
                source_encoded_dict = result.get('source_encoded_dict', source_encoded_dict)
            if self.decode_audio:
                audio_toks_buffer = result['audio_toks_buffer']
                decoded_audio_new = result['decoded_audio_new']
                if decoded_audio_new is not None:
                    audio_segments.append(decoded_audio_new)

                past_key_values = result['past_key_values']
                code = result['code']
            else:
                decoded_audio_new = None

            if frame_idx % 10 == 0 or frame_idx < 3 or gen_text[:, frame_idx].item() == self.model.stt_model.text_eos_id:
                token_str = self.tokenizer.ids_to_text([gen_text[0, frame_idx].item()])
                buffer_status = f"{buffer_fill_level}/{buffer_size_samples}" if buffer_fill_level < buffer_size_samples else "FULL"
                special_label = ""
                if gen_text[0, frame_idx].item() == self.model.stt_model.text_bos_id:
                    special_label = " [BOS]"
                elif gen_text[0, frame_idx].item() == self.model.stt_model.text_eos_id:
                    special_label = " [EOS]"
                elif gen_text[0, frame_idx].item() == self.model.stt_model.text_pad_id:
                    special_label = " [PAD]"
                logging.info(f"Frame {frame_idx:3d}/{total_frames} | Buffer: {buffer_status:20s} | Token: {gen_text[0, frame_idx].item():5d}{special_label} | '{token_str}'")

            # if gen_text[:, frame_idx].item() == self.model.stt_model.text_eos_id:
                # gen_text[:, frame_idx+1:] = self.model.stt_model.text_pad_id
                # total_frames = frame_idx + 1
                # break

            frame_idx += num_frames_per_chunk

        # Prepare results
        elapsed_time = time.time() - start_time
        logging.info("\n" + "=" * 70)
        logging.info("✅ STREAMING INFERENCE COMPLETED")
        logging.info("=" * 70)
        logging.info(f"Total time: {elapsed_time:.2f}s")
        logging.info(f"Audio duration: {total_duration:.2f}s")
        logging.info(f"RTF (Real-Time Factor): {elapsed_time / total_duration:.2f}x")
        logging.info(f"Processed frames: {total_frames}")

        # Trim to actual length
        # TODO: this is currently redundant since we iterate over all frames in the while loop
        gen_text = gen_text[:, :total_frames]
        gen_asr_text = gen_asr_text[:, :total_frames]

        # Decode text
        lengths = torch.tensor([total_frames], dtype=torch.long, device=self.device)
        text_output = tokens_to_str(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.model.stt_model.text_pad_id, eval_text_turn_taking=True)

        # Decode ASR text
        asr_text_output = tokens_to_str(gen_asr_text, lengths, tokenizer=self.tokenizer, pad_id=self.model.stt_model.text_pad_id, eval_text_turn_taking=True)

        # Also create raw versions with <SPECIAL_12> kept for comparison
        text_output_raw = tokens_to_str_raw(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.model.stt_model.text_pad_id)
        asr_text_output_raw = tokens_to_str_raw(gen_asr_text, lengths, tokenizer=self.tokenizer, pad_id=self.model.stt_model.text_pad_id)

        logging.info(f"\n📝 Generated text: {text_output[0]}")
        logging.info(f"\n🎤 Generated ASR text: {asr_text_output[0]}")

        # Save source_encoded tensors if directory is specified
        # Saves to {path_to_enc_save}/{audio_id}_source_encoded.pt
        if path_to_enc_save and source_encoded_dict:
            os.makedirs(path_to_enc_save, exist_ok=True)
            enc_save_path = os.path.join(path_to_enc_save, f"{audio_id}_source_encoded.pt")
            torch.save(source_encoded_dict, enc_save_path)
            logging.info(f"✅ Saved source_encoded tensors ({len(source_encoded_dict)} entries) to: {enc_save_path}")

        ans = {
            "text": text_output,
            "text_raw": text_output_raw,
            "tokens_text": gen_text,
            "tokens_len": lengths,
            "audio": torch.cat(audio_segments, dim=-1) if audio_segments else None,
            "asr_text": asr_text_output,
            "asr_text_raw": asr_text_output_raw,
            "asr_tokens": gen_asr_text,
            "system_prompt": system_prompt if system_prompt else "",
        }

        if self.use_vllm_llm or self.use_vllm_eartts:
            self.abort_request(stream_request_id)

        return ans


def main():
    parser = argparse.ArgumentParser(description="Realtime Streaming Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to eartts's checkpoint with TTS (HF format)")
    parser.add_argument("--llm_checkpoint_path", type=str, required=True,
                       help="Path to checkpoint with LLM/perception (HF format)")
    parser.add_argument("--audio_path", type=str, default=None,
                       help="Path to input audio file (for single-file mode)")
    parser.add_argument("--input_json", type=str, default=None,
                       help="Path to input JSON file containing list of records with audio_filepath and text fields (for batch mode)")
    parser.add_argument("--output_json", type=str, default=None,
                       help="Path to output JSON file with predictions (for batch mode)")
    parser.add_argument("--output_dir", type=str, default="output_streaming",
                       help="Output directory for audio files (for batch mode)")
    parser.add_argument("--pad_audio_to_sec", type=float, default=None,
                       help="Pad audio to this duration in seconds (useful for consistent buffer behavior)")
    parser.add_argument("--speaker_reference", type=str, required=True,
                       help="Path to speaker reference audio file")
    parser.add_argument("--buffer_size_frames", type=int, default=DEFAULT_BUFFER_SIZE_FRAMES,
                       help=f"Size of audio buffer in frames (each frame = 80ms, default: {DEFAULT_BUFFER_SIZE_FRAMES})")
    parser.add_argument("--num_frames_per_chunk", type=int, default=DEFAULT_NUM_FRAMES_PER_CHUNK,
                       help="Number of frames per inference step (default: 1)")
    parser.add_argument("--output_text", type=str, default="output_text_streaming.txt",
                       help="Output text file path")
    parser.add_argument("--output_asr_text", type=str, default="output_asr_text_streaming.txt",
                       help="Output ASR text file path")
    parser.add_argument("--output_audio", type=str, default="generated_audio_streaming.wav",
                       help="Output audio file path")
    parser.add_argument("--decode_audio", action="store_true",
                       help="Whether to decode audio")
    parser.add_argument("--combine_inp_out_audio", action="store_true",
                    help="Whether to decode audio")

    # Perception cache argument
    parser.add_argument("--use_perception_cache", action="store_true",
                       help="Enable cache-aware streaming for perception encoder")
    parser.add_argument("--use_perception_cudagraph", action="store_true",
                       help="Use CUDA graphs for perception encoder (requires --use_perception_cache)")
    parser.add_argument("--path_to_enc_save", type=str, default=None,
                       help="Directory to save source_encoded tensors. Each audio file's encodings "
                            "are saved as {audio_id}_source_encoded.pt (dict with frame idx keys)")

    # vLLM arguments
    parser.add_argument("--engine_type", type=str, default="native", choices=["native", "vllm_llm", "vllm_eartts", "vllm_llm_vllm_eartts"],
                       help="Engine type for inference (default: native)")
    parser.add_argument("--vllm_llm_engine_path", type=str, default=None,
                       help="Path to vLLM-compatible model checkpoint if the path not exists, it will be auto-converted")
    parser.add_argument("--vllm_max_model_len", type=int, default=768,
                       help="Maximum sequence length for vLLM (default: 768)")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, nargs='+', default=[0.4],
                       help="GPU memory utilization for vLLM. Single value shared by both engines; two values assign to LLM and TTS respectively.")
    parser.add_argument("--vllm_llm_dtype", type=str, default="bfloat16",
                       help="Data type for vLLM (default: bfloat16)")

    # vLLM EarTTS arguments
    parser.add_argument("--vllm_eartts_engine_path", type=str, default=None,
                       help="Path to vLLM-compatible EarTTS model checkpoint if the path not exists, it will be auto-converted")
    parser.add_argument("--vllm_eartts_dtype", type=str, default="float32",
                       help="Data type for vLLM (default: float32)")

    # Sampling parameters
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p (nucleus) sampling threshold. 1.0 disables it (greedy). Default: 1.0")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                       help="Repetition penalty for generated tokens. 1.0 disables it. Default: 1.0. Recommended: 1.2")

    # System prompt
    parser.add_argument("--system_prompt", type=str, default=None,
                       help="System prompt to provide context to the model. Can also be specified per-record in input JSON.")
    parser.add_argument("--tts_system_prompt", type=str, default=None,
                       help="System prompt for EARTTS model.")
    args = parser.parse_args()

    # Validate arguments: either audio_path OR input_json must be provided
    if args.audio_path is None and args.input_json is None:
        parser.error("Either --audio_path (single-file mode) or --input_json (batch mode) must be provided")
    if args.audio_path is not None and args.input_json is not None:
        parser.error("Cannot use both --audio_path and --input_json at the same time")

    try:
        import json
        import soundfile as sf

        model_cfg_dict = {
            "model_path": args.model_path,
            "llm_checkpoint_path": args.llm_checkpoint_path,
            "speaker_reference": args.speaker_reference,
            "buffer_size_frames": args.buffer_size_frames,
            "decode_audio": bool(args.decode_audio),
            "engine_type": args.engine_type,
            "use_perception_cache": bool(args.use_perception_cache),
            "use_perception_cudagraph": bool(args.use_perception_cudagraph),
            "path_to_enc_save": args.path_to_enc_save,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "tts_system_prompt": args.tts_system_prompt,
        }

        # Pop GPU memory utilization values: first for LLM, second (or same) for TTS
        _gpu_mem = list(args.vllm_gpu_memory_utilization)
        gpu_mem_llm = _gpu_mem.pop(0)
        gpu_mem_tts = _gpu_mem.pop(0) if _gpu_mem else gpu_mem_llm

        # Add vLLM configuration if using vLLM engine
        if "vllm_llm" in args.engine_type:
            model_cfg_dict["vllm_llm_config"] = {
                "model_path": args.model_path,
                "max_model_len": args.vllm_max_model_len,
                "gpu_memory_utilization": gpu_mem_llm,
                "dtype": args.vllm_llm_dtype,
                "engine_path": args.vllm_llm_engine_path,  # Will auto-convert if needed
                "pretrained_llm": args.llm_checkpoint_path,
            }

        if "vllm_eartts" in args.engine_type:
            model_cfg_dict["vllm_tts_config"] = {
                "model_path": args.model_path, # we use exactly the same whole duplexs2s ckpt
                "max_model_len": args.vllm_max_model_len,
                "gpu_memory_utilization": gpu_mem_tts,
                "dtype": args.vllm_eartts_dtype,
                "engine_path": args.vllm_eartts_engine_path,
                "pretrained_llm": None,
                "skip_tokenizer_init": True
            }

        model_cfg = OmegaConf.create(model_cfg_dict)

        model = NemotronVoicechatInferenceWrapper(model_cfg=model_cfg)

        # =========================================
        # BATCH MODE: Process JSON input
        # =========================================
        if args.input_json is not None:
            logging.info(f"📖 Loading input JSON: {args.input_json}")
            with open(args.input_json, 'r') as f:
                input_records = [json.loads(line) for line in f]

            logging.info(f"Found {len(input_records)} records to process")

            # Create output directory if it doesn't exist
            os.makedirs(args.output_dir, exist_ok=True)

            # Set default output_json paths - create both processed and raw versions
            if args.output_json:
                base_path = args.output_json.rsplit('.', 1)[0] if '.' in args.output_json else args.output_json
                output_json_processed = f"{base_path}_processed.json"
                output_json_raw = f"{base_path}_raw.json"
            else:
                output_json_processed = os.path.join(args.output_dir, "output_results_processed.json")
                output_json_raw = os.path.join(args.output_dir, "output_results_raw.json")

            # Open both output files for incremental writing
            logging.info(f"📝 Output will be saved incrementally to:")
            logging.info(f"   Processed: {output_json_processed}")
            logging.info(f"   Raw: {output_json_raw}")
            output_file_processed = open(output_json_processed, 'w', encoding='utf-8')
            output_file_raw = open(output_json_raw, 'w', encoding='utf-8')

            # Process each record
            output_records = []
            wer_scores = []  # Store WER for each utterance

            try:
                for idx, record in enumerate(input_records):
                    logging.info("\n" + "=" * 70)
                    logging.info(f"📝 Processing record {idx + 1}/{len(input_records)}")
                    logging.info("=" * 70)

                    audio_path = record.get('audio_filepath')
                    ground_truth_text = record.get('text', '')
                    # Get system_prompt from record, fallback to command line arg
                    record_system_prompt = record.get('system_prompt', args.system_prompt)

                    if not audio_path:
                        logging.warning(f"⚠️  Record {idx} missing audio_filepath, skipping...")
                        continue

                    if not os.path.exists(audio_path):
                        logging.warning(f"⚠️  Audio file not found: {audio_path}, skipping...")
                        continue

                    logging.info(f"   Audio: {audio_path}")
                    logging.info(f"   Ground truth: {ground_truth_text}")

                    # Extract ID from audio filename (without extension)
                    audio_id = os.path.splitext(os.path.basename(audio_path))[0]

                    # Run inference with unique request_id per record to avoid vLLM race conditions
                    results = model.inference_realtime_streaming(
                        audio_path,
                        num_frames_per_chunk=args.num_frames_per_chunk,
                        pad_audio_to_sec=args.pad_audio_to_sec,
                        request_id=f"streaming_request_{idx}",
                        audio_id=audio_id,
                        system_prompt=record_system_prompt,
                    )

                    # Extract predictions
                    pred_asr_text = results['asr_text'][0] if 'asr_text' in results else ''
                    pred_asr_text_raw = results['asr_text_raw'][0] if 'asr_text_raw' in results else ''
                    pred_text = results['text'][0] if 'text' in results else ''
                    pred_text_raw = results['text_raw'][0] if 'text_raw' in results else ''

                    # Calculate WER between predicted ASR and ground truth
                    try:
                        cleaned_pred = clean_pred_text(pred_asr_text)
                        cleaned_gt = clean_pred_text(ground_truth_text)
                        if cleaned_gt.strip() and cleaned_pred.strip():
                            utterance_wer = wer(cleaned_gt, cleaned_pred)
                            wer_scores.append(utterance_wer)
                        else:
                            utterance_wer = None
                            logging.warning(f"⚠️  Empty text for WER calculation (ground truth or prediction)")
                    except Exception as e:
                        utterance_wer = None
                        logging.warning(f"⚠️  Error calculating WER: {e}")

                    # Log WER for this utterance
                    if utterance_wer is not None:
                        # Print WER in green color in the terminal
                        print(f"\033[92m📊 WER for utterance {idx + 1}: {utterance_wer:.4f} ({utterance_wer * 100:.2f}%)\033[0m")

                    # Save audio if available and get pred_audio path
                    pred_audio_path = None
                    if args.decode_audio and 'audio' in results and results['audio'] is not None:
                        # Derive output filename from input audio filepath
                        input_basename = os.path.splitext(os.path.basename(audio_path))[0]
                        audio_filename = f"{idx:04d}_{input_basename}_output.wav"
                        pred_audio_path = os.path.join(args.output_dir, audio_filename)

                        audio_np = results['audio'].float().cpu().numpy().flatten()

                        sf.write(pred_audio_path, audio_np, model.target_sample_rate)
                        logging.info(f"✅ Audio saved: {pred_audio_path}")

                        # Save stereo audio (input + output) if requested
                        if args.combine_inp_out_audio:
                            stereo_filename = f"{idx:04d}_{input_basename}_combined.wav"
                            stereo_path_out = os.path.join(args.output_dir, stereo_filename)

                            inp_audio, sr = librosa.load(audio_path, sr=model.target_sample_rate)

                            # Prepend silence to output channel to account for
                            # the one-chunk processing delay: the server can't
                            # produce output until it has received a full input chunk.
                            delay_samples = int(args.num_frames_per_chunk * FRAME_SIZE_SEC * model.target_sample_rate)
                            out_audio_delayed = np.concatenate([np.zeros(delay_samples, dtype=audio_np.dtype), audio_np])

                            max_len = max(len(inp_audio), len(out_audio_delayed))
                            inp_audio_padded = np.pad(inp_audio, (0, max_len - len(inp_audio)))
                            out_audio_padded = np.pad(out_audio_delayed, (0, max_len - len(out_audio_delayed)))

                            # Stereo: channel 0 = input, channel 1 = output
                            stereo_audio = np.stack([inp_audio_padded, out_audio_padded], axis=1)
                            sf.write(stereo_path_out, stereo_audio, model.target_sample_rate)
                            logging.info(f"✅ Stereo audio saved: {stereo_path_out}")

                    # Prepare output records with specific key order for easy diff comparison
                    # Keys: id, target_text, pred_text, pred_audio, src_text, pred_src_text, system_prompt

                    # Get system_prompt from results
                    result_system_prompt = results.get('system_prompt', '')

                    # Processed version (SPECIAL_12 removed)
                    output_record_processed = {
                        'id': audio_id,
                        'target_text': '',
                        'pred_audio': pred_audio_path,
                        'src_text': ground_truth_text,
                        'pred_src_text': pred_asr_text,
                        'pred_text': pred_text,
                        'system_prompt': result_system_prompt,
                    }

                    # Raw version (SPECIAL_12 kept)
                    output_record_raw = {
                        'id': audio_id,
                        'target_text': '',
                        'pred_audio': pred_audio_path,
                        'src_text': ground_truth_text,
                        'pred_src_text': pred_asr_text_raw,
                        'pred_text': pred_text_raw,
                        'system_prompt': result_system_prompt,
                    }

                    output_records.append(output_record_processed)

                    # Write records immediately to both files
                    json.dump(output_record_processed, output_file_processed, ensure_ascii=False)
                    output_file_processed.write('\n')
                    output_file_processed.flush()

                    json.dump(output_record_raw, output_file_raw, ensure_ascii=False)
                    output_file_raw.write('\n')
                    output_file_raw.flush()

                    logging.info(f"✅ Record {idx + 1} completed and saved")

            finally:
                # Always close output files, even if there's an error
                output_file_processed.close()
                output_file_raw.close()

            # Summary
            logging.info("\n" + "=" * 70)
            logging.info("💾 ALL RESULTS SAVED")
            logging.info("=" * 70)
            logging.info(f"✅ Results saved to:")
            logging.info(f"   Processed: {output_json_processed}")
            logging.info(f"   Raw: {output_json_raw}")
            logging.info(f"   Processed {len(output_records)}/{len(input_records)} records successfully")

            # Calculate and display average WER
            if wer_scores:
                avg_wer = np.mean(wer_scores)
                logging.info("\n" + "=" * 70)
                logging.info("📊 WER STATISTICS")
                logging.info("=" * 70)
                logging.info(f"   Total utterances with WER: {len(wer_scores)}")
                logging.info(f"   Average WER: {avg_wer:.4f} ({avg_wer * 100:.2f}%)")
                logging.info(f"   Min WER: {np.min(wer_scores):.4f} ({np.min(wer_scores) * 100:.2f}%)")
                logging.info(f"   Max WER: {np.max(wer_scores):.4f} ({np.max(wer_scores) * 100:.2f}%)")
            else:
                logging.warning("⚠️  No WER scores calculated")

            logging.info("=" * 70)
            logging.info("✅ ALL DONE!")
            logging.info("=" * 70)

        # =========================================
        # SINGLE-FILE MODE: Process single audio file
        # =========================================
        else:
            # Run inference
            results = model.inference_realtime_streaming(
                args.audio_path,
                num_frames_per_chunk=args.num_frames_per_chunk,
                pad_audio_to_sec=args.pad_audio_to_sec,
                system_prompt=args.system_prompt,
            )

            # Save outputs
            logging.info("\n" + "=" * 70)
            logging.info("💾 SAVING OUTPUTS")
            logging.info("=" * 70)

            # Save text
            with open(args.output_text, 'w') as f:
                f.write(results['text'][0].replace("<|", "\n<|"))
            logging.info(f"✅ Text output saved: {args.output_text}")

            # Save ASR text
            if 'asr_text' in results:
                with open(args.output_asr_text, 'w') as f:
                    f.write(results['asr_text'][0])
                logging.info(f"✅ ASR text output saved: {args.output_asr_text}")

            # Save audio if available
            if args.decode_audio and 'audio' in results:
                audio_np = results['audio'].float().cpu().numpy().flatten()

                if args.combine_inp_out_audio:
                    # Load input audio and resample to match the model's target sample rate
                    inp_audio, sr = librosa.load(args.audio_path, sr=model.target_sample_rate)

                    # Prepend silence to output channel to account for
                    # the one-chunk processing delay: the server can't
                    # produce output until it has received a full input chunk.
                    delay_samples = int(args.num_frames_per_chunk * FRAME_SIZE_SEC * model.target_sample_rate)
                    out_audio_delayed = np.concatenate([np.zeros(delay_samples, dtype=audio_np.dtype), audio_np])

                    max_len = max(len(inp_audio), len(out_audio_delayed))
                    inp_audio_padded = np.pad(inp_audio, (0, max_len - len(inp_audio)))
                    out_audio_padded = np.pad(out_audio_delayed, (0, max_len - len(out_audio_delayed)))

                    # Stereo: channel 0 = input, channel 1 = output
                    audio_np = np.stack([inp_audio_padded, out_audio_padded], axis=1)

                import soundfile as sf
                sf.write(args.output_audio, audio_np, model.target_sample_rate)
                logging.info(f"✅ Audio output saved: {args.output_audio}")

                # Verify
                import subprocess
                size_output = subprocess.check_output(['du', '-h', args.output_audio]).decode().split()[0]
                logging.info(f"   File size: {size_output}")

            logging.info("=" * 70)
            logging.info("✅ ALL DONE!")
        logging.info("=" * 70)

    except Exception as e:
        logging.error(f"❌ ERROR during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

