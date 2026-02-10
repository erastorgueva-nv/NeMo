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

import os
import time

import torch
import librosa
from typing import List, Optional
from torch import Tensor
import soundfile as sf
from omegaconf import DictConfig
import math

from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.asr.inference.utils.enums import RequestType
from nemo.collections.asr.inference.streaming.framing.multi_stream import ContinuousBatchedFrameStreamer
from nemo.collections.asr.inference.streaming.buffering.audio_bufferer import BatchedAudioBufferer
from nemo.collections.asr.inference.utils.progressbar import ProgressBar
from nemo.collections.speechlm2.inference.pipelines.s2s_pipeline_interface import S2SPipelineInterface
from nemo.collections.speechlm2.inference.streaming.state.s2s_state import S2SStreamingState
from nemo.collections.speechlm2.inference.model_wrappers.realtime_streaming_wrapper import RealtimeStreamingInference
from nemo.collections.speechlm2.inference.streaming.state.s2s_context_manager import S2SContextManager
from nemo.collections.speechlm2.inference.utils.pipeline_utils import PipelineOutput


class StreamingS2SGenerator(S2SPipelineInterface):
	"""
	Streaming S2S generator.
	"""

	def __init__(self, cfg: DictConfig, s2s_model: RealtimeStreamingInference):
		# ------------------------------------------------------------------
		# Model & device
		# ------------------------------------------------------------------
		self.s2s_model = s2s_model
		self.device = self.s2s_model.device

		# ------------------------------------------------------------------
		# Streaming configuration
		# ------------------------------------------------------------------
		self.streaming_cfg = cfg.get("streaming", {})
		self.input_sample_rate = getattr(self.streaming_cfg, "input_sample_rate", 16000)
		self.output_sample_rate = getattr(self.streaming_cfg, "output_sample_rate", 22050)
		self.batch_size = getattr(self.streaming_cfg, "batch_size", 1)
		self.max_len = getattr(self.streaming_cfg, "max_len", 200)
		

		# ------------------------------------------------------------------
		# Chunk & buffer sizes
		# ------------------------------------------------------------------
		self.chunk_size_in_secs = getattr(self.streaming_cfg, "chunk_size_in_secs", 0.08)
		# Check if self.chunk_size_in_secs is a multiple of 0.08.
		# Because of quirks of floating point arithmetic, the remainder could be either ~0 or ~0.08,
		# so we check for both cases.
		remainder = self.chunk_size_in_secs % 0.08
		if not (math.isclose(remainder, 0, abs_tol=1e-9) or math.isclose(remainder, 0.08, abs_tol=1e-9)):
			raise ValueError(f"Chunk size must be a multiple of 0.08s, but got {self.chunk_size_in_secs}")

		self.num_chunks_per_inference = int(self.chunk_size_in_secs / 0.08)

		# Buffer size controls how much audio context is kept for inference
		# (larger buffer helps online results get closer to offline results)
		# Default: 5.6 seconds (70 * 0.08)
		self.buffer_size_in_secs = getattr(self.streaming_cfg, "buffer_size_in_secs", 70 * 0.08)

		self.att_context_size = getattr(self.streaming_cfg, "att_context_size", [70,0])
		#self.window_size = getattr(self.streaming_cfg, "window_size", 510)
		#self.hop_length = self.enh_model.enh_model_cfg.encoder.hop_length

		# ------------------------------------------------------------------
		# bufferer – reused from ASR utilities
		# ------------------------------------------------------------------
		self.bufferer = BatchedAudioBufferer(
			sample_rate=self.input_sample_rate,
			buffer_size_in_secs=self.buffer_size_in_secs,
		)

		# --------------------------------------------------------------
		# Cache handling helpers
		# --------------------------------------------------------------
		self.use_cache: bool = getattr(self.streaming_cfg, "use_cache", True)

		# ------------------------------------------------------------------
		# System prompt configuration
		# ------------------------------------------------------------------
		s2s_cfg = cfg.get("s2s", {})
		self.system_prompt: Optional[str] = getattr(s2s_cfg, "system_prompt", None)
		if self.system_prompt:
			print(f"📝 System prompt configured: {self.system_prompt[:100]}{'...' if len(self.system_prompt) > 100 else ''}")

		# Context manager
		self.context_manager = S2SContextManager(
			s2s_model=self.s2s_model,
			num_slots=self.batch_size,
			max_len=self.max_len,
			use_cache=self.use_cache,
		)

		#self.window = torch.hamming_window(self.window_size)

		# Output directory for enhanced files
		self.output_dir = getattr(cfg, "output_dir", "./generated")

		# Parse and validate request type early, with a safe default
		req_type_cfg = getattr(self.streaming_cfg, "request_type", "frame")

		# Parse and validate the request type; only 'frame' is supported for s2s.
		self.request_type = RequestType.from_str(req_type_cfg)
		if self.request_type is not RequestType.FRAME:
			raise ValueError(f"Request type {self.request_type} is not supported for s2s.")

		super().__init__()

	# --------------------------------  ----------------------------------
	# State helpers
	# ------------------------------------------------------------------
	def create_state(self) -> S2SStreamingState:
		"""Create new empty state."""
		num_audio_codebooks = getattr(self.s2s_model.model, "_num_codebooks", 1)
		dtype = getattr(self.s2s_model, "compute_dtype", torch.float32)
		state = S2SStreamingState(
			device=self.device,
			dtype=dtype,
			max_len=self.max_len,
			num_audio_codebooks=num_audio_codebooks,
			output_sample_rate=self.output_sample_rate,
		)
		return state


	# ------------------------------------------------------------------
	# Output helpers
	# ------------------------------------------------------------------
	def log_output(self, frames: List[Frame], audio_wave: Tensor, ready_feats: List[bool], text_pieces: List[str], asr_text_pieces: List[str] = None):
		"""Append generated audio waveform and text to per-stream state."""
		for idx, frame in enumerate(frames):
			if not ready_feats[idx]:
				continue
			state = self.get_or_create_state(frame.stream_id)
			# audio_wave is [B, S]; take sample idx
			sample_audio = audio_wave[idx:idx+1, ...]
			# Determine text piece for this index
			piece = None
			if text_pieces and idx < len(text_pieces):
				candidate = text_pieces[idx]
				if isinstance(candidate, str) and candidate:
					piece = candidate
			
			# Determine ASR text piece
			asr_piece = None
			if asr_text_pieces and idx < len(asr_text_pieces):
				candidate = asr_text_pieces[idx]
				if isinstance(candidate, str) and candidate:
					asr_piece = candidate

			state.update_state(sample_audio, output_text=piece, output_asr_text=asr_piece)


	def inner_generate_step(self, frames: List[Frame], buffers: List[Tensor], left_paddings: List[int], ready_feats: List[bool]):
		"""Enhance chunks in *batch* using a shared ContextManager."""
		if len(frames) == 0:
			return

		stream_ids = [f.stream_id for f in frames]
		eos_flags = [f.is_last for f in frames]
		bos_flags = [f.is_first for f in frames]

		print(f"{stream_ids=} {bos_flags=} {eos_flags=}")

		if len(frames) == 0:
			return
		if len(frames) != 1:
			raise NotImplementedError("RealtimeStreamingInference currently supports batch_size == 1")

		# Retrieve per-stream cache slices
		# If this is the first frame of a stream, ensure we have a fresh context
		tts_prefill_code = None
		has_prompt = False
		if bos_flags[0]:
			print(f"🎬 Starting new stream {stream_ids[0]} - ensuring clean state")
			# Recreate context_manager entirely to ensure fresh KV cache (BS=1)
			self.context_manager = S2SContextManager(
				s2s_model=self.s2s_model,
				num_slots=self.batch_size,
				max_len=self.max_len,
				use_cache=self.use_cache,
			)
			
			# Prefill TTS speaker embedding and system prompt for new stream
			# Returns TTS prefill output code if vLLM EarTTS prefill happened
			tts_prefill_code = self._prefill_system_prompt(stream_ids[0])
			# Track whether a system prompt was prefilled so that infer_one_step
			# uses pad embedding (not BOS) at frame 0 — matching the standalone pipeline.
			has_prompt = bool(self.system_prompt)
		
		request_id = self._request_id_for_stream(stream_ids[0])
		
		context, _ = self.context_manager.get_context(stream_ids)
		
		# NOTE: Do NOT update context.code with tts_prefill_code!
		# The batch approach (inference_streaming_realtime.py) uses first_tts_code_input
		# (the INPUT codes from speaker reference), not the prefill OUTPUT codes.
		# Using prefill output codes causes audio quality degradation (mumbling).
		# The context.code is already correctly initialized to first_tts_code_input
		# in context_manager._create_context().
		if tts_prefill_code is not None:
			print(f"   TTS prefill generated codes shape: {tts_prefill_code.shape} (not used - using first_tts_code_input instead)")

		# Debug: print context_manager contents and sizes
		print(f"📊 S2SContextManager state:")
		print(f"   streamidx2slotidx: {self.context_manager.streamidx2slotidx}")
		print(f"   slotidx2streamidx: {self.context_manager.slotidx2streamidx}")
		print(f"   free_slots qsize: {self.context_manager.free_slots.qsize()}")
		print(f"   slot_contexts: {[c is not None for c in self.context_manager.slot_contexts]}")
		print(f"📊 Current context for stream {stream_ids[0]}:")
		print(f"   frame_idx: {context.frame_idx}")
		print(f"   gen_text: {context.gen_text.shape if context.gen_text is not None else None}")
		print(f"   gen_asr_text: {context.gen_asr_text.shape if context.gen_asr_text is not None else None}")
		print(f"   audio_toks_buffer: {context.audio_toks_buffer.shape if context.audio_toks_buffer is not None else None}")
		print(f"   input_embeds_history: {len(context.input_embeds_history)} items, shapes: {[e.shape for e in context.input_embeds_history[:3]]}{'...' if len(context.input_embeds_history) > 3 else ''}")
		print(f"   dynamic_cache: {type(context.dynamic_cache).__name__ if context.dynamic_cache is not None else None}, len={len(context.dynamic_cache) if context.dynamic_cache is not None else 0}")
		print(f"   past_key_values: {type(context.past_key_values).__name__ if context.past_key_values is not None else None}")
		print(f"   code: {context.code.shape if context.code is not None else None}")
		print(f"   subword_mask: {context.subword_mask.shape if context.subword_mask is not None else None}")

		audio_buffer = buffers[0]
		if audio_buffer.dim() == 1:
			audio_buffer = audio_buffer.unsqueeze(0)
		audio_buffer = audio_buffer.to(self.s2s_model.device, dtype=self.s2s_model.dtype)
		
		# Trim the buffer to exclude left padding (zeros at the beginning before buffer is filled)
		left_pad = left_paddings[0]
		if left_pad > 0:
			audio_buffer = audio_buffer[:, left_pad:]

		result = self.s2s_model.infer_one_step(
			audio_input=audio_buffer,
			num_frames_per_inference=self.num_chunks_per_inference,
			frame_idx=context.frame_idx,
			gen_text=context.gen_text,
			audio_toks_buffer=context.audio_toks_buffer,
			input_embeds_history=context.input_embeds_history,
			dynamic_cache=context.dynamic_cache,
			past_key_values=context.past_key_values,
			code=context.code,
			subword_mask=context.subword_mask,
			gen_asr_text=context.gen_asr_text,
			request_id=request_id,
			perception_cache=context.perception_cache,
			has_prompt=has_prompt,
		)

		# Persist updated cache & clean finished streams
		self.context_manager.update_context(stream_ids, result, self.num_chunks_per_inference)
		self.context_manager.reset_slots(stream_ids, eos_flags)
		
		# Explicitly clean up bufferer and state for finished streams
		for stream_id, eos_flag in zip(stream_ids, eos_flags):
			if eos_flag:
				print(f"🏁 Ending stream {stream_id} - cleaning up bufferer and context")
				self.bufferer.rm_bufferer(stream_id)
				self._abort_stream_request(stream_id)
				# Note: We keep the state in _state_pool until finalization to save audio
				# It will be cleaned up in close_session()
		
		# Log audio and attach text to state
		self.log_output(frames, result["decoded_audio_new"], ready_feats, result["predicted_text_strs"], result.get("asr_predicted_text_strs"))

	def generate_step(self, frames: List[Frame]):
		"""Main streaming API similar to *transcribe_step* in recognizers."""
		buffers, left_paddings = self.bufferer.update(frames)
		# For now, treat all buffered features as ready
		ready_feats = [True] * len(frames)

		with torch.no_grad(), torch.inference_mode():
			self.inner_generate_step(frames, buffers, left_paddings, ready_feats)
		
	# ------------------------------------------------------------------
	# Finalization helpers
	# ------------------------------------------------------------------
	# TODO - update from enhancement to s2s
	def _finalize_and_save_finished_streams(
		self,
		frames: List[Frame],
		audio_filepaths: List[str],
		saved_paths_by_stream: dict[int, str],
	) -> None:
		"""Finalize any streams that ended in this batch and save their audio."""
		for frame in frames:
			if frame.is_last:
				stream_id = frame.stream_id
				state = self.get_or_create_state(stream_id)

				# Flush remaining buffered samples and assemble waveform
				if hasattr(state, "finalize"):
					state.finalize()
				# Concatenate emitted chunks and squeeze (B=1,C=1) to mono waveform
				generated_audio = torch.cat(state.speech_frames, dim=-1)
				# Ensure 1D mono waveform and float32 dtype for soundfile
				if generated_audio.dim() == 3 and generated_audio.size(0) == 1 and generated_audio.size(1) == 1:
					generated_audio = generated_audio.squeeze(0).squeeze(0)
				elif generated_audio.dim() == 2 and generated_audio.size(0) == 1:
					generated_audio = generated_audio.squeeze(0)
				generated_audio = generated_audio.to(torch.float32)

				# Build output path mirroring input filename under output_dir
				in_path = audio_filepaths[stream_id]
				base = os.path.splitext(os.path.basename(in_path))[0]
				out_path = os.path.join(self.output_dir, f"{base}.wav")

				# Write audio to disk
				if generated_audio.numel() > 0:
					sf.write(out_path, generated_audio.detach().cpu().numpy(), self.output_sample_rate)

				# Also save a stereo file with input (ch0) and output (ch1)
				# Load input with librosa (handles mono conversion and resampling)
				input_np, _ = librosa.load(in_path, sr=self.output_sample_rate, mono=True)
				input_audio = torch.from_numpy(input_np).to(torch.float32)
				# Length-match to generated output
				gen_cpu = generated_audio.detach().cpu().to(input_audio.dtype)
				gen_len = int(gen_cpu.shape[-1])
				in_len = int(input_audio.shape[-1])
				if in_len < gen_len:
					pad = torch.zeros(gen_len - in_len, dtype=input_audio.dtype)
					input_audio = torch.cat([input_audio, pad], dim=-1)
				elif in_len > gen_len:
					input_audio = input_audio[:gen_len]
				# Assemble stereo [samples, channels] and save
				stereo = torch.stack([input_audio, gen_cpu], dim=0).transpose(0, 1)
				stereo_path = os.path.join(self.output_dir, f"{base}_input_output.wav")
				sf.write(stereo_path, stereo.detach().cpu().numpy(), self.output_sample_rate)

				# Also save accumulated text next to audio
				text_out = state.get_output_text() if hasattr(state, "get_output_text") else ""
				if isinstance(text_out, str):
					text_path = os.path.join(self.output_dir, f"{base}.txt")
					try:
						with open(text_path, "w", encoding="utf-8") as f:
							f.write(text_out)
					except Exception:
						pass

				# Also save accumulated ASR text
				asr_text_out = state.get_output_asr_text() if hasattr(state, "get_output_asr_text") else ""
				if isinstance(asr_text_out, str) and asr_text_out:
					asr_text_path = os.path.join(self.output_dir, f"{base}_asr.txt")
					try:
						with open(asr_text_path, "w", encoding="utf-8") as f:
							f.write(asr_text_out)
					except Exception:
						pass

				saved_paths_by_stream[stream_id] = out_path

				# Keep state until outputs are assembled; will be cleared on close_session


	# ------------------------------------------------------------------
	# Session helpers (extend BaseGenerator)
	# ------------------------------------------------------------------

	def reset_session(self) -> None:
		"""Reset feature buffer and ContextManager together."""
		for stream_id in list(self.context_manager.streamidx2slotidx.keys()):
			self._abort_stream_request(stream_id)
		self.bufferer.reset()
		self.context_manager.reset()

		super().reset_session() # clears state pool

	# ------------------------------------------------------------------
	# Orchestrator – mirrors recognizers' *run* method
	# ------------------------------------------------------------------
	def run(
		self,
		audio_filepaths: List[str],
		progress_bar: Optional[ProgressBar] = None,
	) -> PipelineOutput:
		"""Stream all *audio_filepaths* through the generator and save outputs.

		Saves one generated ``.wav`` per input under ``self.output_dir`` and
		returns their paths in ``PipelineOutput.texts``.
		"""
		if progress_bar and not isinstance(progress_bar, ProgressBar):
			raise ValueError("progress_bar must be an instance of ProgressBar.")
		
		streamer = ContinuousBatchedFrameStreamer(
			n_frames_per_stream=1,
			frame_size_in_secs=self.chunk_size_in_secs,
			sample_rate=self.input_sample_rate,
			batch_size=self.batch_size,
			pad_last_frame=True,
		)
		
		options = [None] * len(audio_filepaths)
		streamer.set_audio_filepaths(audio_filepaths, options)
		streamer.set_progress_bar(progress_bar)

		# Ensure output directory exists
		os.makedirs(self.output_dir, exist_ok=True)

		# Track saved paths by stream id to preserve input order
		saved_paths_by_stream: dict[int, str] = {}

		self.open_session()
		for frames in streamer:
			self.generate_step(frames)
			self._finalize_and_save_finished_streams(frames, audio_filepaths, saved_paths_by_stream)
		# Build outputs before closing the session
		texts = []
		words = []
		asr_texts = []
		for idx in range(len(audio_filepaths)):
			state = self.get_or_create_state(idx)
			text_value = state.get_output_text() if hasattr(state, "get_output_text") else ""
			if not text_value:
				text_value = saved_paths_by_stream.get(idx, "")
			texts.append(text_value)
			per_stream_words = state.get_output_words() if hasattr(state, "get_output_words") else []
			words.append(per_stream_words)
			asr_text_value = state.get_output_asr_text() if hasattr(state, "get_output_asr_text") else ""
			asr_texts.append(asr_text_value)

		self.close_session()

		return PipelineOutput(texts=texts, words=words, asr_texts=asr_texts)

	def _prefill_system_prompt(self, stream_id: int) -> Optional[torch.Tensor]:
		"""Prefill the system prompt for a new stream.
		
		This prepares the system prompt embeddings and processes them through
		the LLM to update the KV cache before audio streaming begins.
		Also prefills the TTS model with speaker embeddings when using vLLM EarTTS.
		
		Note on TTS prefill codes:
			The TTS prefill generates output codes, but these should NOT be used
			to initialize context.code for inference. The batch approach uses
			first_tts_code_input (INPUT codes from speaker reference) instead.
			Using prefill OUTPUT codes causes audio quality issues (mumbling).
		
		Returns:
			Optional[torch.Tensor]: The TTS prefill output codes if vLLM EarTTS prefill
			happened, None otherwise. These are returned for logging/debugging but
			should NOT be used to update context.code.
		"""
		request_id = self._request_id_for_stream(stream_id)
		engine_type = getattr(self.s2s_model, "engine_type", "native")
		tts_output_code = None
		
		# Prefill TTS with speaker embedding when using vLLM EarTTS
		# This initializes the vLLM TTS engine with the speaker context via prompt_token_ids
		use_vllm_eartts = "vllm_eartts" in engine_type.lower()
		if use_vllm_eartts:
			tts_init_inputs = getattr(self.s2s_model, "tts_init_inputs", None)
			tts_prompt_token_ids = getattr(self.s2s_model, "tts_prompt_token_ids", None)
			if tts_init_inputs is not None and tts_prompt_token_ids is not None:
				print(f"🔊 Prefilling TTS speaker embedding for stream {stream_id}...")
				start_tts_prefill = time.time()
				with torch.no_grad():
					# Clone tts_init_inputs to avoid any tensor sharing issues
					import copy
					tts_inputs_copy = copy.deepcopy(tts_init_inputs)
					tts_result = self.s2s_model.model.tts_model.tts_model(
						tts_inputs_copy,
						request_id=request_id,
						prompt_token_ids=tts_prompt_token_ids
					)
					# Capture the generated codes to sync context with vLLM state
					if hasattr(tts_result, 'codes') and tts_result.codes is not None:
						tts_output_code = tts_result.codes.detach().clone()
						print(f"   TTS prefill generated codes shape: {tts_output_code.shape}")
				print(f"   Time taken to prefill TTS speaker embedding: {time.time() - start_tts_prefill:.3f}s")
				print(f"   ✅ TTS speaker embedding prefilled")
			else:
				print(f"   ⚠️  TTS init inputs not available, skipping TTS prefill")
		
		if not self.system_prompt:
			return tts_output_code
		
		print(f"📝 Prefilling system prompt for stream {stream_id}...")
		start_get_prompt_embeddings = time.time()
		prompt_embedded, prompt_len = self.s2s_model._prepare_system_prompt_embeddings(self.system_prompt)
		print(f"   Time taken to get prompt embeddings: {time.time() - start_get_prompt_embeddings:.3f}s")
		
		if prompt_embedded is None:
			print(f"   ⚠️  System prompt embedding returned None, skipping prefill")
			return tts_output_code
		
		# Check if using vLLM for LLM (matches vllm_llm, vllm_llm_vllm_eartts, etc.)
		use_vllm_llm = "vllm_llm" in engine_type.lower()
		
		if use_vllm_llm:
			# For vLLM LLM: process prompt embeddings sequentially
			# vLLM manages its own KV cache internally
			print(f"   Processing {prompt_len} prompt embeddings for vLLM LLM...")
			start_prefill = time.time()
			with torch.no_grad():
				for t in range(prompt_len):
					frame_emb = prompt_embedded[:, t:t+1, :]
					_ = self.s2s_model.model_llm_interface(
						frame_emb,
						request_id=request_id,
						generated_tokens=None,
						current_step=t
					)
			print(f"   Time taken to prefill LLM: {time.time() - start_prefill:.3f}s")
			print(f"   ✅ System prompt processed ({prompt_len} tokens)")
		
		elif self.use_cache:
			# For native cache mode: process prompt through LLM to update cache
			context, _ = self.context_manager.get_context([stream_id])
			with torch.no_grad():
				llm_cache = context.dynamic_cache
				ans = self.s2s_model.model_llm_interface(
					prompt_embedded,
					cache=llm_cache,
					generated_tokens=None,
					current_step=0
				)
				# Update context's cache
				context.dynamic_cache = ans.get("cache", llm_cache)
			print(f"   ✅ System prompt processed, cache updated ({prompt_len} tokens)")
		
		else:
			# For no-cache mode (Nemotron): add prompt embeddings to history
			context, _ = self.context_manager.get_context([stream_id])
			for t in range(prompt_len):
				context.input_embeds_history.append(prompt_embedded[:, t:t+1, :])
			print(f"   ✅ Added {prompt_len} prompt embeddings to input_embeds_history")
		
		return tts_output_code

	def _request_id_for_stream(self, stream_id: int) -> str:
		return str(stream_id)

	def _abort_stream_request(self, stream_id: int) -> None:
		request_id = self._request_id_for_stream(stream_id)
		abort_fn = getattr(self.s2s_model, "abort_request", None)
		if callable(abort_fn):
			try:
				abort_fn(request_id)
			except Exception as exc:
				print(f"⚠️  Failed to abort request {request_id} for stream {stream_id}: {exc}")
