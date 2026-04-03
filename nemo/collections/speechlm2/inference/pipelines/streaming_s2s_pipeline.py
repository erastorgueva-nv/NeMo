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

import copy
import os
import time

import torch
import librosa
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
from nemo.collections.speechlm2.inference.model_wrappers.nemotron_voicechat_inference_wrapper import NemotronVoicechatInferenceWrapper
from nemo.collections.speechlm2.parts.text_utils import tokens_to_str
from nemo.collections.speechlm2.inference.streaming.state.s2s_context_manager import S2SContextManager
from nemo.collections.speechlm2.inference.streaming.framing.s2s_request_options import S2SRequestOptions
from nemo.collections.speechlm2.inference.utils.pipeline_utils import PipelineOutput
from nemo.utils import logging


class StreamingS2SPipeline(S2SPipelineInterface):
    """
    Streaming S2S pipeline.
    """

    def __init__(self, cfg: DictConfig, s2s_model: NemotronVoicechatInferenceWrapper):
        # ------------------------------------------------------------------
        # Model & device
        # ------------------------------------------------------------------
        self.s2s_model = s2s_model
        self.device = self.s2s_model.device
        self.decode_audio = self.s2s_model.decode_audio
        self.collect_debug = False

        # ------------------------------------------------------------------
        # Streaming configuration
        # ------------------------------------------------------------------
        self.streaming_cfg = cfg.get("streaming", {})
        self.input_sample_rate = getattr(self.streaming_cfg, "input_sample_rate", 16000)
        self.output_sample_rate = getattr(self.streaming_cfg, "output_sample_rate", 22050)
        self.batch_size = getattr(self.streaming_cfg, "batch_size", 1)
        self.max_len = getattr(self.streaming_cfg, "max_len", 200)
        if self.batch_size != 1:
            raise ValueError(
                "StreamingS2SPipeline currently supports only single-stream inference "
                "(streaming.batch_size must be 1)."
            )

        # ------------------------------------------------------------------
        # Chunk & buffer sizes
        # Terminology: "frame" = 80ms audio unit, "chunk" = 1 or more frames
        # A chunk is the amount of audio that is processed per inference step.
        # ------------------------------------------------------------------
        self.chunk_size_in_secs = getattr(self.streaming_cfg, "chunk_size_in_secs", 0.08)
        # Check if self.chunk_size_in_secs is a multiple of 0.08.
        # Because of quirks of floating point arithmetic, the remainder could be either ~0 or ~0.08,
        # so we check for both cases.
        remainder = self.chunk_size_in_secs % 0.08
        if not (math.isclose(remainder, 0, abs_tol=1e-9) or math.isclose(remainder, 0.08, abs_tol=1e-9)):
            raise ValueError(f"Chunk size must be a multiple of 0.08s, but got {self.chunk_size_in_secs}")

        self.num_frames_per_chunk = int(self.chunk_size_in_secs / 0.08)

        # Buffer size determines how much audio is passed to the perception encoder
        # Default: 5.68 seconds (71 * 0.08). This is the minimum valid buffer size without the perception cache.
        # i.e. att_context_size[0] + att_context_size[1] + 1 frames = 70+0+1 = 71 frames = 5.68 seconds
        self.buffer_size_in_secs = getattr(self.streaming_cfg, "buffer_size_in_secs", 71 * 0.08)

        self.att_context_size = getattr(self.streaming_cfg, "att_context_size", [70,0])

        # ------------------------------------------------------------------
        # bufferer – reused from ASR utilities
        # ------------------------------------------------------------------
        self.bufferer = BatchedAudioBufferer(
            sample_rate=self.input_sample_rate,
            buffer_size_in_secs=self.buffer_size_in_secs,
        )

        # ------------------------------------------------------------------
        # System prompt & sampling defaults (from YAML s2s block)
        # ------------------------------------------------------------------
        s2s_cfg = cfg.get("s2s", {})
        self.system_prompt: str | None = getattr(s2s_cfg, "system_prompt", None)
        if self.system_prompt:
            logging.info(f"System prompt configured: {self.system_prompt[:100]}{'...' if len(self.system_prompt) > 100 else ''}")

        self._default_top_p: float | None = getattr(s2s_cfg, "top_p", None)
        self._default_temperature: float | None = getattr(s2s_cfg, "temperature", None)
        self._default_repetition_penalty: float | None = getattr(s2s_cfg, "repetition_penalty", None)

        # Context manager
        self.context_manager = S2SContextManager(
            s2s_model=self.s2s_model,
            num_slots=self.batch_size,
            max_len=self.max_len,
        )

        # Output directory for generated files
        self.output_dir = getattr(cfg, "output_dir", "./generated")

        # Parse and validate request type early, with a safe default
        req_type_cfg = getattr(self.streaming_cfg, "request_type", "frame")

        # Parse and validate the request type; only 'frame' is supported for s2s.
        self.request_type = RequestType.from_str(req_type_cfg)
        if self.request_type is not RequestType.FRAME:
            raise ValueError(f"Request type {self.request_type} is not supported for s2s.")

        self._stream_has_prompt: bool = False

        # ------------------------------------------------------------------
        # Input audio padding (silence appended after real audio)
        # ------------------------------------------------------------------
        self.pad_audio_to_sec: float | None = cfg.get("pad_audio_to_sec", None)
        self.pad_silence_ratio: float | None = cfg.get("pad_silence_ratio", None)
        self.pad_audio_by_sec: float | None = cfg.get("pad_audio_by_sec", None)
        if sum(x is not None for x in [self.pad_audio_to_sec, self.pad_silence_ratio, self.pad_audio_by_sec]) > 1:
            raise ValueError("Set at most one of: pad_audio_to_sec, pad_silence_ratio, pad_audio_by_sec")

        super().__init__()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def create_state(self, options: S2SRequestOptions | None = None) -> S2SStreamingState:
        """Create new empty state with optional per-stream options."""
        dtype = getattr(self.s2s_model, "dtype", torch.float32)
        return S2SStreamingState(
            device=self.device,
            dtype=dtype,
            output_sample_rate=self.output_sample_rate,
            options=options or S2SRequestOptions(),
        )

    def _init_state(self, stream_id: int, options: S2SRequestOptions | None = None) -> None:
        """Initialize a new stream: resolve defaults, create state, create context, prefill.

        This is the S2S equivalent of ASR's ``init_state()`` in ``BasePipeline``.
        Called automatically by :meth:`generate_step` when a frame has
        ``is_first=True``.

        The method always runs stream initialization (state creation,
        context-manager allocation, KV-cache prefill).  If the triggering
        frame also carries audio, :meth:`generate_step` will process it
        immediately after this method returns.  For latency-sensitive
        deployments (real-time voice chat), callers should send the first
        frame with **empty audio** so that prefill completes before the
        user starts speaking — this prevents audio from queuing up during
        the expensive prefill phase.
        """
        # Normalize empty-string prompts to None so augment_with_defaults
        # fills in the YAML default instead of treating "" as "set".
        raw_opts = options or S2SRequestOptions()
        if raw_opts.system_prompt is not None and not raw_opts.system_prompt.strip():
            raw_opts = S2SRequestOptions(
                system_prompt=None,
                top_p=raw_opts.top_p,
                temperature=raw_opts.temperature,
                repetition_penalty=raw_opts.repetition_penalty,
            )
        opts = raw_opts.augment_with_defaults(
            default_system_prompt=self.system_prompt,
            default_top_p=self._default_top_p,
            default_temperature=self._default_temperature,
            default_repetition_penalty=self._default_repetition_penalty,
        )
        self.get_or_create_state(stream_id, options=opts)

        self.context_manager = S2SContextManager(
            s2s_model=self.s2s_model,
            num_slots=self.batch_size,
            max_len=self.max_len,
        )

        # Prefill can take hundreds of ms, or even tens of seconds if the
        # prompt is long and the model is not warmed up.
        prompt = opts.system_prompt
        start_prefill = time.time()
        with torch.no_grad(), torch.inference_mode():
            self._prefill_system_prompt(stream_id, prompt)
        torch.cuda.synchronize()
        logging.info(f"_init_state: stream_id={stream_id}, prefill={1000*(time.time()-start_prefill):.1f}ms")

        # Will tell generate_step_for_frames whether the KV cache already contains
        # a system prompt, so it can choose the right first-frame embedding
        # (PAD tokens if prefilled, BOS tokens if not).  Consumed and
        # cleared on the first audio frame.
        self._stream_has_prompt = bool(prompt)


    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def log_output(self, frames: list[Frame], audio_wave: Tensor, text_pieces: list[str], asr_text_pieces: list[str] = None):
        """Append generated audio waveform and text to per-stream state."""
        for idx, frame in enumerate(frames):
            state = self.get_state(frame.stream_id)
            # audio_wave is [B, S]; take sample idx (None when decode_audio=False)
            sample_audio = audio_wave[idx:idx+1, ...] if audio_wave is not None else None
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

            state.append_step_output(sample_audio, text=piece, asr_text=asr_piece)


    def generate_step_for_frames(self, frames: list[Frame], buffers: list[Tensor]):
        """Generate speech for audio Frames using a shared ContextManager.

        This is the S2S equivalent of ASR's ``transcribe_step_for_frames``
        in ``BasePipeline``.  Like its ASR counterpart, it is never called
        directly — :meth:`generate_step` (the public API, analogous to
        ``transcribe_step``) handles stream init and then delegates here
        for the actual audio processing.

        Stream initialization (state, context, prefill) is always handled
        by :meth:`_init_state` *before* this method is called.
        """
        if len(frames) == 0:
            return

        stream_ids = [f.stream_id for f in frames]
        eos_flags = [f.is_last for f in frames]

        logging.debug(f"stream_ids={stream_ids} eos_flags={eos_flags}")

        if len(frames) != 1:
            raise NotImplementedError("NemotronVoicechatInferenceWrapper currently supports batch_size == 1")

        has_prompt = self._stream_has_prompt
        self._stream_has_prompt = False

        request_id = self._request_id_for_stream(stream_ids[0])

        context, _ = self.context_manager.get_context(stream_ids)

        audio_buffer = buffers[0]
        if audio_buffer.dim() == 1:
            audio_buffer = audio_buffer.unsqueeze(0)
        audio_buffer = audio_buffer.to(self.s2s_model.device, dtype=self.s2s_model.dtype)

        # Sampling overrides were resolved by _init_state via augment_with_defaults
        # and stored on state.options.  Build the dict for infer_one_step.
        pipeline_state = self.get_state(stream_ids[0])
        if pipeline_state is None:
            raise RuntimeError(
                f"No state initialized for stream {stream_ids[0]}. "
                "Clients must send an is_first=True frame before streaming audio."
            )
        sampling_params = {
            k: getattr(pipeline_state.options, k)
            for k in ("top_p", "temperature", "repetition_penalty")
            if getattr(pipeline_state.options, k, None) is not None
        }

        result = self.s2s_model.infer_one_step(
            audio_input=audio_buffer,
            num_frames_per_chunk=self.num_frames_per_chunk,
            state=context,
            request_id=request_id,
            has_prompt=has_prompt,
            return_debug=self.collect_debug,
            sampling_params=sampling_params or None,
        )

        if self.collect_debug and result.debug is not None:
            state = self.get_or_create_state(stream_ids[0])
            if not hasattr(state, "debug_steps"):
                state.debug_steps = []
            state.debug_steps.append(result.debug)

        # Persist updated cache & clean finished streams
        self.context_manager.update_context(stream_ids, result, self.num_frames_per_chunk)

        # Save full token tensors to state before the context is destroyed,
        # so we can run tokens_to_str post-hoc.
        for stream_id, eos_flag in zip(stream_ids, eos_flags):
            if eos_flag:
                ctx = self.context_manager.slot_contexts[
                    self.context_manager.streamidx2slotidx[stream_id]
                ]
                if ctx is not None:
                    state = self.get_or_create_state(stream_id)
                    state.save_token_tensors(ctx.gen_text, ctx.gen_asr_text, ctx.frame_idx,
                                             gen_function_text=ctx.gen_function_text)

        self.context_manager.reset_slots(stream_ids, eos_flags)
        
        # Explicitly clean up bufferer and state for finished streams
        for stream_id, eos_flag in zip(stream_ids, eos_flags):
            if eos_flag:
                logging.debug(f"Ending stream {stream_id} - cleaning up bufferer and context")
                self.bufferer.rm_bufferer(stream_id)
                self._abort_stream_request(stream_id)
                # Note: We keep the state in _state_pool until finalization to save audio
                # It will be cleaned up in close_session()
        
        # Log audio and attach text to state
        self.log_output(frames, result.decoded_audio, result.predicted_text_strs, result.asr_predicted_text_strs)

    _WARMUP_FALLBACK_PROMPT = "Mock system prompt for warmup."

    def warmup(self, system_prompt: str | None = None) -> None:
        """Run a throwaway inference cycle to warm up the entire pipeline.

        The very first call through each stage incurs one-time overhead
        (e.g. CUDA graph compilation, memory pool allocation,
        DynamicCache initialization, torch.compile).  Sending a silence
        frame with ``is_first=True`` exercises the full path — prefill,
        perception, LLM decode, TTS, and codec — so the first real
        client request is fast.

        Args:
            system_prompt: Prompt text to use for warmup.  Falls back to
                the YAML-configured ``self.system_prompt``, then to a
                short fallback string so the LLM prefill path is always
                exercised.
        """
        prompt = system_prompt if system_prompt is not None else self.system_prompt
        if not prompt:
            prompt = self._WARMUP_FALLBACK_PROMPT
            logging.info(f"No system prompt configured — using fallback prompt for warmup: \"{prompt}\"")

        warmup_stream_id = -1
        chunk_samples = int(self.chunk_size_in_secs * self.input_sample_rate)

        logging.info("Running pipeline warmup (prefill + one silence chunk)...")
        t0 = time.time()

        warmup_frame = Frame(
            samples=torch.zeros(chunk_samples),
            stream_id=warmup_stream_id,
            is_first=True,
            is_last=True,
            options=S2SRequestOptions(system_prompt=prompt),
        )
        self.generate_step([warmup_frame])

        # Tear down everything so the engine is clean for real traffic
        self.reset_session()
        self._stream_has_prompt = False

        logging.info(f"Pipeline warmup complete in {time.time() - t0:.3f}s")

    def generate_step(self, frames: list[Frame]):
        """Main streaming API — handles both init and audio processing.

        Mirrors ASR's ``transcribe_step``: on ``is_first`` frames, the
        stream is initialized via :meth:`_init_state` (state creation,
        context-manager allocation, KV-cache prefill).  If the frame also
        carries audio, it is processed in the same call.  If there is no
        audio (e.g. a server prefill-only request), the method returns
        after init.

        For latency-sensitive deployments, send the ``is_first`` frame
        with **empty audio** so that the expensive prefill completes
        before the user starts speaking.  For batch/offline usage the
        first frame can carry real audio — init and first-chunk
        processing simply happen back-to-back in one call.
        """
        # Init phase — like ASR's `if request.is_first: self.init_state(...)`
        for frame in frames:
            if frame.is_first:
                self._init_state(frame.stream_id, frame.options)

        # Audio phase — skip if no audio (e.g. server prefill-only request)
        non_empty_frames = [f for f in frames if f.samples.numel() > 0]
        if not non_empty_frames:
            return

        buffers, left_paddings = self.bufferer.update(non_empty_frames)
        # This is a workaround for the fact that the audio buffer does left
        # padding, but the rest of the code requires no padding at all.
        buffers = [b[lp:] for b, lp in zip(buffers, left_paddings)]
        with torch.no_grad(), torch.inference_mode():
            self.generate_step_for_frames(non_empty_frames, buffers)
        
    # ------------------------------------------------------------------
    # Finalization helpers
    # ------------------------------------------------------------------
    def _finalize_and_save_finished_streams(
        self,
        frames: list[Frame],
        audio_filepaths: list[str],
        saved_paths_by_stream: dict[int, str],
    ) -> None:
        """Finalize any streams that ended in this batch and save their outputs."""
        for frame in frames:
            if frame.is_last:
                stream_id = frame.stream_id
                state = self.get_or_create_state(stream_id)

                if hasattr(state, "finalize"):
                    state.finalize()

                in_path = audio_filepaths[stream_id]
                base = os.path.splitext(os.path.basename(in_path))[0]
                txt_dir = os.path.join(self.output_dir, "txt")
                os.makedirs(txt_dir, exist_ok=True)

                out_path = None
                if self.decode_audio:
                    # Squeeze (B=1,C=1) to 1D mono waveform for soundfile
                    generated_audio = state.audio_buffer
                    if generated_audio.dim() == 3 and generated_audio.size(0) == 1 and generated_audio.size(1) == 1:
                        generated_audio = generated_audio.squeeze(0).squeeze(0)
                    elif generated_audio.dim() == 2 and generated_audio.size(0) == 1:
                        generated_audio = generated_audio.squeeze(0)
                    generated_audio = generated_audio.to(torch.float32)

                    wav_dir = os.path.join(self.output_dir, "wav")
                    stereo_dir = os.path.join(self.output_dir, "stereo")
                    os.makedirs(wav_dir, exist_ok=True)
                    os.makedirs(stereo_dir, exist_ok=True)

                    out_path = os.path.join(wav_dir, f"{base}.wav")
                    if generated_audio.numel() > 0:
                        sf.write(out_path, generated_audio.detach().cpu().numpy(), self.output_sample_rate)

                    # Save a stereo file with input (ch0) and output (ch1)
                    input_np, _ = librosa.load(in_path, sr=self.output_sample_rate, mono=True)
                    input_audio = torch.from_numpy(input_np).to(torch.float32)
                    gen_cpu = generated_audio.detach().cpu().to(input_audio.dtype)

                    # Prepend silence to output channel to account for the one-chunk
                    # processing delay: the server can't produce output until it has
                    # received a full input chunk.
                    delay_samples = int(self.chunk_size_in_secs * self.output_sample_rate)
                    silence = torch.zeros(delay_samples, dtype=gen_cpu.dtype)
                    gen_cpu = torch.cat([silence, gen_cpu], dim=-1)

                    # Pad the shorter channel so both have equal length
                    gen_len = int(gen_cpu.shape[-1])
                    in_len = int(input_audio.shape[-1])
                    max_len = max(gen_len, in_len)
                    if in_len < max_len:
                        input_audio = torch.cat([input_audio, torch.zeros(max_len - in_len, dtype=input_audio.dtype)], dim=-1)
                    if gen_len < max_len:
                        gen_cpu = torch.cat([gen_cpu, torch.zeros(max_len - gen_len, dtype=gen_cpu.dtype)], dim=-1)
                    stereo = torch.stack([input_audio, gen_cpu], dim=0).transpose(0, 1)
                    stereo_path = os.path.join(stereo_dir, f"{base}_input_output.wav")
                    sf.write(stereo_path, stereo.detach().cpu().numpy(), self.output_sample_rate)

                text_out = state.output_text_str
                if isinstance(text_out, str):
                    try:
                        with open(os.path.join(txt_dir, f"{base}.txt"), "w", encoding="utf-8") as f:
                            f.write(text_out)
                    except Exception:
                        pass

                asr_text_out = state.output_asr_text_str
                if isinstance(asr_text_out, str) and asr_text_out:
                    try:
                        with open(os.path.join(txt_dir, f"{base}_asr.txt"), "w", encoding="utf-8") as f:
                            f.write(asr_text_out)
                    except Exception:
                        pass

                saved_paths_by_stream[stream_id] = out_path
                # Keep state in _state_pool until _build_pipeline_output;
                # it will be cleared on close_session().


    # ------------------------------------------------------------------
    # Session helpers (extend S2SPipelineInterface)
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
        audio_filepaths: list[str],
        options: list[S2SRequestOptions] | None = None,
        progress_bar: ProgressBar | None = None,
    ) -> PipelineOutput:
        """Stream all *audio_filepaths* through the pipeline and save outputs.

        Saves one generated ``.wav`` per input under ``self.output_dir`` and
        returns their paths in ``PipelineOutput.texts``.
        """
        if progress_bar and not isinstance(progress_bar, ProgressBar):
            raise ValueError("progress_bar must be an instance of ProgressBar.")

        if options is None:
            options = [S2SRequestOptions() for _ in audio_filepaths]

        streamer = ContinuousBatchedFrameStreamer(
            n_frames_per_stream=1,
            frame_size_in_secs=self.chunk_size_in_secs,
            sample_rate=self.input_sample_rate,
            batch_size=self.batch_size,
            pad_last_frame=True,
        )
        streamer.set_audio_filepaths(audio_filepaths, options)
        streamer.set_progress_bar(progress_bar)

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Track saved paths by stream id to preserve input order
        saved_paths_by_stream: dict[int, str] = {}
        chunk_samples = int(self.chunk_size_in_secs * self.input_sample_rate)

        self.open_session()
        for frames in streamer:
            frames, pad_targets = self._apply_padding(frames, streamer)
            self.generate_step(frames)
            self._finalize_and_save_finished_streams(frames, audio_filepaths, saved_paths_by_stream)
            self._generate_silence_padding(pad_targets, chunk_samples, audio_filepaths, saved_paths_by_stream)

        output = self._build_pipeline_output(audio_filepaths, saved_paths_by_stream)
        self.close_session()
        return output

    # ------------------------------------------------------------------
    # run() helpers
    # ------------------------------------------------------------------

    def _apply_padding(
        self,
        frames: list[Frame],
        streamer: ContinuousBatchedFrameStreamer,
    ) -> tuple[list[Frame], dict[int, float]]:
        """If padding is configured, intercept last frames so the bufferer and
        context stay alive for the silence-padding phase.  Returns the
        (possibly modified) frames and a dict mapping ``stream_id`` to the
        remaining seconds of silence to append.
        """
        pad_targets: dict[int, float] = {}
        if not (self.pad_audio_to_sec or self.pad_silence_ratio or self.pad_audio_by_sec):
            return frames, pad_targets

        processed_frames = []
        for frame in frames:
            if frame.is_last:
                elapsed = streamer.elapsed_durations[frame.stream_id]
                remaining = self._padding_remaining_secs(elapsed)
                if remaining > 0:
                    processed_frames.append(Frame(
                        samples=frame.samples,
                        stream_id=frame.stream_id,
                        is_first=frame.is_first,
                        is_last=False,
                        length=frame.length,
                        options=frame.options,
                    ))
                    pad_targets[frame.stream_id] = remaining
                    continue
            processed_frames.append(frame)
        return processed_frames, pad_targets

    def _generate_silence_padding(
        self,
        pad_targets: dict[int, float],
        chunk_samples: int,
        audio_filepaths: list[str],
        saved_paths_by_stream: dict[int, str],
    ) -> None:
        """Generate silence-padding frames for streams that need them.

        Must run in the same iteration as the real last frame to avoid the next
        stream's setup destroying this stream's context.
        """
        for stream_id, remaining_secs in pad_targets.items():
            num_pad_frames = max(1, round(remaining_secs / self.chunk_size_in_secs))
            for i in range(num_pad_frames):
                is_last = (i == num_pad_frames - 1)
                silence_frame = Frame(
                    samples=torch.zeros(chunk_samples),
                    stream_id=stream_id,
                    is_first=False,
                    is_last=is_last,
                    length=chunk_samples,
                )
                self.generate_step([silence_frame])
                if is_last:
                    self._finalize_and_save_finished_streams(
                        [silence_frame], audio_filepaths, saved_paths_by_stream
                    )

    def _build_pipeline_output(
        self,
        audio_filepaths: list[str],
        saved_paths_by_stream: dict[int, str],
    ) -> PipelineOutput:
        """Assemble final ``PipelineOutput`` from accumulated per-stream state.

        Must be called *before* ``close_session()`` since it reads from the
        state pool.
        """
        texts = []
        words = []
        asr_texts = []
        texts_with_timestamps = []
        asr_texts_with_timestamps = []
        raw_texts = []
        raw_asr_texts = []
        token_texts = []
        token_asr_texts = []
        token_function_texts = []
        token_lengths = []
        audio_paths = []

        tokenizer = self.s2s_model.tokenizer
        pad_id = self.s2s_model.model.stt_model.text_pad_id

        for idx in range(len(audio_filepaths)):
            state = self.get_or_create_state(idx)
            text_value = state.output_text_str or saved_paths_by_stream.get(idx, "")
            texts.append(text_value)
            audio_paths.append(saved_paths_by_stream.get(idx))
            words.append(list(state.output_words))
            asr_texts.append(state.output_asr_text_str)

            token_data = state.get_token_tensors()
            if token_data is not None:
                gen_text, gen_asr_text, total_frames, gen_function_text = token_data
                token_texts.append(gen_text)
                token_asr_texts.append(gen_asr_text)
                token_function_texts.append(gen_function_text)
                token_lengths.append(total_frames)
                lengths = torch.tensor([total_frames], dtype=torch.long)
                texts_with_timestamps.append(
                    tokens_to_str(gen_text, lengths, tokenizer=tokenizer, pad_id=pad_id, eval_text_turn_taking=True)[0]
                )
                asr_texts_with_timestamps.append(
                    tokens_to_str(gen_asr_text, lengths, tokenizer=tokenizer, pad_id=pad_id, eval_text_turn_taking=True)[0]
                )
                raw_texts.append(
                    tokens_to_str(gen_text, lengths, tokenizer=tokenizer, pad_id=pad_id, keep_pad=True)[0]
                )
                raw_asr_texts.append(
                    tokens_to_str(gen_asr_text, lengths, tokenizer=tokenizer, pad_id=pad_id, keep_pad=True)[0]
                )
                if gen_function_text is not None:
                    fc_text = tokens_to_str(gen_function_text, lengths, tokenizer=tokenizer, pad_id=pad_id, eval_text_turn_taking=False)[0]
                    fc_text_raw = tokens_to_str(gen_function_text, lengths, tokenizer=tokenizer, pad_id=pad_id, keep_pad=True)[0]
                    logging.info(f"Function calling channel: {fc_text}")
            else:
                token_texts.append(None)
                token_asr_texts.append(None)
                token_function_texts.append(None)
                token_lengths.append(None)
                texts_with_timestamps.append("")
                asr_texts_with_timestamps.append("")
                raw_texts.append("")
                raw_asr_texts.append("")

        debug_data = []
        if self.collect_debug:
            for idx in range(len(audio_filepaths)):
                state = self.get_or_create_state(idx)
                debug_data.append(getattr(state, "debug_steps", []))

        return PipelineOutput(
            texts=texts,
            words=words,
            asr_texts=asr_texts,
            texts_with_timestamps=texts_with_timestamps,
            asr_texts_with_timestamps=asr_texts_with_timestamps,
            raw_texts=raw_texts,
            raw_asr_texts=raw_asr_texts,
            token_texts=token_texts,
            token_asr_texts=token_asr_texts,
            token_function_texts=token_function_texts,
            token_lengths=token_lengths,
            audio_filepaths=audio_paths,
            debug_data=debug_data if debug_data else None,
        )

    def _prefill_system_prompt(self, stream_id: int, system_prompt: str | None = None) -> torch.Tensor | None:
        """Prefill the system prompt for a new stream.
        
        This prepares the system prompt embeddings and processes them through
        the LLM to update the KV cache before audio streaming begins.
        Also prefills the TTS model with speaker embeddings when using vLLM EarTTS.
        
        Args:
            stream_id: The stream identifier.
            system_prompt: The system prompt text for this stream. If *None*,
                TTS prefill still runs (for vLLM EarTTS) but no LLM prompt
                is injected.

        Note on TTS prefill codes:
            The TTS prefill generates output codes, but these should NOT be used
            to initialize context.tts_code for inference. The batch approach uses
            first_tts_code_input (INPUT codes from speaker reference) instead.
            Using prefill OUTPUT codes causes audio quality issues (mumbling).
        
        Returns:
            torch.Tensor | None: The TTS prefill output codes if vLLM EarTTS prefill
            happened, None otherwise. These are returned for logging/debugging but
            should NOT be used to update context.tts_code.
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
                logging.info(f"Prefilling TTS speaker embedding for stream {stream_id}...")
                start_tts_prefill = time.time()
                with torch.no_grad():
                    tts_inputs_copy = copy.deepcopy(tts_init_inputs)
                    tts_result = self.s2s_model.model.tts_model.tts_model(
                        tts_inputs_copy,
                        request_id=request_id,
                        prompt_token_ids=tts_prompt_token_ids
                    )
                    # Capture the generated codes to sync context with vLLM state
                    if hasattr(tts_result, 'codes') and tts_result.codes is not None:
                        tts_output_code = tts_result.codes.detach().clone()
                        logging.debug(f"TTS prefill generated codes shape: {tts_output_code.shape}")
                logging.info(f"TTS speaker embedding prefilled in {time.time() - start_tts_prefill:.3f}s")
            else:
                logging.warning("TTS init inputs not available, skipping TTS prefill")
        
        if not system_prompt:
            return tts_output_code
        
        logging.info(f"Prefilling system prompt for stream {stream_id}...")
        start_get_prompt_embeddings = time.time()
        prompt_embedded, prompt_len = self.s2s_model._prepare_system_prompt_embeddings(system_prompt)
        logging.debug(f"Time taken to get prompt embeddings: {time.time() - start_get_prompt_embeddings:.3f}s")
        
        if prompt_embedded is None:
            logging.warning("System prompt embedding returned None, skipping prefill")
            return tts_output_code
        
        # Check if using vLLM for LLM (matches vllm_llm, vllm_llm_vllm_eartts, etc.)
        use_vllm_llm = "vllm_llm" in engine_type.lower()
        
        if use_vllm_llm:
            # For vLLM LLM: prefill all prompt embeddings in one shot
            # (decode_steps=0 triggers a single bulk prefill in the vLLM engine)
            logging.info(f"Prefilling {prompt_len} prompt embeddings for vLLM LLM...")
            start_prefill = time.time()
            with torch.no_grad():
                _ = self.s2s_model.model_llm_interface(
                    prompt_embedded,
                    request_id=request_id,
                    decode_steps=0,
                    prompt_token_ids=None,
                )
            logging.info(f"System prompt prefilled ({prompt_len} tokens) in {time.time() - start_prefill:.3f}s")
        
        else:
            context, _ = self.context_manager.get_context([stream_id])
            if context.llm_cache is not None:
                # Native cache mode: process prompt through LLM to update KV cache
                with torch.no_grad():
                    cache_pos = torch.arange(prompt_len, device=self.s2s_model.device)
                    llm_cache = context.llm_cache
                    ans = self.s2s_model.model_llm_interface(
                        prompt_embedded,
                        cache=llm_cache,
                        cache_position=cache_pos,
                        generated_tokens=None,
                        current_step=0
                    )
                    context.llm_cache = ans.get("cache", llm_cache)
                context.llm_cache_position_offset = prompt_len
                logging.info(f"System prompt processed, cache updated ({prompt_len} tokens, offset={prompt_len})")
            else:
                for t in range(prompt_len):
                    context.input_embeds_history.append(prompt_embedded[:, t:t+1, :])
                logging.info(f"Added {prompt_len} prompt embeddings to input_embeds_history")
        
        return tts_output_code

    def _padding_remaining_secs(self, elapsed_secs: float) -> float:
        """Return how many seconds of silence padding are still needed."""
        if self.pad_audio_to_sec is not None:
            return max(0.0, self.pad_audio_to_sec - elapsed_secs)
        if self.pad_silence_ratio is not None:
            return elapsed_secs * self.pad_silence_ratio
        if self.pad_audio_by_sec is not None:
            return self.pad_audio_by_sec
        return 0.0

    def _request_id_for_stream(self, stream_id: int) -> str:
        return str(stream_id)

    def _abort_stream_request(self, stream_id: int) -> None:
        request_id = self._request_id_for_stream(stream_id)
        abort_fn = getattr(self.s2s_model, "abort_request", None)
        if callable(abort_fn):
            try:
                abort_fn(request_id)
            except Exception as exc:
                logging.warning(f"Failed to abort request {request_id} for stream {stream_id}: {exc}")
