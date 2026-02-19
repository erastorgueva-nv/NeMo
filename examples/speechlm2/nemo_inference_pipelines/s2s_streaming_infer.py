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

"""
S2S Streaming Inference Client

Usage:
    python s2s_streaming_infer.py \
        audio_file=/path/to/audio_or_directory \
        s2s.model_path=/path/to/eartts_ckpt \
        s2s.llm_checkpoint_path=/path/to/llm_ckpt \
        s2s.speaker_reference=/path/to/speaker.wav \
        streaming.chunk_size_in_secs=0.08 \
        streaming.buffer_size_in_secs=5.6
"""

import json
import os
import re
from time import time
from typing import List, Optional

import hydra
import soundfile as sf
from jiwer import wer as compute_wer
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.collections.speechlm2.inference.factory.s2s_pipeline_builder import S2SPipelineBuilder
from nemo.collections.speechlm2.inference.streaming.framing.s2s_request_options import S2SRequestOptions
from nemo.collections.speechlm2.inference.utils.pipeline_utils import PipelineOutput
from nemo.utils import logging
from omegaconf import DictConfig
import torch


def prepare_audio_data(
    audio_file: str,
    default_system_prompt: str | None = None,
    sort_by_duration: bool = True,
) -> tuple[List[str], List[S2SRequestOptions], List[str | None]]:
    """
    Get audio filepaths and per-stream options from a folder, single file, or manifest.

    When the input is a JSON manifest, each line may contain:
        {"audio_filepath": "clip.wav", "text": "...", "system_prompt": "..."}
    If ``system_prompt`` is absent on a line, *default_system_prompt* is used.

    Returns:
        (filepaths, options, ground_truths) -- parallel lists of audio paths,
        per-stream options, and ground-truth texts (None when unavailable).
    """
    audio_file = audio_file.strip()
    if not os.path.isabs(audio_file):
        audio_file = os.path.abspath(audio_file)

    options: List[S2SRequestOptions] = []
    ground_truths: List[str | None] = []

    if os.path.isdir(audio_file):
        filepaths = [os.path.join(audio_file, x) for x in os.listdir(audio_file) if x.endswith(".wav")]
        options = [S2SRequestOptions(system_prompt=default_system_prompt) for _ in filepaths]
        ground_truths = [None] * len(filepaths)
    elif audio_file.endswith(".wav"):
        filepaths = [audio_file]
        options = [S2SRequestOptions(system_prompt=default_system_prompt)]
        ground_truths = [None]
    elif audio_file.endswith((".json", ".jsonl")):
        samples = []
        with open(audio_file, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    samples.append(json.loads(line))
        filepaths = [get_full_path(entry["audio_filepath"], audio_file) for entry in samples]
        options = [
            S2SRequestOptions(
                system_prompt=entry.get("system_prompt", default_system_prompt),
            )
            for entry in samples
        ]
        ground_truths = [entry.get("text", None) for entry in samples]
    else:
        raise ValueError(f"audio_file `{audio_file}` needs to be a folder, audio file, or manifest file")

    if sort_by_duration:
        durations = [sf.SoundFile(fp).frames for fp in filepaths]
        order = sorted(range(len(filepaths)), key=lambda i: durations[i])
        filepaths = [filepaths[i] for i in order]
        options = [options[i] for i in order]
        ground_truths = [ground_truths[i] for i in order]

    return filepaths, options, ground_truths


def calculate_duration(audio_filepaths: List[str]) -> float:
    """Calculate the total duration of the audio files in seconds."""
    total_dur = 0
    for audio_filepath in audio_filepaths:
        sound = sf.SoundFile(audio_filepath)
        total_dur += sound.frames / sound.samplerate
    return total_dur


def calculate_padded_duration(
    audio_filepaths: List[str],
    pad_to_duration_secs: float | None = None,
    pad_silence_ratio: float | None = None,
) -> float:
    """Calculate total duration including padding for RTFX reporting."""
    total = 0.0
    for fp in audio_filepaths:
        sound = sf.SoundFile(fp)
        orig = sound.frames / sound.samplerate
        if pad_to_duration_secs is not None:
            total += max(orig, pad_to_duration_secs)
        elif pad_silence_ratio is not None:
            total += orig * (1 + pad_silence_ratio)
        else:
            total += orig
    return total


def clean_pred_text(text: str) -> str:
    """Clean prediction text by removing special markers, timestamps, punctuation, and lowercasing.

    Mirrors the normalization in nemotron_voicechat_inference_wrapper.py for
    fair WER comparison.
    """
    if not text:
        return ""
    text = text.lstrip('^')
    text = re.sub(r'<\$[\d.]+\$>', '', text)
    text = re.sub(r'<\|[\d.]+\|>', '', text)
    text = re.sub(r'<SPECIAL_12>', '', text)
    text = text.replace('\u0120', ' ')
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())


def dump_output(
    audio_filepaths: List[str],
    output: PipelineOutput,
    output_filename: str,
    options: List[S2SRequestOptions],
    output_ctm_dir: Optional[str] = None,
) -> None:
    """
    Dump the transcriptions to an output file.
    Args:
        audio_filepaths: List of audio file paths
        output: Pipeline output
        output_filename: Path to the output file
        options: Per-stream request options (carries the system prompt)
        output_ctm_dir: Path to the output CTM directory
    """
    if output_ctm_dir is None:
        output_ctm_dir = os.path.join(os.path.dirname(output_filename), "ctm")

    os.makedirs(output_ctm_dir, exist_ok=True)

    asr_texts = output.asr_texts if output.asr_texts is not None else [None] * len(audio_filepaths)

    with open(output_filename, 'w') as fout:
        for audio_filepath, text, words, asr_text, opts in zip(
            audio_filepaths, output.texts, output.words, asr_texts, options,
        ):
            stem = os.path.splitext(os.path.basename(audio_filepath))[0]
            ctm_filepath = os.path.abspath(os.path.join(output_ctm_dir, f"{stem}.ctm"))
            with open(ctm_filepath, 'w') as ctm_fout:
                for word in words:
                    ctm_line = f"A {round(word.start, 2)} {round(word.duration, 2)} {word.text} {word.conf}"
                    ctm_fout.write(f"{stem} {ctm_line}\n")

            item = {
                "audio_filepath": audio_filepath,
                "text": text,
                "ctm_filepath": ctm_filepath,
            }
            if asr_text is not None:
                item["asr_text"] = asr_text
            if opts.system_prompt is not None:
                item["system_prompt"] = opts.system_prompt
            json.dump(item, fout, ensure_ascii=False)
            fout.write('\n')
            fout.flush()


@hydra.main(config_path="./conf", config_name="s2s_streaming", version_base=None)
def main(cfg: DictConfig):
    default_system_prompt = cfg.get("s2s", {}).get("system_prompt", None)
    audio_filepaths, options, ground_truths = prepare_audio_data(
        cfg.audio_file, default_system_prompt=default_system_prompt,
    )
    logging.info(f"Found {len(audio_filepaths)} audio files to generate")

    # Set matmul precision
    matmul_precision = cfg.get("matmul_precision", "high")
    torch.set_float32_matmul_precision(matmul_precision)
    logging.info(f"Using matmul precision: {matmul_precision}")

    pipeline = S2SPipelineBuilder.build_pipeline(cfg)

    start = time()
    output = pipeline.run(audio_filepaths, options=options)
    exec_dur = time() - start
    logging.info(f"Generated {len(audio_filepaths)} files in {exec_dur:.2f}s")

    # Log RTFX (accounts for padding when configured)
    pad_to = cfg.get("pad_to_duration_secs", None)
    pad_ratio = cfg.get("pad_silence_ratio", None)
    if pad_to or pad_ratio:
        data_dur = calculate_padded_duration(audio_filepaths, pad_to, pad_ratio)
    else:
        data_dur = calculate_duration(audio_filepaths)
    rtfx = data_dur / exec_dur if exec_dur > 0 else float('inf')
    logging.info(f"RTFX: {rtfx:.2f} ({data_dur:.2f}s / {exec_dur:.2f}s)")

    # Compute WER when ground-truth texts are available
    asr_texts = output.asr_texts or [None] * len(audio_filepaths)
    wer_scores = []
    for gt, asr_text in zip(ground_truths, asr_texts):
        if gt and asr_text:
            cleaned_gt = clean_pred_text(gt)
            cleaned_pred = clean_pred_text(asr_text)
            if cleaned_gt.strip() and cleaned_pred.strip():
                wer_scores.append(compute_wer(cleaned_gt, cleaned_pred))
    if wer_scores:
        avg_wer = sum(wer_scores) / len(wer_scores)
        logging.info(
            f"WER: avg={avg_wer:.4f} ({avg_wer * 100:.2f}%), "
            f"n={len(wer_scores)}, "
            f"min={min(wer_scores):.4f}, max={max(wer_scores):.4f}"
        )

    # Dump the transcriptions and CTMs
    dump_output(audio_filepaths, output, cfg.output_filename, options, cfg.output_ctm_dir)
    logging.info(f"Transcriptions written to {cfg.output_filename}")


if __name__ == "__main__":
    main()
