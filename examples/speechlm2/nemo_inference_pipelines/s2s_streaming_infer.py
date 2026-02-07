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
from time import time
from typing import List, Optional

import hydra
import soundfile as sf
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.collections.speechlm2.inference.factory.s2s_pipeline_builder import S2SPipelineBuilder
from nemo.collections.speechlm2.inference.utils.pipeline_utils import PipelineOutput
from nemo.utils import logging
from omegaconf import DictConfig
import torch


def get_audio_filepaths(audio_file: str, sort_by_duration: bool = True) -> List[str]:
    """
    Get audio filepaths from a folder, a single audio file, or a manifest file.
    Args:
        audio_file: Path to the audio file, folder, or manifest file
        sort_by_duration: If True, sort the audio files by duration from shortest to longest
    Returns:
        List of audio filepaths
    """
    audio_file = audio_file.strip()
    if not os.path.isabs(audio_file):
        audio_file = os.path.abspath(audio_file)
    if os.path.isdir(audio_file):
        filepaths = [os.path.join(audio_file, x) for x in os.listdir(audio_file) if x.endswith(".wav")]
    elif audio_file.endswith(".wav"):
        filepaths = [audio_file]
    elif audio_file.endswith(".json"):
        samples = []
        with open(audio_file, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    samples.append(json.loads(line))
        filepaths = [get_full_path(entry["audio_filepath"], audio_file) for entry in samples]
    else:
        raise ValueError(f"audio_file `{audio_file}` needs to be a folder, audio file, or manifest file")

    if sort_by_duration:
        durations = [sf.SoundFile(fp).frames for fp in filepaths]
        filepaths = [fp for fp, _ in sorted(zip(filepaths, durations), key=lambda x: x[1])]
    return filepaths


def calculate_duration(audio_filepaths: List[str]) -> float:
    """Calculate the total duration of the audio files in seconds."""
    total_dur = 0
    for audio_filepath in audio_filepaths:
        sound = sf.SoundFile(audio_filepath)
        total_dur += sound.frames / sound.samplerate
    return total_dur


def dump_output(
    audio_filepaths: List[str],
    output: PipelineOutput,
    output_filename: str,
    output_ctm_dir: Optional[str] = None,
) -> None:
    """
    Dump the transcriptions to an output file.
    Args:
        audio_filepaths: List of audio file paths
        output: Pipeline output
        output_filename: Path to the output file
        output_ctm_dir: Path to the output CTM directory
    """
    if output_ctm_dir is None:
        output_ctm_dir = os.path.join(os.path.dirname(output_filename), "ctm")

    os.makedirs(output_ctm_dir, exist_ok=True)

    asr_texts = output.asr_texts if output.asr_texts is not None else [None] * len(audio_filepaths)

    with open(output_filename, 'w') as fout:
        for audio_filepath, text, words, asr_text in zip(audio_filepaths, output.texts, output.words, asr_texts):
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
            json.dump(item, fout, ensure_ascii=False)
            fout.write('\n')
            fout.flush()


@hydra.main(config_path="./conf", config_name="s2s_streaming", version_base=None)
def main(cfg: DictConfig):
    audio_filepaths = get_audio_filepaths(cfg.audio_file)
    logging.info(f"Found {len(audio_filepaths)} audio files to generate")

    # Set matmul precision
    matmul_precision = cfg.get("matmul_precision", "high")
    torch.set_float32_matmul_precision(matmul_precision)
    logging.info(f"Using matmul precision: {matmul_precision}")

    generator = S2SPipelineBuilder.build_pipeline(cfg)

    start = time()
    output = generator.run(audio_filepaths)
    exec_dur = time() - start
    logging.info(f"Generated {len(audio_filepaths)} files in {exec_dur:.2f}s")

    # Log RTFX
    data_dur = calculate_duration(audio_filepaths)
    rtfx = data_dur / exec_dur if exec_dur > 0 else float('inf')
    logging.info(f"RTFX: {rtfx:.2f} ({data_dur:.2f}s / {exec_dur:.2f}s)")

    # Dump the transcriptions and CTMs
    dump_output(audio_filepaths, output, cfg.output_filename, cfg.output_ctm_dir)
    logging.info(f"Transcriptions written to {cfg.output_filename}")


if __name__ == "__main__":
    main()
