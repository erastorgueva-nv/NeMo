#!/bin/bash
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

# Start Triton Inference Server for S2S voicechat model.
#
# Shares the same s2s_streaming.yaml config used by s2s_streaming_infer.py.
# Fields marked ??? in the YAML are resolved from environment variables below.
#
# Usage:
#   S2S_MODEL_PATH=/path/to/hf_checkpoint \
#   S2S_SPEAKER_NAME=MySpeaker \
#   ./start_triton.sh
#
# Environment variables (required):
#   S2S_MODEL_PATH              - Path to the HF-format checkpoint directory
#
# Environment variables (speaker identity — set at least one):
#   S2S_SPEAKER_REFERENCE       - Path to a speaker reference .wav file
#   S2S_SPEAKER_NAME            - Registered speaker name from the checkpoint
#
# Environment variables (optional):
#   S2S_ENGINE_TYPE             - Engine type (default: native)
#   S2S_DETERMINISTIC           - "true"/"false": deterministic mode (default: false)
#   S2S_USE_LLM_CACHE           - "true"/"false": LLM KV cache (default: true)
#   S2S_USE_TTS_SUBWORD_CACHE   - "true"/"false": TTS subword cache (default: false)
#   S2S_SYSTEM_PROMPT           - LLM system prompt text (default: none)
#   S2S_TTS_SYSTEM_PROMPT       - TTS system prompt (default: none)
#   S2S_CHUNK_SIZE_IN_SECS      - Chunk size in seconds, multiple of 0.08 (default: 0.08)
#   S2S_BUFFER_SIZE_IN_SECS     - Audio buffer size in seconds (default: 5.6)
#   S2S_TRITON_CONFIG_PATH      - Override the YAML config file path
#   MODEL_REPO_DIR              - Override the Triton model repository path

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# All variables below are exported so they are visible to the Triton Python
# backend (infer_streaming.py reads them via os.environ).

# ========================
# Model path (required)
# ========================
export S2S_MODEL_PATH="${S2S_MODEL_PATH:?Please set S2S_MODEL_PATH to the HF-format checkpoint directory}"

# ========================
# Speaker identity (set at least one)
# ========================
export S2S_SPEAKER_REFERENCE="${S2S_SPEAKER_REFERENCE:-}"
export S2S_SPEAKER_NAME="${S2S_SPEAKER_NAME:-}"
if [ -z "${S2S_SPEAKER_REFERENCE}" ] && [ -z "${S2S_SPEAKER_NAME}" ]; then
    echo "ERROR: Set at least one of S2S_SPEAKER_REFERENCE or S2S_SPEAKER_NAME"
    exit 1
fi

# ========================
# Optional overrides
# ========================
export S2S_ENGINE_TYPE="${S2S_ENGINE_TYPE:-native}"
export S2S_DETERMINISTIC="${S2S_DETERMINISTIC:-}"
export S2S_USE_LLM_CACHE="${S2S_USE_LLM_CACHE:-}"
export S2S_USE_TTS_SUBWORD_CACHE="${S2S_USE_TTS_SUBWORD_CACHE:-}"
export S2S_SYSTEM_PROMPT="${S2S_SYSTEM_PROMPT:-}"
export S2S_TTS_SYSTEM_PROMPT="${S2S_TTS_SYSTEM_PROMPT:-}"
export S2S_CHUNK_SIZE_IN_SECS="${S2S_CHUNK_SIZE_IN_SECS:-0.08}"
export S2S_BUFFER_SIZE_IN_SECS="${S2S_BUFFER_SIZE_IN_SECS:-5.6}"
export S2S_TRITON_CONFIG_PATH="${S2S_TRITON_CONFIG_PATH:-${SCRIPT_DIR}/../conf/s2s_streaming.yaml}"
export MODEL_REPO_DIR="${MODEL_REPO_DIR:-${SCRIPT_DIR}/model_repo_s2s}"


echo "=== S2S Triton Server ==="
echo "  S2S_MODEL_PATH:          ${S2S_MODEL_PATH}"
echo "  S2S_SPEAKER_REFERENCE:   ${S2S_SPEAKER_REFERENCE:-<not set>}"
echo "  S2S_SPEAKER_NAME:        ${S2S_SPEAKER_NAME:-<not set>}"
echo "  S2S_ENGINE_TYPE:         ${S2S_ENGINE_TYPE}"
echo "  S2S_DETERMINISTIC:       ${S2S_DETERMINISTIC:-<default>}"
echo "  S2S_USE_LLM_CACHE:       ${S2S_USE_LLM_CACHE:-<default>}"
echo "  S2S_USE_TTS_SUBWORD_CACHE: ${S2S_USE_TTS_SUBWORD_CACHE:-<default>}"
echo "  S2S_CHUNK_SIZE_IN_SECS:  ${S2S_CHUNK_SIZE_IN_SECS}"
echo "  S2S_BUFFER_SIZE_IN_SECS: ${S2S_BUFFER_SIZE_IN_SECS}"
echo "  S2S_SYSTEM_PROMPT:       ${S2S_SYSTEM_PROMPT:-<not set>}"
echo "  S2S_TTS_SYSTEM_PROMPT:   ${S2S_TTS_SYSTEM_PROMPT:-<not set>}"
echo "  MODEL_REPO_DIR:          ${MODEL_REPO_DIR}"
echo "  S2S_TRITON_CONFIG_PATH:  ${S2S_TRITON_CONFIG_PATH}"
echo "========================="

TRITON_BIN="${TRITON_BIN:-/opt/tritonserver/bin/tritonserver}"
if [ ! -x "${TRITON_BIN}" ]; then
    echo "ERROR: Triton server not found at ${TRITON_BIN}"
    echo "       Are you running inside a Triton container?"
    exit 1
fi

"${TRITON_BIN}" --model-repository="${MODEL_REPO_DIR}"
