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

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NEMO_DIR=/home/vklimkov/NemoDuplexRealtimeInference
INPUT_AUDIO_PATH="/home/vklimkov/moshi_client_nemo_20251117_151731_input_sf.wav"
CKPT_PATH="/home/vklimkov/duplex-eartts-2mim_sw_et_eos_dp_eos_dup_fp32_1delay_ind_prompts_nfisher_h2_david_16k_steps_main_branch-stt-AR3_12556_new_branch_load_fixed"
CACHE_DIR="/tmp/cache"

# ──────────────────────────────────────────────────────────────────────────────
# Environment preparation — these exports apply to this script and all child
# processes (checkpoint conversion, inference, etc.)
# ──────────────────────────────────────────────────────────────────────────────
export WANDB_API_KEY="${WANDB}"
export WANDB_MODE=offline
export AIS_ENDPOINT="http://asr.iad.oci.aistore.nvidia.com:51080"
export PYTHONPATH="${NEMO_DIR}:${PYTHONPATH}"
export HF_HOME="${CACHE_DIR}"
export TORCH_HOME="${CACHE_DIR}"
export NEMO_CACHE_DIR="${CACHE_DIR}"
export NEMO_NLP_TMP="${CACHE_DIR}/nemo_nlp_tmp"
export TRITON_CACHE_DIR="${CACHE_DIR}"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE=0.3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export TORCH_CUDNN_V8_API_ENABLED=1
umask 000

# ──────────────────────────────────────────────────────────────────────────────
# Convert checkpoint (skip if already done)
# ──────────────────────────────────────────────────────────────────────────────
ENGINES_DIR="${SCRIPT_DIR}/engines/"
if [ ! -d "${ENGINES_DIR}/eartts" ]; then
    python ${SCRIPT_DIR}/../scripts/convert_eartts_checkpoint.py \
        --config ${CKPT_PATH}/config.json \
        --model ${CKPT_PATH}/model.safetensors \
        --outdir ${ENGINES_DIR}/eartts
fi
if [ ! -d "${ENGINES_DIR}/llm" ]; then
    python ${SCRIPT_DIR}/../scripts/convert_nemotronllm_checkpoint.py \
        --config ${CKPT_PATH}/config.json \
        --checkpoint ${CKPT_PATH}/model.safetensors \
        --output-dir ${ENGINES_DIR}/llm \
        --dtype bfloat16
fi
if [ ! -d "${ENGINES_DIR}/fastconformer" ]; then
    python ${SCRIPT_DIR}/../scripts/convert_fastconformer_checkpoint.py \
        --s2s-checkpoint ${CKPT_PATH} \
        --outdir ${ENGINES_DIR}/fastconformer
fi

# ──────────────────────────────────────────────────────────────────────────────
# Run inference
# ──────────────────────────────────────────────────────────────────────────────
python3 ${SCRIPT_DIR}/infer.py \
    --audio ${INPUT_AUDIO_PATH} \
    --tts-prompt-data ${SCRIPT_DIR}/eartts_prefill_data.pt \
    --llm-prompt-data ${SCRIPT_DIR}/llm_prefill_data.pt \
    --output ${SCRIPT_DIR}/generated.wav \
    --fastconformer ${ENGINES_DIR}/fastconformer \
    --llm ${ENGINES_DIR}/llm \
    --eartts ${ENGINES_DIR}/eartts \
    --s2s-ckpt ${CKPT_PATH} \
    --gpu-mems 0.25 0.45 0.15 \
    --max-model-len 280
