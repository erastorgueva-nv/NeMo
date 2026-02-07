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

from omegaconf.dictconfig import DictConfig

from nemo.utils import logging as logger
from nemo.collections.speechlm2.inference.pipelines.streaming_s2s_pipeline import StreamingS2SGenerator
from nemo.collections.speechlm2.inference.model_wrappers.realtime_streaming_wrapper import RealtimeStreamingInference


class S2SPipelineBuilder:
    """Factory that builds a streaming S2S pipeline."""

    @classmethod
    def build_pipeline(
            cls,
            cfg: DictConfig
    ) -> RealtimeStreamingInference:
        """
        Build the streaming S2S pipeline based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns StreamingS2SGenerator object
        """
        # building enhancement model
        gen_model = RealtimeStreamingInference(model_cfg=cfg.s2s)

        logger.info(f"Duplex model `{cfg.s2s.model_path}` loaded")

        generator = StreamingS2SGenerator(
            cfg,
            gen_model,
        )
        logger.info(f"`{type(generator).__name__}` generator loaded")
        return generator

