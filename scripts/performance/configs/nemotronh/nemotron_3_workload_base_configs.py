# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Parallelism presets for Nemotron 3 Nano performance configs.

Config naming convention:
    {MODEL}_{SIZE}_{TASK}_CONFIG_{GPU}_{PRECISION}_{VERSION}

V1: 30B_a3b

Use --config_variant to select a variant.
Use --list_config_variants to see available variants interactively.
"""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_NEMOTRON_3_NANO_CONFIG = WorkloadBaseConfig(
    num_gpus=8,
    global_batch_size=512,
    tensor_model_parallel_size=1,
    expert_tensor_parallel_size=1,
    expert_model_parallel_size=8,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "mamba", "moe_router", "moe_preprocess"],
)

NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_BF16_V1 = replace(
    BASE_NEMOTRON_3_NANO_CONFIG,
    micro_batch_size=4,
)
NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_FP8_MX_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_BF16_V1
NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_NVFP4_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_BF16_V1

NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_BF16_V1 = replace(
    BASE_NEMOTRON_3_NANO_CONFIG,
    micro_batch_size=2,
)
NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_FP8_MX_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_BF16_V1
NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_NVFP4_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_BF16_V1

NEMOTRON_3_NANO_PRETRAIN_CONFIG_VR200_BF16_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_BF16_V1
NEMOTRON_3_NANO_PRETRAIN_CONFIG_VR200_FP8_MX_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_FP8_MX_V1
NEMOTRON_3_NANO_PRETRAIN_CONFIG_VR200_NVFP4_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_NVFP4_V1

NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_BF16_V1 = replace(
    BASE_NEMOTRON_3_NANO_CONFIG,
    micro_batch_size=4,
)
NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_FP8_MX_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_BF16_V1
NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_NVFP4_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_BF16_V1

NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_BF16_V1 = replace(
    BASE_NEMOTRON_3_NANO_CONFIG,
    micro_batch_size=2,
)
NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_FP8_MX_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_BF16_V1
NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_NVFP4_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_BF16_V1

_NEMOTRON_3_NANO_PRETRAIN_CONFIG_H100 = replace(
    BASE_NEMOTRON_3_NANO_CONFIG,
    num_gpus=16,
    global_batch_size=1024,
    micro_batch_size=1,
    cuda_graph_impl="transformer_engine",
)

NEMOTRON_3_NANO_PRETRAIN_CONFIG_H100_BF16_V1 = replace(
    _NEMOTRON_3_NANO_PRETRAIN_CONFIG_H100,
    recompute_modules=["moe", "layernorm"],
    cuda_graph_scope=["attn", "mamba"],
)
NEMOTRON_3_NANO_PRETRAIN_CONFIG_H100_FP8_CS_V1 = replace(
    _NEMOTRON_3_NANO_PRETRAIN_CONFIG_H100,
    cuda_graph_scope=["mamba"],
    recompute_modules=["moe", "layernorm", "core_attn", "moe_act"],
)


# =============================================================================
# Nemotron 3 Super Pretrain - V1
# =============================================================================

BASE_NEMOTRON_3_SUPER_CONFIG = WorkloadBaseConfig(
    num_gpus=64,
    global_batch_size=512,
    micro_batch_size=1,
    tensor_model_parallel_size=1,
    expert_tensor_parallel_size=1,
    expert_model_parallel_size=64,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "mamba", "moe_router", "moe_preprocess"],
)

NEMOTRON_3_SUPER_PRETRAIN_CONFIG_GB300_BF16_V1 = BASE_NEMOTRON_3_SUPER_CONFIG
NEMOTRON_3_SUPER_PRETRAIN_CONFIG_GB300_FP8_MX_V1 = BASE_NEMOTRON_3_SUPER_CONFIG
NEMOTRON_3_SUPER_PRETRAIN_CONFIG_GB300_NVFP4_V1 = BASE_NEMOTRON_3_SUPER_CONFIG


BASE_NEMOTRON_3_SUPER_CONFIG_GB200 = replace(
    BASE_NEMOTRON_3_SUPER_CONFIG,
    tensor_model_parallel_size=2,
)

NEMOTRON_3_SUPER_PRETRAIN_CONFIG_GB200_BF16_V1 = BASE_NEMOTRON_3_SUPER_CONFIG_GB200
NEMOTRON_3_SUPER_PRETRAIN_CONFIG_GB200_FP8_MX_V1 = BASE_NEMOTRON_3_SUPER_CONFIG_GB200
NEMOTRON_3_SUPER_PRETRAIN_CONFIG_GB200_NVFP4_V1 = BASE_NEMOTRON_3_SUPER_CONFIG_GB200

NEMOTRON_3_SUPER_PRETRAIN_CONFIG_VR200_BF16_V1 = NEMOTRON_3_SUPER_PRETRAIN_CONFIG_GB200_BF16_V1
NEMOTRON_3_SUPER_PRETRAIN_CONFIG_VR200_FP8_MX_V1 = NEMOTRON_3_SUPER_PRETRAIN_CONFIG_GB200_FP8_MX_V1
NEMOTRON_3_SUPER_PRETRAIN_CONFIG_VR200_NVFP4_V1 = NEMOTRON_3_SUPER_PRETRAIN_CONFIG_GB200_NVFP4_V1


BASE_NEMOTRON_3_SUPER_CONFIG_B300 = replace(
    BASE_NEMOTRON_3_SUPER_CONFIG,
    expert_model_parallel_size=8,
)
NEMOTRON_3_SUPER_PRETRAIN_CONFIG_B300_BF16_V1 = replace(
    BASE_NEMOTRON_3_SUPER_CONFIG_B300,
    recompute_modules=["moe_act", "layernorm"],
)
NEMOTRON_3_SUPER_PRETRAIN_CONFIG_B300_FP8_MX_V1 = BASE_NEMOTRON_3_SUPER_CONFIG_B300
NEMOTRON_3_SUPER_PRETRAIN_CONFIG_B300_NVFP4_V1 = NEMOTRON_3_SUPER_PRETRAIN_CONFIG_B300_BF16_V1


BASE_NEMOTRON_3_SUPER_CONFIG_B200 = replace(
    BASE_NEMOTRON_3_SUPER_CONFIG,
    expert_model_parallel_size=64,
    cuda_graph_impl="none",
    recompute_modules=["moe_act", "layernorm"],
)

NEMOTRON_3_SUPER_PRETRAIN_CONFIG_B200_BF16_V1 = replace(
    BASE_NEMOTRON_3_SUPER_CONFIG_B200,
    recompute_modules=["moe_act", "moe", "layernorm", "core_attn"],
)
NEMOTRON_3_SUPER_PRETRAIN_CONFIG_B200_FP8_MX_V1 = BASE_NEMOTRON_3_SUPER_CONFIG_B200
NEMOTRON_3_SUPER_PRETRAIN_CONFIG_B200_NVFP4_V1 = replace(
    BASE_NEMOTRON_3_SUPER_CONFIG_B200,
    tensor_model_parallel_size=2,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["mamba", "attn", "moe_router", "moe_preprocess"],
    recompute_modules=None,
)


# =============================================================================
# Nemotron 3 Ultra Pretrain - V1
# =============================================================================
#
# Mirrors the reference Megatron-LM GB300 launch script
# (Megatron-LM-Private/examples/nt3/gb300_nt3.sh): 64 GB300 nodes x 4 GPUs = 256
# GPUs, TP1 / PP1 / CP1 / EP64 / ETP1, GBS 256 / MBS 1, Megatron-FSDP (HSDP),
# CuteDSL fused grouped MLP, HybridEP dispatcher, selective recompute of moe_act,
# and CUDA graphs disabled.

BASE_NEMOTRON_3_ULTRA_CONFIG = WorkloadBaseConfig(
    num_gpus=256,
    global_batch_size=256,
    micro_batch_size=1,
    tensor_model_parallel_size=1,
    expert_tensor_parallel_size=1,
    expert_model_parallel_size=64,
    use_megatron_fsdp=True,
    moe_flex_dispatcher_backend="hybridep",
    cutedsl_fused_grouped_mlp=True,
    cuda_graph_impl="none",
    cuda_graph_scope=None,
    recompute_modules=["moe_act"],
    # this argument helps simulate the behaviour of offload_modules = ["moe_act"]
    # by only offloading tensors larger than 500MB. This works for the current NT3 Ultra config seqlen=8192,
    # but may need to be updated for other seqlens / batch sizes.
    min_offloaded_tensor_size=500000000,
)

NEMOTRON_3_ULTRA_PRETRAIN_CONFIG_GB300_FP8_MX_V1 = BASE_NEMOTRON_3_ULTRA_CONFIG


__all__ = [
    "NEMOTRON_3_ULTRA_PRETRAIN_CONFIG_GB300_FP8_MX_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_BF16_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_FP8_MX_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_NVFP4_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_BF16_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_FP8_MX_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_NVFP4_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_VR200_BF16_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_VR200_FP8_MX_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_VR200_NVFP4_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_BF16_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_FP8_MX_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_NVFP4_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_BF16_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_FP8_MX_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_NVFP4_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_H100_BF16_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_H100_FP8_CS_V1",
    "NEMOTRON_3_SUPER_PRETRAIN_CONFIG_GB300_BF16_V1",
    "NEMOTRON_3_SUPER_PRETRAIN_CONFIG_GB300_FP8_MX_V1",
    "NEMOTRON_3_SUPER_PRETRAIN_CONFIG_GB300_NVFP4_V1",
    "NEMOTRON_3_SUPER_PRETRAIN_CONFIG_GB200_BF16_V1",
    "NEMOTRON_3_SUPER_PRETRAIN_CONFIG_GB200_FP8_MX_V1",
    "NEMOTRON_3_SUPER_PRETRAIN_CONFIG_GB200_NVFP4_V1",
    "NEMOTRON_3_SUPER_PRETRAIN_CONFIG_VR200_BF16_V1",
    "NEMOTRON_3_SUPER_PRETRAIN_CONFIG_VR200_FP8_MX_V1",
    "NEMOTRON_3_SUPER_PRETRAIN_CONFIG_VR200_NVFP4_V1",
    "NEMOTRON_3_SUPER_PRETRAIN_CONFIG_B300_BF16_V1",
    "NEMOTRON_3_SUPER_PRETRAIN_CONFIG_B300_FP8_MX_V1",
    "NEMOTRON_3_SUPER_PRETRAIN_CONFIG_B300_NVFP4_V1",
    "NEMOTRON_3_SUPER_PRETRAIN_CONFIG_B200_BF16_V1",
    "NEMOTRON_3_SUPER_PRETRAIN_CONFIG_B200_FP8_MX_V1",
    "NEMOTRON_3_SUPER_PRETRAIN_CONFIG_B200_NVFP4_V1",
]
