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

import logging
import os

import torch

from megatron.core.quantization.utils import load_quantization_recipe
from utils.overrides import set_workload_base_configs
from utils.precision import get_precision_config
from utils.utils import get_workload_base_config

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from megatron.bridge.recipes.nemotronh.nemotron_3_super import nemotron_3_super_pretrain_config
from megatron.bridge.recipes.nemotronh.nemotron_3_ultra import nemotron_3_ultra_pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import nemotron_3_super_bf16_with_nvfp4_mixed


logger = logging.getLogger(__name__)


def set_nemotron_3_nano_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all Nemotron 3 Nano configs."""
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = True


def set_nemotron_3_super_common_configs(cfg: ConfigContainer, precision: str) -> None:
    """Common Nemotron 3 Super pretrain perf settings; restores model recipe fields for ``precision``."""
    if precision.lower() == "nvfp4":
        cfg.mixed_precision = nemotron_3_super_bf16_with_nvfp4_mixed()
        # Disabled until MCore PR 4358 lands.
        cfg.mixed_precision.fp4_param_gather = False
        cfg.model.quant_recipe = load_quantization_recipe(os.path.join(os.path.dirname(__file__), "te_quant.cfg"))

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = True

    cfg.checkpoint.async_save = False

    if precision.lower() in ("nvfp4", "fp8_mx"):
        cfg.model.moe_router_padding_for_quantization = True


# Number of GPUs in a single GB300 multi-node NVLink (MNNVL) domain
# (16 nodes x 4 GPUs). Each Megatron-FSDP optimizer instance is sharded within one
# NVLink domain (HSDP), matching `--num-distributed-optimizer-instances $((nodes/16))`
# in the reference Megatron-LM launch script.
_GB300_NVLINK_DOMAIN_GPUS = 64


def set_nemotron_3_ultra_common_configs(cfg: ConfigContainer, precision: str) -> None:
    """Common Nemotron 3 Ultra pretrain perf settings; restores model recipe fields for ``precision``."""
    if precision.lower() == "nvfp4":
        cfg.mixed_precision = nemotron_3_super_bf16_with_nvfp4_mixed()
        # Disabled until MCore PR 4358 lands.
        cfg.mixed_precision.fp4_param_gather = False
        cfg.model.quant_recipe = load_quantization_recipe(os.path.join(os.path.dirname(__file__), "te_quant.cfg"))

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = True

    cfg.checkpoint.async_save = False

    if precision.lower() in ("nvfp4", "fp8_mx"):
        cfg.model.moe_router_padding_for_quantization = True


def _apply_nemotron_3_ultra_fsdp_hsdp(cfg: ConfigContainer, num_gpus: int) -> None:
    """Apply Megatron-FSDP HSDP settings that the generic overrides do not cover.

    Mirrors the ``fsdp_options`` block of gb300_nt3.sh: shard params/grads/optimizer
    within each NVLink domain and replicate (optimizer-sharded) across domains, with
    BF16 gradient comm, FP32 main params, and BF16 main grads.
    """
    # --num-distributed-optimizer-instances $((SLURM_JOB_NUM_NODES / 16))
    cfg.ddp.num_distributed_optimizer_instances = max(1, num_gpus // _GB300_NVLINK_DOMAIN_GPUS)
    # --outer-dp-sharding-strategy optim (HSDP across NVLink domains)
    cfg.ddp.outer_dp_sharding_strategy = "optim"
    # --data-parallel-sharding-strategy optim_grads_params is set by _set_megatron_fsdp_overrides.
    # --megatron-fsdp-grad-comm-dtype bf16 / --megatron-fsdp-main-params-dtype fp32 /
    # --megatron-fsdp-main-grads-dtype bf16
    cfg.ddp.megatron_fsdp_grad_comm_dtype = torch.bfloat16
    cfg.ddp.megatron_fsdp_main_params_dtype = torch.float32
    cfg.ddp.megatron_fsdp_main_grads_dtype = torch.bfloat16
    # --no-gradient-accumulation-fusion (incompatible with BF16 FSDP main grads)
    cfg.model.gradient_accumulation_fusion = False
    # --ckpt-format fsdp_dtensor
    cfg.checkpoint.ckpt_format = "fsdp_dtensor"
    # --cross-entropy-fusion-impl native (the common perf override forces "te")
    cfg.model.cross_entropy_fusion_impl = "native"


def nemotron_3_ultra_pretrain_config_gb300(
    precision: str = "fp8_mx", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config (matches Megatron-LM examples/nt3/gb300_nt3.sh)."""
    base_cfg = get_workload_base_config(
        model_family_name="nemotronh",
        model_recipe_name="nemotron_3_ultra",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = nemotron_3_ultra_pretrain_config()
    cfg.mixed_precision = precision_config
    set_nemotron_3_ultra_common_configs(cfg, precision)
    set_workload_base_configs(cfg, base_cfg)
    if base_cfg.moe_flex_dispatcher_backend is not None:
        cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    # Apply HSDP / FSDP dtype overrides last so they win over the generic FSDP overrides.
    _apply_nemotron_3_ultra_fsdp_hsdp(cfg, base_cfg.num_gpus)

    return cfg


def nemotron_3_super_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="nemotronh",
        model_recipe_name="nemotron_3_super",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = nemotron_3_super_pretrain_config()
    cfg.mixed_precision = precision_config
    set_nemotron_3_super_common_configs(cfg, precision)
    set_workload_base_configs(cfg, base_cfg)
    if base_cfg.moe_flex_dispatcher_backend is not None:
        cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    return cfg


def nemotron_3_super_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="nemotronh",
        model_recipe_name="nemotron_3_super",
        gpu="gb200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = nemotron_3_super_pretrain_config()
    cfg.mixed_precision = precision_config
    set_nemotron_3_super_common_configs(cfg, precision)
    set_workload_base_configs(cfg, base_cfg)
    if base_cfg.moe_flex_dispatcher_backend is not None:
        cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    return cfg


def nemotron_3_super_pretrain_config_vr200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """VR200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="nemotronh",
        model_recipe_name="nemotron_3_super",
        gpu="vr200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = nemotron_3_super_pretrain_config()
    cfg.mixed_precision = precision_config
    set_nemotron_3_super_common_configs(cfg, precision)
    set_workload_base_configs(cfg, base_cfg)
    if base_cfg.moe_flex_dispatcher_backend is not None:
        cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    return cfg


def nemotron_3_super_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="nemotronh",
        model_recipe_name="nemotron_3_super",
        gpu="b300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = nemotron_3_super_pretrain_config()
    cfg.mixed_precision = precision_config
    set_nemotron_3_super_common_configs(cfg, precision)
    set_workload_base_configs(cfg, base_cfg)
    if base_cfg.moe_flex_dispatcher_backend is not None:
        cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    return cfg


def nemotron_3_super_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="nemotronh",
        model_recipe_name="nemotron_3_super",
        gpu="b200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = nemotron_3_super_pretrain_config()
    cfg.mixed_precision = precision_config
    set_nemotron_3_super_common_configs(cfg, precision)
    set_workload_base_configs(cfg, base_cfg)
    if base_cfg.moe_flex_dispatcher_backend is not None:
        cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    return cfg


def nemotron_3_nano_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="nemotronh",
        model_recipe_name="nemotron_3_nano",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = precision_config
    set_nemotron_3_nano_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if base_cfg.moe_flex_dispatcher_backend is not None:
        cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    return cfg


def nemotron_3_nano_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="nemotronh",
        model_recipe_name="nemotron_3_nano",
        gpu="gb200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = precision_config
    set_nemotron_3_nano_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if base_cfg.moe_flex_dispatcher_backend is not None:
        cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    return cfg


def nemotron_3_nano_pretrain_config_vr200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """VR200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="nemotronh",
        model_recipe_name="nemotron_3_nano",
        gpu="vr200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = precision_config
    set_nemotron_3_nano_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if base_cfg.moe_flex_dispatcher_backend is not None:
        cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    return cfg


def nemotron_3_nano_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="nemotronh",
        model_recipe_name="nemotron_3_nano",
        gpu="b300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = precision_config
    set_nemotron_3_nano_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if base_cfg.moe_flex_dispatcher_backend is not None:
        cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    return cfg


def nemotron_3_nano_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="nemotronh",
        model_recipe_name="nemotron_3_nano",
        gpu="b200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = precision_config
    set_nemotron_3_nano_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if base_cfg.moe_flex_dispatcher_backend is not None:
        cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    return cfg


def nemotron_3_nano_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """H100, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="nemotronh",
        model_recipe_name="nemotron_3_nano",
        gpu="h100",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = precision_config
    set_nemotron_3_nano_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if base_cfg.moe_flex_dispatcher_backend is not None:
        cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    return cfg
