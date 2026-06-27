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


import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.config import ConfigContainer


NEMOTRON_3_ULTRA_HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"


def nemotron_3_ultra_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Nemotron 3 Ultra (550B-A55B LatentMoE).

    This is a hybrid Mamba / attention Latent MoE model with Multi-Token Prediction
    (MTP). The settings here mirror the reference Megatron-LM GB300 launch script
    (``examples/nt3/gb300_nt3.sh``): TP=1, PP=1, CP=1, EP=64, ETP=1 with Megatron-FSDP
    (HSDP) sharding, BF16 + MXFP8 mixed precision, and the CuteDSL / HybridEP /
    activation-offloading performance feature set.

    Returns:
        ConfigContainer: Pre-training configuration for Nemotron 3 Ultra.
    """
    cfg = _pretrain_common()

    # Model Configuration (LatentMoE with MTP) — derived from HF config via AutoBridge.
    # MoE specifics (512 experts, top-k 22, sigmoid router, latent size 2048, shared
    # expert, seq_aux_loss balancing, squared-relu, hybrid Mamba/attention pattern, ...)
    # all come from the HF model definition.
    cfg.model = AutoBridge.from_hf_pretrained(NEMOTRON_3_ULTRA_HF_MODEL_ID).to_megatron_provider(load_weights=False)

    # Parallelism Settings (match gb300_nt3.sh: TP1 / PP1 / CP1 / EP64 / ETP1).
    # These are the recipe defaults; the GB300 performance config re-applies them
    # via the workload base config.
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = 8192

    # Tokenizer (--tokenizer-model). Nemotron 3 family shares the Nano tokenizer.
    cfg.tokenizer.tokenizer_model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

    # Dataset Configuration (--mock-data --no-mmap-bin-files --num-workers 1)
    cfg.dataset.seq_length = 8192
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 1
    cfg.dataset.mmap_bin_files = False

    # MoE Token Dispatcher Settings (--moe-token-dispatcher-type flex
    # --moe-flex-dispatcher-backend hybridep). The flex backend is applied by the
    # performance layer; share-expert overlap stays off to match the bash script. #TODO: check if this is correct
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_router_fusion = True

    # Training Configuration (--train-samples 80000 / --global-batch-size 256 /
    # --micro-batch-size 1, with manual GC every 100 steps).
    cfg.train.train_iters = 312
    cfg.train.global_batch_size = 256
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100 # TODO: try with 5 same as dsv3 and see if jitter goes away

    # Validation (--eval-interval 250 --eval-iters 14)
    cfg.validation.eval_interval = 250
    cfg.validation.eval_iters = 14

    # Transformer Engine (TE)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph disabled to match the reference run (--enable-cuda-graph is commented
    # out in gb300_nt3.sh).
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = []

    # Kernel Selections
    cfg.model.attention_backend = "fused"
    # Native cross-entropy fusion (--cross-entropy-fusion-impl native). TE fusion has
    # known stability issues and is rejected by Megatron-LM arg validation.
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.use_fused_weighted_squared_relu = True
    cfg.model.use_te_rng_tracker = False

    # CuteDSL fused grouped MLP + TE op fuser (--use-transformer-engine-op-fuser,
    # NVTE_CUTEDSL_FUSED_GROUPED_MLP=1). interleave size mirrors the perf override.
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_mlp_glu_interleave_size = 32

    # Fine-grained activation offloading (--fine-grained-activation-offloading
    # --offload-modules fused_group_mlp). Requires NVTE_CPU_OFFLOAD_V1=1 in the
    # environment. We use the fused_group_mlp offload point (not moe_act) because
    # use_transformer_engine_op_fuser=True collapses fc1+activation+fc2 into a
    # single TE op, so the unfused moe_act activation-input tensor does not exist
    # as a separable handle. The MLM validator enforces this: moe_act/expert_fc1
    # offloads are rejected with the fused impl, and fused_group_mlp requires
    # use_transformer_engine_op_fuser.

    # NOTE: offload and recompute target *different* tensors in the expert MLP:
    # offload (fused_group_mlp) moves the whole fused-grouped-MLP activation input
    # to CPU; recompute (moe_act) drops the activation output and recomputes it in
    # backward inside the fused op via ScaledSReLU(activation_recompute_in_mlp=True).
    # Together they minimize expert-MLP peak activation memory. Matches gb300_nt3.sh.
    cfg.model.fine_grained_activation_offloading = True
    cfg.model.offload_modules = ["fused_group_mlp"]

    # Selective recompute of MoE activation (--recompute-granularity selective
    # --recompute-modules moe_act). Threads through to the fused TE op as
    # activation_recompute_in_mlp; safe to keep as "moe_act" with the fused path.
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe_act"]

    # High priority NCCL stream for the EP communicator (--high-priority-stream-groups ep).
    cfg.dist.high_priority_stream_groups = ["ep"]
    # --distributed-timeout-minutes 30
    cfg.dist.distributed_timeout_minutes = 30

    # MTP Settings (--mtp-num-layers 2 --mtp-use-repeated-layer
    # --calculate-per-token-loss --mtp-loss-scaling-factor 0.3)
    cfg.model.mtp_num_layers = 2
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.calculate_per_token_loss = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.mtp_use_repeated_layer = True

    # Mixed Precision (default BF16 + MXFP8; the performance layer swaps this per
    # --compute_dtype). Mirrors mxfp8_options in gb300_nt3.sh.
    cfg.mixed_precision = "bf16_with_mxfp8_mixed"

    # Optimizer hyperparameters (--lr 8e-4 --min-lr 8e-6 --weight-decay 0.1
    # --adam-beta1 0.9 --adam-beta2 0.95 --lr-decay-style WSD)
    cfg.optimizer.lr = 8.0e-4
    cfg.optimizer.min_lr = 8.0e-6
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.scheduler.lr_warmup_iters = 15
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.lr_decay_style = "WSD"

    # Checkpoint Configuration
    cfg.checkpoint.save_interval = 2000
    cfg.checkpoint.ckpt_assume_constant_structure = True
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.checkpoint.async_save = True

    # DDP Configuration (--overlap-grad-reduce --overlap-param-gather
    # --use-distributed-optimizer --ddp-num-buckets 48 --grad-reduce-in-bf16)
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.average_in_collective = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.num_buckets = 48

    cfg.model.init_method_std = 0.0099
    cfg.model.apply_rope_fusion = False
    cfg.model.gradient_accumulation_fusion = True

    return cfg


__all__ = [
    "nemotron_3_ultra_pretrain_config",
]
