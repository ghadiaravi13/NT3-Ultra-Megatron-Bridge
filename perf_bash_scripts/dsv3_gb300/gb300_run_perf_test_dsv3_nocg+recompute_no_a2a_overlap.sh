HF_TOKEN="${HF_TOKEN:?Environment variable HF_TOKEN is not set}"
CONTAINER="/lustre/fsw/coreai_dlalgo_llm/dingqingy/2606release/containers/nemo-26.06.rc6.sqsh"
MBRIDGE_PATH="../../"

JOB_NAME="dsv3_gb300_toy"
RESULTS_DIR="${MBRIDGE_PATH}/results/dsv3_gb300/${JOB_NAME}"

# Toy DeepSeek-V3 config: 2 nodes (8x GB200), TP=1, PP=2 (no VP), EP=4, CP=1 (none).
# Smaller model: 6 layers total, first layer dense MLP + next 5 layers MoE.
#   - EP is capped at 4 here: with TP=1, PP=2, CP=1 -> DP=4, and expert parallel
#     must divide TP*DP=4 (expert_tensor_parallel_size=1), so EP=8 is invalid.
#   - VP is explicitly disabled via "-vp None".
#   - num_moe_experts is cut from 256 to 32 (8 experts/GPU at EP=4). The full 256
#     experts OOM: expert optimizer states are unsharded (expert-DP = world/(TP*PP*
#     CP*EP) = 1), costing ~16 B/param * 64 experts/GPU * 4 MoE-eq layers ~= 180 GB
#     on the heavier PP stage alone. This dominates memory, not batch size/layers.
# CUDA graph is turned off completely ("--cuda_graph_impl none"), and the
# full-iteration-CG MoE padding/paged-stash machinery (enabled by the fp8_mx
# base config) is disabled via Hydra overrides so nothing CG-related remains.
# A2A expert-parallel comm overlap is turned off via a Hydra override
# (the "--moe_a2a_overlap false" CLI flag is a no-op; it only acts when true).
#
# The DeepSeek-V3 PP layout is hardcoded for the full 61-layer model and has no
# (PP=2, VP=1) preset, so for an 8-layer toy we pass an explicit 2-stage layout
# via Hydra. It is given as a list-of-lists (full LayerType names) rather than
# the "Etttt|ttttmL" string DSL because the generated launcher re-parses args
# inside `bash -c '...'`, where '|' would become a shell pipe and '*' a glob.
# Layout: stage0 = embedding + 3 decoders, stage1 = 3 decoders + MTP + loss.

uv run --no-project --with nemo-run --with numpy python ${MBRIDGE_PATH}/scripts/performance/setup_experiment.py \
  --account coreai_dlalgo_llm \
  --container_image ${CONTAINER} \
  --partition gb300 \
  --model_family_name deepseek \
  --model_recipe_name deepseek_v3 \
  --log_dir ${RESULTS_DIR} \
  --num_gpus 8 \
  --gpus_per_node 4 \
  --time_limit "00:15:00" \
  --gpu gb300 \
  --compute_dtype fp8_mx \
  --max_steps 20 \
  --num_layers 6 \
  --first_k_dense_replace 1 \
  --global_batch_size 256 \
  -tp 1 \
  -pp 2 \
  -vp None \
  -ep 4 \
  --cuda_graph_impl none \
  --pytorch_profiler true \
  --profiling_start_step 15 \
  --profiling_stop_step 16 \
  --packager none \
  --wandb_key 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 \
  --wandb_project_name DSv3_GB300_performance \
  --wandb_experiment_name pt_prof_dsv3_gb300_toy_nocg+recompute_no_a2a_overlap \
  --hf_token ${HF_TOKEN} \
  comm_overlap.overlap_moe_expert_parallel_comm=false \
  comm_overlap.delay_wgrad_compute=false \
  model.moe_pad_experts_for_cuda_graph_inference=false \
  model.moe_paged_stash=false \
  model.num_moe_experts=32 \
  'model.pipeline_model_parallel_layout=[[embedding,decoder,decoder,decoder],[decoder,decoder,decoder,mtp,loss]]'
  # --enable_nsys \