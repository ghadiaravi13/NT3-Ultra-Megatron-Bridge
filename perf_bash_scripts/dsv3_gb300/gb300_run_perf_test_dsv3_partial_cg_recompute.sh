HF_TOKEN="${HF_TOKEN:?Environment variable HF_TOKEN is not set}"
CONTAINER="/lustre/fsw/coreai_dlalgo_llm/dingqingy/2606release/containers/nemo-26.06.rc6.sqsh"
MBRIDGE_PATH="../../"

JOB_NAME="dsv3_gb200_partial_cg_recompute"
RESULTS_DIR="${MBRIDGE_PATH}/results/dsv3_gb200/${JOB_NAME}"

uv run --no-project --with nemo-run --with numpy python ${MBRIDGE_PATH}/scripts/performance/setup_experiment.py \
  --account coreai_dlalgo_llm \
  --container_image ${CONTAINER} \
  --partition gb200 \
  --model_family_name deepseek \
  --model_recipe_name deepseek_v3 \
  --log_dir ${RESULTS_DIR} \
  --num_gpus 256 \
  --gpus_per_node 4 \
  --time_limit "00:15:00" \
  --gpu gb200 \
  --compute_dtype fp8_mx \
  --cuda_graph_impl transformer_engine \
  --recompute_modules mla_up_proj \
  'model.moe_paged_stash=false' \
  'model.moe_expert_rank_capacity_factor=null' \
  'model.moe_pad_experts_for_cuda_graph_inference=false' \
  --max_steps 15 \
  --enable_nsys \
  --profiling_start_step 10 \
  --profiling_stop_step 11 \
  --packager none \
  --wandb_key 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 \
  --wandb_project_name DSv3_GB200_performance \
  --wandb_experiment_name dsv3_gb200_partial_cg_recompute \
  --hf_token ${HF_TOKEN}