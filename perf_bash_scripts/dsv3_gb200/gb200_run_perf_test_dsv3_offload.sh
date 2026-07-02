HF_TOKEN="${HF_TOKEN:?Environment variable HF_TOKEN is not set}"
CONTAINER="/lustre/fsw/coreai_dlalgo_llm/dingqingy/2606release/containers/nemo-26.06.rc6.sqsh"
MBRIDGE_PATH="../../"

JOB_NAME="with_profile_dsv3_gb200_mxfp8_fullcg_offl_core_attn_and_attn_proj"
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
  --max_steps 20 \
  --packager none \
  --wandb_key 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 \
  --wandb_project_name DSv3_GB200_performance \
  --wandb_experiment_name dsv3_gb200_mxfp8_fullcg_offl_core_attn_and_attn_proj_max_inflt_offl_4 \
  --custom_env_vars NVTE_CPU_OFFLOAD_V1=1 \
  --hf_token ${HF_TOKEN} \
  model.fine_grained_activation_offloading=true \
  model.fine_grained_offloading_max_inflight_offloads=4 \
  'model.offload_modules=[core_attn,attn_proj]' \
  'model.recompute_modules=[]' \
  --enable_nsys \
  --profiling_start_step 15 \
  --profiling_stop_step 16 \
  # 'model.recompute_granularity=null' \