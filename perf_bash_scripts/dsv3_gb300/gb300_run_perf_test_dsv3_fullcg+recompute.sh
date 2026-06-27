HF_TOKEN="${HF_TOKEN:?Environment variable HF_TOKEN is not set}"
WANDB_KEY="${WANDB_KEY:?Environment variable WANDB_KEY is not set}"
CONTAINER="/lustre/fsw/coreai_dlalgo_llm/dingqingy/2606release/containers/nemo-26.06.rc6.sqsh"
MBRIDGE_PATH="../../"

JOB_NAME="dsv3_gb300"
RESULTS_DIR="${MBRIDGE_PATH}/results/dsv3_gb300/${JOB_NAME}"

uv run --no-project --with nemo-run --with numpy python ${MBRIDGE_PATH}/scripts/performance/setup_experiment.py \
  --account coreai_dlalgo_llm \
  --container_image ${CONTAINER} \
  --partition gb300 \
  --model_family_name deepseek \
  --model_recipe_name deepseek_v3 \
  --log_dir ${RESULTS_DIR} \
  --num_gpus 256 \
  --gpus_per_node 4 \
  --time_limit "00:15:00" \
  --gpu gb300 \
  --compute_dtype fp8_mx \
  --max_steps 20 \
  --enable_nsys \
  --profiling_start_step 15 \
  --profiling_stop_step 16 \
  --packager none \
  --wandb_key ${WANDB_KEY} \
  --wandb_project_name DSv3_GB300_performance \
  --wandb_experiment_name dsv3_gb300_fullcg+recompute \
  --hf_token ${HF_TOKEN}