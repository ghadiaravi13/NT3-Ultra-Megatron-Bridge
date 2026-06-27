HF_TOKEN="${HF_TOKEN:?Environment variable HF_TOKEN is not set}"
CONTAINER="/lustre/fsw/coreai_dlalgo_llm/dingqingy/2606release/containers/nemo-26.06.rc6.sqsh"
MBRIDGE_PATH="./Megatron-Bridge/"

JOB_NAME="qwen3_30b_gb200"
RESULTS_DIR="${MBRIDGE_PATH}/results/${JOB_NAME}"

uv run --no-project --with nemo-run --with numpy python ${MBRIDGE_PATH}/scripts/performance/setup_experiment.py \
  --account coreai_dlalgo_llm \
  --container_image ${CONTAINER} \
  --partition gb200 \
  --model_family_name qwen \
  --model_recipe_name qwen3_30b_a3b \
  --log_dir ${RESULTS_DIR} \
  --num_gpus 8 \
  --gpus_per_node 4 \
  --time_limit "00:15:00" \
  --gpu gb200 \
  --compute_dtype fp8_mx \
  --max_steps 5 \
  --enable_nsys \
  --profiling_start_step 3 \
  --profiling_stop_step 4 \
  --packager none \
  --hf_token ${HF_TOKEN}