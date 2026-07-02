HF_TOKEN="${HF_TOKEN:?Environment variable HF_TOKEN is not set}"
WANDB_KEY="${WANDB_KEY:?Environment variable WANDB_KEY is not set}"
CONTAINER="/lustre/fsw/coreai_dlalgo_llm/rghadia/vr200_bringup/images/nemo-deepep-rubin-devel-06-25-cutedsl.sqsh"
MBRIDGE_PATH="../../"

JOB_NAME="qwen3_30b_vr200"
RESULTS_DIR="${MBRIDGE_PATH}/results/${JOB_NAME}"

uv run --no-project --with nemo-run --with numpy python ${MBRIDGE_PATH}/scripts/performance/setup_experiment.py \
  --account coreai_dlalgo_llm \
  --container_image ${CONTAINER} \
  --partition batch-xdr \
  --model_family_name qwen \
  --model_recipe_name qwen3_30b_a3b \
  --log_dir ${RESULTS_DIR} \
  --num_gpus 8 \
  --gpus_per_node 4 \
  --time_limit "00:15:00" \
  --gpu vr200 \
  --compute_dtype fp8_mx \
  --max_steps 20 \
  --enable_nsys \
  --profiling_start_step 15 \
  --profiling_stop_step 16 \
  --packager none \
  --hf_token ${HF_TOKEN} \
  --packager none \
  --wandb_key ${WANDB_KEY} \
  --wandb_project_name Qwen3_30B_VR200_performance \
  --wandb_experiment_name qwen3_30b_vr200_full_iter_cg_test \
  -E LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/compat/lib.real:/usr/local/cuda/lib64:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  -E NSYS_CONFIG_DIRECTIVES='CuptiUseRawGpuTimestamps=false' \
  --cuda_graph_impl full_iteration \
  --cuda_graph_scope "[]"