#!/bin/bash
#SBATCH --job-name=u4_reference_paper
#SBATCH --output=logs/u4_reference_paper_%A_%a.out
#SBATCH --error=logs/u4_reference_paper_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=h100-47
#SBATCH --partition=gpu-long
#
# Reference reproduction run using the untouched third_party/scaling-crl/train.py.
# This is the closest codepath to the paper implementation.
# - Environment: ant_u4_maze (train=eval)
# - Recipe: 100M env steps, 512 envs, batch size 512
# - Model: standard untied baseline from the paper trainer
# - Width: 256
# - Depth: 16
# - W&B logging happens once per epoch, so NUM_EPOCHS controls logging cadence.
#
# Usage:
#   WANDB_ENTITY=<your-team-or-user> sbatch --array=0-0 job_u4_reference_paper_array.slurm

set -e

mkdir -p logs
OUT_DIR="results/u4_reference_paper_recipe"
mkdir -p "$OUT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate online

export WANDB_DIR="${SLURM_TMPDIR:-/tmp}"

ENV_ID="ant_u4_maze"
EVAL_ENV_ID="ant_u4_maze"
SEEDS=(1001)

NUM_EPOCHS="${NUM_EPOCHS:-100}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-recurrent_resnet_u4}"
WANDB_GROUP="${WANDB_GROUP:-u4_reference_paper_recipe}"

WANDB_ARGS=(
  --wandb_mode online
  --wandb_project_name "$WANDB_PROJECT_NAME"
  --wandb_group "$WANDB_GROUP"
)
if [ -n "${WANDB_ENTITY:-}" ]; then
  WANDB_ARGS+=(--wandb_entity "$WANDB_ENTITY")
fi

task_id=${SLURM_ARRAY_TASK_ID:-0}
if [ "$task_id" -ge "${#SEEDS[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$task_id for ${#SEEDS[@]} seeds" >&2
  exit 2
fi
SEED="${SEEDS[$task_id]}"

NAME="u4_reference_paper_baseline_d16_w256"
NODE="$(hostname)"
GPU_NAME="$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo N/A)"

LOG="logs/${NAME}_s${SEED}_${SLURM_JOB_ID:-local}_${task_id}.log"
OUT="${OUT_DIR}/${NAME}_s${SEED}_${SLURM_JOB_ID:-local}_${task_id}.csv"
echo "config,seed,node,gpu_name,env_steps_final,time_at_goal_final,success_rate_final,success_any_final,episode_dist_final,eval_reward_final,training_time_hours" > "$OUT"

echo "[$(date +%H:%M:%S)] Running $NAME..."
set +e
(
  cd third_party/scaling-crl
  python train.py \
    --exp_name "$NAME" \
    --seed "$SEED" \
    --env_id "$ENV_ID" \
    --eval_env_id "$EVAL_ENV_ID" \
    --num_epochs "$NUM_EPOCHS" \
    --total_env_steps 100000000 \
    --num_envs 512 \
    --batch_size 512 \
    --critic_network_width 256 \
    --actor_network_width 256 \
    --critic_depth 16 \
    --actor_depth 16 \
    --actor_skip_connections 4 \
    --critic_skip_connections 4 \
    --actor_lr 0.0003 \
    --critic_lr 0.0003 \
    --logsumexp_penalty_coeff 0.1 \
    --vis_length 1000 \
    --save_buffer 0 \
    --wandb_dir "$WANDB_DIR" \
    --no-capture_vis \
    --no-checkpoint \
    "${WANDB_ARGS[@]}"
) > "$LOG" 2>&1
run_status=$?
set -e
if [ "$run_status" -ne 0 ]; then
  echo "[$(date +%H:%M:%S)] WARNING: training exited non-zero ($run_status). Writing partial CSV row." >> "$LOG"
fi

final_line=$(grep "epoch .* complete\\. metrics:" "$LOG" | tail -1 || true)

env_steps_final=$(echo "$final_line" | grep -oP "'training/envsteps': \\K[0-9.]+|'training/envsteps': \\Knan" || echo "N/A")
time_at_goal_final=$(echo "$final_line" | grep -oP "'eval/episode_success': \\K[0-9.]+|'eval/episode_success': \\Knan" || echo "N/A")
success_rate_final=$(echo "$final_line" | grep -oP "'eval/success_rate': \\K[0-9.]+|'eval/success_rate': \\Knan" || echo "N/A")
success_any_final=$(echo "$final_line" | grep -oP "'eval/episode_success_any': \\K[0-9.]+|'eval/episode_success_any': \\Knan" || echo "N/A")
episode_dist_final=$(echo "$final_line" | grep -oP "'eval/episode_dist': \\K[-0-9.]+|'eval/episode_dist': \\Knan" || echo "N/A")
eval_reward_final=$(echo "$final_line" | grep -oP "'eval/episode_reward': \\K[-0-9.]+|'eval/episode_reward': \\Knan" || echo "N/A")
train_time=$(echo "$final_line" | grep -oP "'training/walltime': \\K[0-9.]+" || echo "N/A")
if [ "$train_time" != "N/A" ]; then
  train_time=$(python - <<'PY' "$train_time"
import sys
try:
    x = float(sys.argv[1])
    print(round(x / 3600.0, 5))
except Exception:
    print("N/A")
PY
)
fi

echo "$NAME,$SEED,$NODE,$GPU_NAME,$env_steps_final,$time_at_goal_final,$success_rate_final,$success_any_final,$episode_dist_final,$eval_reward_final,$train_time" >> "$OUT"
echo "[$(date +%H:%M:%S)] Done. Row written to $OUT"
exit "$run_status"
