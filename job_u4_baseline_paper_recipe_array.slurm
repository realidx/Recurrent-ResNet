#!/bin/bash
#SBATCH --job-name=u4_baseline_paper
#SBATCH --output=logs/u4_baseline_paper_%A_%a.out
#SBATCH --error=logs/u4_baseline_paper_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=h100-47
#SBATCH --partition=gpu-long
#
# Paper-style U4 baseline run with the standard untied ResNet encoder.
# - Environment: ant_u4_maze (train=eval)
# - Recipe: 100M env steps, 512 envs, batch size 512, UTD 40
# - Model: untied_resnet baseline on actor and critic
# - Width: 256
# - Depth: 16 (maps to 4 residual blocks in the current implementation)
# - W&B logging happens once per epoch, so NUM_EPOCHS controls logging cadence.
#
# Usage:
#   sbatch --array=0-0 job_u4_baseline_paper_recipe_array.slurm
#
# Optional:
#   WANDB_ENTITY=<your-team-or-user> sbatch --array=0-0 job_u4_baseline_paper_recipe_array.slurm

set -e

mkdir -p logs
OUT_DIR="results/u4_baseline_paper_recipe"
mkdir -p "$OUT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate online

export WANDB_DIR="${SLURM_TMPDIR:-/tmp}"

ENV_ID="ant_u4_maze"
EVAL_ENV_ID="ant_u4_maze"
SEEDS=(1001)

NUM_EPOCHS="${NUM_EPOCHS:-100}"
TIME_PRINT_INTERVAL_EPOCHS="${TIME_PRINT_INTERVAL_EPOCHS:-10}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-recurrent_resnet_u4}"
WANDB_GROUP="${WANDB_GROUP:-u4_baseline_paper_recipe}"

WANDB_ARGS=(
  --wandb-mode online
  --wandb-project-name "$WANDB_PROJECT_NAME"
  --wandb-group "$WANDB_GROUP"
)
if [ -n "${WANDB_ENTITY:-}" ]; then
  WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

task_id=${SLURM_ARRAY_TASK_ID:-0}
if [ "$task_id" -ge "${#SEEDS[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$task_id for ${#SEEDS[@]} seeds" >&2
  exit 2
fi
SEED="${SEEDS[$task_id]}"

NAME="u4_paper_recipe_baseline_d16_w256"
NODE="$(hostname)"
GPU_NAME="$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo N/A)"

LOG="logs/${NAME}_s${SEED}_${SLURM_JOB_ID:-local}_${task_id}.log"
OUT="${OUT_DIR}/${NAME}_s${SEED}_${SLURM_JOB_ID:-local}_${task_id}.csv"
echo "config,seed,node,gpu_name,git_rev,args_sha1,actor_width,critic_width,recur_steps,env_steps_final,time_at_goal_final,success_rate_final,success_any_final,eval_reward_final,env_steps_tailmean,time_at_goal_tailmean,success_rate_tailmean,success_any_tailmean,eval_reward_tailmean,env_steps_best,time_at_goal_best,success_rate_best,success_any_best,eval_reward_best,param_count,contrastive_loss,pos_dist,logits_pos,logits_neg,training_time_hours" > "$OUT"

echo "[$(date +%H:%M:%S)] Running $NAME..."
set +e
python scripts/crl_train.py \
  --exp-name "$NAME" \
  --seed "$SEED" \
  --no-capture-vis \
  --no-checkpoint \
  --total-env-steps 100000000 \
  --num-epochs "$NUM_EPOCHS" \
  --num-envs 512 \
  --batch-size 512 \
  --min-replay-size 1000 \
  --max-replay-size 10000 \
  --episode-length 1000 \
  --num-sgd-batches-per-training-step 40 \
  --logsumexp-penalty-coeff 0.1 \
  --actor-lr 0.0003 \
  --critic-lr 0.0003 \
  --env-id "$ENV_ID" \
  --eval-env-id "$EVAL_ENV_ID" \
  --compact-metrics \
  --summary-tail-epochs 10 \
  --no-print-args \
  --time-print-interval-epochs "$TIME_PRINT_INTERVAL_EPOCHS" \
  --wandb-dir "$WANDB_DIR" \
  --actor-network-width 256 \
  --critic-network-width 256 \
  --actor-depth 16 \
  --critic-depth 16 \
  --encoder-type untied_resnet \
  --recur-apply-to-actor 0 \
  --recur-apply-to-critic 0 \
  "${WANDB_ARGS[@]}" \
  > "$LOG" 2>&1
run_status=$?
set -e
if [ "$run_status" -ne 0 ]; then
  echo "[$(date +%H:%M:%S)] WARNING: training exited non-zero ($run_status). Writing partial CSV row." >> "$LOG"
fi

final_line=$(grep "final_summary:" "$LOG" | tail -1 || true)
tail_line=$(grep "tail_mean_summary:" "$LOG" | tail -1 || true)
best_line=$(grep "best_summary:" "$LOG" | tail -1 || true)
if [ -z "$final_line" ]; then
  final_line=$(grep "summary:" "$LOG" | tail -1 || true)
fi
if [ -z "$tail_line" ]; then
  tail_line="$final_line"
fi
if [ -z "$best_line" ]; then
  best_line="$final_line"
fi

env_steps_final=$(echo "$final_line" | grep -oP "'training/envsteps': \\K[0-9.]+|'training/envsteps': \\Knan" || echo "N/A")
time_at_goal_final=$(echo "$final_line" | grep -oP "'eval/time_at_goal': \\K[0-9.]+|'eval/time_at_goal': \\Knan" || echo "N/A")
success_rate_final=$(echo "$final_line" | grep -oP "'eval/success_rate': \\K[0-9.]+|'eval/success_rate': \\Knan" || echo "N/A")
success_any_final=$(echo "$final_line" | grep -oP "'eval/episode_success_any': \\K[0-9.]+|'eval/episode_success_any': \\Knan" || echo "N/A")
eval_reward_final=$(echo "$final_line" | grep -oP "'eval/episode_reward': \\K[-0-9.]+|'eval/episode_reward': \\Knan" || echo "N/A")

env_steps_tailmean=$(echo "$tail_line" | grep -oP "'training/envsteps': \\K[0-9.]+|'training/envsteps': \\Knan" || echo "N/A")
time_at_goal_tailmean=$(echo "$tail_line" | grep -oP "'eval/time_at_goal': \\K[0-9.]+|'eval/time_at_goal': \\Knan" || echo "N/A")
success_rate_tailmean=$(echo "$tail_line" | grep -oP "'eval/success_rate': \\K[0-9.]+|'eval/success_rate': \\Knan" || echo "N/A")
success_any_tailmean=$(echo "$tail_line" | grep -oP "'eval/episode_success_any': \\K[0-9.]+|'eval/episode_success_any': \\Knan" || echo "N/A")
eval_reward_tailmean=$(echo "$tail_line" | grep -oP "'eval/episode_reward': \\K[-0-9.]+|'eval/episode_reward': \\Knan" || echo "N/A")

env_steps_best=$(echo "$best_line" | grep -oP "'training/envsteps': \\K[0-9.]+|'training/envsteps': \\Knan" || echo "N/A")
time_at_goal_best=$(echo "$best_line" | grep -oP "'eval/time_at_goal': \\K[0-9.]+|'eval/time_at_goal': \\Knan" || echo "N/A")
success_rate_best=$(echo "$best_line" | grep -oP "'eval/success_rate': \\K[0-9.]+|'eval/success_rate': \\Knan" || echo "N/A")
success_any_best=$(echo "$best_line" | grep -oP "'eval/episode_success_any': \\K[0-9.]+|'eval/episode_success_any': \\Knan" || echo "N/A")
eval_reward_best=$(echo "$best_line" | grep -oP "'eval/episode_reward': \\K[-0-9.]+|'eval/episode_reward': \\Knan" || echo "N/A")

param_count=$(echo "$final_line" | grep -oP "'model/param_count': \\K[0-9.]+|'model/param_count': \\Knan" || echo "N/A")
contrastive_loss=$(echo "$tail_line" | grep -oP "'training/contrastive_loss': \\K[0-9.]+|'training/contrastive_loss': \\Knan" || echo "N/A")
pos_dist=$(echo "$tail_line" | grep -oP "'training/pos_dist': \\K[0-9.]+|'training/pos_dist': \\Knan" || echo "N/A")
logits_pos=$(echo "$tail_line" | grep -oP "'training/logits_pos': \\K[-0-9.]+|'training/logits_pos': \\Knan" || echo "N/A")
logits_neg=$(echo "$tail_line" | grep -oP "'training/logits_neg': \\K[-0-9.]+|'training/logits_neg': \\Knan" || echo "N/A")
git_rev=$(echo "$final_line" | grep -oP "'meta/git_rev': '\\K[^']+" || echo "N/A")
args_sha1=$(echo "$final_line" | grep -oP "'meta/args_sha1': '\\K[^']+" || echo "N/A")
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

echo "$NAME,$SEED,$NODE,$GPU_NAME,$git_rev,$args_sha1,256,256,0,$env_steps_final,$time_at_goal_final,$success_rate_final,$success_any_final,$eval_reward_final,$env_steps_tailmean,$time_at_goal_tailmean,$success_rate_tailmean,$success_any_tailmean,$eval_reward_tailmean,$env_steps_best,$time_at_goal_best,$success_rate_best,$success_any_best,$eval_reward_best,$param_count,$contrastive_loss,$pos_dist,$logits_pos,$logits_neg,$train_time" >> "$OUT"
echo "[$(date +%H:%M:%S)] Done. Row written to $OUT"
exit "$run_status"
