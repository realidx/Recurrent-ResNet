#!/bin/bash
#SBATCH --job-name=phase3_u4_k8
#SBATCH --output=logs/phase3_u4_k8_%A_%a.out
#SBATCH --error=logs/phase3_u4_k8_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#
# Phase 3: K=8 tie-patterns, stability + parameter-efficiency
# - Train=eval on U4 (no generalization split)
# - Actor: untied_resnet depth=32 (fixed)
# - Critic: tied_mlp applied to critic only, with effective steps=8
#   Patterns (blocks×iters): {1×8, 2×4, 4×2, 8×1}
# - No truncated BPTT (tbptt=0) because stop_gradient TBPTT breaks learning here.
# - Includes a small width sweep for 8×1 to get a param-matched stacked baseline.
#
# Usage:
#   sbatch --array=0-17 job_phase3_u4_k8_tiepattern_array.slurm
#   sbatch --array=0-17%6 job_phase3_u4_k8_tiepattern_tbptt_array.slurm   # throttle
#
# Optional (pin a GPU type; pick one your cluster supports):
#   sbatch --gres=gpu:h100-47:1 --array=0-23 job_phase3_u4_k8_tiepattern_tbptt_array.slurm
#
# Outputs:
#   - Per-task logs: logs/*.log
#   - Per-task CSV row: results/phase3/

set -e

mkdir -p logs
OUT_DIR="results/phase3"
mkdir -p "$OUT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate recurrent

export WANDB_MODE=disabled
export WANDB_DIR="${SLURM_TMPDIR:-/tmp}"

ENV_ID="ant_u4_maze"
EVAL_ENV_ID="ant_u4_maze"

# Adjust if you want different seeds.
SEEDS=(1001 1003 1004)

# Config set (6):
#   - core tie patterns at width=256: 1×8, 2×4, 4×2, 8×1
#   - param-match sweep for 8×1: width ∈ {192, 160}
NUM_CONFIGS=6

COMMON_ARGS="--no-track \
  --no-capture-vis \
  --no-checkpoint \
  --total-env-steps 100000000 \
  --num-envs 512 \
  --batch-size 512 \
  --min-replay-size 1000 \
  --max-replay-size 10000 \
  --episode-length 1000 \
  --num-sgd-batches-per-training-step 40 \
  --logsumexp-penalty-coeff 0.1 \
  --actor-lr 0.0003 \
  --critic-lr 0.0003 \
  --env-id $ENV_ID \
  --eval-env-id $EVAL_ENV_ID \
  --compact-metrics \
  --summary-tail-epochs 10 \
  --no-print-args \
  --no-log-flush \
  --time-print-interval-epochs 10 \
  --wandb-dir $WANDB_DIR \
  --actor-network-width 256 \
  --actor-depth 32 \
  --critic-depth 32 \
  --encoder-type tied_mlp \
  --recur-steps 8 \
  --recur-apply-to-actor 0 \
  --recur-apply-to-critic 1 \
  --critic-recur-steps 8"

task_id=${SLURM_ARRAY_TASK_ID:-0}
seed_idx=$((task_id / NUM_CONFIGS))
cfg_idx=$((task_id % NUM_CONFIGS))
if [ "$seed_idx" -ge "${#SEEDS[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$task_id for ${#SEEDS[@]} seeds and $NUM_CONFIGS configs" >&2
  exit 2
fi
SEED="${SEEDS[$seed_idx]}"

case "$cfg_idx" in
  0) NAME="phase3_u4_d32_k8_C1x8_w256";  CRITIC_BLOCKS=1; CRITIC_ITERS=8; CRITIC_WIDTH=256 ;;
  1) NAME="phase3_u4_d32_k8_C2x4_w256";  CRITIC_BLOCKS=2; CRITIC_ITERS=4; CRITIC_WIDTH=256 ;;
  2) NAME="phase3_u4_d32_k8_C4x2_w256";  CRITIC_BLOCKS=4; CRITIC_ITERS=2; CRITIC_WIDTH=256 ;;
  3) NAME="phase3_u4_d32_k8_C8x1_w256";  CRITIC_BLOCKS=8; CRITIC_ITERS=1; CRITIC_WIDTH=256 ;;
  4) NAME="phase3_u4_d32_k8_C8x1_w192";  CRITIC_BLOCKS=8; CRITIC_ITERS=1; CRITIC_WIDTH=192 ;;
  5) NAME="phase3_u4_d32_k8_C8x1_w160";  CRITIC_BLOCKS=8; CRITIC_ITERS=1; CRITIC_WIDTH=160 ;;
  *) echo "Unknown config index $cfg_idx (expected 0-5)" >&2; exit 2 ;;
esac

NODE="$(hostname)"
GPU_NAME="$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo N/A)"

LOG="logs/${NAME}_s${SEED}_${SLURM_JOB_ID:-local}_${task_id}.log"
OUT="${OUT_DIR}/${NAME}_s${SEED}_${SLURM_JOB_ID:-local}_${task_id}.csv"
echo "config,seed,node,gpu_name,git_rev,args_sha1,critic_width,critic_blocks,critic_iters,critic_steps,env_steps_final,success_rate_final,success_any_final,eval_reward_final,env_steps_tailmean,success_rate_tailmean,success_any_tailmean,eval_reward_tailmean,env_steps_best,success_rate_best,success_any_best,eval_reward_best,param_count,contrastive_loss,pos_dist,logits_pos,logits_neg,training_time_hours" > "$OUT"

echo "[$(date +%H:%M:%S)] Running $NAME..."
set +e
python scripts/crl_train.py \
  --exp-name "$NAME" \
  $COMMON_ARGS \
  --seed "$SEED" \
  --critic-network-width "$CRITIC_WIDTH" \
  --tied-mlp-ffn-mult 2.667 \
  --tied-mlp-layerscale-init 0.01 \
  --tied-mlp-step-embed-dim 0 \
  --tied-mlp-trunc-bptt 0 \
  --critic-tied-mlp-blocks "$CRITIC_BLOCKS" \
  --critic-tied-mlp-iters "$CRITIC_ITERS" \
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
success_rate_final=$(echo "$final_line" | grep -oP "'eval/success_rate': \\K[0-9.]+|'eval/success_rate': \\Knan" || echo "N/A")
success_any_final=$(echo "$final_line" | grep -oP "'eval/episode_success_any': \\K[0-9.]+|'eval/episode_success_any': \\Knan" || echo "N/A")
eval_reward_final=$(echo "$final_line" | grep -oP "'eval/episode_reward': \\K[-0-9.]+|'eval/episode_reward': \\Knan" || echo "N/A")

env_steps_tailmean=$(echo "$tail_line" | grep -oP "'training/envsteps': \\K[0-9.]+|'training/envsteps': \\Knan" || echo "N/A")
success_rate_tailmean=$(echo "$tail_line" | grep -oP "'eval/success_rate': \\K[0-9.]+|'eval/success_rate': \\Knan" || echo "N/A")
success_any_tailmean=$(echo "$tail_line" | grep -oP "'eval/episode_success_any': \\K[0-9.]+|'eval/episode_success_any': \\Knan" || echo "N/A")
eval_reward_tailmean=$(echo "$tail_line" | grep -oP "'eval/episode_reward': \\K[-0-9.]+|'eval/episode_reward': \\Knan" || echo "N/A")

env_steps_best=$(echo "$best_line" | grep -oP "'training/envsteps': \\K[0-9.]+|'training/envsteps': \\Knan" || echo "N/A")
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

echo "$NAME,$SEED,$NODE,$GPU_NAME,$git_rev,$args_sha1,$CRITIC_WIDTH,$CRITIC_BLOCKS,$CRITIC_ITERS,8,$env_steps_final,$success_rate_final,$success_any_final,$eval_reward_final,$env_steps_tailmean,$success_rate_tailmean,$success_any_tailmean,$eval_reward_tailmean,$env_steps_best,$success_rate_best,$success_any_best,$eval_reward_best,$param_count,$contrastive_loss,$pos_dist,$logits_pos,$logits_neg,$train_time" >> "$OUT"
echo "[$(date +%H:%M:%S)] Done. Row written to $OUT"
exit "$run_status"
