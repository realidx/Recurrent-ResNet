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
# Phase 3: diagnose K=8 tie-patterns (+ trunc BPTT)
# - Train=eval on U4 (no generalization split)
# - Actor: untied_resnet depth=32 (fixed)
# - Critic: tied_mlp applied to critic only, with effective steps=8
#   Patterns (blocks×iters): subset of {1×8, 2×4, 4×2, 8×1}
# - Trunc BPTT sweep: subset of {0, 4}
#
# Usage:
#   sbatch --array=0-14 job_phase3_u4_k8_tiepattern_tbptt_array.slurm
#   sbatch --array=0-14%6 job_phase3_u4_k8_tiepattern_tbptt_array.slurm   # throttle
#
# Optional (pin a GPU type; pick one your cluster supports):
#   sbatch --gres=gpu:h100-47:1 --array=0-23 job_phase3_u4_k8_tiepattern_tbptt_array.slurm
#
# Outputs:
#   - Per-task logs: logs/*.log
#   - Per-task CSV row: results/phase3/array_u4_k8/${SLURM_JOB_ID}/

set -e

mkdir -p logs
OUT_DIR="results/phase3/array_u4_k8/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate recurrent

export WANDB_MODE=disabled
export WANDB_DIR="${SLURM_TMPDIR:-/tmp}"

ENV_ID="ant_u4_maze"
EVAL_ENV_ID="ant_u4_maze"

# Adjust if you want different seeds.
SEEDS=(1001 1003 1004)

# Config set (currently 5):
#   - tbptt0:  2×4, 4×2
#   - tbptt4:  1×8, 2×4, 4×2
NUM_CONFIGS=5

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
  --critic-network-width 256 \
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
  0) NAME="u4_d32_k8_C2x4_tbptt0";  CRITIC_BLOCKS=2; CRITIC_ITERS=4; TBPTT=0 ;;
  1) NAME="u4_d32_k8_C4x2_tbptt0";  CRITIC_BLOCKS=4; CRITIC_ITERS=2; TBPTT=0 ;;
  2) NAME="u4_d32_k8_C1x8_tbptt4";  CRITIC_BLOCKS=1; CRITIC_ITERS=8; TBPTT=4 ;;
  3) NAME="u4_d32_k8_C2x4_tbptt4";  CRITIC_BLOCKS=2; CRITIC_ITERS=4; TBPTT=4 ;;
  4) NAME="u4_d32_k8_C4x2_tbptt4";  CRITIC_BLOCKS=4; CRITIC_ITERS=2; TBPTT=4 ;;
  *) echo "Unknown config index $cfg_idx (expected 0-4)" >&2; exit 2 ;;
esac

NODE="$(hostname)"
GPU_NAME="$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo N/A)"

LOG="logs/${NAME}_s${SEED}_${SLURM_JOB_ID:-local}_${task_id}.log"
OUT="${OUT_DIR}/${NAME}_s${SEED}_${SLURM_JOB_ID:-local}_${task_id}.csv"
echo "config,seed,node,gpu_name,git_rev,args_sha1,critic_blocks,critic_iters,critic_steps,tbptt,env_steps,success_rate,success_any,param_count,contrastive_loss,pos_dist,logits_pos,logits_neg,eval_reward,training_time_hours" > "$OUT"

echo "[$(date +%H:%M:%S)] Running $NAME..."
python scripts/crl_train.py \
  --exp-name "$NAME" \
  $COMMON_ARGS \
  --seed "$SEED" \
  --tied-mlp-ffn-mult 2.667 \
  --tied-mlp-layerscale-init 0.01 \
  --tied-mlp-step-embed-dim 0 \
  --tied-mlp-trunc-bptt "$TBPTT" \
  --critic-tied-mlp-blocks "$CRITIC_BLOCKS" \
  --critic-tied-mlp-iters "$CRITIC_ITERS" \
  > "$LOG" 2>&1

last_summary=$(grep "final_summary:" "$LOG" | tail -1 || true)
if [ -z "$last_summary" ]; then
  last_summary=$(grep "summary:" "$LOG" | tail -1 || true)
fi

env_steps=$(echo "$last_summary" | grep -oP "'training/envsteps': \\K[0-9.]+|'training/envsteps': \\Knan" || echo "N/A")
success_rate=$(echo "$last_summary" | grep -oP "'eval/success_rate': \\K[0-9.]+|'eval/success_rate': \\Knan" || echo "N/A")
success_any=$(echo "$last_summary" | grep -oP "'eval/episode_success_any': \\K[0-9.]+|'eval/episode_success_any': \\Knan" || echo "N/A")
param_count=$(echo "$last_summary" | grep -oP "'model/param_count': \\K[0-9.]+|'model/param_count': \\Knan" || echo "N/A")
contrastive_loss=$(echo "$last_summary" | grep -oP "'training/contrastive_loss': \\K[0-9.]+|'training/contrastive_loss': \\Knan" || echo "N/A")
pos_dist=$(echo "$last_summary" | grep -oP "'training/pos_dist': \\K[0-9.]+|'training/pos_dist': \\Knan" || echo "N/A")
logits_pos=$(echo "$last_summary" | grep -oP "'training/logits_pos': \\K[-0-9.]+|'training/logits_pos': \\Knan" || echo "N/A")
logits_neg=$(echo "$last_summary" | grep -oP "'training/logits_neg': \\K[-0-9.]+|'training/logits_neg': \\Knan" || echo "N/A")
eval_reward=$(echo "$last_summary" | grep -oP "'eval/episode_reward': \\K[-0-9.]+|'eval/episode_reward': \\Knan" || echo "N/A")
git_rev=$(echo "$last_summary" | grep -oP "'meta/git_rev': '\\K[^']+" || echo "N/A")
args_sha1=$(echo "$last_summary" | grep -oP "'meta/args_sha1': '\\K[^']+" || echo "N/A")
train_time=$(echo "$last_summary" | grep -oP "'training/walltime': \\K[0-9.]+" || echo "N/A")
if [ "$train_time" != "N/A" ]; then
  train_time=$(echo "scale=2; $train_time / 3600" | bc)
fi

echo "$NAME,$SEED,$NODE,$GPU_NAME,$git_rev,$args_sha1,$CRITIC_BLOCKS,$CRITIC_ITERS,8,$TBPTT,$env_steps,$success_rate,$success_any,$param_count,$contrastive_loss,$pos_dist,$logits_pos,$logits_neg,$eval_reward,$train_time" >> "$OUT"
echo "[$(date +%H:%M:%S)] Done. Row written to $OUT"
