#!/bin/bash
#SBATCH --job-name=phase2_u4_recurrent
#SBATCH --output=logs/phase2_u4_recurrent_%A_%a.out
#SBATCH --error=logs/phase2_u4_recurrent_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#
# Usage:
#   sbatch --array=0-5 job_phase2_u4_recurrent_d16_d32_array.slurm
#   sbatch --array=0-5%6 job_phase2_u4_recurrent_d16_d32_array.slurm   # throttle
#
# Optional (pin a GPU type; pick one your cluster supports):
#   sbatch --gres=gpu:h100-47:1 --array=0-5 job_phase2_u4_recurrent_d16_d32_array.slurm
#
# Runs recurrent SwiGLU critics at D=16 and D=32 on standard U4 (train=eval).
# - Actor: untied_resnet depth=D
# - Critic: tied_mlp applied to critic only, with one repeated block and K=D/4 steps

set -e

mkdir -p logs
OUT_DIR="results/phase2/array_u4_recurrent/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate recurrent

export WANDB_MODE=disabled
export WANDB_DIR="${SLURM_TMPDIR:-/tmp}"
# Note: XLA determinism flags are version-dependent. Some clusters error out on
# unknown flags (fatal). If your XLA supports determinism, set it explicitly at
# submission time, e.g.:
#   sbatch --export=ALL,XLA_FLAGS="--xla_gpu_deterministic_ops" ...

ENV_ID="ant_u4_maze"
EVAL_ENV_ID="ant_u4_maze"
SEEDS=(1001 1003 1004)
NUM_CONFIGS=2

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
  --wandb-dir $WANDB_DIR"

TIED_MLP_ARGS="--tied-mlp-ffn-mult 2.667 \
  --tied-mlp-layerscale-init 0.01 \
  --tied-mlp-trunc-bptt 0"

task_id=${SLURM_ARRAY_TASK_ID:-0}
seed_idx=$((task_id / NUM_CONFIGS))
cfg_idx=$((task_id % NUM_CONFIGS))
if [ "$seed_idx" -ge "${#SEEDS[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$task_id for ${#SEEDS[@]} seeds and $NUM_CONFIGS configs" >&2
  exit 2
fi
SEED="${SEEDS[$seed_idx]}"

case "$cfg_idx" in
  0)
    NAME="u4_recurrent_d16_k4"
    ACTOR_DEPTH=16
    CRITIC_ITERS=4
    CRITIC_STEPS=4
    ;;
  1)
    NAME="u4_recurrent_d32_k8"
    ACTOR_DEPTH=32
    CRITIC_ITERS=8
    CRITIC_STEPS=8
    ;;
  *)
    echo "Unknown config index $cfg_idx (expected 0-1)" >&2
    exit 2
    ;;
esac

NODE="$(hostname)"
GPU_NAME="$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo N/A)"

LOG="logs/${NAME}_s${SEED}_${SLURM_JOB_ID:-local}_${task_id}.log"
OUT="${OUT_DIR}/${NAME}_s${SEED}_${SLURM_JOB_ID:-local}_${task_id}.csv"
echo "config,seed,node,gpu_name,git_rev,args_sha1,actor_depth,critic_iters,critic_steps,env_steps_final,success_rate_final,success_any_final,eval_reward_final,env_steps_tailmean,success_rate_tailmean,success_any_tailmean,eval_reward_tailmean,param_count,training_time_hours" > "$OUT"

echo "[$(date +%H:%M:%S)] Running $NAME on 1 GPU..."
python scripts/crl_train.py \
  --exp-name "$NAME" \
  $COMMON_ARGS \
  --seed "$SEED" \
  --actor-depth "$ACTOR_DEPTH" \
  --critic-depth "$ACTOR_DEPTH" \
  --encoder-type tied_mlp \
  --recur-apply-to-actor 0 \
  --recur-apply-to-critic 1 \
  --critic-recur-steps "$CRITIC_STEPS" \
  $TIED_MLP_ARGS \
  --critic-tied-mlp-iters "$CRITIC_ITERS" \
  > "$LOG" 2>&1

final_line=$(grep "final_summary:" "$LOG" | tail -1 || true)
tail_line=$(grep "tail_mean_summary:" "$LOG" | tail -1 || true)
if [ -z "$final_line" ]; then
  final_line=$(grep "summary:" "$LOG" | tail -1 || true)
fi
if [ -z "$tail_line" ]; then
  tail_line="$final_line"
fi
env_steps_final=$(echo "$final_line" | grep -oP "'training/envsteps': \\K[0-9.]+|'training/envsteps': \\Knan" || echo "N/A")
success_rate_final=$(echo "$final_line" | grep -oP "'eval/success_rate': \\K[0-9.]+|'eval/success_rate': \\Knan" || echo "N/A")
success_any_final=$(echo "$final_line" | grep -oP "'eval/episode_success_any': \\K[0-9.]+|'eval/episode_success_any': \\Knan" || echo "N/A")
eval_reward_final=$(echo "$final_line" | grep -oP "'eval/episode_reward': \\K[-0-9.]+|'eval/episode_reward': \\Knan" || echo "N/A")

env_steps_tailmean=$(echo "$tail_line" | grep -oP "'training/envsteps': \\K[0-9.]+|'training/envsteps': \\Knan" || echo "N/A")
success_rate_tailmean=$(echo "$tail_line" | grep -oP "'eval/success_rate': \\K[0-9.]+|'eval/success_rate': \\Knan" || echo "N/A")
success_any_tailmean=$(echo "$tail_line" | grep -oP "'eval/episode_success_any': \\K[0-9.]+|'eval/episode_success_any': \\Knan" || echo "N/A")
eval_reward_tailmean=$(echo "$tail_line" | grep -oP "'eval/episode_reward': \\K[-0-9.]+|'eval/episode_reward': \\Knan" || echo "N/A")

param_count=$(echo "$final_line" | grep -oP "'model/param_count': \\K[0-9.]+|'model/param_count': \\Knan" || echo "N/A")
git_rev=$(echo "$final_line" | grep -oP "'meta/git_rev': '\\K[^']+" || echo "N/A")
args_sha1=$(echo "$final_line" | grep -oP "'meta/args_sha1': '\\K[^']+" || echo "N/A")
train_time=$(echo "$final_line" | grep -oP "'training/walltime': \\K[0-9.]+" || echo "N/A")
if [ "$train_time" != "N/A" ]; then
  train_time=$(echo "scale=2; $train_time / 3600" | bc)
fi

echo "$NAME,$SEED,$NODE,$GPU_NAME,$git_rev,$args_sha1,$ACTOR_DEPTH,$CRITIC_ITERS,$CRITIC_STEPS,$env_steps_final,$success_rate_final,$success_any_final,$eval_reward_final,$env_steps_tailmean,$success_rate_tailmean,$success_any_tailmean,$eval_reward_tailmean,$param_count,$train_time" >> "$OUT"
echo "[$(date +%H:%M:%S)] Done. Row written to $OUT"
