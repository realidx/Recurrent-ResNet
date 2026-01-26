#!/bin/bash
#SBATCH --job-name=phase2_u4_array
#SBATCH --output=logs/phase2_u4_array_%A_%a.out
#SBATCH --error=logs/phase2_u4_array_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#
# Usage:
#   sbatch --array=0-5 job_phase2_u4_depthmatch_array.slurm
#
# Each array task runs one config on one GPU, allowing SLURM to schedule many
# tasks concurrently (up to your QOS per-user GPU cap).

set -e

mkdir -p logs
mkdir -p results/phase2/array_u4

source ~/miniconda3/etc/profile.d/conda.sh
conda activate recurrent

export WANDB_MODE=disabled
export WANDB_DIR="${SLURM_TMPDIR:-/tmp}"

ENV_ID="ant_u4_maze"
EVAL_ENV_ID="ant_u4_maze"
SEED=1000

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
  --seed $SEED \
  --compact-metrics \
  --summary-tail-epochs 10 \
  --no-print-args \
  --no-log-flush \
  --time-print-interval-epochs 10 \
  --wandb-dir $WANDB_DIR"

TIED_MLP_ARGS="--tied-mlp-ffn-mult 2.667 \
  --tied-mlp-layerscale-init 0.01 \
  --tied-mlp-step-embed-dim 0 \
  --tied-mlp-trunc-bptt 0"

task_id=${SLURM_ARRAY_TASK_ID:-0}

case "$task_id" in
  0)
    NAME="u4_untied_resnet_d16"
    CMD=(python scripts/crl_train.py --exp-name "$NAME" $COMMON_ARGS \
      --encoder-type untied_resnet --recur-apply-to-actor 0 --recur-apply-to-critic 0 --actor-depth 16 --critic-depth 16)
    ;;
  1)
    NAME="u4_untied_resnet_d32"
    CMD=(python scripts/crl_train.py --exp-name "$NAME" $COMMON_ARGS \
      --encoder-type untied_resnet --recur-apply-to-actor 0 --recur-apply-to-critic 0 --actor-depth 32 --critic-depth 32)
    ;;
  2)
    NAME="u4_untied_resnet_d64"
    CMD=(python scripts/crl_train.py --exp-name "$NAME" $COMMON_ARGS \
      --encoder-type untied_resnet --recur-apply-to-actor 0 --recur-apply-to-critic 0 --actor-depth 64 --critic-depth 64)
    ;;
  3)
    NAME="u4_tiedcritic_d16_C1x4"
    CMD=(python scripts/crl_train.py --exp-name "$NAME" $COMMON_ARGS \
      --actor-depth 16 --critic-depth 16 \
      --encoder-type tied_mlp --recur-apply-to-actor 0 --recur-apply-to-critic 1 \
      --critic-recur-steps 4 $TIED_MLP_ARGS --critic-tied-mlp-blocks 1 --critic-tied-mlp-iters 4)
    ;;
  4)
    NAME="u4_tiedcritic_d32_C2x4"
    CMD=(python scripts/crl_train.py --exp-name "$NAME" $COMMON_ARGS \
      --actor-depth 32 --critic-depth 32 \
      --encoder-type tied_mlp --recur-apply-to-actor 0 --recur-apply-to-critic 1 \
      --critic-recur-steps 8 $TIED_MLP_ARGS --critic-tied-mlp-blocks 2 --critic-tied-mlp-iters 4)
    ;;
  5)
    NAME="u4_tiedcritic_d64_C4x4"
    CMD=(python scripts/crl_train.py --exp-name "$NAME" $COMMON_ARGS \
      --actor-depth 64 --critic-depth 64 \
      --encoder-type tied_mlp --recur-apply-to-actor 0 --recur-apply-to-critic 1 \
      --critic-recur-steps 16 $TIED_MLP_ARGS --critic-tied-mlp-blocks 4 --critic-tied-mlp-iters 4)
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=$task_id (expected 0-5)" >&2
    exit 2
    ;;
esac

LOG="logs/${NAME}_${SLURM_JOB_ID:-local}_${task_id}.log"
OUT="results/phase2/array_u4/${NAME}_${SLURM_JOB_ID:-local}_${task_id}.csv"
echo "config,seed,env_steps,success_rate,success_any,param_count,eval_reward,training_time_hours" > "$OUT"

echo "[$(date +%H:%M:%S)] Running $NAME on 1 GPU..."
"${CMD[@]}" > "$LOG" 2>&1

last_summary=$(grep "summary:" "$LOG" | tail -1 || true)
env_steps=$(echo "$last_summary" | grep -oP "'training/envsteps': \\K[0-9.]+|'training/envsteps': \\Knan" || echo "N/A")
success_rate=$(echo "$last_summary" | grep -oP "'eval/success_rate': \\K[0-9.]+|'eval/success_rate': \\Knan" || echo "N/A")
success_any=$(echo "$last_summary" | grep -oP "'eval/episode_success_any': \\K[0-9.]+|'eval/episode_success_any': \\Knan" || echo "N/A")
param_count=$(echo "$last_summary" | grep -oP "'model/param_count': \\K[0-9.]+|'model/param_count': \\Knan" || echo "N/A")
eval_reward=$(echo "$last_summary" | grep -oP "'eval/episode_reward': \\K[-0-9.]+|'eval/episode_reward': \\Knan" || echo "N/A")
train_time=$(echo "$last_summary" | grep -oP "'training/walltime': \\K[0-9.]+" || echo "N/A")
if [ "$train_time" != "N/A" ]; then
  train_time=$(echo "scale=2; $train_time / 3600" | bc)
fi

echo "$NAME,$SEED,$env_steps,$success_rate,$success_any,$param_count,$eval_reward,$train_time" >> "$OUT"
echo "[$(date +%H:%M:%S)] Done. Row written to $OUT"

