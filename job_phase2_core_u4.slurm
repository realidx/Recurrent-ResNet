#!/bin/bash
#SBATCH --job-name=phase2_core_u4
#SBATCH --output=logs/phase2_core_%j.out
#SBATCH --error=logs/phase2_core_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

# ============================================================================
# Phase 2 Core Comparison (U4, 1 seed)
# ============================================================================
# Goal: Compare a TIED critic (1×K) vs an UNTIED/STACKED critic (K×1) at the
# same effective step budget K, while keeping the actor fixed and untied.
#
# Actor: untied_resnet (fixed depth/width)
# Critic: tied_mlp family (SwiGLU + FiLM + LayerScale), varying blocks/iters
#
# Runs:
#   K=4:  critic 1×4 vs 4×1
#   K=8:  critic 1×8 vs 8×1
#   K=16: critic 1×16 vs 16×1
# ============================================================================

set -e

mkdir -p logs
mkdir -p results/phase2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate recurrent

export WANDB_MODE=disabled
export WANDB_DIR="${SLURM_TMPDIR:-/tmp}"

ENV_ID="ant_u4_maze_gen_1f2f3f"
EVAL_ENV_ID="ant_u4_maze"
SEED=1000

# Keep actor fixed/untied (ResNet-style). Depth is total Dense layers.
ACTOR_DEPTH=16

RAW_RESULTS_FILE="results/phase2/core_u4_${SLURM_JOB_ID:-local}_raw.csv"
RESULTS_FILE="results/phase2/core_u4_${SLURM_JOB_ID:-local}.csv"
echo "config,seed,actor_depth,critic_blocks,critic_iters,critic_steps,env_steps,success_rate,success_any,param_count,logits_pos,logits_neg,pos_dist,eval_reward,training_time_hours" > "$RAW_RESULTS_FILE"

extract_metrics() {
    local log_file=$1
    local config=$2
    local seed=$3
    local actor_depth=$4
    local c_blocks=$5
    local c_iters=$6
    local c_steps=$7

    local last_summary
    last_summary=$(grep "summary:" "$log_file" | tail -1)

    local success_rate
    success_rate=$(echo "$last_summary" | grep -oP "'eval/success_rate': \\K[0-9.]+|'eval/success_rate': \\Knan" || echo "N/A")
    local success_any
    success_any=$(echo "$last_summary" | grep -oP "'eval/episode_success_any': \\K[0-9.]+|'eval/episode_success_any': \\Knan" || echo "N/A")
    local eval_reward
    eval_reward=$(echo "$last_summary" | grep -oP "'eval/episode_reward': \\K[-0-9.]+|'eval/episode_reward': \\Knan" || echo "N/A")
    local env_steps
    env_steps=$(echo "$last_summary" | grep -oP "'training/envsteps': \\K[0-9.]+|'training/envsteps': \\Knan" || echo "N/A")
    local train_time
    train_time=$(echo "$last_summary" | grep -oP "'training/walltime': \\K[0-9.]+" || echo "N/A")
    local param_count
    param_count=$(echo "$last_summary" | grep -oP "'model/param_count': \\K[0-9.]+|'model/param_count': \\Knan" || echo "N/A")
    local logits_pos
    logits_pos=$(echo "$last_summary" | grep -oP "'training/logits_pos': \\K[-0-9.]+|'training/logits_pos': \\Knan" || echo "N/A")
    local logits_neg
    logits_neg=$(echo "$last_summary" | grep -oP "'training/logits_neg': \\K[-0-9.]+|'training/logits_neg': \\Knan" || echo "N/A")
    local pos_dist
    pos_dist=$(echo "$last_summary" | grep -oP "'training/pos_dist': \\K[0-9.]+|'training/pos_dist': \\Knan" || echo "N/A")

    if [ "$train_time" != "N/A" ]; then
        train_time=$(echo "scale=2; $train_time / 3600" | bc)
    fi

    echo "$config,$seed,$actor_depth,$c_blocks,$c_iters,$c_steps,$env_steps,$success_rate,$success_any,$param_count,$logits_pos,$logits_neg,$pos_dist,$eval_reward,$train_time" >> "$RAW_RESULTS_FILE"
}

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

run_one() {
    local config=$1
    local c_blocks=$2
    local c_iters=$3
    local c_steps=$4
    local log_file=$5

    echo ""
    echo "=============================================="
    echo "[$(date +%H:%M:%S)] Starting $config..."
    echo "Actor: untied_resnet depth=$ACTOR_DEPTH | Critic: tied_mlp $c_blocks×$c_iters (steps=$c_steps)"
    echo "=============================================="

    python scripts/crl_train.py \
        --exp-name "$config" \
        $COMMON_ARGS \
        --actor-depth "$ACTOR_DEPTH" \
        --encoder-type tied_mlp \
        --recur-apply-to-actor 0 \
        --recur-apply-to-critic 1 \
        --critic-recur-steps "$c_steps" \
        $TIED_MLP_ARGS \
        --critic-tied-mlp-blocks "$c_blocks" \
        --critic-tied-mlp-iters "$c_iters" \
        > "$log_file" 2>&1

    extract_metrics "$log_file" "$config" "$SEED" "$ACTOR_DEPTH" "$c_blocks" "$c_iters" "$c_steps"
    echo "[$(date +%H:%M:%S)] $config complete!"
}

# K=4
run_one "core_u4_K4_critic_tied_1x4" 1 4 4 "logs/core_u4_K4_critic_tied_1x4_${SLURM_JOB_ID:-local}.log"
run_one "core_u4_K4_critic_untied_4x1" 4 1 4 "logs/core_u4_K4_critic_untied_4x1_${SLURM_JOB_ID:-local}.log"

# K=8
run_one "core_u4_K8_critic_tied_1x8" 1 8 8 "logs/core_u4_K8_critic_tied_1x8_${SLURM_JOB_ID:-local}.log"
run_one "core_u4_K8_critic_untied_8x1" 8 1 8 "logs/core_u4_K8_critic_untied_8x1_${SLURM_JOB_ID:-local}.log"

# K=16
run_one "core_u4_K16_critic_tied_1x16" 1 16 16 "logs/core_u4_K16_critic_tied_1x16_${SLURM_JOB_ID:-local}.log"
run_one "core_u4_K16_critic_untied_16x1" 16 1 16 "logs/core_u4_K16_critic_untied_16x1_${SLURM_JOB_ID:-local}.log"

python - <<'PY'
import csv
from pathlib import Path

raw_path = Path(r"""'"$RAW_RESULTS_FILE"'""")
out_path = Path(r"""'"$RESULTS_FILE"'""")

rows = []
with raw_path.open(newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

out_fieldnames = [
    "config",
    "seed",
    "actor_depth",
    "critic_blocks",
    "critic_iters",
    "critic_steps",
    "env_steps",
    "success_rate",
    "success_any",
    "param_count",
    "logits_pos",
    "logits_neg",
    "pos_dist",
    "eval_reward",
    "training_time_hours",
]
with out_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out_fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)
PY

echo ""
echo "=============================================="
echo "Core U4 Experiments Complete!"
echo "=============================================="
echo "Raw results:     $RAW_RESULTS_FILE"
echo "Final results:   $RESULTS_FILE"
echo ""
echo "Final Comparison Table:"
echo "=============================================="
column -t -s',' "$RESULTS_FILE"
echo "=============================================="

