#!/bin/bash
#SBATCH --job-name=phase2_tied_mlp
#SBATCH --output=logs/phase2_%j.out
#SBATCH --error=logs/phase2_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

# ============================================================================
# Phase 2: Tied-MLP Experiments
# ============================================================================
# This script runs Loop-1, Loop-2, and Loop-4 experiments to evaluate
# weight-tied MLP architectures with SwiGLU, FiLM conditioning, and LayerScale.
#
# All configurations have the same compute (4 block applications):
#   - Loop-1: 1 block × 4 iterations (fully tied)
#   - Loop-2: 2 blocks × 2 iterations (partially tied)
#   - Loop-4: 4 blocks × 1 iteration (untied, equivalent to D=16 baseline)
#
# Training: ant_u4_maze_gen_1f2f3f (short paths: 1-3 units)
# Evaluation: ant_u4_maze (full maze: ~6 units)
#
# Architecture per block:
#   u = (1 + γ_k) ⊙ LayerNorm(x) + β_k   # FiLM conditioning on step k
#   a = W_a @ u                           # SwiGLU gate projection
#   g = W_g @ u                           # SwiGLU value projection
#   h = a ⊙ SiLU(g)                       # SwiGLU activation
#   Δ = W_o @ h                           # Output projection
#   x = x + α * Δ                         # LayerScale residual
# ============================================================================

set -e  # Exit on error

# Create logs directory
mkdir -p logs
mkdir -p results/phase2

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate recurrent

# Record start time
START_TIME=$(date +%s)
echo "=============================================="
echo "Phase 2: Tied-MLP Experiments"
echo "Start time: $(date)"
echo "=============================================="

# Common arguments for Phase 2
ENV_ID="ant_u4_maze_gen_1f2f3f"
EVAL_ENV_ID="ant_u4_maze"
SEED=1000

# Results file for comparison
RESULTS_FILE="results/phase2/comparison_${SLURM_JOB_ID:-local}.csv"
echo "config,num_blocks,num_iters,env_steps,success_rate,success_hard_rate,eval_reward,training_time_hours" > $RESULTS_FILE

# Function to extract final metrics from log
extract_metrics() {
    local log_file=$1
    local config=$2
    local num_blocks=$3
    local num_iters=$4

    # Get the last summary line
    local last_summary=$(grep "summary:" "$log_file" | tail -1)

    # Extract metrics using grep/sed
    local success_rate=$(echo "$last_summary" | grep -oP "'eval/success_rate': \K[0-9.]+|'eval/success_rate': \Knan" || echo "N/A")
    local success_hard=$(echo "$last_summary" | grep -oP "'eval/success_hard_rate': \K[0-9.]+|'eval/success_hard_rate': \Knan" || echo "N/A")
    local eval_reward=$(echo "$last_summary" | grep -oP "'eval/episode_reward': \K[-0-9.]+|'eval/episode_reward': \Knan" || echo "N/A")
    local env_steps=$(echo "$last_summary" | grep -oP "'training/envsteps': \K[0-9.]+|'training/envsteps': \Knan" || echo "N/A")
    local train_time=$(echo "$last_summary" | grep -oP "'training/walltime': \K[0-9.]+" || echo "N/A")

    # Convert walltime to hours if available
    if [ "$train_time" != "N/A" ]; then
        train_time=$(echo "scale=2; $train_time / 3600" | bc)
    fi

    echo "$config,$num_blocks,$num_iters,$env_steps,$success_rate,$success_hard,$eval_reward,$train_time" >> $RESULTS_FILE
}

# Common training arguments
COMMON_ARGS="--no-track \
    --no-capture-vis \
    --checkpoint \
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
    --encoder-type tied_mlp \
    --env-id $ENV_ID \
    --eval-env-id $EVAL_ENV_ID \
    --seed $SEED \
    --compact-metrics \
    --summary-tail-epochs 10"

# Tied-MLP specific arguments (common across all configs)
TIED_MLP_ARGS="--tied-mlp-ffn-mult 2.667 \
    --tied-mlp-layerscale-init 0.01 \
    --tied-mlp-step-embed-dim 0 \
    --tied-mlp-trunc-bptt 0"

# ============================================================================
# Run Loop-1: 1 block × 4 iterations (fully tied)
# ============================================================================
echo ""
echo "=============================================="
echo "[$(date +%H:%M:%S)] Starting Loop-1 (1 block × 4 iters)..."
echo "=============================================="

LOOP1_LOG="logs/phase2_loop1_${SLURM_JOB_ID:-local}.log"

python scripts/crl_train.py \
    --exp-name phase2_loop1 \
    $COMMON_ARGS \
    $TIED_MLP_ARGS \
    --tied-mlp-blocks 1 \
    --tied-mlp-iters 4 \
    2>&1 | tee $LOOP1_LOG

echo "[$(date +%H:%M:%S)] Loop-1 complete!"
extract_metrics $LOOP1_LOG "loop1" 1 4

# ============================================================================
# Run Loop-2: 2 blocks × 2 iterations (partially tied)
# ============================================================================
echo ""
echo "=============================================="
echo "[$(date +%H:%M:%S)] Starting Loop-2 (2 blocks × 2 iters)..."
echo "=============================================="

LOOP2_LOG="logs/phase2_loop2_${SLURM_JOB_ID:-local}.log"

python scripts/crl_train.py \
    --exp-name phase2_loop2 \
    $COMMON_ARGS \
    $TIED_MLP_ARGS \
    --tied-mlp-blocks 2 \
    --tied-mlp-iters 2 \
    2>&1 | tee $LOOP2_LOG

echo "[$(date +%H:%M:%S)] Loop-2 complete!"
extract_metrics $LOOP2_LOG "loop2" 2 2

# ============================================================================
# Run Loop-4: 4 blocks × 1 iteration (untied, equivalent to baseline)
# ============================================================================
echo ""
echo "=============================================="
echo "[$(date +%H:%M:%S)] Starting Loop-4 (4 blocks × 1 iter)..."
echo "=============================================="

LOOP4_LOG="logs/phase2_loop4_${SLURM_JOB_ID:-local}.log"

python scripts/crl_train.py \
    --exp-name phase2_loop4 \
    $COMMON_ARGS \
    $TIED_MLP_ARGS \
    --tied-mlp-blocks 4 \
    --tied-mlp-iters 1 \
    2>&1 | tee $LOOP4_LOG

echo "[$(date +%H:%M:%S)] Loop-4 complete!"
extract_metrics $LOOP4_LOG "loop4" 4 1

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$(echo "scale=2; $ELAPSED / 3600" | bc)

echo ""
echo "=============================================="
echo "Phase 2 Experiments Complete!"
echo "=============================================="
echo "Total runtime: ${ELAPSED_HOURS} hours"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Final Comparison Table:"
echo "=============================================="
column -t -s',' $RESULTS_FILE
echo "=============================================="
echo ""
echo "Interpretation:"
echo "  Loop-1: Fully tied (1 block × 4 iters) - tests max weight sharing"
echo "  Loop-2: Partially tied (2 blocks × 2 iters) - balance between tying and capacity"
echo "  Loop-4: Untied (4 blocks × 1 iter) - equivalent compute to D=16 baseline"
echo ""
echo "If Loop-1 or Loop-2 match or exceed Loop-4, weight tying is beneficial!"
echo "=============================================="
