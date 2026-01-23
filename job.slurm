#!/bin/bash
#SBATCH --job-name=phase1_stitching
#SBATCH --output=logs/phase1_%j.out
#SBATCH --error=logs/phase1_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

# ============================================================================
# Phase 1: Stitching Reproduction - Baseline Experiments
# ============================================================================
# This script runs depth 4, 16, and 64 experiments sequentially to reproduce
# the "stitching" phenomenon from "1000 Layer Networks for Self-Supervised RL"
#
# Training: ant_u4_maze_gen_1f2f3f (short paths: 1-3 units)
# Evaluation: ant_u4_maze (full maze: ~6 units)
#
# Expected results (from paper Figure 11):
#   - D=4:  Poor stitching, low success rate on long paths
#   - D=16: Moderate stitching capability
#   - D=64: Good stitching, high success rate on long paths
# ============================================================================

set -e  # Exit on error

# Create logs directory
mkdir -p logs
mkdir -p results/phase1

# Activate conda environment
# Initialize conda (adjust path if your conda is installed elsewhere)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate recurrent

# Record start time
START_TIME=$(date +%s)
echo "=============================================="
echo "Phase 1: Stitching Reproduction Experiments"
echo "Start time: $(date)"
echo "=============================================="

# Common arguments for Phase 1
ENV_ID="ant_u4_maze_gen_1f2f3f"
EVAL_ENV_ID="ant_u4_maze"
SEED=1000

# Results file for comparison
RESULTS_FILE="results/phase1/comparison_${SLURM_JOB_ID:-local}.csv"
echo "depth,env_steps,success_rate,success_hard_rate,eval_reward,training_time_hours" > $RESULTS_FILE

# Function to extract final metrics from log
extract_metrics() {
    local log_file=$1
    local depth=$2

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

    echo "$depth,$env_steps,$success_rate,$success_hard,$eval_reward,$train_time" >> $RESULTS_FILE
}

# ============================================================================
# Run D=4 Baseline
# ============================================================================
echo ""
echo "=============================================="
echo "[$(date +%H:%M:%S)] Starting D=4 baseline..."
echo "=============================================="

D4_LOG="logs/phase1_d4_${SLURM_JOB_ID:-local}.log"

python scripts/crl_train.py \
    --config configs/crl/phase1_baseline_d4.yaml \
    --env-id $ENV_ID \
    --eval-env-id $EVAL_ENV_ID \
    --seed $SEED \
    --compact-metrics \
    --summary-tail-epochs 10 \
    2>&1 | tee $D4_LOG

echo "[$(date +%H:%M:%S)] D=4 baseline complete!"
extract_metrics $D4_LOG 4

# ============================================================================
# Run D=16 Baseline
# ============================================================================
echo ""
echo "=============================================="
echo "[$(date +%H:%M:%S)] Starting D=16 baseline..."
echo "=============================================="

D16_LOG="logs/phase1_d16_${SLURM_JOB_ID:-local}.log"

python scripts/crl_train.py \
    --config configs/crl/phase1_baseline_d16.yaml \
    --env-id $ENV_ID \
    --eval-env-id $EVAL_ENV_ID \
    --seed $SEED \
    --compact-metrics \
    --summary-tail-epochs 10 \
    2>&1 | tee $D16_LOG

echo "[$(date +%H:%M:%S)] D=16 baseline complete!"
extract_metrics $D16_LOG 16

# ============================================================================
# Run D=64 Baseline
# ============================================================================
echo ""
echo "=============================================="
echo "[$(date +%H:%M:%S)] Starting D=64 baseline..."
echo "=============================================="

D64_LOG="logs/phase1_d64_${SLURM_JOB_ID:-local}.log"

python scripts/crl_train.py \
    --config configs/crl/phase1_baseline_d64.yaml \
    --env-id $ENV_ID \
    --eval-env-id $EVAL_ENV_ID \
    --seed $SEED \
    --compact-metrics \
    --summary-tail-epochs 10 \
    2>&1 | tee $D64_LOG

echo "[$(date +%H:%M:%S)] D=64 baseline complete!"
extract_metrics $D64_LOG 64

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$(echo "scale=2; $ELAPSED / 3600" | bc)

echo ""
echo "=============================================="
echo "Phase 1 Experiments Complete!"
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
echo "Expected trend (from paper):"
echo "  D=4:  Low success rate (poor stitching)"
echo "  D=16: Moderate success rate"
echo "  D=64: High success rate (good stitching)"
echo ""
echo "If D=64 >> D=4 in success_rate, stitching is confirmed!"
echo "=============================================="
