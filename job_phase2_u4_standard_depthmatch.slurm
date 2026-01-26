#!/bin/bash
#SBATCH --job-name=phase2_u4_depthmatch
#SBATCH --output=logs/phase2_u4_%j.out
#SBATCH --error=logs/phase2_u4_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

# ============================================================================
# Phase 2 (U4 standard): depth-matched untied vs tied-critic (1 seed)
# ============================================================================
# Train on U4, evaluate on U4 (no explicit generalization split).
#
# Untied baselines (paper-style depth scaling within this repo):
#   - encoder_type=untied_resnet, depth D ∈ {16, 32, 64}
#   - depth D is total Dense layers (4-layer residual blocks)
#
# Tied-critic variants (new architecture in critic only, actor untied_resnet):
#   - D=16: critic 1×4  (steps=4)
#   - D=32: critic 2×4  (steps=8)
#   - D=64: critic 4×4  (steps=16)
# where "steps" are TiedMLP block applications and we map depth D = 4 * steps.
# ============================================================================

set -e

mkdir -p logs
mkdir -p results/phase2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate recurrent

export WANDB_MODE=disabled
export WANDB_DIR="${SLURM_TMPDIR:-/tmp}"

START_TIME=$(date +%s)
echo "=============================================="
echo "Phase 2: U4 standard depth-matched runs"
echo "Start time: $(date)"
echo "=============================================="

ENV_ID="ant_u4_maze"
EVAL_ENV_ID="ant_u4_maze"
SEED=1000

RAW_RESULTS_FILE="results/phase2/u4_depthmatch_${SLURM_JOB_ID:-local}.csv"
echo "config,seed,actor_depth,critic_mode,critic_depth,critic_blocks,critic_iters,critic_steps,env_steps,success_rate,success_any,param_count,logits_pos,logits_neg,pos_dist,eval_reward,training_time_hours" > "$RAW_RESULTS_FILE"

extract_metrics() {
    local log_file=$1
    local config=$2
    local seed=$3
    local actor_depth=$4
    local critic_mode=$5
    local critic_depth=$6
    local c_blocks=$7
    local c_iters=$8
    local c_steps=$9

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

    echo "$config,$seed,$actor_depth,$critic_mode,$critic_depth,$c_blocks,$c_iters,$c_steps,$env_steps,$success_rate,$success_any,$param_count,$logits_pos,$logits_neg,$pos_dist,$eval_reward,$train_time" >> "$RAW_RESULTS_FILE"
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

run_untied() {
    local depth=$1
    local config="u4_untied_resnet_d${depth}"
    local log_file="logs/${config}_${SLURM_JOB_ID:-local}.log"

    echo ""
    echo "=============================================="
    echo "[$(date +%H:%M:%S)] Starting $config..."
    echo "Actor+Critic: untied_resnet depth=$depth"
    echo "=============================================="

    python scripts/crl_train.py \
        --exp-name "$config" \
        $COMMON_ARGS \
        --encoder-type untied_resnet \
        --recur-apply-to-actor 0 \
        --recur-apply-to-critic 0 \
        --actor-depth "$depth" \
        --critic-depth "$depth" \
        > "$log_file" 2>&1

    extract_metrics "$log_file" "$config" "$SEED" "$depth" "untied_resnet" "$depth" "" "" ""
    echo "[$(date +%H:%M:%S)] $config complete!"
}

run_tied_critic() {
    local depth=$1
    local c_blocks=$2
    local c_iters=$3
    local c_steps=$4
    local config="u4_tiedcritic_d${depth}_C${c_blocks}x${c_iters}"
    local log_file="logs/${config}_${SLURM_JOB_ID:-local}.log"

    echo ""
    echo "=============================================="
    echo "[$(date +%H:%M:%S)] Starting $config..."
    echo "Actor: untied_resnet depth=$depth | Critic: tied_mlp ${c_blocks}×${c_iters} (steps=${c_steps})"
    echo "=============================================="

    python scripts/crl_train.py \
        --exp-name "$config" \
        $COMMON_ARGS \
        --actor-depth "$depth" \
        --critic-depth "$depth" \
        --encoder-type tied_mlp \
        --recur-apply-to-actor 0 \
        --recur-apply-to-critic 1 \
        --critic-recur-steps "$c_steps" \
        $TIED_MLP_ARGS \
        --critic-tied-mlp-blocks "$c_blocks" \
        --critic-tied-mlp-iters "$c_iters" \
        > "$log_file" 2>&1

    extract_metrics "$log_file" "$config" "$SEED" "$depth" "tied_mlp_critic" "$depth" "$c_blocks" "$c_iters" "$c_steps"
    echo "[$(date +%H:%M:%S)] $config complete!"
}

# Untied depth scaling (D=16,32,64)
run_untied 16
run_untied 32
run_untied 64

# Tied-critic depth-matched (D=16,32,64)
run_tied_critic 16 1 4 4
run_tied_critic 32 2 4 8
run_tied_critic 64 4 4 16

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$(echo "scale=2; $ELAPSED / 3600" | bc)

echo ""
echo "=============================================="
echo "Phase 2 U4 standard runs complete!"
echo "=============================================="
echo "Total runtime: ${ELAPSED_HOURS} hours"
echo "Results saved to: $RAW_RESULTS_FILE"
echo ""
echo "Final Comparison Table:"
echo "=============================================="
column -t -s',' "$RAW_RESULTS_FILE"
echo "=============================================="
