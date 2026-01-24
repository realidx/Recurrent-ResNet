#!/bin/bash
#SBATCH --job-name=phase2_critic_partial
#SBATCH --output=logs/phase2_mix_%j.out
#SBATCH --error=logs/phase2_mix_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu

# ============================================================================
# Phase 2 (Critic Partial Tying): Actor fixed, sweep critic tying/steps
# ============================================================================
# Goal: Keep the actor "untied" within the TiedMLP family (4 blocks × 1 iter),
# and sweep critic tying/iterations.
#
# Definitions within the TiedMLP family:
#   - "tied"   = 1 block × 4 iters  (max weight reuse)
#   - "untied" = 4 blocks × 1 iter  (no reuse across blocks)
#
# Actor (fixed):
#   - 4 blocks × 1 iter  (effective steps = 4)
#
# Critic (sweep):
#   - 1×4   (effective steps = 4)
#   - 2×2   (effective steps = 4)
#   - 2×4   (effective steps = 8)
#   - 2×8   (effective steps = 16)
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
echo "Phase 2: Critic Partial Tying (TiedMLP)"
echo "Start time: $(date)"
echo "=============================================="

ENV_ID="ant_u4_maze_gen_1f2f3f"
EVAL_ENV_ID="ant_u4_maze"
SEEDS=(1000 1001 1002)

RAW_RESULTS_FILE="results/phase2/critic_partial_${SLURM_JOB_ID:-local}_raw.csv"
RESULTS_FILE="results/phase2/critic_partial_${SLURM_JOB_ID:-local}.csv"
echo "config,seed,actor_blocks,actor_iters,actor_steps,critic_blocks,critic_iters,critic_steps,env_steps,success_rate,success_any,param_count,logits_pos,logits_neg,pos_dist,eval_reward,training_time_hours" > $RAW_RESULTS_FILE

extract_metrics() {
    local log_file=$1
    local config=$2
    local seed=$3
    local a_blocks=$4
    local a_iters=$5
    local a_steps=$6
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

    echo "$config,$seed,$a_blocks,$a_iters,$a_steps,$c_blocks,$c_iters,$c_steps,$env_steps,$success_rate,$success_any,$param_count,$logits_pos,$logits_neg,$pos_dist,$eval_reward,$train_time" >> $RAW_RESULTS_FILE
}

# Common training arguments
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
    --encoder-type tied_mlp \
    --recur-apply-to-actor 1 \
    --recur-apply-to-critic 1 \
    --recur-steps 4 \
    --recur-eval-steps 4 \
    --actor-recur-steps 4 \
    --actor-recur-eval-steps 4 \
    --no-print-args \
    --no-log-flush \
    --time-print-interval-epochs 100 \
    --wandb-dir $WANDB_DIR \
    --env-id $ENV_ID \
    --eval-env-id $EVAL_ENV_ID \
    --compact-metrics \
    --summary-tail-epochs 10"

TIED_MLP_ARGS="--tied-mlp-ffn-mult 2.667 \
    --tied-mlp-layerscale-init 0.01 \
    --tied-mlp-step-embed-dim 0 \
    --tied-mlp-trunc-bptt 0"

run_one() {
    local config=$1
    local c_blocks=$2
    local c_iters=$3
    local seed=$4
    local log_file=$5

    local a_blocks=4
    local a_iters=1
    local a_steps=$((a_blocks * a_iters))
    local c_steps=$((c_blocks * c_iters))

    echo ""
    echo "=============================================="
    echo "[$(date +%H:%M:%S)] Starting $config..."
    echo "Actor: $a_blocks blocks × $a_iters iters (steps=$a_steps) | Critic: $c_blocks blocks × $c_iters iters (steps=$c_steps)"
    echo "=============================================="

    python scripts/crl_train.py \
        --exp-name "${config}_s${seed}" \
        $COMMON_ARGS \
        $TIED_MLP_ARGS \
        --seed "$seed" \
        --actor-tied-mlp-blocks "$a_blocks" \
        --actor-tied-mlp-iters "$a_iters" \
        --critic-tied-mlp-blocks "$c_blocks" \
        --critic-tied-mlp-iters "$c_iters" \
        --critic-recur-steps "$c_steps" \
        > "$log_file" 2>&1

    echo "[$(date +%H:%M:%S)] $config complete!"
    extract_metrics "$log_file" "$config" "$seed" "$a_blocks" "$a_iters" "$a_steps" "$c_blocks" "$c_iters" "$c_steps"
}

# Actor fixed as AU (4×1). Critic partial tying sweep.
for SEED in "${SEEDS[@]}"; do
    run_one "phase2_partial_C1x4" 1 4 "$SEED" "logs/phase2_partial_C1x4_s${SEED}_${SLURM_JOB_ID:-local}.log"
    run_one "phase2_partial_C2x2" 2 2 "$SEED" "logs/phase2_partial_C2x2_s${SEED}_${SLURM_JOB_ID:-local}.log"
    run_one "phase2_partial_C2x4" 2 4 "$SEED" "logs/phase2_partial_C2x4_s${SEED}_${SLURM_JOB_ID:-local}.log"
    run_one "phase2_partial_C2x8" 2 8 "$SEED" "logs/phase2_partial_C2x8_s${SEED}_${SLURM_JOB_ID:-local}.log"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$(echo "scale=2; $ELAPSED / 3600" | bc)

echo ""
echo "=============================================="
echo "Phase 2 Mix Experiments Complete!"
echo "=============================================="
echo "Total runtime: ${ELAPSED_HOURS} hours"
echo ""
python - <<'PY'
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

raw_path = Path(r"""'"$RAW_RESULTS_FILE"'""")
out_path = Path(r"""'"$RESULTS_FILE"'""")

rows = []
with raw_path.open(newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

group_cols = [
    "config",
    "actor_blocks",
    "actor_iters",
    "actor_steps",
    "critic_blocks",
    "critic_iters",
    "critic_steps",
]

metrics = [
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

def to_float(x):
    if x is None:
        return None
    x = str(x).strip()
    if not x or x == "N/A" or x.lower() == "nan":
        return None
    try:
        return float(x)
    except ValueError:
        return None

groups = defaultdict(list)
for r in rows:
    key = tuple(r[c] for c in group_cols)
    groups[key].append(r)

out_fieldnames = group_cols + ["n_seeds"]
for m in metrics:
    out_fieldnames.append(f"{m}_mean")
    out_fieldnames.append(f"{m}_std")

with out_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out_fieldnames)
    w.writeheader()

    for key in sorted(groups.keys()):
        members = groups[key]
        out = {c: v for c, v in zip(group_cols, key)}
        out["n_seeds"] = len({m["seed"] for m in members})

        for m in metrics:
            vals = [to_float(r.get(m)) for r in members]
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            if not vals:
                out[f"{m}_mean"] = ""
                out[f"{m}_std"] = ""
            elif len(vals) == 1:
                out[f"{m}_mean"] = f"{vals[0]:.6g}"
                out[f"{m}_std"] = "0"
            else:
                out[f"{m}_mean"] = f"{statistics.mean(vals):.6g}"
                out[f"{m}_std"] = f"{statistics.stdev(vals):.6g}"

        w.writerow(out)
PY

echo "Raw per-seed results saved to: $RAW_RESULTS_FILE"
echo "Mean±std summary saved to: $RESULTS_FILE"
echo ""
echo "Final Comparison Table:"
echo "=============================================="
column -t -s',' $RESULTS_FILE
echo "=============================================="
