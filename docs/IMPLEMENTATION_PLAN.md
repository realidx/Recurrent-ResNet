# Implementation and Experiment Plan

This is the concrete plan for adding a weight-tied recurrent residual core to the CRL codebase and running the depth-scaling experiments. Keep this as the source of truth when decisions get fuzzy.

## Goals
- Test whether depth gains in CRL come from iterative refinement instead of parameter count.
- Compare deep untied residual stacks vs tied recurrent residual blocks at matched compute.
- Verify whether larger K at inference improves goal-reaching without retraining.

## Integration strategy (minimize risk)
- Treat `third_party/scaling-crl` as the baseline reference.
- Create a forked training entrypoint in this repo (not inside the submodule) so we can freely edit without dirtying the submodule history.
- Keep baseline behavior identical when `encoder_type=untied_resnet` so reproduction remains possible.

## Files to create/modify
- New training script: `scripts/crl_train.py` (copy from `third_party/scaling-crl/train.py`).
- New Flax modules for recurrence: `src/recurrent_resnet/jax/recurrent_core.py` (tied/untied blocks).
- Update `scripts/benchmark.py` later to invoke `scripts/crl_train.py` with suite configs.
- Add new config templates in `configs/` as experiments stabilize.

## Phase 0: Baseline reproduction
- Environment: use JAX/Flax + Brax as described in `third_party/scaling-crl/README.md`.
- Apply the two Brax fixes before running.
- Sanity run: low `num_envs`, short `total_env_steps`, shallow depth.
- Baseline run: reproduce 1-2 depth-sensitive tasks with default depth scaling.

## Phase 1: Add recurrent residual core

### 1) Implement recurrent residual block
- Add `RecurrentResidualBlock` in `src/recurrent_resnet/jax/recurrent_core.py`.
- Parameters:
  - `width`, `steps`, `tie_weights`, `use_relu`, `norm_type`.
  - `init_identity` (use zero-init on the last Dense in block so `f(x) ≈ 0`).
- Tied case: create one `ResidualBlock` instance and reuse in a loop.
- Untied case: create a list of `ResidualBlock`s, one per step.
- Use `jax.lax.fori_loop` (dynamic steps) or `nn.scan` (fast + clean). Both are fine.

### 2) Wire into encoders and actor
In `scripts/crl_train.py`:
- Replace the for-loop residual stack in:
  - `SA_encoder.__call__`
  - `G_encoder.__call__`
  - `Actor.__call__`
- New behavior:
  - `x = Dense(width) -> norm -> activation`
  - `x = core(x, steps=K)`
  - `x = output_head`
- Gate with flags:
  - `encoder_type`: `untied_resnet` | `recurrent_tied` | `recurrent_untied`
  - `recur_steps` (K)
  - `recur_apply_to_actor` (bool)
  - `recur_apply_to_critic` (bool)

### 3) Depth dropout + truncated BPTT
- Add `recur_depth_dropout` (float in [0,1]).
- If `recur_depth_dropout > 0`, sample `steps` per update:
  - `steps = randint(K_min, K_max)` or `steps = max(1, int(K_max * (1 - dropout)))`.
- Optional: `recur_trunc_bptt` (int). If > 0, stop gradients every N steps.

### 4) Logging and parameter counts
- Log total parameter count for actor and critic encoders.
- Log steps/sec and throughput to guard against “more compute” critiques.
- Include encoder type + K in run name.

## Phase 2: Baseline vs recurrent experiments

### Core comparisons
- Untied deep baseline (paper-style): depth `D` (e.g., 64).
- Tied recurrent: one residual block unrolled `K` steps.
- Untied recurrent (control): `K` independent blocks to isolate effect of tying.

### Compute matching
- In this code, `network_depth` is the number of Dense layers.
- Each residual block uses 4 Dense layers, so `num_blocks = depth / 4`.
- Choose `K ≈ D / 4` for compute-matched comparisons.

### Ablations
- Tied vs untied at same K.
- Fixed K vs stochastic K (depth dropout).
- Apply recurrence to critic only vs actor+critic.
- Inference-only scaling: evaluate same checkpoint at larger K.

### Suggested tasks
Pick 2 depth-sensitive tasks for the initial reproduction:
- `humanoid` (if memory allows)
- `ant_maze` or `ant_ball`
- `arm_binpick_hard` (if using manipulation envs)

## Metrics and plots
- Success rate vs environment steps.
- Wall-clock vs success threshold.
- Parameter count + steps/sec.
- Stability: NaN checks, critic loss spikes.

## Go / No-Go checkpoints
- Checkpoint A (week 2-3): recurrent K=2/4 trains stably and reaches non-trivial success.
- Checkpoint B (week 4): tied recurrence matches untied depth at comparable wall-clock or is clearly more parameter-efficient.

## Known risks and mitigations
- Vanishing/exploding gradients: identity init + LayerNorm; optional gradient clipping.
- Compute bottleneck: lower `num_sgd_batches_per_training_step` for pilot runs.
- Memory: use smaller K for most runs, truncated BPTT if needed.

## Decision log (to fill)
- Final list of environments:
- Baseline depth D:
- Compute-matched K:
- UTD settings:
