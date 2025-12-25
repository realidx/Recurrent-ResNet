# Recurrent-ResNet

Research codebase to test weight-tied recurrent residual blocks inside CRL (online self-supervised RL), focusing on whether depth-scaling gains come from iterative refinement rather than parameter count.

## Goals
- Compare deep untied residual stacks vs weight-tied recurrent residual blocks.
- Evaluate compute-accuracy tradeoffs by varying refinement steps (K) at inference.
- Match paper baselines where depth scaling mattered, then run tied/untied ablations.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Submodule (paper reference implementation)
```bash
git submodule update --init --recursive
```

## Project layout
- `configs/`: experiment configs (baseline + recurrent variants)
- `scripts/`: entrypoints for training/eval/benchmarks
- `benchmarks/`: task suites and benchmark tracking
- `src/recurrent_resnet/`: core model, agent, and utilities
- `third_party/`: external dependencies (paper repo as submodule)
- `docs/`: design notes and experiment planning

## Status
Skeleton in place. Next step is implementing the CRL baseline wrapper and recurrent residual block.
