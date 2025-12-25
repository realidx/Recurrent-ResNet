# Benchmarks

`benchmarks/suites.yaml` defines task suites used for quick debugging and depth-scaling experiments.

Suggested usage:
- Start with `quick_debug` for sanity checks.
- Promote to `depth_scaling_focus` once stability is confirmed.

Each suite stores the environment list, steps, and recommended `num_envs` for matching the paper's compute profile.

## Running suites
Dry-run to verify commands:
```bash
python scripts/benchmark.py --suite quick_debug --config-dir configs/crl --dry-run
```

Run a single config in the active env:
```bash
python scripts/benchmark.py --suite quick_debug --configs configs/crl/sanity.yaml
```
