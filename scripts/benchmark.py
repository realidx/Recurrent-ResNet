import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def collect_config_paths(configs: list[Path], config_dir: Path) -> list[Path]:
    if configs:
        return configs
    if not config_dir.exists():
        raise SystemExit(f"Config dir not found: {config_dir}")
    return sorted(path for path in config_dir.glob("*.yaml"))


def format_args(config: dict) -> list[str]:
    args: list[str] = []
    for key in sorted(config.keys()):
        value = config[key]
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            args.append(flag if value else f"--no-{key.replace('_', '-')}")
        else:
            args.extend([flag, str(value)])
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark suites")
    parser.add_argument("--suite", type=str, default="quick_debug")
    parser.add_argument("--suites", type=Path, default=Path("benchmarks/suites.yaml"))
    parser.add_argument("--configs", type=Path, nargs="*", default=[])
    parser.add_argument("--config-dir", type=Path, default=Path("configs/crl"))
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-args", nargs="*", default=[])
    args = parser.parse_args()

    suites = load_yaml(args.suites)
    suite = suites.get("suites", {}).get(args.suite)
    if suite is None:
        raise SystemExit(f"Unknown suite: {args.suite}")

    config_paths = collect_config_paths(args.configs, args.config_dir)
    if not config_paths:
        raise SystemExit("No configs found.")

    suite_envs = suite.get("envs", [])
    if not suite_envs:
        raise SystemExit(f"No envs defined for suite: {args.suite}")

    for config_path in config_paths:
        config = load_yaml(config_path)
        config_name = config_path.stem
        config.setdefault("exp_name", config_name)
        if "total_env_steps" not in config and "steps" in suite:
            config["total_env_steps"] = suite["steps"]
        if "num_envs" not in config and "num_envs" in suite:
            config["num_envs"] = suite["num_envs"]
        if "num_eval_envs" not in config and "num_eval_envs" in suite:
            config["num_eval_envs"] = suite["num_eval_envs"]

        for env_id in suite_envs:
            run_config = dict(config)
            run_config.setdefault("env_id", env_id)
            if not run_config.get("eval_env_id"):
                run_config["eval_env_id"] = env_id
            cmd = [args.python, "scripts/crl_train.py"]
            cmd.extend(format_args(run_config))
            cmd.extend(args.extra_args)

            print(shlex.join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
