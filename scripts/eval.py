import argparse
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained policy checkpoints")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=False)
    args = parser.parse_args()

    config = load_config(args.config)
    print("Loaded config:")
    print(config)
    print(f"Checkpoint: {args.checkpoint}")
    raise SystemExit("Evaluation pipeline not implemented yet")


if __name__ == "__main__":
    main()
