"""Resolve a ConCal launch configuration without starting training."""

import argparse
import json
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from context_credit.adapter import validate_runtime_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("alfworld", "webshop"), required=True)
    args, overrides = parser.parse_known_args()
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    OmegaConf.resolve(config)
    validate_runtime_config(config)
    print(json.dumps({"validation": "passed", "benchmark": args.benchmark,
        "repo": str(ROOT), "seed": config.env.seed, "data_seed": config.data.seed,
        "pooling_mode": config.algorithm.context_credit.pooling_mode,
        "normalization": config.algorithm.context_credit.local_normalization,
        "gigpo_mode": config.algorithm.gigpo.mode,
        "interaction_limit": config.env.max_steps,
        "run_dir": str(Path(config.trainer.default_local_dir).parent)}))


if __name__ == "__main__":
    main()
