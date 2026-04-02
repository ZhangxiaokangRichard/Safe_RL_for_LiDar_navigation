"""OmniSafe training script for CarGoal Level 2 — supports PPO and safety variants."""

import sys
import random
from pathlib import Path
import argparse

try:
    import omnisafe
except ImportError:
    print("ERROR: omnisafe not installed. Install with: pip install omnisafe")
    sys.exit(1)

try:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
except ImportError:
    print("ERROR: tomli not available for Python < 3.11")
    sys.exit(1)

# Algorithms that require Lagrangian safety config
LAGRANGIAN_ALGOS = {"PPOLag", "TRPOLag", "CPPOPID", "TRPOPID"}


def load_config(config_path: Path) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def build_custom_cfgs(config: dict, algo: str, device: str, seed: int) -> dict:
    """Translate TOML config → OmniSafe custom_cfgs."""
    lr = config["algorithm"]["learning_rate"]

    cfgs: dict = {
        "seed": seed,
        "train_cfgs": {
            "device": device,
            "total_steps": config["training"]["num_epochs"] * config["training"]["steps_per_epoch"],
        },
        "algo_cfgs": {
            "steps_per_epoch": config["training"]["steps_per_epoch"],
            "batch_size": config["training"]["batch_size"],
            "gamma": config["algorithm"]["gamma"],
            "lam": config["algorithm"]["gae_lambda"],
            "clip": config["algorithm"]["clip_ratio"],
            "entropy_coef": config["algorithm"]["entropy_coef"],
        },
        "model_cfgs": {
            "actor": {"lr": lr},
            "critic": {"lr": lr},
        },
        "logger_cfgs": {
            "save_model_freq": config["training"]["save_freq"],
            "use_tensorboard": True,
        },
    }

    # Add Lagrangian safety constraints for safety algorithms
    if algo in LAGRANGIAN_ALGOS:
        safety = config.get("safety", {})
        cfgs["lagrange_cfgs"] = {
            "cost_limit": safety.get("cost_limit", 25.0),
            "lagrangian_multiplier_init": safety.get("lagrangian_multiplier_init", 0.001),
            "lambda_lr": safety.get("lambda_lr", 0.035),
            "lambda_optimizer": safety.get("lambda_optimizer", "Adam"),
        }
        # Safety algorithms also need use_cost = True
        cfgs["algo_cfgs"]["use_cost"] = True

    return cfgs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an OmniSafe RL agent on SafetyCarGoal2-v0"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/ppo_cargoal_default.toml"),
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        help="Algorithm override (e.g. PPO, PPOLag). Defaults to config [meta].algo or PPO.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override (0 = random)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Training device: cpu or cuda:0",
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    # Determine algorithm: CLI > config [meta].algo > "PPO"
    algo = args.algo or config.get("meta", {}).get("algo", "PPO")

    # Validate
    all_algos = [a for algos in omnisafe.ALGORITHMS.values() for a in algos]
    if algo not in all_algos:
        print(f"ERROR: Unknown algorithm '{algo}'. Available: {sorted(set(all_algos))}")
        sys.exit(1)

    # Seed resolution
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    seed = config["training"]["seed"]
    if seed == 0:
        seed = random.randint(1, 10000)

    env_id = config["environment"]["env_id"]
    total_steps = config["training"]["num_epochs"] * config["training"]["steps_per_epoch"]

    print("=" * 65)
    print(f"OmniSafe Training — {algo} on {env_id}")
    print("=" * 65)
    print(f"Config:          {args.config}")
    print(f"Algorithm:       {algo}")
    print(f"Seed:            {seed}")
    print(f"Device:          {args.device}")
    print(f"Total steps:     {total_steps:,}  "
          f"({config['training']['num_epochs']} epochs × "
          f"{config['training']['steps_per_epoch']} steps)")
    if algo in LAGRANGIAN_ALGOS:
        cl = config.get("safety", {}).get("cost_limit", 25.0)
        print(f"Cost limit:      {cl}  (Lagrangian constraint)")
    print("=" * 65 + "\n")

    custom_cfgs = build_custom_cfgs(config, algo, args.device, seed)

    print(f"Initializing {algo} agent...")
    agent = omnisafe.Agent(algo, env_id=env_id, custom_cfgs=custom_cfgs)
    print("[OK] Agent initialized\n")

    print("Starting training...\n")
    try:
        agent.learn()
        print("\n[OK] Training complete.")
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[X] Training error: {e}")
        raise

    print(f"\nRun saved to: runs/{algo}-{{{env_id}}}/seed-{seed:03d}-*/")
    print("Next steps:")
    print(f"  python evaluate.py --exp-dir runs/{algo}-{{{env_id}}}/seed-{seed:03d}-*/")
    print(f"  python compare.py  --runs runs/ --save-dir experiments/base_line/plots/")


if __name__ == "__main__":
    main()
