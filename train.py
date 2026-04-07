"""OmniSafe training script — supports PPO and safety variants on any safety-gymnasium environment."""

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


def _patch_task_classes(extra_goal_reward: float, boundary_cost: float) -> None:
    """Monkey-patch safety-gymnasium task classes to inject env tuning.

    - extra_goal_reward: bonus added to reward each time goal is reached
    - boundary_cost:     extra cost per step when agent exits arena bounds
    """
    if extra_goal_reward != 0.0:
        try:
            from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0
            _orig_goal_reward = GoalLevel0.calculate_reward

            def _patched_goal_reward(self):
                reward = _orig_goal_reward(self)
                if getattr(self, "goal_achieved", False):
                    reward += extra_goal_reward
                return reward

            GoalLevel0.calculate_reward = _patched_goal_reward
            print(f"  [patch] extra_goal_reward = +{extra_goal_reward} on each goal reached")
        except ImportError:
            print("  [warn] GoalLevel0 not found — extra_goal_reward patch skipped")

    if boundary_cost != 0.0:
        try:
            from safety_gymnasium.bases.base_task import BaseTask
            _orig_cost = BaseTask.calculate_cost

            def _patched_cost(self):
                cost = _orig_cost(self)
                try:
                    ext = self.placements_conf.extents[2]  # xmax (symmetric)
                    pos = self.agent.pos[:2]
                    if abs(float(pos[0])) > ext or abs(float(pos[1])) > ext:
                        cost["cost_boundary"] = boundary_cost
                        cost["cost_sum"] = cost.get("cost_sum", 0.0) + boundary_cost
                except Exception:
                    pass
                return cost

            BaseTask.calculate_cost = _patched_cost
            print(f"  [patch] boundary_cost = +{boundary_cost} per step outside arena")
        except ImportError:
            print("  [warn] BaseTask not found — boundary_cost patch skipped")


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
        description="Train an OmniSafe RL agent (algorithm and environment set via config)"
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
    env_tuning = config.get("env_tuning", {})
    extra_goal_reward = float(env_tuning.get("extra_goal_reward", 0.0))
    boundary_cost = float(env_tuning.get("boundary_cost", 0.0))
    if extra_goal_reward != 0.0 or boundary_cost != 0.0:
        print(f"Env tuning:      extra_goal_reward={extra_goal_reward}  boundary_cost={boundary_cost}")
    print("=" * 65 + "\n")

    custom_cfgs = build_custom_cfgs(config, algo, args.device, seed)

    # Apply env-level monkey-patches before agent (and its env) is created
    if extra_goal_reward != 0.0 or boundary_cost != 0.0:
        print("Applying env tuning patches...")
        _patch_task_classes(extra_goal_reward, boundary_cost)
        print()

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
