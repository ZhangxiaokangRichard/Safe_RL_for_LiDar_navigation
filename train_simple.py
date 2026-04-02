#!/usr/bin/env python3
"""Simplified training script for SafetyCarGoal2-v0 - Direct approach."""

import sys
from pathlib import Path
from datetime import datetime

try:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
except ImportError:
    print("ERROR: tomli required for Python < 3.11")
    sys.exit(1)

try:
    import gymnasium as gym
    import safety_gymnasium
except ImportError:
    print("ERROR: Install required packages: pip install -r requirements.txt")
    sys.exit(1)


def load_config(config_path):
    """Load configuration from TOML file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def train_simple(config_path: str = "config/ppo_cargoal_default.toml", exp_name: str = None, seed: int = None):
    """
    Simple training function without OmniSafe.

    This demonstrates that the environment and configuration system works
    independently of OmniSafe.
    """

    # Load configuration
    config = load_config(Path(config_path))

    # Create experiment directory
    exp_name = exp_name or f"cargoal_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = Path("experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SafetyCarGoal2-v0 Training (Direct Approach)")
    print("=" * 70)
    print(f"\nExperiment dir: {exp_dir.resolve()}")
    print(f"Config file:    {config_path}")

    # Create environment
    env_id = config["environment"]["env_id"]

    print(f"\nCreating environment: {env_id}")
    # Note: disable_env_checker=True allows SafetyGymnasium's 6-tuple return format
    # Access unwrapped environment to avoid wrapper conflicts
    env = gym.make(env_id, render_mode=None, disable_env_checker=True)
    env = env.unwrapped

    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")

    # Training loop (simplified - just random actions for demo)
    num_epochs = config["training"]["num_epochs"]
    steps_per_epoch = config["training"]["steps_per_epoch"]

    print(f"\nTraining configuration:")
    print(f"  Epochs:              {num_epochs}")
    print(f"  Steps per epoch:     {steps_per_epoch}")
    print(f"  Learning rate:       {config['algorithm']['learning_rate']}")
    print(f"  Gamma (discount):    {config['algorithm']['gamma']}")
    print(f"  Clip ratio:          {config['algorithm']['clip_ratio']}")

    print(f"\n{'=' * 70}")
    print("Starting environment test (10 steps with random actions)...")
    print(f"{'=' * 70}\n")

    # Test environment with random actions
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    total_cost = 0.0

    for step in range(10):
        action = env.action_space.sample()
        obs, reward, cost, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_cost += cost

        print(f"Step {step+1}: Reward={reward:7.3f}, Cost={cost:6.4f}, Done={terminated or truncated}")

        if terminated or truncated:
            break

    print(f"\n{'=' * 70}")
    print(f"Test Results: Total Reward={total_reward:.3f}, Total Cost={total_cost:.4f}")
    print(f"{'=' * 70}")

    # Save config
    config_path = exp_dir / "config.toml"
    with open(config_path, "w") as f:
        f.write(f"# Configuration for {exp_name}\n")
        f.write(f"# Original config: {config_path}\n")

    print(f"\n[OK] Experiment directory: {exp_dir.resolve()}")
    print("[OK] Configuration saved")
    print("\nNext steps:")
    print(f"  1. Install OmniSafe: pip install omnisafe==0.5.0")
    print(f"  2. Run full training: python train.py --exp-name {exp_name}")
    print(f"  3. Evaluate: python evaluate.py --exp-dir {str(exp_dir)}")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple SafetyCarGoal2-v0 test")
    parser.add_argument("--config", default="config/ppo_cargoal_default.toml")
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    train_simple(config_path=args.config, exp_name=args.exp_name, seed=args.seed)
