"""OmniSafe PPO Policy Adapter for CarGoal Environment."""

import sys
from pathlib import Path

try:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
except ImportError:
    raise ImportError("tomli required for Python < 3.11")

import omnisafe


class CarGoalPPOAgent:
    """
    Adapter class for training PPO agent on CarGoal environment.

    This class integrates OmniSafe's PPO implementation with custom
    configurations for the SafetyCarGoal2-v0 environment.
    """

    def __init__(self, config_path: Path, exp_name: str = None, exp_dir: Path = None):
        """
        Initialize PPO agent with configuration.

        Args:
            config_path: Path to TOML configuration file
            exp_name: Experiment name (if None, auto-generated)
            exp_dir: Experiment directory (if None, auto-determined)
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.exp_name = exp_name
        self.exp_dir = exp_dir
        self.agent = None

    def _load_config(self) -> dict:
        """Load configuration from TOML file."""
        with open(self.config_path, "rb") as f:
            return tomllib.load(f)

    def create_agent(self):
        """Create and initialize OmniSafe PPO agent."""
        # Handle seed: 0 means use random seed for OmniSafe
        seed = self.config["training"]["seed"]
        if seed == 0:
            seed = None

        self.agent = omnisafe.Agent(
            "PPO",
            env_id=self.config["environment"]["env_id"],
            exp_name=self.exp_name,
        )

        # Apply all configurations via custom_cfgs
        self.agent.custom_cfgs = {
            "seed": seed,
            "env_cfgs": {
                "env_kwargs": {
                    "level": self.config["environment"]["level"],
                },
            },
            "algo_cfgs": {
                "learning_rate": self.config["algorithm"]["learning_rate"],
                "gamma": self.config["algorithm"]["gamma"],
                "lam": self.config["algorithm"]["gae_lambda"],
                "clip_ratio": self.config["algorithm"]["clip_ratio"],
                "entropy_coef": self.config["algorithm"]["entropy_coef"],
            },
            "train_cfgs": {
                "steps_per_epoch": self.config["training"]["steps_per_epoch"],
                "total_steps": (
                    self.config["training"]["num_epochs"]
                    * self.config["training"]["steps_per_epoch"]
                ),
                "batch_size": self.config["training"]["batch_size"],
            },
        }

        return self.agent

    def train(self):
        """Start training the agent."""
        if self.agent is None:
            self.create_agent()

        return self.agent.learn()

    def get_policy(self):
        """Get the trained policy."""
        if self.agent is None:
            raise RuntimeError("Agent not trained yet")

        return self.agent.policy

    def save_policy(self, path: Path):
        """Save the trained policy."""
        if self.agent is None:
            raise RuntimeError("Agent not trained yet")

        import torch
        torch.save(self.agent.policy, path)

    def load_policy(self, path: Path):
        """Load a trained policy."""
        import torch
        return torch.load(path)
