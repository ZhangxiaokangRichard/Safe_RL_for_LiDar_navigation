"""Environment wrapper for state preprocessing and normalization."""

import gymnasium as gym
import numpy as np
from typing import Tuple, Any, Dict, Optional


class EnvironmentWrapper(gym.Wrapper):
    """
    Wrapper for SafetyCarGoal2-v0 with state preprocessing and normalization.

    This wrapper handles:
    - State normalization and preprocessing
    - Observation space standardization
    - Reward clipping if needed
    - Cost tracking for safety constraints
    """

    def __init__(
        self,
        env: gym.Env,
        obs_normalize: bool = True,
        reward_scale: float = 1.0,
    ):
        """
        Initialize environment wrapper.

        Args:
            env: The base environment to wrap
            obs_normalize: Whether to normalize observations
            reward_scale: Scaling factor for rewards
        """
        super().__init__(env)
        self.obs_normalize = obs_normalize
        self.reward_scale = reward_scale

        # SafetyGymnasium returns 6-tuple from step(). We must bypass gymnasium wrappers
        # (TimeLimit, EnvChecker) that only accept 5-tuple. Store a reference to the raw env.
        self._base_env = env.unwrapped if hasattr(env, "unwrapped") else env

        # Track observation statistics for normalization
        self.obs_mean = None
        self.obs_std = None
        self.obs_count = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment and return initial observation.

        Args:
            seed: Random seed
            options: Reset options

        Returns:
            Initial observation and info dict
        """
        obs, info = self._base_env.reset(seed=seed, options=options)
        obs = self._preprocess_obs(obs)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: The action to take

        Returns:
            observation, reward, terminated, truncated, info
        """
        obs, reward, cost, terminated, truncated, info = self._base_env.step(action)

        # Preprocess observation
        obs = self._preprocess_obs(obs)

        # Scale reward
        reward = reward * self.reward_scale

        # Track cost for safety metrics (cost is a first-class return value in SafetyGymnasium)
        info["original_reward"] = reward / self.reward_scale
        info["cost"] = cost

        return obs, reward, terminated, truncated, info

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Preprocess observation with normalization if enabled.

        Args:
            obs: Raw observation from environment

        Returns:
            Preprocessed observation
        """
        if not self.obs_normalize:
            return obs

        # Initialize statistics on first observation
        if self.obs_mean is None:
            obs = np.asarray(obs, dtype=np.float32)
            self.obs_mean = np.zeros_like(obs)
            self.obs_std = np.ones_like(obs)
            self.obs_count = 1
            return (obs - self.obs_mean) / (self.obs_std + 1e-8)

        # Update running statistics
        obs = np.asarray(obs, dtype=np.float32)
        delta = obs - self.obs_mean
        self.obs_mean += delta / (self.obs_count + 1)
        self.obs_std = np.sqrt(
            (self.obs_count * self.obs_std**2 + delta * (obs - self.obs_mean)) /
            (self.obs_count + 1) + 1e-8
        )
        self.obs_count += 1

        # Return normalized observation
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)

    def get_obs_stats(self) -> Dict[str, np.ndarray]:
        """
        Get observation statistics for external use (e.g., policy deployment).

        Returns:
            Dictionary with 'mean' and 'std' keys
        """
        return {
            "mean": self.obs_mean.copy() if self.obs_mean is not None else None,
            "std": self.obs_std.copy() if self.obs_std is not None else None,
        }
