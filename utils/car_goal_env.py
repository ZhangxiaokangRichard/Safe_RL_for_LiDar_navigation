"""CarGoal Environment - High-level interface for easy usage.

This module provides a convenient wrapper around EnvironmentWrapper and
MuJoCoVisualizer for quick prototyping and experimentation.

Design: Building on top of the lower-level utilities (env_wrapper.py, visualizer.py)
to simplify common workflows while maintaining full composability.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import gymnasium as gym
import safety_gymnasium
import numpy as np

# Handle TOML parsing for different Python versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError(
            "tomli is required for Python < 3.11. Install it with: pip install tomli"
        )

from .env_wrapper import EnvironmentWrapper
from .visualizer import MuJoCoVisualizer


class CarGoalEnv:
    """
    High-level interface for SafetyCarGoal2-v0 environment.

    This class simplifies common tasks:
    - Configuration management via TOML
    - One-step environment creation with preprocessing
    - Episode simulation and video recording
    - Batch processing and statistics
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize CarGoal environment manager.

        Args:
            config_path: Path to TOML configuration file. If None, uses defaults.
        """
        self.config = self._load_config(config_path)
        self.env: Optional[gym.Env] = None
        self.visualizer: Optional[MuJoCoVisualizer] = None
        self._episode_count = 0

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """
        Load configuration from TOML file or return defaults.

        Args:
            config_path: Path to TOML config file

        Returns:
            Configuration dictionary
        """
        default_config = {
            "environment": {
                "env_id": "SafetyCarGoal2-v0",
                "level": 2,
                "render_mode": None,
            },
            "preprocessing": {
                "obs_normalize": True,
                "reward_scale": 1.0,
            },
            "rendering": {
                "width": 640,
                "height": 480,
                "fps": 30,
            },
            "recording": {
                "enable": False,
                "output_dir": "./videos",
            },
        }

        if config_path is None:
            return default_config

        config_path = Path(config_path)
        if not config_path.exists():
            print(f"⚠ Config file not found: {config_path}. Using defaults.")
            return default_config

        with open(config_path, "rb") as f:
            loaded_config = tomllib.load(f)

        # Deep merge loaded config with defaults
        for key in default_config:
            if key in loaded_config:
                if isinstance(default_config[key], dict):
                    default_config[key].update(loaded_config[key])
                else:
                    default_config[key] = loaded_config[key]

        return default_config

    def create_env(
        self,
        render_mode: Optional[str] = None,
        enable_recording: bool = False,
    ) -> gym.Env:
        """
        Create and initialize the SafetyCarGoal2-v0 environment.

        Args:
            render_mode: "human" for visualization, None for no rendering
            enable_recording: Whether to enable video recording

        Returns:
            The wrapped gymnasium environment
        """
        if render_mode is not None:
            self.config["environment"]["render_mode"] = render_mode
        if enable_recording:
            self.config["recording"]["enable"] = enable_recording

        env_id = self.config["environment"]["env_id"]
        level = self.config["environment"]["level"]
        render_mode = self.config["environment"]["render_mode"]

        # Create base environment
        # Note: level is encoded in env_id (SafetyCarGoal2-v0 = level 2)
        base_env = gym.make(env_id, render_mode=render_mode, disable_env_checker=True)

        # Wrap with preprocessing layer
        obs_normalize = self.config["preprocessing"]["obs_normalize"]
        reward_scale = self.config["preprocessing"]["reward_scale"]

        self.env = EnvironmentWrapper(
            base_env,
            obs_normalize=obs_normalize,
            reward_scale=reward_scale,
        )

        # Setup recording if enabled
        if self.config["recording"]["enable"]:
            output_dir = Path(self.config["recording"]["output_dir"])
            self.visualizer = MuJoCoVisualizer(
                env_id=env_id,
                level=level,
                render_mode="rgb_array",
                output_dir=output_dir,
            )
            self.visualizer.set_fps(self.config["rendering"]["fps"])

        return self.env

    def reset(self, seed: Optional[int] = None):
        """
        Reset the environment.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Initial observation and info dict
        """
        if self.env is None:
            raise RuntimeError("Environment not created. Call create_env() first.")

        return self.env.reset(seed=seed)

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: The action to take

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.env is None:
            raise RuntimeError("Environment not created. Call create_env() first.")

        return self.env.step(action)

    def run_episode(
        self,
        max_steps: Optional[int] = None,
        render: bool = False,
        seed: Optional[int] = None,
        record: bool = False,
        episode_label: str = "",
    ) -> Dict[str, Any]:
        """
        Run a complete episode.

        Args:
            max_steps: Maximum steps per episode
            render: Whether to render during episode
            seed: Random seed
            record: Whether to record video for this episode
            episode_label: Label for video filename

        Returns:
            Episode statistics dictionary
        """
        obs, info = self.reset(seed=seed)
        episode_reward = 0.0
        episode_cost = 0.0
        step_count = 0
        frames = []

        done = False
        while not done:
            # Sample random action (replace with your policy)
            action = self.env.action_space.sample()

            obs, reward, terminated, truncated, info = self.step(action)
            episode_reward += reward
            episode_cost += info.get("cost", 0.0)
            step_count += 1

            # Capture frame if recording
            if record and self.config["environment"]["render_mode"] == "rgb_array":
                frame = self.env.render()
                if frame is not None:
                    frames.append(frame)

            done = terminated or truncated
            if max_steps is not None and step_count >= max_steps:
                break

        self._episode_count += 1

        # Save video if frames were captured
        video_path = None
        if record and frames:
            video_path = self._save_video(frames, episode_label)

        return {
            "episode": self._episode_count - 1,
            "total_reward": episode_reward,
            "total_cost": episode_cost,
            "steps": step_count,
            "video_path": video_path,
        }

    def run_episodes(
        self,
        num_episodes: int,
        max_steps_per_episode: Optional[int] = None,
        render_episodes: Optional[List[int]] = None,
        record_episodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> list:
        """
        Run multiple episodes with optional recording.

        Args:
            num_episodes: Number of episodes to run
            max_steps_per_episode: Max steps per episode
            render_episodes: List of episode indices to render
            record_episodes: List of episode indices to record
            seed: Random seed

        Returns:
            List of episode statistics
        """
        if render_episodes is None:
            render_episodes = []
        if record_episodes is None:
            record_episodes = []

        stats = []
        print(f"\n{'='*70}")
        print(f"Running {num_episodes} episodes")
        print(f"Recording: {record_episodes}")
        print(f"{'='*70}\n")

        for ep in range(num_episodes):
            should_render = ep in render_episodes
            should_record = ep in record_episodes

            stat = self.run_episode(
                max_steps=max_steps_per_episode,
                render=should_render,
                seed=seed + ep if seed is not None else None,
                record=should_record,
                episode_label=f"ep_{ep:03d}",
            )
            stats.append(stat)

            status = "[REC]" if should_record else "[---]"
            print(
                f"{status} Episode {ep + 1:3d}/{num_episodes}: "
                f"Reward={stat['total_reward']:8.2f}  "
                f"Cost={stat['total_cost']:8.2f}  "
                f"Steps={stat['steps']:4d}"
            )

        print(f"\n{'='*70}\n")
        return stats

    def _save_video(self, frames: List[np.ndarray], episode_label: str) -> Path:
        """
        Save frames to video file.

        Args:
            frames: List of RGB frames
            episode_label: Label for filename

        Returns:
            Path to saved video
        """
        try:
            import imageio
        except ImportError:
            print("⚠ imageio not installed. Video recording disabled.")
            return None

        output_dir = Path(self.config["recording"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        video_path = output_dir / f"{episode_label}.mp4"

        frames = np.array(frames, dtype=np.uint8)
        fps = self.config["rendering"]["fps"]

        try:
            imageio.mimwrite(str(video_path), frames, fps=fps)
            print(f"  ✓ Video saved: {video_path}")
        except Exception as e:
            print(f"  ✗ Error saving video: {e}")
            return None

        return video_path

    def close(self) -> None:
        """Close the environment and cleanup resources."""
        if self.env is not None:
            self.env.close()
            self.env = None
        if self.visualizer is not None:
            self.visualizer.close()
            self.visualizer = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
