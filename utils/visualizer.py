"""MuJoCo Visualizer for independent simulation rendering and video recording."""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import warnings

import gymnasium as gym
import numpy as np

try:
    import imageio
except ImportError:
    warnings.warn("imageio not installed. Video recording will be disabled.")
    imageio = None


class MuJoCoVisualizer:
    """
    Independent MuJoCo visualization and video recording engine.

    This class provides:
    - Independent environment rendering (decoupled from training)
    - Policy loading and inference
    - Video recording with customizable FPS and resolution
    - Batch episode rendering for multiple visualizations

    Design principle: **Functional Independence**
    The visualizer operates completely independently of the training loop,
    making it suitable for post-hoc analysis, demo generation, and reproducibility.
    """

    def __init__(
        self,
        env_id: str = "SafetyCarGoal2-v0",
        level: int = 2,
        render_mode: str = "rgb_array",
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the MuJoCo visualizer.

        Args:
            env_id: Environment ID (e.g., "SafetyCarGoal2-v0"). Level is encoded in the ID.
            level: Ignored. Kept for API compatibility — use env_id to specify level.
            render_mode: Rendering mode ("rgb_array" for offline, "human" for live)
            output_dir: Directory to save videos (required for video recording)
        """
        self.env_id = env_id
        self.level = level
        self.render_mode = render_mode
        self.output_dir = Path(output_dir) if output_dir else Path("./videos")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize environment
        # SafetyGymnasium's step() returns 6 values (obs, reward, cost, terminated, truncated, info).
        # disable_env_checker avoids the gymnasium env-checker that rejects non-5-tuple returns.
        # Unwrap past TimeLimit so step() receives the 6-tuple directly.
        _env = gym.make(env_id, render_mode=render_mode, disable_env_checker=True)
        self.env = _env.unwrapped

        # Policy storage
        self.policy: Optional[Callable] = None
        self.policy_path: Optional[Path] = None

        # Rendering configuration
        self.fps = 30
        self.frame_buffer: List[np.ndarray] = []

    def load_policy(self, policy_path: Path, policy_type: str = "pt") -> None:
        """
        Load a trained policy from file.

        Args:
            policy_path: Path to the saved policy file (.pt, .pkl, etc.)
            policy_type: Type of policy file ("pt" for PyTorch, "pkl" for pickle)

        Raises:
            FileNotFoundError: If policy file does not exist
            ImportError: If required libraries for loading are not available
        """
        policy_path = Path(policy_path)
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

        self.policy_path = policy_path

        if policy_type == "pt":
            try:
                import torch
            except ImportError:
                raise ImportError("PyTorch is required to load .pt policies")

            # Load PyTorch model
            self.policy = torch.jit.load(str(policy_path))
            self.policy.eval()

        elif policy_type == "pkl":
            import pickle

            with open(policy_path, "rb") as f:
                self.policy = pickle.load(f)

        print(f"✓ Policy loaded from: {policy_path}")

    def set_policy_fn(self, policy_fn: Callable) -> None:
        """
        Set a policy function directly without loading from file.

        Args:
            policy_fn: Callable that takes observation and returns action
        """
        self.policy = policy_fn
        print("✓ Policy function set directly")

    def render_episode(
        self,
        max_steps: int = 500,
        seed: Optional[int] = None,
        record: bool = False,
        episode_label: str = "",
        action_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Render a single episode.

        Args:
            max_steps: Maximum steps in the episode
            seed: Random seed for reproducibility
            record: Whether to save video
            episode_label: Label for video filename
            action_fn: Custom action function (overrides loaded policy)

        Returns:
            Episode statistics dictionary

        Raises:
            RuntimeError: If recording is requested but policy is not set
                         and action_fn is not provided
        """
        if action_fn is None and self.policy is None:
            raise RuntimeError(
                "Policy or action function required. "
                "Call load_policy() or set_policy_fn()"
            )

        # Reset environment
        obs, info = self.env.reset(seed=seed)
        self.frame_buffer = []

        episode_reward = 0.0
        episode_cost = 0.0
        step_count = 0

        done = False
        while not done and step_count < max_steps:
            # Get action
            if action_fn is not None:
                action = action_fn(obs)
            else:
                action = self._policy_inference(obs)

            # Step environment (SafetyGymnasium returns 6-tuple: obs, reward, cost, terminated, truncated, info)
            obs, reward, cost, terminated, truncated, info = self.env.step(action)

            # Accumulate rewards and costs
            episode_reward += reward
            episode_cost += cost
            step_count += 1

            # Capture frame for recording
            if record and self.render_mode == "rgb_array":
                frame = self.env.render()
                if frame is not None:
                    self.frame_buffer.append(frame)

            done = terminated or truncated

        # Save video if recording
        if record and self.frame_buffer:
            video_path = self._save_video(episode_label, step_count)
            print(f"✓ Video saved: {video_path}")
        else:
            video_path = None

        return {
            "episode_reward": episode_reward,
            "episode_cost": episode_cost,
            "steps": step_count,
            "video_path": video_path,
        }

    def render_episodes(
        self,
        num_episodes: int,
        max_steps: int = 500,
        record_episodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
        action_fn: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Render multiple episodes with optional recording.

        Args:
            num_episodes: Number of episodes to render
            max_steps: Maximum steps per episode
            record_episodes: List of episode indices to record (0-indexed)
            seed: Base random seed (incremented for each episode)
            action_fn: Custom action function

        Returns:
            List of episode statistics
        """
        if record_episodes is None:
            record_episodes = []

        stats_list = []
        print(f"\n{'='*70}")
        print(f"Rendering {num_episodes} episodes (Recording: {record_episodes})")
        print(f"{'='*70}\n")

        for ep in range(num_episodes):
            should_record = ep in record_episodes
            episode_seed = seed + ep if seed is not None else None

            stats = self.render_episode(
                max_steps=max_steps,
                seed=episode_seed,
                record=should_record,
                episode_label=f"ep_{ep:03d}",
                action_fn=action_fn,
            )

            stats_list.append(stats)

            status = "REC" if should_record else "---"
            print(
                f"[{status}] Episode {ep:3d}: "
                f"Reward={stats['episode_reward']:8.2f}  "
                f"Cost={stats['episode_cost']:8.2f}  "
                f"Steps={stats['steps']:4d}"
            )

        print(f"\n{'='*70}\n")
        return stats_list

    def _policy_inference(self, obs: np.ndarray) -> np.ndarray:
        """
        Perform policy inference.

        Args:
            obs: Current observation

        Returns:
            Action to take
        """
        obs_tensor = self._to_tensor(obs)

        if isinstance(self.policy, object) and hasattr(self.policy, "__call__"):
            try:
                import torch
                if isinstance(self.policy, torch.jit.ScriptModule):
                    with torch.no_grad():
                        action = self.policy(obs_tensor)
                    return action.cpu().numpy()
            except (ImportError, AttributeError):
                pass

            # Fallback: treat as regular callable
            action = self.policy(obs)
            return np.asarray(action)

        raise RuntimeError("Policy inference failed")

    def _to_tensor(self, obs: np.ndarray):
        """
        Convert observation to appropriate tensor format.

        Args:
            obs: Observation as numpy array

        Returns:
            Tensor in framework used by policy (PyTorch by default)
        """
        try:
            import torch
            return torch.from_numpy(obs).float().unsqueeze(0)
        except ImportError:
            return obs

    def _save_video(self, episode_label: str, num_frames: int) -> Path:
        """
        Save frame buffer to video file.

        Args:
            episode_label: Label for the video filename
            num_frames: Number of frames recorded

        Returns:
            Path to saved video file

        Raises:
            RuntimeError: If imageio is not installed
        """
        if imageio is None:
            raise RuntimeError("imageio required for video recording. Install with: pip install imageio")

        video_path = self.output_dir / f"{episode_label}.mp4"

        # Ensure frames have correct dtype and shape
        frames = np.array(self.frame_buffer, dtype=np.uint8)

        try:
            imageio.mimwrite(str(video_path), frames, fps=self.fps)
        except Exception as e:
            print(f"✗ Error saving video: {e}")
            raise

        return video_path

    def set_fps(self, fps: int) -> None:
        """Set frames per second for video recording."""
        self.fps = fps

    def close(self) -> None:
        """Close the environment."""
        if self.env is not None:
            self.env.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass
