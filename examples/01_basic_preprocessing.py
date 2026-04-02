"""
示例 1: 基础环境使用与预处理

演示如何直接使用 EnvironmentWrapper 和预处理功能。
"""

import gymnasium as gym
import safety_gymnasium
from utils import EnvironmentWrapper


def example_basic_preprocessing():
    """演示基础的观察值归一化和奖励缩放。"""
    print("="*70)
    print("Example 1: Basic Environment Preprocessing")
    print("="*70)

    # 创建基础环境
    base_env = gym.make("SafetyCarGoal2-v0", level=2, render_mode="rgb_array")

    # 包装上预处理层
    env = EnvironmentWrapper(
        base_env,
        obs_normalize=True,  # 启用观察归一化
        reward_scale=0.1,    # 缩放奖励
    )

    # 运行若干步
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial observation (first 5): {obs[:5]}")

    total_reward = 0.0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if step % 20 == 0:
            print(
                f"\nStep {step:3d}: "
                f"Obs (first 5)={obs[:5]}  "
                f"Reward={reward:.4f}  "
                f"Cost={info['cost']:.4f}"
            )

        if terminated or truncated:
            break

    # 获取观察值统计
    stats = env.get_obs_stats()
    print(f"\n=== Observation Statistics ===")
    print(f"Mean (first 5): {stats['mean'][:5]}")
    print(f"Std  (first 5): {stats['std'][:5]}")
    print(f"Total steps: {step + 1}")
    print(f"Total reward: {total_reward:.4f}")

    env.close()
    print("\n✓ Example 1 completed\n")


if __name__ == "__main__":
    example_basic_preprocessing()
