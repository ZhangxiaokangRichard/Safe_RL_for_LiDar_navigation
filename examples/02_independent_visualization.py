"""
示例 2: 独立可视化引擎

演示如何使用 MuJoCoVisualizer 独立渲染和录制视频。
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import MuJoCoVisualizer


def example_random_policy():
    """演示使用随机策略进行渲染和录制。"""
    print("="*70)
    print("Example 2: Independent Visualization with Random Policy")
    print("="*70)

    # 创建可视化器
    output_dir = Path("./videos/example_02")
    viz = MuJoCoVisualizer(
        env_id="SafetyCarGoal2-v0",
        level=2,
        output_dir=output_dir,
    )

    # 设置视频参数
    viz.set_fps(30)

    # 定义随机策略
    def random_policy(obs):
        """随机动作策略"""
        return np.random.randn(2)  # SafetyCarGoal2-v0 是连续 2D 动作空间

    print(f"\n✓ Visualizer created, output dir: {output_dir.resolve()}\n")

    # 渲染并录制
    stats = viz.render_episodes(
        num_episodes=5,
        max_steps=300,
        record_episodes=[0, 2, 4],  # 仅录制第 0, 2, 4 回合
        seed=42,
        action_fn=random_policy,
    )

    # 分析统计信息
    print(f"\n=== Episode Statistics ===")
    rewards = [s["episode_reward"] for s in stats]
    costs = [s["episode_cost"] for s in stats]
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Cost:   {np.mean(costs):.4f} ± {np.std(costs):.4f}")

    viz.close()
    print("\n✓ Example 2 completed\n")


if __name__ == "__main__":
    example_random_policy()
