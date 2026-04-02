"""
示例 3: 高级接口 (CarGoalEnv) 的完整工作流

演示如何使用 CarGoalEnv 高级接口进行快速原型设计。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import CarGoalEnv


def example_high_level_api():
    """演示 CarGoalEnv 高级接口。"""
    print("="*70)
    print("Example 3: High-Level CarGoalEnv API")
    print("="*70)

    # 创建环境（使用默认配置）
    env = CarGoalEnv()

    # 初始化环境
    env.create_env(render_mode="rgb_array", enable_recording=True)

    print("\n✓ Environment created with recording enabled\n")

    # 运行多个回合，仅录制第 1 和 3 个
    stats = env.run_episodes(
        num_episodes=5,
        max_steps_per_episode=200,
        record_episodes=[1, 3],
        seed=2024,
    )

    # 打印统计信息
    print(f"\n=== Final Statistics ===")
    for i, stat in enumerate(stats):
        print(
            f"  Episode {i}: "
            f"Reward={stat['total_reward']:8.2f}  "
            f"Cost={stat['total_cost']:8.4f}  "
            f"Steps={stat['steps']:4d}"
        )

    env.close()
    print("\n✓ Example 3 completed\n")


if __name__ == "__main__":
    example_high_level_api()
