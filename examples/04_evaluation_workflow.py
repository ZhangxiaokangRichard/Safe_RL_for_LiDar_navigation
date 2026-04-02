"""
示例 4: 完整的评估工作流

演示如何加载训练好的策略、渲染回合并生成评估报告。
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import MuJoCoVisualizer


def example_evaluation_workflow():
    """演示完整的评估工作流。"""
    print("="*70)
    print("Example 4: Complete Evaluation Workflow")
    print("="*70)

    # [Step 1] 创建可视化器
    output_dir = Path("./experiments/eval_20260401/videos")
    viz = MuJoCoVisualizer(
        env_id="SafetyCarGoal2-v0",
        level=2,
        render_mode="rgb_array",
        output_dir=output_dir,
    )
    viz.set_fps(60)  # 高质量录制

    print(f"\n✓ Step 1: Visualizer initialized")
    print(f"  Output dir: {output_dir.resolve()}")

    # [Step 2] 模拟加载策略（这里用随机策略代替）
    # 在实际应用中，这里会是：
    # viz.load_policy(Path("experiments/model_final.pt"), policy_type="pt")

    def trained_policy(obs):
        """模拟训练好的策略"""
        # 实际应用中这会是神经网络推理
        return np.random.randn(2) * 0.5

    viz.set_policy_fn(trained_policy)
    print(f"\n✓ Step 2: Policy loaded (simulated with random policy)")

    # [Step 3] 批量渲染
    print(f"\n✓ Step 3: Rendering episodes...")
    stats = viz.render_episodes(
        num_episodes=10,
        max_steps=500,
        record_episodes=[0, 5, 9],  # 录制第 1, 6, 10 个回合
        seed=2024,
    )

    # [Step 4] 生成评估报告
    print(f"\n{'='*70}")
    print("Evaluation Report")
    print(f"{'='*70}\n")

    rewards = [s["episode_reward"] for s in stats]
    costs = [s["episode_cost"] for s in stats]
    steps_list = [s["steps"] for s in stats]

    print(f"Total Episodes:        {len(stats)}")
    print(f"Recorded Episodes:     {len([s for s in stats if s['video_path']])}")
    print()
    print(f"Reward (Average):      {np.mean(rewards):8.2f}")
    print(f"Reward (Std):          {np.std(rewards):8.2f}")
    print(f"Reward (Min/Max):      {np.min(rewards):8.2f} / {np.max(rewards):8.2f}")
    print()
    print(f"Cost (Average):        {np.mean(costs):8.4f}")
    print(f"Cost (Std):            {np.std(costs):8.4f}")
    print(f"Cost (Min/Max):        {np.min(costs):8.4f} / {np.max(costs):8.4f}")
    print()
    print(f"Steps (Average):       {np.mean(steps_list):8.1f}")
    print(f"Steps (Std):           {np.std(steps_list):8.1f}")
    print()
    print(f"{'='*70}")

    viz.close()
    print("\n✓ Example 4 completed\n")


if __name__ == "__main__":
    example_evaluation_workflow()
