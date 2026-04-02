# Utils Module - 独立仿真组件设计文档

## 1. 设计理念

`utils/` 模块遵循**独立性原则**（Independence Principle），提供完全脱离训练框架的仿真和可视化工具。这一设计的核心优势包括：

- ✅ **完全解耦**: 仿真组件与训练循环完全独立，可在任何环境中独立调用
- ✅ **可复用性**: 支持多种使用场景（评估、演示、调试、论文配图）
- ✅ **功能聚焦**: 每个模块专注单一职责，便于维护和扩展
- ✅ **框架无关性**: 最小化对特定框架（如 OmniSafe）的依赖

---

## 2. 模块架构

```
utils/
├── __init__.py              # 模块导出
├── env_wrapper.py           # [1] 环境包装层
├── visualizer.py            # [2] 可视化引擎
├── car_goal_env.py          # [3] 高级接口（向后兼容）
└── README.md                # 本文件
```

### 模块设计图

```
┌─────────────────────────────────────────────────────────┐
│           用户代码 (Training/Evaluation)                 │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
    ┌─────▼─────────┐      ┌───────▼───────┐
    │ CarGoalEnv    │      │ MuJoCoVisualizer│
    │ (高级接口)     │      │ (独立渲染)      │
    └─────┬─────────┘      └───────┬───────┘
          │                         │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │                         │
    ┌─────▼──────────┐    ┌────────▼──────┐
    │ EnvironmentWrapper │  Policy Loader │
    │ (预处理)         │    (推理).      │
    └─────┬──────────┘    └────────┬──────┘
          │                        │
          └────────────┬───────────┘
                       │
              ┌────────▼─────────┐
              │  gymnasium/      │
              │  SafetyCarGoal2-v0│
              └──────────────────┘
```

---

## 3. 核心模块详解

### 3.1 EnvironmentWrapper - 环境包装层

**文件**: `env_wrapper.py`

**职责**: 状态预处理与动态归一化

#### 主要特性

1. **动态观察值归一化**
   ```python
   obs_normalized = (obs - μ) / (σ + ε)
   ```
   - 使用运行统计（running statistics）动态计算均值和标准差
   - 在线学习观察空间的统计特性
   - 无需预先离线处理数据

2. **奖励缩放**
   ```python
   reward_scaled = reward * reward_scale
   ```
   - 灵活的奖励缩放因子
   - 便于调整学习稳定性

3. **成本追踪**
   - 自动从环境 info 中提取 `cost` 字段
   - 用于安全约束监测

#### 使用示例

```python
from gymnasium import make
from utils import EnvironmentWrapper

# 创建基础环境
base_env = make("SafetyCarGoal2-v0", level=2, render_mode="rgb_array")

# 包装上预处理层
env = EnvironmentWrapper(base_env, obs_normalize=True, reward_scale=0.1)

# 像正常环境一样使用
obs, info = env.reset(seed=42)
for _ in range(500):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break

# 获取观察值统计（用于部署）
stats = env.get_obs_stats()
print(f"Obs mean: {stats['mean']}")
print(f"Obs std: {stats['std']}")
```

---

### 3.2 MuJoCoVisualizer - 独立渲染引擎

**文件**: `visualizer.py`

**职责**: 独立的 MuJoCo 渲染与视频录制

#### 设计要点

1. **功能独立性**
   - 与训练框架完全分离
   - 可加载任意训练好的策略（PyTorch、pickle 等）
   - 支持自定义动作函数

2. **核心 API**

   | 方法 | 功能 |
   |------|------|
   | `load_policy()` | 从文件加载策略权重 |
   | `set_policy_fn()` | 直接设置策略函数 |
   | `render_episode()` | 渲染单个回合 |
   | `render_episodes()` | 批量渲染并选择性录制 |
   | `set_fps()` | 设置视频帧率 |

3. **视频录制流程**
   ```
   Step 1: render_mode="rgb_array" → 获取 RGB 帧
   Step 2: 帧缓冲到内存列表
   Step 3: 使用 imageio 合成 MP4
   ```

#### 使用示例

##### 场景1: 加载 PyTorch 策略并录制视频

```python
from pathlib import Path
from utils import MuJoCoVisualizer

# 创建可视化器
viz = MuJoCoVisualizer(
    env_id="SafetyCarGoal2-v0",
    level=2,
    output_dir=Path("./videos")
)

# 加载训练好的策略
policy_path = Path("experiments/ppo_model_ep50.pt")
viz.load_policy(policy_path, policy_type="pt")

# 设置视频质量
viz.set_fps(60)

# 渲染并录制前 3 个回合
stats = viz.render_episodes(
    num_episodes=5,
    max_steps=500,
    record_episodes=[0, 1, 2],  # 仅录制第 0, 1, 2 回合
    seed=42
)

# 查看结果
for ep, stat in enumerate(stats):
    print(f"Episode {ep}: Reward={stat['episode_reward']:.2f}, Cost={stat['episode_cost']:.2f}")

viz.close()
```

##### 场景2: 使用自定义动作函数（不需要加载策略）

```python
import numpy as np

# 定义自定义动作函数
def random_action_fn(obs):
    """随机策略"""
    return np.random.randn(2)  # 连续的 2D 动作

# 不加载策略，直接使用自定义函数
viz = MuJoCoVisualizer(output_dir=Path("./videos"))
stats = viz.render_episodes(
    num_episodes=3,
    record_episodes=[1],  # 仅录制第 1 个回合
    action_fn=random_action_fn  # 传入自定义函数
)
```

##### 场景3: 上下文管理器（自动清理资源）

```python
with MuJoCoVisualizer(output_dir=Path("./videos")) as viz:
    viz.load_policy(Path("model.pt"))
    stats = viz.render_episodes(num_episodes=5, record_episodes=[0, 2, 4])
    # 退出时自动调用 close()
```

#### 返回数据格式

```python
stats = viz.render_episode(...)
# 返回字典:
{
    "episode_reward": 123.45,        # 累积奖励
    "episode_cost": 0.05,            # 安全约束成本
    "steps": 500,                    # 实际步数
    "video_path": Path("./videos/ep_000.mp4")  # 视频路径（如果录制）
}

stats_list = viz.render_episodes(...)
# 返回列表: [stats_dict_0, stats_dict_1, ...]
```

---

### 3.3 CarGoalEnv - 高级接口

**文件**: `car_goal_env.py`

**职责**: 向后兼容的高级接口，简化常见操作

**特点**:
- 基于 `EnvironmentWrapper` 和 `MuJoCoVisualizer` 构建
- 提供一体化的简便 API
- 支持 TOML 配置管理
- 自动资源清理

---

## 4. 数据流与集成示例

### 完整的评估流程

```python
from pathlib import Path
from utils import MuJoCoVisualizer

# [Step 1] 初始化可视化器
viz = MuJoCoVisualizer(
    env_id="SafetyCarGoal2-v0",
    level=2,
    output_dir=Path("experiments/ppo_baseline_20260401/videos")
)

# [Step 2] 加载训练好的策略
policy_path = Path("experiments/ppo_baseline_20260401/models/epoch_100.pt")
viz.load_policy(policy_path, policy_type="pt")

# [Step 3] 设置视频参数
viz.set_fps(60)

# [Step 4] 批量渲染并选择性录制
stats_list = viz.render_episodes(
    num_episodes=10,
    max_steps=500,
    record_episodes=[0, 5, 9],  # 录制前10个回合中的3个
    seed=2024
)

# [Step 5] 分析结果
total_reward = sum(s["episode_reward"] for s in stats_list)
avg_reward = total_reward / len(stats_list)
print(f"Average Reward: {avg_reward:.2f}")

# [Step 6] 清理资源
viz.close()
```

### 与训练框架的集成

```python
# train.py
from utils import EnvironmentWrapper, MuJoCoVisualizer

# 训练时使用包装层
base_env = gymnasium.make("SafetyCarGoal2-v0", level=2)
env = EnvironmentWrapper(base_env, obs_normalize=True, reward_scale=0.1)

agent = omnisafe.Agent("PPO", ...)
agent.learn(env)  # 训练

# 保存模型
agent.save("experiments/model_final.pt")

# 评估时使用可视化器（完全独立的过程）
viz = MuJoCoVisualizer(output_dir=Path("./final_videos"))
viz.load_policy(Path("experiments/model_final.pt"))
viz.render_episodes(num_episodes=5, record_episodes=[0, 2, 4])
```

---

## 5. 依赖管理

### 核心依赖

```
gymnasium >= 0.26.0
safety-gymnasium
mujoco >= 2.1.0
numpy
```

### 可选依赖

| 库 | 用途 | 安装命令 |
|----|------|--------|
| imageio | 视频录制 | `pip install imageio` |
| torch | PyTorch 策略加载 | `pip install torch` |
| tomli | Python < 3.11 的 TOML 解析 | `pip install tomli` |

### 版本检查

```python
# 检查 Python 版本兼容性
if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib
```

---

## 6. 设计模式说明

### 6.1 Wrapper 模式（EnvironmentWrapper）

```python
class EnvironmentWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs, info = self.env.reset()
        obs = self._preprocess(obs)  # 插入预处理逻辑
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._preprocess(obs)
        return obs, reward, terminated, truncated, info
```

**优势**:
- 无侵入式设计
- 可堆叠多个 Wrapper
- 符合 OpenAI Gym 标准

### 6.2 策略工厂模式（MuJoCoVisualizer）

```python
# 支持多种策略格式
viz.load_policy(path, policy_type="pt")   # PyTorch
viz.load_policy(path, policy_type="pkl")  # Pickle
viz.set_policy_fn(custom_fn)              # 自定义函数
```

**优势**:
- 开放式设计
- 支持任意策略格式
- 易于扩展新的策略类型

### 6.3 枚举模式（记录集）

```python
# render_episodes() 返回统一格式的记录列表
stats_list = [
    {
        "episode_reward": 150.5,
        "episode_cost": 0.1,
        "steps": 500,
        "video_path": Path("...")
    },
    # ... 更多记录
]
```

**优势**:
- 统一的数据格式
- 便于批处理和分析
- 与数据分析库（pandas、numpy）无缝协作

---

## 7. 最佳实践

### 7.1 内存管理

```python
# ✓ 推荐：使用上下文管理器
with MuJoCoVisualizer(...) as viz:
    viz.render_episodes(...)
    # 退出自动清理

# ✗ 不推荐：手动管理
viz = MuJoCoVisualizer(...)
try:
    viz.render_episodes(...)
finally:
    viz.close()  # 容易忘记
```

### 7.2 错误处理

```python
from pathlib import Path

try:
    viz = MuJoCoVisualizer(output_dir=Path("./videos"))
    viz.load_policy(Path("model.pt"))
except FileNotFoundError as e:
    print(f"Policy file not found: {e}")
except RuntimeError as e:
    print(f"Policy inference failed: {e}")
```

### 7.3 大批量渲染

```python
# 对于需要渲染数百个回合的情况：
# 1. 分批处理，避免内存爆炸
# 2. 只录制关键回合

for batch_id in range(num_batches):
    stats = viz.render_episodes(
        num_episodes=50,
        record_episodes=[0, 25, 49]  # 每批仅录3个
    )
    # 处理结果，释放内存
```

---

## 8. 扩展点

### 8.1 添加新的预处理方法

```python
# 在 EnvironmentWrapper 中添加
def _apply_reward_penalty(self, reward, cost):
    """添加成本惩罚"""
    return reward - 0.5 * cost
```

### 8.2 支持图像观察

```python
class CNNWrapper(EnvironmentWrapper):
    def _preprocess_obs(self, obs):
        if obs.ndim == 3:  # RGB 图像
            return self._process_image(obs)
        return super()._preprocess_obs(obs)
```

### 8.3 多环境并行渲染

```python
# 使用 multiprocessing 并行运行
from concurrent.futures import ProcessPoolExecutor

def render_single(episode_id):
    viz = MuJoCoVisualizer(...)
    return viz.render_episode(...)

with ProcessPoolExecutor(max_workers=4) as executor:
    results = executor.map(render_single, range(100))
```

---

## 9. 故障排查

### 问题1: `ModuleNotFoundError: No module named 'tomli'`

```bash
# Python < 3.11 需要安装 tomli
pip install tomli
```

### 问题2: `RuntimeError: imageio required for video recording`

```bash
pip install imageio
```

### 问题3: 视频文件为空或损坏

```python
# 检查帧缓冲是否正确
if not visualizer.frame_buffer:
    print("No frames captured!")

# 确保 render_mode 为 "rgb_array"
viz = MuJoCoVisualizer(render_mode="rgb_array")
```

---

## 10. 性能指标

### 典型性能表现

| 操作 | 平均耗时 | 备注 |
|------|--------|------|
| 单回合渲染（500步） | ~5-10s | 不含录制 |
| 单回合渲染+录制（500步@30fps） | ~15-25s | GPU 加速策略推理 |
| 10回合批量渲染 | ~50-100s | 线性扩展 |

### 优化建议

1. 使用 GPU 推理：减少策略评估时间
2. 降低视频帧率：`set_fps(15)` 而非默认的 30
3. 灵活选择录制对象：不需要录制所有回合

---

## 11. 总结

`utils/` 模块通过清晰的职责划分和函数式设计，提供了：

- ✅ **EnvironmentWrapper**: 轻量级预处理层
- ✅ **MuJoCoVisualizer**: 强大的独立渲染引擎
- ✅ **CarGoalEnv**: 便利的高级接口
- ✅ **开放架构**: 易于扩展和集成

这套设计确保了与 OmniSafe 等框架的最小耦合，同时提供了足够的灵活性用于各种研究和工程场景。
