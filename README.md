# Safe RL for LiDAR Navigation — CarGoal Level 2

基于 **OmniSafe PPO** 的 SafetyCarGoal2-v0 完整训练、评估与可视化系统。

> 遇到错误？查看 [DEBUG.md](DEBUG.md) 中的已知问题和修复记录。

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 验证环境（10 步快速测试）
python train_simple.py --exp-name test

# 3. 正式训练（~30 分钟 CPU）
python train.py --seed 42

# 4. 评估并录制视频
python evaluate.py --exp-dir runs/PPO-{SafetyCarGoal2-v0}/seed-042-*
```

---

## 项目结构

```
Safe_RL_for_LiDar_navigation/
│
├── config/
│   ├── ppo_cargoal_default.toml    # 标准训练配置（100 epoch × 5000 步）
│   └── ppo_cargoal_hq.toml         # 高质量录制配置（60fps，1280×960）
│
├── utils/
│   ├── env_wrapper.py              # Layer 1：观察归一化 + 奖励缩放
│   ├── visualizer.py               # Layer 1：独立 MuJoCo 渲染 + 视频录制
│   ├── car_goal_env.py             # Layer 2：一体化高级接口
│   ├── __init__.py
│   └── README.md                   # utils API 详细文档
│
├── examples/
│   ├── 01_basic_preprocessing.py   # EnvironmentWrapper 用法演示
│   ├── 02_independent_visualization.py  # MuJoCoVisualizer 用法演示
│   ├── 03_high_level_api.py        # CarGoalEnv 用法演示
│   └── 04_evaluation_workflow.py   # 完整工作流演示
│
├── policy/
│   └── __init__.py                 # CarGoalPPOAgent（OmniSafe 适配器）
│
├── runs/                           # OmniSafe 自动生成的训练输出
│   └── PPO-{SafetyCarGoal2-v0}/
│       └── seed-XXX-<timestamp>/
│           ├── torch_save/         # 模型权重（*.pt）
│           ├── progress.csv        # 训练指标
│           └── config.json         # 配置备份
│
├── train.py                        # OmniSafe PPO 训练入口
├── train_simple.py                 # 环境连通性诊断脚本（不依赖 OmniSafe）
├── evaluate.py                     # 评估 + 视频录制入口
└── requirements.txt
```

---

## 完整工作流程

### 阶段 1：训练

```bash
# 使用默认配置
python train.py

# 指定种子（推荐，保证复现性）
python train.py --seed 42

# 使用高质量配置
python train.py --config config/ppo_cargoal_hq.toml --seed 42

# 指定 GPU
python train.py --seed 42 --device cuda:0
```

训练输出保存至：`./runs/PPO-{SafetyCarGoal2-v0}/seed-042-<timestamp>/`

训练指标可用 TensorBoard 查看：
```bash
tensorboard --logdir runs/
```

### 阶段 2：评估与录制

```bash
# 自动查找最新模型并评估，录制回合 0、5、9
python evaluate.py \
    --exp-dir runs/PPO-{SafetyCarGoal2-v0}/seed-042-<timestamp> \
    --num-episodes 20 \
    --record-episodes 0 5 9

# 高质量录制（60fps，1280×960）
python evaluate.py \
    --exp-dir runs/PPO-{SafetyCarGoal2-v0}/seed-042-<timestamp> \
    --fps 60 \
    --record-episodes 0 1 2 \
    --output-dir ./videos_hq
```

### 阶段 3：Python 脚本集成

```python
from utils import MuJoCoVisualizer
from pathlib import Path

viz = MuJoCoVisualizer(
    env_id="SafetyCarGoal2-v0",
    output_dir=Path("./my_videos")
)
viz.load_policy(Path("runs/.../torch_save/epoch-100.pt"))
stats = viz.render_episodes(num_episodes=10, record_episodes=[0, 5, 9], seed=42)

for s in stats:
    print(f"回合 {s['episode']}: 奖励={s['episode_reward']:.2f}, 成本={s['episode_cost']:.4f}")

viz.close()
```

---

## 配置文件

### `config/ppo_cargoal_default.toml`

```toml
[environment]
env_id = "SafetyCarGoal2-v0"   # Level 已编码在 ID 中（2 = level 2）
level = 2                       # 仅文档说明，不传给 gym.make()

[algorithm]
learning_rate = 0.0003          # 映射到 model_cfgs.actor/critic.lr
gamma = 0.99                    # 映射到 algo_cfgs.gamma
gae_lambda = 0.95               # 映射到 algo_cfgs.lam
clip_ratio = 0.2                # 映射到 algo_cfgs.clip
entropy_coef = 0.0              # 映射到 algo_cfgs.entropy_coef

[training]
num_epochs = 100
steps_per_epoch = 5000          # 映射到 algo_cfgs.steps_per_epoch
batch_size = 256                # 映射到 algo_cfgs.batch_size
save_freq = 10
seed = 0                        # 0 = 每次随机生成种子

[rendering]
fps = 30
width = 640
height = 480
```

### OmniSafe 参数映射关系

TOML 参数与 OmniSafe `custom_cfgs` 的对应关系（容易出错，务必注意）：

| TOML 字段 | OmniSafe custom_cfgs 路径 | 备注 |
|---|---|---|
| `algorithm.learning_rate` | `model_cfgs.actor.lr` 和 `model_cfgs.critic.lr` | 不在 algo_cfgs 中 |
| `algorithm.clip_ratio` | `algo_cfgs.clip` | 注意不是 `clip_ratio` |
| `algorithm.gae_lambda` | `algo_cfgs.lam` | 注意不是 `gae_lambda` |
| `training.steps_per_epoch` | `algo_cfgs.steps_per_epoch` | 注意不在 train_cfgs 中 |
| `training.num_epochs × steps_per_epoch` | `train_cfgs.total_steps` | |
| `training.seed` | `seed`（顶层） | |

---

## 模块 API 说明

### 三层 API 设计

```
Layer 3（最简单） → train.py / evaluate.py      一行命令运行
Layer 2（中等）  → CarGoalEnv                   一体化Python接口
Layer 1（最灵活）→ EnvironmentWrapper + MuJoCoVisualizer   逐组件定制
```

### Layer 1：EnvironmentWrapper

```python
import gymnasium as gym
from utils import EnvironmentWrapper

# 注意：必须 disable_env_checker=True（SafetyGymnasium 返回 6-tuple）
base_env = gym.make("SafetyCarGoal2-v0", disable_env_checker=True)
env = EnvironmentWrapper(base_env, obs_normalize=True, reward_scale=1.0)

obs, info = env.reset(seed=42)
# step() 返回标准 5-tuple；cost 存入 info["cost"]
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(f"cost: {info['cost']}")
```

### Layer 1：MuJoCoVisualizer

```python
from utils import MuJoCoVisualizer
from pathlib import Path

viz = MuJoCoVisualizer(
    env_id="SafetyCarGoal2-v0",
    render_mode="rgb_array",
    output_dir=Path("./videos")
)
viz.load_policy(Path("model.pt"), policy_type="pt")
viz.set_fps(30)

stats = viz.render_episodes(
    num_episodes=10,
    max_steps=500,
    record_episodes=[0, 5, 9],
    seed=42
)
viz.close()
```

主要方法：

| 方法 | 说明 |
|---|---|
| `load_policy(path, policy_type)` | 加载策略文件（pt 或 pkl） |
| `set_policy_fn(fn)` | 设置自定义动作函数 |
| `render_episode(record, seed)` | 渲染单个回合 |
| `render_episodes(n, record_episodes)` | 批量渲染 |
| `set_fps(fps)` | 设置视频帧率 |
| `close()` | 释放环境资源 |

### Layer 2：CarGoalEnv

```python
from utils import CarGoalEnv

# 使用默认配置
env = CarGoalEnv()
env.create_env(render_mode="rgb_array", enable_recording=True)

stats = env.run_episodes(num_episodes=5, record_episodes=[0, 2, 4])
for s in stats:
    print(f"回合 {s['episode']}: 奖励={s['total_reward']:.2f}")

env.close()

# 使用 TOML 配置
env2 = CarGoalEnv(config_path="config/ppo_cargoal_default.toml")
env2.create_env()
env2.run_episodes(10)
```

---

## SafetyGymnasium 环境说明

### 环境 ID 与 Level 的关系

Level 编码在环境 ID 中，**不接受单独的 `level=` 参数**：

| 环境 ID | Level |
|---|---|
| `SafetyCarGoal0-v0` | Level 0（最简单） |
| `SafetyCarGoal1-v0` | Level 1 |
| `SafetyCarGoal2-v0` | Level 2（最难，本项目使用） |

### Step 返回值

SafetyGymnasium 的 `step()` 返回 **6 个值**（多一个 cost）：

```python
obs, reward, cost, terminated, truncated, info = env.unwrapped.step(action)
```

使用 `gym.make()` 包装器时必须加 `disable_env_checker=True` 并使用 `.unwrapped`，否则 gymnasium 内置的 TimeLimit wrapper 会因 6-tuple 报错。

---

## 依赖

```
omnisafe==0.5.0       # PPO 训练框架
safety-gymnasium      # 安全约束环境
gymnasium             # Gym API
mujoco                # 物理引擎
tomli >= 1.1.0        # TOML 解析（Python < 3.11）
imageio >= 2.9.0      # 视频编码
tensorboard           # 训练曲线可视化
```

安装：
```bash
pip install -r requirements.txt
```

---

## 性能参考

| 操作 | 耗时（CPU） |
|---|---|
| 环境测试（10 步） | < 1 秒 |
| 单回合渲染（500 步） | ~1-2 秒 |
| 单回合 + 视频录制（30fps） | ~5 秒 |
| 完整训练（100 epoch × 5000 步） | ~30-120 分钟 |

---

## 常见问题

**Q: `ModuleNotFoundError: No module named 'tomli'`**
```bash
pip install tomli
```

**Q: 训练太慢**
- 减少 `num_epochs`（快速验证用 10）
- 增大 `batch_size`（256 → 512）
- 使用 GPU：`python train.py --device cuda:0`

**Q: 视频录制为空**
- 确保 `render_mode="rgb_array"`（不是 `"human"`）
- 使用 `env.unwrapped` 访问底层环境

**Q: OmniSafe 训练报错**
- 查看 [DEBUG.md](DEBUG.md) 中 OmniSafe API 兼容性修复记录
- 先运行 `python train_simple.py` 验证环境连通性

---

## 版本信息

```
环境:    SafetyCarGoal2-v0 Level 2
框架:    OmniSafe 0.5.0 + Safety-Gymnasium
Python:  3.10+
状态:    训练已验证可运行
```
