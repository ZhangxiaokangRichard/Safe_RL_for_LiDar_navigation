# DEBUG.md — 已知问题与修复记录

本文件记录项目开发过程中遇到的所有错误，及其根本原因和最终修复方案。

---

## Bug 1：TOML 配置文件 `seed = None` 语法错误

**错误信息**
```
tomli._parser.TOMLDecodeError: Invalid value (at line 33, column 8)
```

**根本原因**
TOML 格式不支持 `None` 值（这是 Python 概念，不是 TOML 概念）。

**修复**

`config/ppo_cargoal_default.toml`：
```toml
# 之前（错误）
seed = None

# 之后（正确）
seed = 0   # 0 作为哨兵值，代码中将其转换为随机种子
```

`train.py` 中添加转换逻辑：
```python
seed = config["training"]["seed"]
if seed == 0:
    import random
    seed = random.randint(1, 10000)
```

---

## Bug 2：OmniSafe `env_level` 参数不被接受

**错误信息**
```
TypeError: AlgoWrapper.__init__() got an unexpected keyword argument 'env_level'
```

**根本原因**
`omnisafe.Agent.__init__` 只接受 4 个参数：
```python
Agent(algo, env_id, train_terminal_cfgs=None, custom_cfgs=None)
```
不接受 `env_level`、`seed`、`exp_name` 等其他参数。

**修复**

将所有非标准参数移入 `custom_cfgs`，在构造函数中直接传入：
```python
agent = omnisafe.Agent(
    "PPO",
    env_id="SafetyCarGoal2-v0",
    custom_cfgs={
        "seed": seed,
        "algo_cfgs": { ... },
        ...
    },
)
```

**受影响文件**：`train.py`、`policy/__init__.py`

---

## Bug 3：OmniSafe `seed` 参数不被接受

**错误信息**
```
TypeError: AlgoWrapper.__init__() got an unexpected keyword argument 'seed'
```

**根本原因**
同 Bug 2，`seed` 也不是 `Agent.__init__` 的直接参数，需放在 `custom_cfgs` 顶层。

**修复**
```python
custom_cfgs = {
    "seed": seed,   # 顶层，不是嵌套在其他 cfgs 中
    "train_cfgs": { ... },
    ...
}
```

---

## Bug 4：OmniSafe `exp_name` 参数不被接受

**错误信息**
```
TypeError: AlgoWrapper.__init__() got an unexpected keyword argument 'exp_name'
```

**根本原因**
同 Bug 2，`exp_name` 不是直接参数。OmniSafe 自动生成实验名，格式为：
```
PPO-{SafetyCarGoal2-v0}
```
保存路径：`./runs/PPO-{SafetyCarGoal2-v0}/seed-XXX-<timestamp>/`

**修复**
移除 `exp_name` 参数，接受 OmniSafe 自动命名。

---

## Bug 5：OmniSafe custom_cfgs 键名不匹配

**错误信息**
```
KeyError / AssertionError（custom_cfgs 中存在不支持的键）
```

**根本原因**
OmniSafe 的 `_init_config()` 调用 `recursive_check_config()` 验证 custom_cfgs 中的所有键必须存在于默认配置中。几个键名与直觉不同。

**正确的 OmniSafe custom_cfgs 结构（通过 `get_default_kwargs_yaml` 验证）**：

```python
custom_cfgs = {
    "seed": 42,
    "train_cfgs": {
        "device": "cpu",
        "total_steps": 500000,       # = num_epochs × steps_per_epoch
        # 注意：steps_per_epoch 不在这里！
    },
    "algo_cfgs": {
        "steps_per_epoch": 5000,     # 在 algo_cfgs，不在 train_cfgs
        "batch_size": 256,
        "gamma": 0.99,
        "lam": 0.95,                 # 不是 gae_lambda
        "clip": 0.2,                 # 不是 clip_ratio
        "entropy_coef": 0.0,
    },
    "model_cfgs": {
        "actor": {"lr": 0.0003},     # lr 在 model_cfgs 里，不在 algo_cfgs
        "critic": {"lr": 0.0003},
    },
    "logger_cfgs": {
        "save_model_freq": 10,
        "use_tensorboard": True,
    },
}
```

**受影响文件**：`train.py`（完整重写）

---

## Bug 6：SafetyCarGoal2-v0 不接受 `level=` 参数

**错误信息**
```
TypeError: Builder.__init__() got an unexpected keyword argument 'level'
```

**根本原因**
SafetyGymnasium 的 `Builder.__init__` 签名为：
```python
Builder(task_id, config=None, render_mode=None, width=256, height=256, ...)
```
不接受 `level` 参数。**Level 已编码在环境 ID 中**：`SafetyCarGoal2-v0` 本身就是 level 2。

**完整的可用环境 ID**：
```
SafetyCarGoal0-v0   Level 0
SafetyCarGoal1-v0   Level 1
SafetyCarGoal2-v0   Level 2（本项目）
```

**修复**：移除所有传给 `gym.make()` 的 `level=level` 参数。

**受影响文件**：`train_simple.py`、`utils/visualizer.py`、`utils/car_goal_env.py`

---

## Bug 7：SafetyGymnasium step() 返回 6-tuple，gymnasium wrapper 报错

**错误信息**
```
gymnasium.error.Error: Expected `Env.step` to return a four or five element tuple,
actual number of elements returned: 6.

# 或
ValueError: too many values to unpack (expected 5)
```

**根本原因**
SafetyGymnasium 的 `step()` 返回 **6 个值**（多出一个 `cost`）：
```python
obs, reward, cost, terminated, truncated, info = env.step(action)
```

而 gymnasium 的内置 wrapper（TimeLimit、EnvChecker）每次 step 都强制拆包为 5 个值，导致上游 wrapper 崩溃：
```python
# TimeLimit wrapper 中（gymnasium 源码）
observation, reward, terminated, truncated, info = self.env.step(action)
```

**修复方案**

直接绕过所有 gymnasium wrapper，访问原始 `Builder` 环境：
```python
import gymnasium as gym

env = gym.make("SafetyCarGoal2-v0", render_mode="rgb_array", disable_env_checker=True)
base_env = env.unwrapped  # 绕过 TimeLimit / EnvChecker

obs, info = base_env.reset(seed=42)
obs, reward, cost, terminated, truncated, info = base_env.step(action)
```

`disable_env_checker=True` 移除 EnvChecker wrapper，`.unwrapped` 绕过 TimeLimit wrapper。虽然两个都要；单独 `disable_env_checker` 尚存 TimeLimit 问题。

**受影响文件及修改**：

| 文件 | 修改内容 |
|---|---|
| `train_simple.py` | 改用 `env.unwrapped`，step 拆包改为 6 值 |
| `utils/env_wrapper.py` | 添加 `self._base_env = env.unwrapped`，step 内部拆包 6 值，对外仍返回 5-tuple（cost 放入 info） |
| `utils/visualizer.py` | 构造时 `self.env = gym.make(..., disable_env_checker=True).unwrapped`，step 改为 6 值 |
| `utils/car_goal_env.py` | `gym.make()` 加 `disable_env_checker=True` |

---

## Bug 8：Windows 终端 Unicode 编码错误

**错误信息**
```
UnicodeEncodeError: 'gbk' codec can't encode character '\u2713' in position 2:
illegal multibyte sequence
```

**根本原因**
Windows 默认终端使用 GBK 编码，无法显示 `✓`（U+2713）等 Unicode 字符。

**修复**：将 `✓` / `✗` / `⚠` 等符号替换为 ASCII：
```python
# 之前
print("✓ Agent initialized")

# 之后
print("[OK] Agent initialized")
```

**受影响文件**：`train_simple.py`、`train.py`

---

## 调试流程建议

遇到新问题时，建议按以下顺序排查：

```bash
# Step 1：验证 SafetyGymnasium 环境本身是否可用
python -c "
import gymnasium as gym, safety_gymnasium
env = gym.make('SafetyCarGoal2-v0', disable_env_checker=True).unwrapped
obs, info = env.reset(seed=42)
r = env.step(env.action_space.sample())
print(f'step 返回 {len(r)} 个值: OK')
"

# Step 2：验证 utils 模块
python -c "
from utils import EnvironmentWrapper, MuJoCoVisualizer, CarGoalEnv
print('utils 导入: OK')
"

# Step 3：验证 train_simple（不依赖 OmniSafe）
python train_simple.py --exp-name test

# Step 4：验证 OmniSafe Agent 初始化
python -c "
import omnisafe
agent = omnisafe.Agent('PPO', env_id='SafetyCarGoal2-v0', custom_cfgs={'seed': 42})
print('OmniSafe Agent 初始化: OK')
"

# Step 5：完整训练
python train.py --seed 42
```

---

## 已验证可正常运行的状态（2026-04-03）

| 组件 | 状态 |
|---|---|
| SafetyCarGoal2-v0 环境创建 | OK |
| EnvironmentWrapper（6-tuple 处理） | OK |
| MuJoCoVisualizer（unwrapped env） | OK |
| CarGoalEnv 高级接口 | OK |
| train_simple.py 环境测试 | OK |
| OmniSafe Agent 初始化 | OK |
| train.py 完整训练 | OK（已到达 epoch 1） |
