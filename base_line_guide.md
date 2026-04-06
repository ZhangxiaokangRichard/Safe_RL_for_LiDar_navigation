# Safe RL Baseline Experiment — Comprehensive Design Guide

> **范围**：PPO / PPOLag 在 SafetyCarGoal2-v0 上的训练、迁移至 SafetyCarGoal1-v0 的评估、在含动态障碍物 SafetyCarButton1-v0 上的训练，以及六类综合输出的完整生成流程。
>
> **参阅**：[doc/ch2_01.md](doc/ch2_01.md) · [doc/ch2_02.md](doc/ch2_02.md) · [doc/ch3_03.md](doc/ch3_03.md)

---

## 一、整体实验管线

```
┌─ Phase 1 ──────────────────────────────────────────────────────────┐
│  CarGoal2 训练                                                      │
│  train.py ──► runs/PPO-{CarGoal2}/     (100 epochs, 500K steps)    │
│  train.py ──► runs/PPOLag-{CarGoal2}/  (100 epochs, 500K steps)    │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                                     ▼
┌─ Phase 2 ──────────────┐          ┌─ Phase 3 ──────────────────────┐
│  CarGoal1 迁移评估      │          │  CarButton1 训练               │
│  cross_eval.py          │          │  train.py ──► PPO  (300 epoch) │
│  runs/PPO-transfer-/   │          │  train.py ──► PPOLag(300 epoch) │
│  runs/PPOLag-transfer-/ │          └──────────────┬─────────────────┘
└────────────┬────────────┘                         │
             │                                      │
             └──────────────────┬───────────────────┘
                                ▼
             ┌─ Phase 4 ─────────────────────────────────────────────┐
             │  输出生成（六类）                                        │
             │  compare.py         → 奖励/代价训练曲线                 │
             │  evaluate.py        → 最优回合录像 (MP4)                │
             │  trajectory_viz.py  → 2D 路径地图                      │
             │  cross_eval.py      → 迁移评估报告                      │
             └───────────────────────────────────────────────────────┘
```

---

## 二、代码模块架构

### 2.1 现有模块

| 脚本 | 职责 | 关键参数 |
|------|------|---------|
| `train.py` | OmniSafe 训练入口，TOML → custom_cfgs | `--config` `--algo` `--seed` `--device` |
| `evaluate.py` | OmniSafe Evaluator 加载模型、评估、录视频 | `--exp-dir` `--num-episodes` `--record-episodes` |
| `compare.py` | OmniSafe Plotter 生成奖励/代价曲线 | `--runs` `--legend` `--cost-limit` `--save-dir` |
| `cross_eval.py` | CarGoal2 权重迁移至 CarGoal1，patched config | `--ppo-dir` `--ppolag-dir` `--num-episodes` |

### 2.2 新增模块

| 脚本 | 职责 | 关键参数 |
|------|------|---------|
| `trajectory_viz.py` | 2D 俯视轨迹可视化，代价着色，多回合叠加 | `--exp-dir` `--env-id` `--num-episodes` `--best-only` `--overlay` |

### 2.3 配置文件

| 文件 | 环境 | 算法 | 说明 |
|------|------|------|------|
| `experiments/base_line/ppo_config.toml` | CarGoal2 | PPO | 参照组（已有结果） |
| `experiments/base_line/ppollag_config.toml` | CarGoal2 | PPOLag | 安全基准（已有结果） |
| `experiments/base_line/ppo_button_config.toml` | CarButton1 | PPO | 动态障碍实验 |
| `experiments/base_line/ppolag_button_config.toml` | CarButton1 | PPOLag | 安全+动态障碍 |

---

## 三、实验配置参数全表

### 3.1 算法超参数

| 参数 | 符号 | CarGoal2 | CarButton1 | 调整原因 |
|------|------|---------|-------------|---------|
| 学习率 | $\alpha$ | 3e-4 | 3e-4 | 网络宽度不变 |
| 折扣因子 | $\gamma$ | 0.99 | 0.99 | — |
| GAE λ | $\lambda_{\text{GAE}}$ | 0.95 | 0.95 | — |
| PPO clip | $\varepsilon$ | 0.2 | 0.2 | — |
| 熵系数 | $\beta$ | 0.0 | **0.01** | Button 探索更难 |
| Batch size | — | 256 | 256 | — |

### 3.2 训练规模

| 阶段 | 环境 | Epochs | Steps/Epoch | 总步数 | 保存频率 |
|------|------|--------|------------|-------|---------|
| Phase 1 | CarGoal2 | 100 | 5,000 | 500K | 10 |
| Phase 3 | CarButton1 | **300** | 5,000 | **1.5M** | 20 |

### 3.3 PPOLag 安全参数

| 参数 | CarGoal | CarButton1 | 说明 |
|------|---------|-------------|------|
| `cost_limit` $d$ | 25.0 | 25.0 | 与 OmniSafe 标准基准对齐 |
| `lagrangian_multiplier_init` $\lambda_0$ | 0.001 | 0.001 | 初始值远小于 1 |
| `lambda_lr` | 0.035 | 0.035 | 乘子更新步长 |
| `lambda_optimizer` | Adam | Adam | — |

### 3.4 环境参数对比

| 参数 | CarGoal1 | CarGoal2 | CarButton1 |
|------|---------|---------|-------------|
| 竞技场 | 3m×3m | 4m×4m | 3m×3m |
| Obs 维度 | 72 | 72 | **88** |
| Action 维度 | 2 | 2 | 2 |
| Hazards 数 | 8 | 10 | 4 |
| 动态障碍物 | — | — | **4 Gremlins** |
| 代价类型数 | 1 | 2 | **3** |
| 目标机制 | 随机传送 | 随机传送 | **4 按钮中随机选一** |
| 感知盲视期 | — | — | **10 步** |

---

## 四、六类输出设计

### 4.1 奖励/代价训练曲线（`compare.py`）

**生成工具**：OmniSafe 内置 `Plotter`，读 `progress.csv`

**输出格式**：PNG（默认），可选 PDF/SVG

| 图表 | 文件名 | 内容 |
|------|-------|------|
| 奖励曲线 | `reward_curve.png` | 各算法 EpRet 随 Steps 变化（滚动均值平滑） |
| 代价曲线 | `cost_curve.png` | 各算法 EpCost + cost_limit=25 参考虚线 |

**命令**：

```bash
# Phase 1 结果对比（CarGoal2）
python compare.py \
    --runs runs/PPO-{SafetyCarGoal2-v0}/seed-042-* \
           runs/PPOLag-{SafetyCarGoal2-v0}/seed-042-* \
    --legend "PPO (CarGoal2)" "PPOLag (CarGoal2)" \
    --save-dir experiments/base_line/plots/

# Phase 3 结果对比（CarButton1）
python compare.py \
    --runs runs/PPO-{SafetyCarButton1-v0}/seed-042-* \
           runs/PPOLag-{SafetyCarButton1-v0}/seed-042-* \
    --legend "PPO (Button)" "PPOLag (Button)" \
    --save-dir experiments/base_line/plots_button/

# 跨任务四线对比（CarGoal1 迁移 + Button 训练）
python compare.py \
    --runs runs/PPO-{SafetyCarGoal2-v0}/seed-042-* \
           runs/PPOLag-{SafetyCarGoal2-v0}/seed-042-* \
           runs/PPO-{SafetyCarButton1-v0}/seed-042-* \
           runs/PPOLag-{SafetyCarButton1-v0}/seed-042-* \
    --legend "PPO-CarGoal2" "PPOLag-CarGoal2" "PPO-Button" "PPOLag-Button" \
    --save-dir experiments/base_line/plots_cross/
```

**预期曲线特征**：

```
CarGoal2:
  PPO奖励    — 前20epoch快速爬升至10~15，100epoch稳定在15~25
  PPOLag奖励 — 爬升更慢，100epoch约1~5（代价约束压低奖励）
  PPOLag代价 — 从200+下降，100epoch约50~120（未完全收敛）

CarButton1（更难，需300epoch）：
  PPO奖励    — epoch100前缓慢，100~200出现跳跃，最终15~30
  PPOLag代价 — Lagrange乘子epoch50后开始显著上升
```

---

### 4.2 回合录像（`evaluate.py`）

**生成工具**：OmniSafe Evaluator → moviepy → MP4

**最优回合选择**：评估多回合后选 reward 最高者录制

| 录像类型 | 命名 | 说明 |
|---------|------|------|
| 普通评估帧 | `eval-episode-N.mp4` | OmniSafe 默认录制，前 N 回合 |
| 最优回合 | 选取 reward 最大回合 | 通过 `--record-episodes` 控制数量 |

**命令（以 CarGoal2 PPO 为例）**：

```bash
PPO_DIR="runs/PPO-{SafetyCarGoal2-v0}/seed-042-2026-04-03-03-37-24"

# 评估 20 回合，录制最高奖励的 3 回合
python evaluate.py \
    --exp-dir "$PPO_DIR" \
    --num-episodes 20 \
    --record-episodes 3 \
    --width 640 --height 480

# CarButton1 PPOLag 评估
python evaluate.py \
    --exp-dir "runs/PPOLag-{SafetyCarButton1-v0}/seed-042-*" \
    --num-episodes 20 \
    --record-episodes 3
```

**输出路径**：`$EXP_DIR/videos/eval-episode-{0,1,2}.mp4`

---

### 4.3 2D 路径地图（`trajectory_viz.py`）

**生成工具**：新增脚本，matplotlib 俯视图

**可视化元素**：

| 元素 | 颜色/形状 | 说明 |
|------|----------|------|
| 竞技场边界 | 灰边黄底方框 | 真实物理边界 |
| Hazard 区域 | 红色半透明圆 + 红圈 | 中心标"+" |
| Vase（CarGoal） | 灰色圆 | 不透明度 0.4 |
| 按钮（Button） | 紫色圆 + 编号 | 4 个固定位置 |
| Gremlin 路径圈 | 橙色虚线圆 | 圆心为初始位置，半径=travel=0.35m |
| 目标位置 | 黄色五角星 | 每次刷新后记录新位置 |
| 轨迹（安全步） | 绿色线段 | cost=0 时 |
| 轨迹（违规步） | 红色线段 | cost>0 时 |
| 起点/终点 | 绿圆/红三角 | 明显标注 |

**命令**：

```bash
# CarGoal2 PPO — 5 回合，每回合单独保存
python trajectory_viz.py \
    --exp-dir runs/PPO-{SafetyCarGoal2-v0}/seed-042-* \
    --num-episodes 5

# CarGoal1 迁移 — 仅保存最优回合 + 叠加图
python trajectory_viz.py \
    --exp-dir runs/PPO-transfer-{SafetyCarGoal1-v0}/seed-042-* \
    --env-id SafetyCarGoal1-v0 \
    --num-episodes 10 \
    --best-only --overlay

# CarButton1 PPOLag — 5 回合叠加（展示 Gremlin 规避）
python trajectory_viz.py \
    --exp-dir runs/PPOLag-{SafetyCarButton1-v0}/seed-042-* \
    --num-episodes 5 \
    --overlay
```

**输出路径**：`$EXP_DIR/trajectories/`

```
trajectories/
    trajectory_ep01.png   # 单回合地图（绿/红轨迹）
    trajectory_ep02.png
    trajectory_ep03.png
    ...
    trajectory_overlay.png  # 所有回合叠加（tab10 配色）
    best_trajectory.png     # （--best-only 模式）
```

**示例图说明**：

```
[CarGoal1 PPO 轨迹图 — 典型安全回合]

  竞技场 3m×3m，8 个红色 Hazard 圆（r=0.2m）
  分布在竞技场各角落，Goal ★ 随机出现
  绿线段：直线冲向目标
  红线段：穿越 Hazard 时出现（PPO 无安全约束）
  轨迹密集处 = 多次 Goal 区域

[CarButton1 PPOLag 轨迹图 — 安全策略特征]
  4 个橙色虚线圆 = Gremlin 运动轨迹
  蓝紫色按钮标 0~3，Goal ★ 跳跃指向目标按钮
  PPOLag 轨迹绕过 Gremlin 圆圈外侧（代价规避行为可见）
```

---

### 4.4 迁移评估报告（`cross_eval.py`）

**原理**：复制 `config.json ` 并修改 `env_id`，CarGoal2→CarGoal1 两者 obs/action 维度相同（均 72 维），权重无需任何转换。

**命令**：

```bash
python cross_eval.py \
    --ppo-dir   runs/PPO-{SafetyCarGoal2-v0}/seed-042-2026-04-03-03-37-24 \
    --ppolag-dir runs/PPOLag-{SafetyCarGoal2-v0}/seed-042-2026-04-03-05-45-15 \
    --num-episodes 10 --record-episodes 3
```

**输出**：

```
runs/
  PPO-transfer-{SafetyCarGoal1-v0}/seed-042-*/
      config.json          ← patched (env_id=SafetyCarGoal1-v0)
      torch_save/epoch-100.pt
      videos/
          eval-episode-{0,1,2}.mp4
          transfer_result.json   ← reward/cost 统计
  PPOLag-transfer-{SafetyCarGoal1-v0}/seed-042-*/
      ...（同上）
```

**已有实验结果**：

| 算法 | 源环境 | 目标环境 | 奖励均值 | 代价均值 | 零代价率 |
|------|-------|---------|---------|---------|---------|
| PPO | CarGoal2 | CarGoal1 | 27.65 | 60.5 | 20% |
| PPOLag | CarGoal2 | CarGoal1 | 1.76 | 27.2 | 60% |

迁移到更简单的 CarGoal1 后：奖励提升（+24% / +33%），代价显著下降（-69% / -77%）。

---

### 4.5 Lagrange 乘子曲线（PPOLag 专属）

**数据来源**：`progress.csv` 中的 `Metrics/LagrangeMultiplier` 列

**生成命令**（直接读 CSV，用 matplotlib 绘制）：

```python
import pandas as pd, matplotlib.pyplot as plt, glob

# 找到 PPOLag 的 progress.csv
csv_files = glob.glob("runs/PPOLag-*/**/progress.csv", recursive=True)
for path in csv_files:
    df = pd.read_csv(path)
    lag_col = [c for c in df.columns if "Lag" in c]
    if not lag_col:
        continue
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(df["Train/Epoch"], df["Metrics/EpCost/Mean"], label="EpCost")
    ax1.axhline(25, linestyle="--", color="red", label="cost_limit=25")
    ax2.plot(df["Train/Epoch"], df[lag_col[0]], color="orange", label="λ")
    ax1.set_ylabel("Episode Cost"); ax2.set_ylabel("Lagrange λ")
    ax2.set_xlabel("Epoch")
    plt.tight_layout(); plt.savefig("lagrange_curve.png", dpi=150)
```

**预期曲线形状**：

```
Cost (上图):
  ████████████████ → 逐渐下降 → 震荡收敛至 cost_limit 附近

λ (下图):
  0.001 → 缓慢上升 → 加速 → 趋于稳定的正值
```

---

### 4.6 综合评估表（横向比较）

所有实验完成后，汇总同一个对比表：

```python
# 从 progress.csv 读取各算法最终性能
import pandas as pd, json, glob, numpy as np

rows = []
for cfg_path in glob.glob("runs/**/config.json", recursive=True):
    with open(cfg_path) as f:
        cfg = json.load(f)
    csv_path = cfg_path.replace("config.json", "progress.csv")
    if not __import__("os").path.exists(csv_path): continue
    df = pd.read_csv(csv_path)
    rows.append({
        "Algo": cfg.get("algo"),
        "Env": cfg.get("env_id"),
        "Reward_mean": df["Metrics/EpRet/Mean"].mean(),
        "Reward_final": df["Metrics/EpRet/Mean"].iloc[-1],
        "Cost_mean": df["Metrics/EpCost/Mean"].mean(),
        "Cost_final": df["Metrics/EpCost/Mean"].iloc[-1],
        "Epochs": len(df),
    })
summary = pd.DataFrame(rows)
print(summary.to_markdown(index=False))
```

---

## 五、阶段化执行手册

### Phase 1：CarGoal2 训练（已完成）

```bash
# 结果已在 runs/ 下：
#   PPO-{SafetyCarGoal2-v0}/seed-042-2026-04-03-03-37-24  (100 epochs)
#   PPOLag-{SafetyCarGoal2-v0}/seed-042-2026-04-03-05-45-15 (100 epochs)
# 如需重新训练：
python train.py --config experiments/base_line/ppo_config.toml
python train.py --config experiments/base_line/ppollag_config.toml
```

### Phase 2：CarGoal1 迁移评估（已完成）

```bash
# 结果已在 runs/ 下：
#   PPO-transfer-{SafetyCarGoal1-v0}/  PPOLag-transfer-{SafetyCarGoal1-v0}/
# 如需重新执行：
python cross_eval.py \
    --ppo-dir    runs/PPO-{SafetyCarGoal2-v0}/seed-042-2026-04-03-03-37-24 \
    --ppolag-dir runs/PPOLag-{SafetyCarGoal2-v0}/seed-042-2026-04-03-05-45-15

# 轨迹可视化
python trajectory_viz.py \
    --exp-dir runs/PPO-transfer-{SafetyCarGoal1-v0}/seed-042-* \
    --num-episodes 10 --overlay
python trajectory_viz.py \
    --exp-dir runs/PPOLag-transfer-{SafetyCarGoal1-v0}/seed-042-* \
    --num-episodes 10 --overlay
```

### Phase 3：CarButton1 训练（待执行）

```bash
# 步骤 3a — 训练 PPO（约 2~3 小时/CPU）
python train.py --config experiments/base_line/ppo_button_config.toml

# 步骤 3b — 评估 + 录像
python evaluate.py \
    --exp-dir runs/PPO-{SafetyCarButton1-v0}/seed-042-* \
    --num-episodes 20 --record-episodes 3

# 步骤 3c — 轨迹图
python trajectory_viz.py \
    --exp-dir runs/PPO-{SafetyCarButton1-v0}/seed-042-* \
    --num-episodes 5 --overlay

# 步骤 3d — 训练 PPOLag（约 2~3 小时/CPU）
python train.py --config experiments/base_line/ppolag_button_config.toml

# 步骤 3e — 评估 + 录像 + 轨迹
python evaluate.py \
    --exp-dir runs/PPOLag-{SafetyCarButton1-v0}/seed-042-* \
    --num-episodes 20 --record-episodes 3
python trajectory_viz.py \
    --exp-dir runs/PPOLag-{SafetyCarButton1-v0}/seed-042-* \
    --num-episodes 5 --overlay
```

### Phase 4：汇总输出

```bash
# 训练曲线（CarGoal2 Phase 1）
python compare.py \
    --runs runs/PPO-{SafetyCarGoal2-v0}/seed-042-* \
           runs/PPOLag-{SafetyCarGoal2-v0}/seed-042-* \
    --legend "PPO(CarGoal2)" "PPOLag(CarGoal2)" \
    --save-dir experiments/base_line/plots/

# 训练曲线（CarButton1 Phase 3）
python compare.py \
    --runs runs/PPO-{SafetyCarButton1-v0}/seed-042-* \
           runs/PPOLag-{SafetyCarButton1-v0}/seed-042-* \
    --legend "PPO(Button)" "PPOLag(Button)" \
    --save-dir experiments/base_line/plots_button/

# 跨任务四线对比
python compare.py \
    --runs runs/PPO-{SafetyCarGoal2-v0}/seed-042-* \
           runs/PPOLag-{SafetyCarGoal2-v0}/seed-042-* \
           runs/PPO-{SafetyCarButton1-v0}/seed-042-* \
           runs/PPOLag-{SafetyCarButton1-v0}/seed-042-* \
    --legend "PPO-CarGoal2" "PPOLag-CarGoal2" "PPO-Button" "PPOLag-Button" \
    --save-dir experiments/base_line/plots_cross/
```

---

## 六、输出文件树（完整）

```
runs/
├── PPO-{SafetyCarGoal2-v0}/seed-042-*/          [Phase 1 ✅]
│   ├── config.json / progress.csv / torch_save/
│   ├── videos/eval-episode-{0,1,2}.mp4
│   └── trajectories/trajectory_ep{01..05}.png · overlay.png
│
├── PPOLag-{SafetyCarGoal2-v0}/seed-042-*/        [Phase 1 ✅]
│   └── ... (同上)
│
├── PPO-transfer-{SafetyCarGoal1-v0}/seed-042-*/  [Phase 2 ✅]
│   ├── config.json (patched) / torch_save/
│   ├── videos/eval-episode-{0,1,2}.mp4 · transfer_result.json
│   └── trajectories/...
│
├── PPOLag-transfer-{SafetyCarGoal1-v0}/seed-042-/ [Phase 2 ✅]
│   └── ...
│
├── PPO-{SafetyCarButton1-v0}/seed-042-*/        [Phase 3 ⏳]
│   ├── config.json / progress.csv / torch_save/
│   ├── videos/eval-episode-{0,1,2}.mp4
│   └── trajectories/trajectory_ep{01..05}.png · overlay.png
│
└── PPOLag-{SafetyCarButton1-v0}/seed-042-*/     [Phase 3 ⏳]
    └── ...

experiments/base_line/
├── ppo_config.toml / ppollag_config.toml         [CarGoal2]
├── ppo_button_config.toml / ppolag_button_config.toml [Button]
├── plots/
│   ├── reward_curve.png     ←  CarGoal2 PPO vs PPOLag
│   └── cost_curve.png
├── plots_button/
│   ├── reward_curve.png     ←  CarButton1 PPO vs PPOLag
│   └── cost_curve.png
└── plots_cross/
    ├── reward_curve.png     ←  4 条线跨任务
    └── cost_curve.png
```

---

## 七、预期综合结果表（目标基准）

| 指标 | PPO-CarGoal2 | PPOLag-CarGoal2 | PPO→CarGoal1 | PPOLag→CarGoal1 | PPO-Button | PPOLag-Button |
|------|------------|----------------|-------------|----------------|-----------|--------------|
| 训练奖励(mean) | ~14 | ~1.5 | — | — | ~10 | ~3 |
| 训练代价(mean) | ~200 | ~86 | — | — | ~250 | ~40 |
| 评估奖励(mean) | ~22 | ~1.3 | **27.65** ✅ | **1.76** ✅ | ~20 | ~5 |
| 评估代价(mean) | ~194 | ~119 | **60.5** ✅ | **27.2** ✅ | ~200 | ~30 |
| 零代价率 | — | — | **20%** ✅ | **60%** ✅ | <5% | ~40% |
| Gremlin 规避 | — | — | — | — | 无意识 | 主动绕行 |

> **✅** = 已有实验结果；其余为预期区间，待 Phase 3 完成后更新。
>
> Button 任务代价普遍高于 CarGoal，因 Gremlin 周期运动会主动经过智能体路径。PPOLag 在 Button 环境中预期更节制但奖励偏低（动态障碍难以完全规避）。

---

## 八、trajectory_viz.py 工作原理简介

```
用户调用 trajectory_viz.py
│
├─ 1. 解析 --exp-dir 和 --env-id
│      若 env-id 与 config.json 不同 → 创建 tmp dir + patched config
│
├─ 2. omnisafe.Evaluator().load_saved()
│      内部完成：读 config.json → 重建 actor 网络 → 加载 epoch-N.pt 权重
│
├─ 3. 访问 evaluator._env（内部包装环境）
│      _unwrap_to_base() 逐层 .env 直到 safety-gymnasium Builder
│
├─ 4. 手动 Episode 循环（N 回合）
│      每步：
│        env.unwrapped.task.agent.pos[:2]  → 记录 agent (x, y)
│        env.unwrapped.task.hazards.pos     → 记录障碍物位置（重置后固定）
│        env.unwrapped.task.goal.pos[:2]   → 记录 Goal 当前位置
│        actor.predict(obs, deterministic=True) → 动作
│        env.step(act)                     → 更新状态，收集 cost
│
└─ 5. matplotlib 渲染（每回合一图 + 叠加图）
       轨迹线段用 LineCollection 按 cost>0 着色（RdYlGn_r colormap）
       静态障碍物圆圈 + Gremlin 运动路径环
       保存 PNG 至 <exp-dir>/trajectories/
```

**注意**：若 `evaluator._actor` 属性名在 OmniSafe 版本间有差异，脚本已添加 `_agent` 后备路径。如遇 `AttributeError`，可检查 `dir(evaluator)` 输出并在 `trajectory_viz.py` 第 92 行附近更新属性名。

---

## 九、快速参考卡

```bash
# ══ 已有结果可直接生成的输出 ══════════════════════════════════════

# 1. CarGoal2 训练曲线
python compare.py --runs runs/PPO-{SafetyCarGoal2-v0}/seed-042-* \
    runs/PPOLag-{SafetyCarGoal2-v0}/seed-042-* \
    --legend "PPO" "PPOLag" --save-dir experiments/base_line/plots/

# 2. CarGoal1 迁移轨迹
python trajectory_viz.py \
    --exp-dir runs/PPO-transfer-{SafetyCarGoal1-v0}/seed-042-* \
    --num-episodes 10 --overlay
python trajectory_viz.py \
    --exp-dir runs/PPOLag-transfer-{SafetyCarGoal1-v0}/seed-042-* \
    --num-episodes 10 --overlay

# ══ 待完成（Phase 3 Button 训练后）══════════════════════════════

# 3. Button PPO 训练
python train.py --config experiments/base_line/ppo_button_config.toml

# 4. Button PPOLag 训练
python train.py --config experiments/base_line/ppolag_button_config.toml

# 5. Button 评估 + 录像 + 轨迹（训练完成后）
python evaluate.py --exp-dir runs/PPO-{SafetyCarButton1-v0}/seed-042-* \
    --num-episodes 20 --record-episodes 3
python trajectory_viz.py --exp-dir runs/PPO-{SafetyCarButton1-v0}/seed-042-* \
    --num-episodes 5 --overlay
python evaluate.py --exp-dir runs/PPOLag-{SafetyCarButton1-v0}/seed-042-* \
    --num-episodes 20 --record-episodes 3
python trajectory_viz.py --exp-dir runs/PPOLag-{SafetyCarButton1-v0}/seed-042-* \
    --num-episodes 5 --overlay

# 6. 最终跨任务综合对比曲线
python compare.py \
    --runs runs/PPO-{SafetyCarGoal2-v0}/seed-042-* \
           runs/PPOLag-{SafetyCarGoal2-v0}/seed-042-* \
           runs/PPO-{SafetyCarButton1-v0}/seed-042-* \
           runs/PPOLag-{SafetyCarButton1-v0}/seed-042-* \
    --legend "PPO-CarGoal2" "PPOLag-CarGoal2" "PPO-Button" "PPOLag-Button" \
    --save-dir experiments/base_line/plots_cross/
```
