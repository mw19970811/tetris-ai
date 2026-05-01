# Tetris AI 使用文档

> **版本**: v1.3  
> **算法**: Rainbow DQN (Double + Dueling + PER + Noisy Nets + N-step TD) + PPO 备选  
> **目标**: 训练 AI 智能体在标准俄罗斯方块（Tetris Guideline）中达到人类顶尖水平

---

## 一、项目概述

Tetris AI 是一套完整的**深度强化学习**俄罗斯方块智能引擎，包含环境模拟、策略训练、模型导出、浏览器端部署的全链路能力。

核心流程：

```
┌───────────┐     ┌────────────┐     ┌──────────────┐
│ 训练环境   │ ──▶ │ Rainbow DQN│ ──▶ │ ONNX 模型     │
│ (Python)  │     │ 训练 (GPU) │     │ 导出          │
└───────────┘     └────────────┘     └──────┬───────┘
                                            │
┌───────────┐     ┌────────────┐            │
│ 浏览器游戏 │ ◀── │ ONNX Runtime│ ◀─────────┘
│ code.html │     │ Web 推理    │
└───────────┘     └────────────┘
```

### 核心特性

- **Rainbow DQN** 六合一算法：Double DQN + Dueling Network + Prioritized Experience Replay + Noisy Nets + N-step TD + Action Masking
- **PPO 备选**：Actor-Critic 架构 + GAE 优势估计
- **Placement-based 动作空间**：直接决策方块的最终放置位置，大幅压缩动作序列
- **混合状态编码**：CNN 空间特征 + 53 维手工特征（Dellacherie 六大特征）
- **浏览器端部署**：ONNX Runtime Web 在浏览器内实时推理，Agent 直接操控 `code.html` 游戏
- **三层 AI 交互模式**：手动 / AI 自动 / AI 建议

---

## 二、项目架构

```
agent-ai/
├── tetris/
│   └── code.html               # 浏览器俄罗斯方块游戏（含 AI Agent 集成）
│
├── env/                         # ──────── 游戏环境 ────────
│   ├── core/                    # C++ bitboard 参考实现（header-only，未编译）
│   ├── tetris_env.py            # Gymnasium 标准 RL 环境（纯 Python + numpy）
│   ├── state_encoder.py         # 状态编码（bitmap + 手工特征）
│   └── reward_calculator.py     # 奖励函数计算
│
├── agent/                       # ──────── AI 算法 ────────
│   ├── model.py                 # 神经网络 (DuelingDQN / ActorCritic)
│   ├── dqn.py                   # Rainbow DQN 智能体
│   ├── ppo.py                   # PPO 智能体
│   ├── noisy_layers.py          # NoisyLinear 噪声层
│   ├── memory.py                # 优先经验回放 (SumTree)
│   ├── nstep_buffer.py          # N-step return 缓冲
│   ├── action_mask.py           # 动作掩码（过滤非法动作）
│   └── pretrain.py              # 模仿学习预训练
│
├── trainer/                     # ──────── 训练系统 ────────
│   ├── trainer.py               # 主训练循环
│   ├── config.py                # 超参数配置 dataclass
│   ├── hardware_probe.py        # 硬件探测（CPU/GPU/RAM）
│   ├── resource_planner.py      # 资源规划与配置推荐
│   ├── evaluator.py             # 评估循环
│   ├── checkpoint.py            # Checkpoint 管理
│   └── logger.py                # WandB 日志
│
├── inference/                   # ──────── 推理部署 ────────
│   └── python/
│       ├── infer.py             # Python 推理引擎
│       └── export.py            # PyTorch → ONNX 导出
│
├── scripts/                     # ──────── 入口脚本 ────────
│   ├── train.py                 # 训练入口
│   ├── probe.py                 # 硬件探测入口
│   ├── eval.py                  # 评估入口
│   ├── play.py                  # 终端观战
│   └── export_model.py          # ONNX 导出入口
│
├── tests/                       # ──────── 单元测试 ────────
├── docs/                        # ──────── 文档 ────────
│   ├── tetris-rl-engine-design.md  # 详细设计文档
│   └── usage-guide.md           # 本文档
│
├── CLAUDE.md                    # Claude Code 项目指南
└── README.md                    # 项目简介
```

### 数据流

```
训练阶段:
┌──────────┐  State (numpy)  ┌──────────┐  Batch (GPU)  ┌───────────┐
│ TetrisEnv│ ──────────────▶ │ Rainbow  │ ────────────▶ │ PyTorch   │
│ (Python) │ ◀────────────── │ DQN      │ ◀──────────── │ GPU       │
└──────────┘  Action (int)   └──────────┘  Gradients    └───────────┘

部署阶段:
┌──────────┐  State (tensor) ┌───────────┐  Q-values   ┌────────────┐
│ code.html│ ──────────────▶ │ ONNX      │ ──────────▶ │ AI 决策     │
│ (Browser)│                 │ Runtime   │             │ → 按键执行  │
└──────────┘                 └───────────┘             └────────────┘
```

---

## 三、安装

### 环境要求

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CUDA 推荐，CPU 也可)
- 浏览器（Chrome / Edge / Firefox，用于运行 `code.html`）

### 安装步骤

```bash
# 1. 进入项目目录
cd agent-ai

# 2. 安装核心依赖
pip install torch numpy gymnasium

# 3. 安装推理依赖
pip install onnx onnxruntime

# 4. (可选) 训练日志可视化
pip install wandb

# 5. (可选) 安装测试依赖
pip install pytest
```

---

## 四、使用指南

### 4.0 训练前硬件检测

在正式训练之前，建议先运行硬件探测以确定最佳环境配置：

```bash
# 完整诊断报告
python scripts/probe.py

# 一行摘要
python scripts/probe.py --brief

# JSON 输出（供脚本解析）
python scripts/probe.py --json

# 探测并立即开始训练
python scripts/train.py --probe
```

探测报告包含：
- **系统信息**：OS、Python/PyTorch/CUDA 版本
- **CPU**：型号、物理/逻辑核心数、可用内存
- **GPU**：型号、显存总量/剩余量、计算能力
- **推荐配置**：训练模式（CPU/CUDA/多GPU）、并行环境数、batch size、replay buffer 容量
- **内存预算**：各部分内存占用的分项估算
- **警告与建议**：资源不足提醒、优化建议
- **快速启动命令**：直接可用的训练命令

**训练模式自动判定**：

| 条件 | 推荐模式 | 说明 |
|------|---------|------|
| ≥2 GPU, ≥10GB 显存 | `cuda_multi_gpu` | 多卡 DataParallel/DDP |
| 1 GPU, ≥2GB 显存 | `cuda` | 单卡 CUDA |
| 无 GPU 或显存 <2GB | `cpu` | CPU-only，自动确定线程数 |

**资源分级（Tier）**：

| 级别 | 条件 | envs | batch | replay | 说明 |
|------|------|------|-------|--------|------|
| High | GPU ≥20GB + RAM ≥48GB | 64–128 | 32–64 | 1M–2M | 理想训练环境 |
| Medium | GPU ≥6GB + RAM ≥12GB | 32–64 | 32 | 500K–1M | 标准训练环境 |
| Low | GPU ≥2GB 或 RAM ≥8GB | 16–32 | 16–32 | 200K–500K | 受限环境 |
| Minimal | 其他 | 1–8 | 8–16 | 100K | 仅验证可用 |

浏览器端也可检测：打开 `tetris/code.html`，在右侧 AI 面板点击"环境检测"按钮，查看 WebGL/WebAssembly 支持情况和推荐配置。

### 4.1 训练模型

**CLI 参数一览**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--algo` | `dqn` | 算法选择：`dqn` / `ppo` |
| `--steps` | 50,000,000 | 总训练步数 |
| `--envs` | 64 | 并行环境数 |
| `--device` | `cuda` | 训练设备 |
| `--seed` | 42 | 随机种子 |
| `--wandb` | 否 | 启用 Weights & Biases 日志 |
| `--probe` | 否 | 训练前先运行硬件探测并显示推荐配置 |
| `--no-pretrain` | 否 | 跳过模仿学习预训练 |
| `--resume` | 否 | 自动从 `--checkpoint-dir` 中最新 checkpoint 恢复 |
| `--resume-from` | — | 从指定 `.pt` 文件恢复 |
| `--checkpoint-dir` | `checkpoints` | Checkpoint 保存/读取目录 |

```bash
# 默认 DQN 训练（50M 步）
python scripts/train.py

# 指定训练步数和算法
python scripts/train.py --algo dqn --steps 10000000

# 使用 PPO 训练
python scripts/train.py --algo ppo --steps 20000000

# 启用 WandB 实时监控
python scripts/train.py --algo dqn --wandb --wandb-project tetris-ai

# 自动从最新 checkpoint 恢复训练
python scripts/train.py --resume

# 从指定 checkpoint 恢复
python scripts/train.py --resume-from checkpoints/step_005000000.pt

# 恢复后训练到更多步数
python scripts/train.py --resume --steps 100000000
```

训练过程中，checkpoint 会自动保存到 `checkpoints/` 目录，每 50,000 步保存一次。每次保存包含模型权重、optimizer 状态、训练步数计数器，以及 replay buffer（存为独立 `_buffer.pt` 文件）。

**训练日志格式**：

```
[2026-04-30 08:42:30]  Step    50,000/50,000,000 (1.0%)  |  Avg100R:     123.4  |  FPS:   32,000  |  Elapsed: 1h02m03s  |  ETA: 1h30m00s
```

每行包含：当前时间戳、步数/总步数（进度百分比）、最近100局的均分、采样速度（FPS）、已用时间和预计剩余时间（ETA）。评估时也会打印时间戳和已用时间。

> **注意**：首次训练前会自动运行 Dellacherie 模仿学习预训练（收集 ~1000 局专家对局），如果不需要可加 `--no-pretrain` 跳过。

### 4.2 评估模型

```bash
# 评估 PyTorch checkpoint（100 局）
python scripts/eval.py --model checkpoints/step_010000000.pt --episodes 100

# 评估 ONNX 模型
python scripts/eval.py --model tetris_ai.onnx --backend onnx --episodes 50
```

输出指标包括：平均得分、最大得分、平均消行数、Tetris 率等。

### 4.3 终端观战

```bash
# AI 自动游玩（终端 ANSI 渲染）
python scripts/play.py --model checkpoints/step_010000000.pt --delay 100

# 使用 ONNX 模型
python scripts/play.py --model tetris_ai.onnx --backend onnx --delay 50

# 人类手动游玩
python scripts/play.py --human
```

### 4.4 导出 ONNX 模型

```bash
python scripts/export_model.py checkpoints/step_010000000.pt -o tetris_ai.onnx
```

导出完成后会在当前目录生成 `tetris_ai.onnx` 文件（约 5-15 MB）。

### 4.5 浏览器 AI 部署

这是本项目的核心亮点：将训练好的 AI 加载到 `tetris/code.html` 游戏中进行实时推理。

**步骤**：

1. 导出 ONNX 模型（见 4.4）
2. 用浏览器打开 `tetris/code.html`
3. 在右侧 AI 面板中点击文件选择器，加载 `.onnx` 模型
4. 等待状态栏显示 "AI Auto ✓"（加载完成）
5. 点击 **"AI 自动"** 按钮
6. 点击 **"开始游戏"** — Agent 开始自动操作

**三种交互模式**：

| 模式 | 说明 |
|------|------|
| **手动** | 经典人类游玩，键盘/触摸操作 |
| **AI 自动** | Agent 自动决策和执行，可调速度 (1x-Instant) |
| **AI 建议** | 人类操作，AI 实时显示推荐的热力图和 Top-3 动作 |

**AI 面板说明**：

| 元素 | 功能 |
|------|------|
| 热力图 | 棋盘上每列的 Q 值分布，绿色=最佳列，蓝色=其他列 |
| 决策面板 | 当前选中的旋转、列、Hold、Q 值 |
| Top-3 动作 | Q 值最高的三个候选放置位置 |
| 推理延迟 | 单次 ONNX 推理耗时 |
| 置信度 | 最优动作与次优动作的 Q 值差 |
| 速度滑块 | 1x 慢速 → 4x 即时 |

**游戏统计面板**（左侧）：

| 指标 | 说明 |
|------|------|
| 时间 | 从开始到当前的游戏时间（MM:SS） |
| 方块数 | 已放置的方块总数 |
| 速度 | 每秒放置方块数（pps），衡量 AI / 人类操作效率 |

游戏结束时覆盖层会显示：最终分数、用时、放置方块数、放置速度、最终等级。通过这些指标可以评估 AI 在不同速度档位下的实际吞吐量和生存能力。

> **参考数据**：AI 慢速 1x 约 0.3 pps（每秒 0.3 个方块），即时 4x 可达 5-10 pps。人类熟练玩家约 0.5-1.0 pps。

---

## 五、算法说明

### 5.1 马尔可夫决策过程（MDP）建模

将俄罗斯方块形式化为六元组 \(\langle S, A, P, R, \gamma, \rho_0 \rangle\)。

#### 状态空间 \(S\)

```
S = (Board, CurrentPiece, HoldPiece, NextQueue)
```

| 分量 | 维度 | 说明 |
|------|------|------|
| Board | 22 × 10 二值网格 | 每格有/无方块，含 2 行隐藏区 |
| CurrentPiece | 7 种 × 4 旋转 | 当前活动的 Tetromino |
| HoldPiece | 8 种 (含空) | 保持槽中的方块 |
| NextQueue | 4 × 7 | 预览队列中的下一个方块 |

状态空间约 \(2^{220}\) 数量级，实际有效状态远小于此。

#### 动作空间 \(A\)

采用 **placement-based（基于放置）** 动作空间：

```
A = { (rotation ∈ {0,1,2,3}, column ∈ [-2, …, 11], hold ∈ {true, false}) }
```

- 最大动作数：4 × 14 × 2 = **112** 个
- 实际合法动作：通常 10-30 个（由棋盘状态决定）
- 每个动作对应一个方块从生成到锁定的完整过程

> **为什么选择 placement-based 而非 frame-based（逐帧按键）？**
>
> 1. 决策粒度与结果对齐：一次决策 = 一次放置，信用分配更清晰
> 2. 动作空间小得多：~40 合法动作 vs 理论上无限的帧序列
> 3. 避免学习 DAS、重力计时等低级操作
> 4. 在部署时通过**路径规划器**将 placement 转换为按键序列

#### 奖励函数 \(R\)

```
r = r_clear × level           # 消行奖励（原生分数 × 等级）
  + r_death                    # 死亡惩罚 (-100)
  - w_h × Σ heights            # 高度惩罚 (w_h = 0.3)
  - w_o × holes_count          # 孔洞惩罚 (w_o = 1.5)
  - w_b × bumpiness            # 崎岖度惩罚 (w_b = 0.2)
  - w_w × max_well_depth²      # 井深惩罚 (w_w = 0.5)
  + 0.01                       # 存活奖励
```

| 奖励项 | 含义 | 权重 | 设计理由 |
|--------|------|------|----------|
| 高度惩罚 | 每列柱高之和 | -0.3 | 高棋盘更容易死亡 |
| 孔洞惩罚 | 被方块封住的空格 | -1.5 | 孔洞是最致命的布局缺陷 |
| 崎岖度惩罚 | 相邻列高度差的绝对值之和 | -0.2 | 平坦表面更容易放置方块 |
| 井深惩罚 | 最大井深（二次惩罚） | -0.5 | 深井极难填补，二次惩罚挤压策略 |
| 存活奖励 | 每次放置未死亡 | +0.01 | 微小的正向信号延长存活 |

**消行奖励**（游戏原生分 × 等级）：

| 消行数 | 基础分 |
|--------|--------|
| Single (1) | 100 |
| Double (2) | 300 |
| Triple (3) | 500 |
| Tetris (4) | 800 |

#### 折扣因子

\(\gamma = 0.99\) — 在短期收益与长期生存之间取得平衡。俄罗斯方块中每次放置的影响可以延续数百步。

---

### 5.2 Rainbow DQN（主算法）

Rainbow DQN 是 DQN 的六个改进的组合体，本项目的实现包含全部六个组件。

#### 5.2.1 Double DQN — 解耦动作选择与评估

**问题**：标准 DQN 使用同一个网络同时选择和评估动作，导致系统性 Q 值高估。

**解决**：

\[
y_t = r_t + \gamma \cdot Q_{\text{target}}\left(s_{t+1}, \arg\max_a Q_{\text{online}}(s_{t+1}, a)\right)
\]

- Online 网络选择最优动作
- Target 网络评估该动作的值
- Target 网络每 8000 步硬更新（或使用 Polyak 软更新，\(\tau = 0.005\)）

**代码位置**：`agent/dqn.py` 第 139-145 行

#### 5.2.2 Dueling Network — 状态价值与动作优势分解

**核心思想**：将 Q 函数分解为状态价值 \(V(s)\) 和动作优势 \(A(s, a)\)：

\[
Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|A|}\sum_{a'} A(s, a')\right)
\]

**为什么有效**：在俄罗斯方块中，许多状态的价值与具体动作选择关系不大（例如棋盘几乎全满时，任何动作价值都很低）。Dueling 架构可以更高效地学习状态价值。

**网络结构**：

```
Board (1,22,10) ──▶ CNN (3层 Conv 32→64→64) ──▶ GlobalAvgPool → 64d ──┐
                                                                          ├── Concat(128d) ──▶ Dueling Head
Features (53d)  ──▶ MLP (53→128→64) ──────────────────────────────────┘
                                                                          │
                                                    ┌── Value Stream ────┤
                                                    │  128→64→1          │
                                                    │                    ├── Q(s,a)
                                                    └── Advantage Stream─┘
                                                       128→64→|A|
```

**代码位置**：`agent/model.py` 第 65-118 行

#### 5.2.3 Prioritized Experience Replay (PER) — 按重要性采样

**问题**：均匀采样浪费了大量时间在低学习价值的转移上。

**方法**：按 TD-error 绝对值加权的采样概率：

\[
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad p_i = |\delta_i| + \epsilon
\]

并引入**重要性采样权重**修正分布偏移：

\[
w_i = \left(\frac{1}{N} \cdot \frac{1}{P(i)}\right)^\beta / \max_j w_j
\]

**超参数**：
- \(\alpha = 0.6\) — 控制优先级程度
- \(\beta\)：0.4 → 1.0 — 从训练开始到结束线性增长
- 缓冲区容量：1,000,000 条转移
- 数据结构：SumTree（线段树），实现 \(O(\log N)\) 采样

**代码位置**：`agent/memory.py` — `PrioritizedReplayBuffer` 类

#### 5.2.4 Noisy Networks — 学习型探索

**替代 ε-greedy**：在线性层权重中注入可学习的高斯噪声：

\[
y = (b + W \odot \epsilon^b) + (W + W \odot \epsilon^w) x
\]

其中 \(\epsilon^b, \epsilon^w\) 使用 Factorised Gaussian Noise 以降低参数量。

**优势**：
- 探索策略随训练自适应：收敛区域减少噪声（利用），不确定区域增加噪声（探索）
- 不需要手工设计 ε 衰减曲线
- 噪声参数 \(\sigma\) 初始化为 0.017

**代码位置**：`agent/noisy_layers.py` — `NoisyLinear` 类

#### 5.2.5 N-step TD — 加速信用传播

使用 n-step return 替代单步 TD：

\[
R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_a Q(s_{t+n}, a)
\]

- \(n = 5\)
- 对 Tetris 特别重要：为 Tetris 准备 I 块列需要约 5-8 步的规划，n-step 让这种长链奖励更快传播

**代码位置**：`agent/nstep_buffer.py` — `NStepBuffer` 类

#### 5.2.6 Action Masking — 过滤非法动作

在 Q 网络输出的 112 维向量上，将非法动作的 Q 值设为 \(-\infty\)（实际用 \(-10^9\)）：

\[\text{masked\_q}[i] = \begin{cases} q[i] & \text{if action } i \text{ is legal} \\ -10^9 & \text{otherwise} \end{cases}\]

**代码位置**：`agent/action_mask.py`

#### 完整训练算法

```
Algorithm: Rainbow DQN for Tetris

Initialize:
    online network Q_θ, target network Q_θ⁻ ← Q_θ
    prioritized replay buffer D (capacity = 1M, SumTree)
    optimizer: Adam(lr=6.25e-5)

Loop until convergence:
    s = env.reset()
    Loop per episode:
        a = argmax_a masked_Q_θ(s, a)  [with noisy net exploration]
        s', r, done = env.step(a)
        δ = r + γ·Q_θ⁻(s', argmax Q_θ(s')) - Q_θ(s, a)
        D.add((s,a,r,s',done), priority=|δ|)

        if step % 4 == 0:
            batch, indices, weights = D.sample(32)
            Compute n-step targets via Double DQN
            L = Σ w_i · HuberLoss(target_i - Q_θ(s_i, a_i))
            Optimize(L)
            Update priorities in D

        if step % 8000 == 0:
            Q_θ⁻ ← Q_θ  (hard sync)

        s = s'
        if done: break
```

### 5.3 PPO（备选算法）

当 DQN 出现严重 Q 值高估或训练不稳定时，切换到 PPO。

#### 5.3.1 PPO-Clip 目标函数

\[
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[\min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]
\]

其中 \(r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}\) 是新旧策略概率比，\(\epsilon = 0.2\)。

#### 5.3.2 GAE 优势估计

\[
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}
\]

其中 \(\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\)，\(\lambda = 0.95\)。

#### 5.3.3 PPO 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| \(\gamma\) | 0.99 | 折扣因子 |
| \(\lambda\) (GAE) | 0.95 | 偏差-方差权衡 |
| \(\epsilon\) (clip) | 0.2 | PPO clip 范围 |
| 学习率 | 2.5e-4 | Adam |
| 批大小 | 256 | 每次更新的样本数 |
| 小批量 | 64 | Mini-batch 大小 |
| 更新轮数 | 4 | 每批数据重复训练 |
| 收集步数 | 2048 | 每次更新的交互步数 |
| 并行环境数 | 64 | |

**代码位置**：`agent/ppo.py`

---

### 5.4 状态编码方案

采用**混合表示（方案 C）**，结合位图和手工特征。

#### 分支 A：二值位图（CNN 输入）

```
Shape: (1, 22, 10), dtype=float32
```
- 22 × 10 网格，每格 0（空）或 1（占据）
- CNN 自动学习空间模式（平/崎岖/孔洞结构）

#### 分支 B：手工特征向量（MLP 输入）

共 **53 维**，包含 Dellacherie 六大特征和块身份编码：

| 索引 | 特征 | 维度 | 说明 |
|------|------|------|------|
| 0 | 累计高度 | 1 | \(\sum_{c=0}^{9} h_c\) |
| 1 | 消行数占位 | 1 | 预留给放置后的消行数 |
| 2 | 孔洞数 | 1 | 被封住的空位数量 |
| 3 | 崎岖度 | 1 | \(\sum_{c=0}^{8} |h_c - h_{c+1}|\) |
| 4 | 最大井深 | 1 | \(\max_c \text{well}(c)\) |
| 5 | 高度变化占位 | 1 | 预留 |
| 6–12 | 当前块 one-hot | 7 | I/O/T/S/Z/J/L |
| 13–16 | 旋转状态 one-hot | 4 | 0/1/2/3 |
| 17–24 | Hold 块 one-hot | 8 | 7 种 + 空 |
| 25–52 | Next 队列 one-hots | 28 | 4 × 7 = 28 |

**代码位置**：`env/state_encoder.py` 和 `tetris/code.html`（JS 版本）

> ⚠️ **关键要求**：Python 训练环境和浏览器 JS 推理端的编码必须**完全一致**。特征计算的任何偏差都会导致模型输出错误。

---

### 5.5 辅助训练技术

#### 模仿学习预训练（Imitation Learning）

用 Dellacherie 启发式算法预训练策略网络，实现冷启动，避免 RL 初期随机探索的无效样本。

**Dellacherie 评分函数**：

```
Score(placement) = -4.500 × landing_height
                  + 3.418 × cleared_lines
                  - 7.899 × holes
                  - 3.386 × bumpiness
                  - 3.129 × max_well_depth
                  - 2.000 × row_transitions
```

选择得分最高的合法放置作为专家动作。收集 10,000 局专家数据 → 监督学习预训练。

**代码位置**：`agent/pretrain.py`

#### 课程学习（Curriculum Learning）

分阶段增加重力速度：

| 阶段 | 重力 | 训练步数 |
|------|------|----------|
| Stage 1 | 2000ms (极慢) | 500 万 |
| Stage 2 | 1000ms (Level 1) | 500 万 |
| Stage 3 | 500ms (Level 5) | 500 万 |
| Stage 4 | 200ms (Level 10) | 500 万 |
| Stage 5 | 50ms (Level 15) | 500 万 |
| Stage 6 | 9ms (Level 20) | 持续 |

---

## 六、配置参考

### 6.1 训练配置 (TrainingConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `algorithm` | `"dqn"` | 算法选择：`"dqn"` 或 `"ppo"` |
| `total_steps` | 50,000,000 | 总训练步数 |
| `eval_every` | 10,000 | 评估间隔（步） |
| `eval_episodes` | 100 | 每次评估的局数 |
| `save_every` | 50,000 | Checkpoint 保存间隔 |
| `num_envs` | 64 | 并行环境数 |
| `device` | `"cuda"` | 训练设备 |
| `seed` | 42 | 随机种子 |

### 6.2 Rainbow DQN 配置 (DQNConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gamma` | 0.99 | 折扣因子 |
| `n_step` | 5 | N-step return 步数 |
| `lr` | 6.25e-5 | 学习率 |
| `batch_size` | 32 | 训练批大小 |
| `train_every` | 4 | 每 N 环境步训练一次 |
| `target_update_freq` | 8000 | 目标网络硬更新频率 |
| `target_update_tau` | 0.005 | 软更新系数（与硬更新互斥） |
| `replay_capacity` | 1,000,000 | 回放缓冲区容量 |
| `per_alpha` | 0.6 | PER 优先级指数 |
| `per_beta_start` | 0.4 | IS 修正初始值 |
| `per_beta_end` | 1.0 | IS 修正最终值 |
| `per_beta_frames` | 10,000,000 | β 衰减帧数 |
| `grad_clip_norm` | 10.0 | 梯度裁剪阈值 |

### 6.3 PPO 配置 (PPOConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE λ 参数 |
| `clip_epsilon` | 0.2 | PPO clip 范围 |
| `value_coef` | 0.5 | 价值损失权重 |
| `entropy_coef` | 0.01 | 熵正则系数 |
| `lr` | 2.5e-4 | 学习率 |
| `batch_size` | 256 | 批大小 |
| `mini_batch_size` | 64 | 小批大小 |
| `n_epochs` | 4 | 更新轮数 |
| `rollout_steps` | 2048 | 收集步数 |

### 6.4 环境配置 (EnvConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cols` | 10 | 棋盘列数 |
| `rows` | 20 | 可见行数 |
| `hidden_rows` | 2 | 隐藏行数 |
| `next_queue_size` | 4 | Next 队列可见长度 |
| `max_steps` | 10000 | 单局最大步数（truncation） |
| `reward_weights.w_height` | 0.3 | 高度惩罚权重 |
| `reward_weights.w_holes` | 1.5 | 孔洞惩罚权重 |
| `reward_weights.w_bumpiness` | 0.2 | 崎岖度惩罚权重 |
| `reward_weights.w_well` | 0.5 | 井深惩罚权重 |

---

## 七、硬件建议

| 组件 | 推荐配置 | 最低配置 |
|------|----------|----------|
| GPU | NVIDIA RTX 4090 / A100 | RTX 3060 (12GB) |
| CPU | 32+ cores | 8 cores |
| RAM | 64 GB+ | 16 GB |
| 存储 | 500 GB NVMe | 100 GB |
| 预期训练时长 | 2-4 天 | 7-14 天 |

纯 CPU 训练可实现但速度极慢，不推荐用于正式训练。

---

## 八、Python 与 C++ 的关系

### 当前状态：纯 Python 运行

**当前所有训练、推理和游戏逻辑均以纯 Python + NumPy + PyTorch 运行。** `env/core/` 中的 C++ 头文件是一套并行的参考实现（bitboard 棋盘、碰撞检测、行消除），但 Python 训练环境 `env/tetris_env.py` 并未导入或使用它们。

```
当前运行时:
┌──────────────────┐
│ env/tetris_env.py │  ← 纯 Python + NumPy
│ agent/*.py       │  ← PyTorch
│ trainer/*.py     │  ← PyTorch
│ inference/python/ │  ← ONNX Runtime / PyTorch
└──────────────────┘
        ↑ 实际执行路径

env/core/*.h        ← C++ 参考实现（未编译，未导入）
env/core/*.cpp      ← 已精简为 stub 文件（实现均在 .h 中）
```

### 为什么没有使用 C++ 加速

1. **PyTorch 本身已高度优化**：卷积、矩阵运算等核心计算由 CUDA/cuDNN 加速，外层 Python 开销占比极小
2. **训练瓶颈在 GPU**：64 并行环境 × 神经网络推理，GPU 利用率是主要制约，环境步进的开销可忽略
3. **ONNX Runtime Web 替代了 C++ 推理**：浏览器部署使用 ONNX Runtime Web（WebAssembly 后端），无需 C++

### 如何启用 C++ 加速（可选）

如果确实需要 C++ bitboard 环境以进一步提升采样速度（>50K FPS），需要：

1. 安装 pybind11：`pip install pybind11`
2. 在 `CMakeLists.txt` 中取消 `add_subdirectory(env/core)` 和 `add_subdirectory(env/bindings)` 的注释
3. 在 `env/bindings/tetris_bindings.cpp` 中实现 Python 绑定
4. 在 `env/tetris_env.py` 中添加回退逻辑：优先导入编译后的 C++ 模块，不可用时使用纯 Python

> 当前项目已将 C++ 代码精简为 header-only 参考实现（.cpp 文件为 stub），bindings 目录保留骨架供后续开发。

---

## 九、常见问题

### Q: 训练不收敛怎么办？
1. 检查奖励函数的权重是否合理（孔洞惩罚应该是最高的）
2. 确保使用了模仿学习预训练（`use_pretrain: true`）
3. 尝试降低学习率到 `2.5e-5`
4. 从课程学习的低速阶段重新开始

### Q: 训练意外中断后如何恢复？
```bash
# 自动从最新 checkpoint 恢复
python scripts/train.py --resume

# 从指定 checkpoint 恢复并继续训练
python scripts/train.py --resume-from checkpoints/step_005000000.pt --steps 100000000
```
恢复时会同时加载：模型权重、optimizer 状态、训练步数计数器、replay buffer。预训练和初始评估会被跳过。

### Q: 浏览器中 AI 不工作？
1. 打开浏览器控制台（F12），检查 ONNX 模型加载日志
2. 确认模型是用 `dynamo=False` 导出的（脚本已配置）
3. 确认模型文件路径正确（支持本地文件，通过文件选择器加载）
4. 如果 ONNX Runtime Web 加载失败，检查网络是否能访问 CDN（jsdelivr.net）

### Q: 如何判断模型质量？
- **得分 < 10,000**：基本没有学会，仍在随机探索
- **得分 10,000 - 100,000**：学会了基本消行和存活
- **得分 100,000 - 1,000,000**：掌握了布局优化
- **得分 > 10,000,000**：达到人类顶尖水平
- **辅助指标**：浏览器游戏中的"速度"面板显示 pps（方块/秒），高 pps 同时保持高分数说明模型推理快速且决策质量高

### Q: 如何衡量 AI 实际游戏速度？
浏览器游戏左侧面板实时显示：
- **时间**：游戏已运行的 MM:SS
- **方块数**：已放置的方块总数
- **速度**：每秒放置方块数（pps）

游戏结束覆盖层汇总全部数据。通过调整 AI 模式下的速度滑块（慢速→即时），可以对比不同推理延迟下的吞吐量。

### Q: 如何运行测试？
```bash
# 运行所有测试
pytest tests/ -v

# 根目录的 test_env.py 为历史遗留副本，请使用 tests/ 下的正式测试
```
注意：根目录下的 `test_env.py` 是 `tests/test_env.py` 的历史重复文件，统一使用 `tests/` 目录下的测试。

### Q: 已知限制
1. **PPO 训练**：当前 Rainbow DQN 是主要验证的算法路径。PPO 可通过 `--algo ppo` 使用，但训练效果未经充分验证
2. **C++ 加速**：`env/core/` 中的 C++ 代码是参考实现，当前不参与运行时。所有训练和推理均为纯 Python
3. **仅 DQN 支持 ONNX 导出**：PPO 模型无法通过 `export_model.py` 导出
4. **根目录 `test_env.py`** 是 `tests/test_env.py` 的历史副本，建议手动删除

---

## 十、参考

| 论文 | 内容 |
|------|------|
| Mnih et al. (2015), *Nature* | DQN — Human-level control through deep RL |
| Van Hasselt et al. (2016), *AAAI* | Double DQN |
| Wang et al. (2016), *ICML* | Dueling Network Architectures |
| Schaul et al. (2016), *ICLR* | Prioritized Experience Replay |
| Fortunato et al. (2018), *ICLR* | Noisy Networks for Exploration |
| Hessel et al. (2018), *AAAI* | Rainbow: Combining Improvements in DQN |
| Schulman et al. (2017) | PPO — Proximal Policy Optimization Algorithms |
| Dellacherie (2003) | Tetris AI with heuristic evaluation |
