# 俄罗斯方块智能引擎 — 完整强化学习设计方案

> **版本**: v1.0  
> **目标**: 基于深度强化学习训练一个在标准 Tetris Guideline 规则下达到人类顶尖水平的俄罗斯方块 AI  
> **参考实现**: `tetris/code.html`（10×20 棋盘, SRS 旋转系统, 7-bag 随机, 锁定延迟 500ms, 15 次移动重置上限）

---

## 第一章　引言与问题定义

### 1.1 项目背景与目标

俄罗斯方块是强化学习领域的经典基准测试问题。其状态空间巨大（约 \(2^{200}\) 量级）、奖励稀疏（消行才产生分数）、且具有长期规划需求（布局优化需要跨多块的决策链），全面考验 RL 算法的样本效率、信用分配和泛化能力。

**引擎目标**：

| 维度 | 目标 |
|------|------|
| 游戏表现 | 平均单局得分 ≥ 10,000,000（人类顶尖水平） |
| 生存能力 | 在 Level 15+ 重力下持续消行不死亡 |
| 效率 | 单块决策延迟 ≤ 1ms |
| 样本效率 | 5000 万帧内收敛到可用策略 |
| 泛化 | 跨棋盘尺寸、跨重力曲线的迁移能力 |

### 1.2 马尔可夫决策过程（MDP）建模

将俄罗斯方块形式化为六元组 \(\langle S, A, P, R, \gamma, \rho_0 \rangle\)：

#### 1.2.1 状态空间 \(S\)

```
S = (Board, CurrentPiece, HoldPiece, NextQueue)
```

- **Board**: \(22 \times 10\) 的二值网格（每个格子有/无方块）。用 bitboard 表示则为 22 个 10-bit 整数，共 220 位。
- **CurrentPiece**: 当前活动方块类型（7 种）和当前旋转状态（4 种）。
- **HoldPiece**: 已保持的方块类型（7 种 + 空 = 8 种）。
- **NextQueue**: 可见的下一个方块序列（本项目显示 4 个）。

**状态空间大小估计**：
- 棋盘布局：\(2^{220} \approx 1.68 \times 10^{66}\)（实际有效状态远少于此，但仍极为巨大）
- 加上方块类型信息：\(\times 7 \times 4 \times 8 \times 7^4 \approx 3.2 \times 10^6\) 组合因子

#### 1.2.2 动作空间 \(A\)

采用 **placement-based（基于放置）** 动作空间，即每个动作对应 "将当前方块以某种旋转状态放置到某个列位置"：

```
A = { (rotation, column) | rotation ∈ {0,1,2,3}, column ∈ {0,...,9},
      使得该放置位置合法的所有组合 }
```

- 理论最大动作数：\(4 \times 10 = 40\) 个
- 实际合法动作数：通常 10~30 个（取决于当前方块形状和棋盘状态）
- 特殊动作：Hold（保持方块），将当前方块与 Hold 槽交换

> **设计决策**：选择 placement-based 而非 frame-based（逐帧操作），原因如下：
> 1. 决策粒度与最终结果对齐（一次决策 = 一次放置）
> 2. 动作空间小得多（~40 vs 理论上无限帧序列）
> 3. 避免学习 DAS/移动等低级操作
> 4. 但也需额外输出 Hold 决策（二元选择）

#### 1.2.3 状态转移 \(P\)

状态转移是确定性与随机性的混合：

- **确定性部分**：方块放置后的棋盘更新（碰撞检测、行消除、方块锁定）完全由规则决定。
- **随机性部分**：7-bag 随机生成器决定下一个方块类型。严格来说这是一个确定性的随机过程（每 7 块一个周期，Fisher-Yates 打乱）。

#### 1.2.4 奖励函数 \(R\)

奖励函数设计是决定训练成败的关键。本项目采用**多层奖励体系**：

**基础奖励（游戏原生分数）**：

| 事件 | 奖励 |
|------|------|
| 单消（Single） | \(+100 \times \text{level}\) |
| 双消（Double） | \(+300 \times \text{level}\) |
| 三消（Triple） | \(+500 \times \text{level}\) |
| 四消（Tetris） | \(+800 \times \text{level}\) |
| 硬降（Hard Drop） | \(+2\) per cell |
| 软降（Soft Drop） | \(+1\) per cell |
| T-Spin Mini | \(+100 \times \text{level}\) |
| T-Spin Single/Double/Triple | \(+800/+1200/+1600 \times \text{level}\) |

**塑性奖励（Reward Shaping）** —— 在每步（每次放置）提供即时反馈：

| 特征 | 公式 | 权重 | 含义 |
|------|------|------|------|
| 行消除奖励 | \(R_{\text{clear}}\) | 见上表 | 鼓励消行 |
| 游戏结束惩罚 | \(R_{\text{death}}\) | \(-100\) | 惩罚死亡 |
| 高度惩罚 | \(R_{\text{height}} = -\sum_{c} h_c\) | \(-0.5\) | 惩罚柱高累积 |
| 崎岖度惩罚 | \(R_{\text{bump}} = -\sum_{c} \text{abs[} h_c - h_{c+1} \text{]} \) | \(-0.3\) | 惩罚表面崎岖 |
| 孔洞惩罚 | \(R_{\text{holes}} = -\sum \text{holes}\) | \(-1.0\) | 惩罚封闭空洞 |
| 井深惩罚 | \(R_{\text{well}} = -\sum_{c} \text{well}(c)^2\) | \(-0.8\) | 惩罚深井 |

**总即时奖励**：
\[
r_t = R_{\text{clear}} + R_{\text{death}} + w_h R_{\text{height}} + w_b R_{\text{bump}} + w_o R_{\text{holes}} + w_w R_{\text{well}}
\]

> **关键设计原则**：确保塑性奖励满足 Potential-Based Shaping 约束，即存在势函数 \(\Phi(s)\) 使得 \(F(s, a, s') = \gamma\Phi(s') - \Phi(s)\)。否则可能导致 reward hacking——AI 学会刷奖励而非真正玩游戏。

#### 1.2.5 折扣因子 \(\gamma\)

设置 \(\gamma = 0.99\)，在短期收益与长期生存之间取得平衡。对于俄罗斯方块，每一步的决策影响可能延续数百步，因此需要较高的折扣因子。

#### 1.2.6 初始状态分布 \(\rho_0\)

每局游戏从空棋盘开始，7-bag 随机初始化，第一块随机生成。

### 1.3 关键难点分析

| 难点 | 描述 | 对策 |
|------|------|------|
| **巨大状态空间** | 有效棋盘状态约 \(10^{30}\) 量级 | CNN 特征提取 + Bitboard 高效编码 + 函数逼近 |
| **稀疏奖励** | 原生分数仅在消行时产生 | 精心设计的塑性奖励 + N-step TD 加速信用传播 |
| **长期信用分配** | 一个好的布局决策可能需要数十步后才产生 Tetris | Prioritized Experience Replay + TD(λ) |
| **7-bag 随机性** | 方块序列具有周期性，但排列随机 | 提供 Next 队列+Hold 信息作为状态输入 |
| **死亡螺旋** | 糟糕的布局一旦形成就难以逆转 | 预训练（模仿学习冷启动）+ 惩罚孔洞/高度 |
| **探索困难** | 良好的 Tetris 开局需要十几步的精确序列 | Noisy Networks + ε-greedy 渐进衰减 + 课程学习 |

---

## 第二章　游戏环境设计

### 2.1 环境接口规范

环境遵循 **Gymnasium (gym)** 标准接口，方便与 stable-baselines3 / cleanRL / rllib 等框架集成。

```python
# 核心接口
class TetrisEnv(gym.Env):
    def reset(self, seed=None) -> State          # 重置环境，返回初始状态
    def step(self, action: Action) -> StepResult  # 执行动作，返回 (state, reward, done, info)
    def render(self) -> None                      # 可选：渲染当前棋盘
    def get_legal_actions(self) -> List[Action]   # 返回当前状态下的合法动作列表
    def get_state(self) -> np.ndarray             # 返回状态表示（特征/位图）
```

**StepResult 结构**：
```
StepResult = (state, reward, terminated, truncated, info)
- state: 新的状态表示
- reward: 即时奖励（包含塑性奖励）
- terminated: 是否游戏结束（死亡）
- truncated: 是否超出最大步数（训练时设置上限，如 10,000 步）
- info: 诊断信息（消行数、得分、当前level等）
```

**Action 数据结构**：
```python
@dataclass
class Action:
    rotation: int   # 0, 1, 2, 3（对应 SRS 四种旋转状态）
    column: int     # 放置列（0~9），该列为方块最左端的列坐标
    hold: bool      # 是否先执行 Hold 操作
```

**State 数据结构**：
```python
@dataclass
class State:
    board: np.ndarray       # shape: (22, 10), dtype=bool
    current_piece: int      # 0~6: I,O,T,S,Z,J,L
    current_rotation: int   # 0~3
    hold_piece: int         # 0~6 or -1 (empty)
    next_queue: List[int]   # 长度为 4，值为 0~6
    can_hold: bool          # 当前块是否可 Hold
```

### 2.2 状态表示方案

提供**三种表示方案**，默认使用混合方案 C。

#### 方案 A：二值位图（Raw Bitboard）

```
Shape: (22, 10), dtype=bool
总计: 220 维二进制特征
```

| 优点 | 缺点 |
|------|------|
| 信息无损 | 维度较高 |
| CNN 可自动学习空间模式 | 需要更多训练样本 |
| 无需手工特征工程 | 泛化到不同尺寸棋盘需要调整网络 |

#### 方案 B：手工特征向量（BCTS / Dellacherie 六大特征）

Dellacherie 算法是公认的启发式 Tetris AI 基准，其六大特征如下：

| 特征 | 公式 / 描述 | 维度 |
|------|-------------|------|
| 累计高度 | \(\sum_{c=0}^{9} h_c\)，\(h_c\) 为列 \(c\) 的柱高 | 1 |
| 消行数 | 当前放置消除的行数 | 1 |
| 孔洞数 | 被方块覆盖的空位数量 | 1 |
| 崎岖度 | \(\sum_{c=0}^{8} |h_c - h_{c+1}|\) | 1 |
| 最大井深 | \(\max_c \text{well}(c)\)，井深 = 相邻列高度差 | 1 |
| 高度均值变化 | 放置前后的平均高度变化 | 1 |

再加上：当前方块类型（one-hot 7 维）、旋转状态（one-hot 4 维）、Hold 方块（one-hot 8 维）、Next 队列前 4 个（4 × one-hot 7 维）

**总计**: \(1 \times 6 + 7 + 4 + 8 + 4 \times 7 = 53\) 维

| 优点 | 缺点 |
|------|------|
| 极低维度 | 信息有损 |
| 样本效率高 | 需要领域知识设计 |
| 泛化能力强 | 难以捕捉复杂空间模式 |

#### 方案 C：混合表示（推荐）

将方案 A 和方案 B 结合：

- **CNN 分支**接收 \(22 \times 10\) 位图，提取空间特征
- **MLP 分支**接收 53 维手工特征
- 两个分支的特征向量在融合层 Concat，然后输入 Dueling Head

```
Input: (board 22×10, features 53-dim)
      │
      ├── CNN (3 conv layers, 32/64/64 channels, stride=1, padding=1)
      │    └── GlobalAvgPool → 64-dim vector
      │
      └── MLP (53 → 128 → 64) → 64-dim vector
      
      Concat → 128-dim → Dueling Head → Q(s,a) for each action
```

### 2.3 动作空间设计：Placement-Based 详解

#### 2.3.1 动作生成算法

对于每个可能的组合（rotation, column），计算该放置是否合法：

```
function get_legal_actions(state):
    actions = []
    for rotation in [0, 1, 2, 3]:
        for column in [-2, ..., 10]:       # 负数列允许部分方块在左侧界外
            ghost_y = compute_ghost_y(state, rotation, column)
            if is_valid_placement(state, rotation, column, ghost_y):
                actions.append(Action(rotation, column, hold=False))
    
    if state.can_hold:
        for rotation in [0, 1, 2, 3]:
            for column in [-2, ..., 10]:
                ghost_y = compute_ghost_y_after_hold(state, rotation, column)
                if is_valid_placement(state, rotation, column, ghost_y):
                    actions.append(Action(rotation, column, hold=True))
    
    return actions
```

- 典型合法动作数：I 块 ~18 个，O 块~10 个，T/S/Z/J/L 块 ~18-34 个
- 动作空间动态变化，需要**动作掩码（Action Masking）**过滤非法动作

#### 2.3.2 动作掩码实现

在 Q 网络输出端对非法动作施加 \(-\infty\) 掩码：

```python
def masked_q_values(q_values, legal_actions_mask):
    """q_values: (batch, num_actions), mask: (batch, num_actions) bool"""
    q_values = q_values.clone()
    q_values[~legal_actions_mask] = -float('inf')
    return q_values
```

在 PPO 中，类似地对非法动作的 logits 施加掩码。

### 2.4 奖励函数工程（详细）

#### 2.4.1 高度惩罚

```
R_height = -w_h × Σ_c h_c / (COLS × ROWS)
```
每列高度求和后归一化，默认权重 \(w_h = 0.3\)。

#### 2.4.2 孔洞检测算法

孔洞定义为：在某一列中，某行上方存在方块覆盖且该位置为空。

```
function count_holes(board):
    holes = 0
    for col in 0..COLS-1:
        found_block = false
        for row in 0..ROWS-1:
            if board[row][col] == 1:
                found_block = true
            elif found_block:
                holes += 1
    return holes
```

```
R_holes = -w_o × count_holes(board)
```
默认权重 \(w_o = 1.5\)（孔洞是最致命的布局缺陷）。

#### 2.4.3 崎岖度

```
R_bumpiness = -w_b × Σ_{c=0}^{COLS-2} |h_c - h_{c+1}|
```
默认权重 \(w_b = 0.2\)。

#### 2.4.4 井深

井定义为：某列高度显著低于相邻列，形成深槽。

```
function compute_well_depth(heights, col):
    left_diff = (heights[col-1] - heights[col]) if col > 0 else 0
    right_diff = (heights[col+1] - heights[col]) if col < COLS-1 else 0
    return max(0, min(left_diff, right_diff))

R_well = -w_w × Σ_c compute_well_depth(heights, c)²
```
默认权重 \(w_w = 0.5\)。对深井施加二次惩罚，因为深井极难填补。

#### 2.4.5 生存奖励

```
R_survival = +0.01  (每放置一个方块未死亡)
```
微小的正向奖励鼓励 AI 延长存活时间。这实际上提高了有效折扣因子，让远期奖励更容易传播。

#### 2.4.6 T-Spin 奖励

T-Spin 是高级 Tetris 技术，在竞技比赛中分值极高：

```
R_tspin = see SCORE_TABLE above (× level factor)
```

T-Spin 判定标准（3-corner rule）：T 块的四个对角中至少有 3 个被占据。

#### 2.4.7 完整奖励汇总

```
r = r_clear × level                    # 消行奖励（游戏原生 × 等级倍率）
  + r_death                             # 死亡惩罚 (−100)
  + w_h × r_height                      # 高度惩罚
  + w_o × r_holes                       # 孔洞惩罚
  + w_b × r_bumpiness                   # 崎岖度惩罚
  + w_w × r_well                        # 井深惩罚
  + 0.01                                # 存活奖励
  + r_tspin                             # T-Spin 奖励
```

所有超参数（\(w_h, w_o, w_b, w_w\)）通过 Hydra 配置文件暴露，支持 Optuna 自动调优。

### 2.5 环境加速方案

#### 2.5.1 Bitboard 表示

核心思想：用**位运算**替代循环实现碰撞检测和行消除。

```
棋盘表示: uint16_t board[22]   // 每行一个 10-bit 整数
方块表示: uint16_t piece[4]     // 每个旋转状态的 4 行掩码
```

**碰撞检测**：
```cpp
bool collides(uint16_t board[22], uint16_t piece[4], int x, int y) {
    for (int r = 0; r < 4; r++) {
        int row = y + r;
        if (row < 0) continue;
        if (row >= 22) return true;
        uint16_t shifted = piece[r] >> (10 - x);  // 移位到目标列
        if (shifted & board[row]) return true;     // 与棋盘做按位与
        if (shifted & 0xFC00) return true;         // 检查右侧越界（bit 10+）
    }
    return false;
}
```

**行消除**：
```cpp
int clear_lines(uint16_t board[22]) {
    constexpr uint16_t FULL_ROW = (1 << 10) - 1;  // 0x3FF
    int cleared = 0;
    for (int y = 21; y >= 0; y--) {
        if (board[y] == FULL_ROW) {
            // 移除该行，上方所有行下移
            memmove(&board[1], &board[0], y * sizeof(uint16_t));
            board[0] = 0;
            cleared++;
            y++;  // 重新检查当前行
        }
    }
    return cleared;
}
```

#### 2.5.2 性能目标

| 指标 | 纯 Python | Python + C++ Bitboard |
|------|-----------|----------------------|
| 单步 FPS | ~5,000 | ~50,000+ |
| 单环境 | — | — |
| 64 并行环境 | — | ~3,000,000 FPS |
| 单次放置推理 | — | ~0.05ms |

#### 2.5.3 多环境并行架构

采用 Python `multiprocessing` + 共享内存实现 Vectorized Environment：

```
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Env 0    │  │ Env 1    │  │ Env N-1  │   ← C++ 环境 (独立进程)
│ (C++     │  │ (C++     │  │ (C++     │
│  core)   │  │  core)   │  │  core)   │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │              │              │
     └──────────────┼──────────────┘
                    │
          ┌─────────▼─────────┐
          │  Shared Memory    │   ← numpy 数组
          │  (states, actions,│
          │   rewards, dones) │
          └─────────┬─────────┘
                    │
          ┌─────────▼─────────┐
          │  Trainer (Python) │   ← GPU 训练
          │  PyTorch          │
          └───────────────────┘
```

### 2.6 环境参数配置化

使用 Hydra 进行配置管理，所有环境参数外置：

```yaml
# configs/env/tetris_standard.yaml
env:
  cols: 10
  rows: 20
  hidden_rows: 2
  bag_type: "7bag"           # 7bag / random / fixed_sequence
  lock_delay_ms: 500
  lock_moves_max: 15
  gravity_curve: [1000,800,650,500,400,320,250,180,130,90,70,55,45,35,28,22,17,14,11,9]
  level_up_lines: 10
  next_queue_size: 4
  scoring:
    lines: {1:100, 2:300, 3:500, 4:800}
    hard_drop: 2
    soft_drop: 1
  reward_weights:
    height: 0.3
    holes: 1.5
    bumpiness: 0.2
    well: 0.5
    survival: 0.01
    death: -100.0
```

---

## 第三章　核心 AI 算法选型与设计

### 3.1 算法选型综述

| 算法 | 样本效率 | 最终表现 | 训练稳定性 | 工程复杂度 | 推荐度 |
|------|----------|----------|------------|------------|--------|
| **DQN + Extensions** | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ | **主方案** |
| PPO | ★★★☆☆ | ★★★★☆ | ★★★★★ | ★★☆☆☆ | 备选方案 |
| A3C/A2C | ★★☆☆☆ | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ | 不推荐 |
| AlphaZero (MCTS) | ★★★★★ | ★★★★★ | ★★★☆☆ | ★★★★★ | 远期目标 |
| SAC (连续动作) | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | ★★★☆☆ | 不适用 |

**选择理由**：

1. **DQN + Extensions（主方案）**：俄罗斯方块的动作空间是离散的（~40 actions），DQN 天然适合。Double DQN + Dueling + PER + Noisy Nets 的组合已在多项 Tetris RL 工作中被验证有效（如 DeepTamer）。off-policy 特性允许充分复用历史经验，样本效率高。

2. **PPO（备选方案）**：on-policy 算法，训练更稳定但样本效率较低。当 DQN 出现严重高估或训练不稳定时切换至此方案。PPO 的 clip 机制天然防止策略崩溃。

3. **AlphaZero-style MCTS（远期目标）**：结合 MCTS 的前向搜索与神经网络的评估能力，理论上能达到最强表现。但工程复杂度极高，暂不纳入第一期。

### 3.2 深度神经网络架构设计（主方案：Dueling DQN）

#### 3.2.1 网络总体结构

```
                     ┌──────────────────────┐
                     │   Input Layer         │
                     │   board: (22,10,1)    │
                     │   features: (53,)     │
                     └──────┬───────┬───────┘
                            │       │
               ┌────────────▼─┐  ┌──▼──────────────┐
               │  CNN Branch   │  │  MLP Branch      │
               │               │  │                   │
               │ Conv2d(       │  │ Linear(53→128)    │
               │  1→32,3×3,   │  │   + ReLU          │
               │  pad=1)      │  │ Linear(128→64)    │
               │   + ReLU     │  │   + ReLU          │
               │ Conv2d(       │  │                   │
               │  32→64,3×3, │  └────────┬──────────┘
               │  pad=1)      │           │
               │   + ReLU     │           │
               │ Conv2d(       │           │
               │  64→64,3×3, │           │
               │  pad=1)      │           │
               │   + ReLU     │           │
               │ GlobalAvgPool │           │
               │  → 64-dim    │           │
               └──────┬───────┘           │
                      │                   │
                      └───────┬───────────┘
                              │ Concat → 128-dim
                     ┌────────▼──────────┐
                     │  Dueling Head      │
                     │                    │
                     │  ┌── Value Stream ─┐│
                     │  │ 128→64→1       ││
                     │  └────────────────┘│
                     │  ┌── Advantage Str.┐│
                     │  │ 128→64→|A|    ││
                     │  └────────────────┘│
                     │                    │
                     │  Q(s,a) = V(s) +  │
                     │  A(s,a) - mean(A) │
                     └────────┬──────────┘
                              │
                     ┌────────▼──────────┐
                     │  Q-values         │
                     │  shape: (N, 41)   │
                     │  (40 placement + 1│
                     │   for hold flag)  │
                     └───────────────────┘
```

#### 3.2.2 Dueling Network 详解

将 Q 函数分解为状态价值 \(V(s)\) 和动作优势 \(A(s, a)\)：

\[
Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|A|}\sum_{a'} A(s, a')\right)
\]

- **为什么用 Dueling**：在俄罗斯方块中，许多状态的价值与具体动作选择关系不大（例如棋盘几近全满时，任何动作价值都很低）。Dueling 可以更高效地学习状态价值，而不必为每个动作单独估计。
- **优势函数均值归零化**：使用 mean 而非 max 做基线，保证数值稳定性且不改变相对排序。

#### 3.2.3 Noisy Networks 替代 ε-Greedy

在网络的最后一层全连接中使用带噪声的权重：

\[
y = (b + W \odot \epsilon^b) + (W + W \odot \epsilon^w) x
\]

其中 \(\epsilon^b, \epsilon^w\) 是独立高斯噪声，可学习参数为 \(\mu\) 和 \(\sigma\)（每个权重对应一对），噪声使用 Factorised Gaussian Noise 参数化以降低参数量。

```python
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def forward(self, x):
        # Factorised Gaussian noise
        epsilon_in = self._f(torch.randn(self.in_features, device=x.device))
        epsilon_out = self._f(torch.randn(self.out_features, device=x.device))
        weight = self.weight_mu + self.weight_sigma * torch.outer(epsilon_out, epsilon_in)
        bias = self.bias_mu + self.bias_sigma * epsilon_out
        return F.linear(x, weight, bias)

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()
```

**优势**：Noisy Nets 可以学习到状态相关的探索策略——在高原值区域减少噪声（利用），在不确定区域增加噪声（探索），不需要手工设计 ε 衰减曲线。

#### 3.2.4 Prioritized Experience Replay（PER）

**核心思想**：不是均匀地从回放缓冲区采样，而是按 TD-error 的绝对值加权采样。

**采样概率**（rank-based）：
\[
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad p_i = \frac{1}{\text{rank}(i)}
\]

**重要性采样权重**（修正分布偏移）：
\[
w_i = \left(\frac{1}{N} \cdot \frac{1}{P(i)}\right)^\beta / \max_j w_j
\]

**超参数设置**：
- \(\alpha = 0.6\)：控制优先级程度（0=均匀, 1=严格按TD-error）
- \(\beta = 0.4 \rightarrow 1.0\)：从训练开始到结束线性增长（最终完全修正偏差）
- 缓冲区大小：\(N = 1,000,000\)（100 万条转移）

**数据结构**：使用 SumTree 实现 \(O(\log N)\) 的采样和更新：

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)  # 线段树实现
        self.alpha = alpha
        self.max_priority = 1.0

    def add(self, transition, td_error=None):
        priority = (abs(td_error) + 1e-6) ** self.alpha if td_error else self.max_priority
        self.tree.add(priority, transition)

    def sample(self, batch_size, beta=0.4):
        batch = []
        indices = []
        weights = []
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            prob = priority / self.tree.total()
            weight = (len(self.tree) * prob) ** (-beta)
            batch.append(data)
            indices.append(idx)
            weights.append(weight)
        weights = np.array(weights) / max(weights)
        return batch, indices, weights
```

#### 3.2.5 Double DQN

解耦动作选择和动作评估：

\[
y_t = r_t + \gamma Q_{\text{target}}\left(s_{t+1}, \arg\max_a Q_{\text{online}}(s_{t+1}, a)\right)
\]

- Online 网络 \(\theta\)：选择最优动作
- Target 网络 \(\theta^-\)：评估该动作的值
- Target 网络每 \(C = 8000\) 步硬更新一次（或使用 Polyak 平均软更新 \(\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-\)，\(\tau=0.005\)）

#### 3.2.6 Multi-Step TD (N-step Return)

使用 N-step return 加速奖励传播：

\[
R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_a Q_{\text{target}}(s_{t+n}, a)
\]

- 默认 \(n = 5\)（俄罗斯方块中，Tetris 的准备需要约 5-8 步）
- N-step 对 Tetris 特别重要，因为它让"为 Tetris 准备 I 块列"这种需要多步规划的信用能更快传播

#### 3.2.7 完整 DQN 训练算法（伪代码）

```
Algorithm: Rainbow DQN for Tetris

Initialize:
    online network Q_θ, target network Q_θ⁻ ← Q_θ
    prioritized replay buffer D (capacity = 1M)
    optimizer: Adam(lr=6.25e-5)
    global_step = 0

Loop until convergence:
    s = env.reset()
    Loop per episode:
        a = argmax_a Q_θ(s, a) with noisy net exploration
        s', r, done, info = env.step(a)
        δ = r + γ*Q_θ⁻(s', argmax Q_θ(s')) - Q_θ(s, a)
        D.add((s, a, r, s', done), priority=|δ|)
        
        if global_step % 4 == 0:        # 每 4 个环境步训练一次
            batch, indices, weights = D.sample(batch_size=32, β=β)
            Compute n-step targets y_t using Double DQN
            L = Σ w_i * huber_loss(y_i - Q_θ(s_i, a_i))
            Optimizer.zero_grad()
            L.backward()
            clip_grad_norm_(max_norm=10)
            Optimizer.step()
            Update priorities in D with new |δ_i|
        
        if global_step % 8000 == 0:     # 目标网络更新
            Q_θ⁻ ← Q_θ   (hard update)
        
        Decay ε → 0.01 (if using ε-greedy fallback)
        Anneal β → 1.0
        global_step += 1
        s = s'
        if done: break

    Evaluate every 10K env steps: play 100 episodes (no noise)
```

### 3.3 PPO 设计（备选方案）

若 DQN 出现高估崩溃或策略振荡无法收敛，切换至 PPO。

#### 3.3.1 网络结构（Actor-Critic）

```
                     ┌──────────────────────┐
                     │   Shared Backbone     │
                     │   (同 DQN 的 CNN+MLP) │
                     └──────┬───────────────┘
                            │
               ┌────────────┼────────────┐
               │                         │
     ┌─────────▼─────────┐   ┌──────────▼─────────┐
     │  Policy Head      │   │  Value Head         │
     │  128→64→|A|     │   │  128→64→1          │
     │  (输出 π(a|s))    │   │  (输出 V(s))        │
     └───────────────────┘   └────────────────────┘
```

#### 3.3.2 PPO-Clip 目标函数

\[
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[\min\left(r_t(\theta) \hat{A}_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]
\]

其中 \(r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}\) 是新旧策略概率比，\(\epsilon = 0.2\)。

#### 3.3.3 GAE (Generalized Advantage Estimation)

\[
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}
\]

其中 \(\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\)，\(\lambda = 0.95\)。

#### 3.3.4 PPO 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| \(\gamma\) | 0.99 | 折扣因子 |
| \(\lambda\) (GAE) | 0.95 | GAE 平滑参数 |
| \(\epsilon\) (clip) | 0.2 | PPO clip 范围 |
| 学习率 | 2.5e-4 | Adam |
| 批大小 | 256 | 每次更新的样本数 |
| 小批量大小 | 64 | mini-batch |
| 更新轮数 | 4 | 每批数据重复训练轮数 |
| 熵系数 | 0.01 | 鼓励探索 |
| value loss coeff | 0.5 | 价值损失权重 |
| 收集步数 | 2048 | 每次更新前收集的交互步数 |
| 并行环境数 | 64 | |

### 3.4 辅助训练技术

#### 3.4.1 模仿学习预训练（Imitation Learning）

**目标**：用 Dellacherie 启发式算法预训练策略网络，实现冷启动，避免 RL 初期随机探索的大量无效样本。

**方法**：
1. 运行 Dellacherie 算法玩 10,000 局，收集 \((s, a)\) 对
2. 使用 Behavior Cloning（监督学习）预训练策略网络
3. 损失函数：\(L = -\sum_{(s,a)} \log \pi_\theta(a|s)\)

**Dellacherie 算法的评分函数**：
```
Score(placement) = w₁ × landing_height
                 + w₂ × cleared_lines
                 + w₃ × holes
                 + w₄ × bumpiness
                 + w₅ × max_well_depth
                 + w₆ × row_transitions

Weights (by particle swarm optimization):
w₁=-4.500, w₂=3.418, w₃=-7.899, w₄=-3.386, w₅=-3.129, w₆=-2.000
```

选择得分最高的合法放置作为 Dellacherie 的输出动作。

#### 3.4.2 课程学习（Curriculum Learning）

分阶段增加难度，帮助策略逐步学习：

| 阶段 | 重力速度 | 棋盘高度 | 训练步数 |
|------|----------|----------|----------|
| Stage 1 | 2,000ms (极慢) | 20 行 | 500 万 |
| Stage 2 | 1,000ms (Level 1) | 20 行 | 500 万 |
| Stage 3 | 500ms (Level 5) | 20 行 | 500 万 |
| Stage 4 | 200ms (Level 10) | 20 行 | 500 万 |
| Stage 5 | 50ms (Level 15) | 20 行 | 500 万 |
| Stage 6 | 9ms (Level 20) | 20 行 | 持续 |

每个阶段开始时加载上一阶段的最优 checkpoint 继续训练。

#### 3.4.3 Self-Play / Population-Based Training（远期）

训练一个包含多个策略的种群，相互竞争（在相同随机种子下比较得分）：

1. 维护 N=10 个策略的种群
2. 定期评估，淘汰表现最差的 20%
3. 用表现最好的策略变异（Exploit & Explore）替换被淘汰者
4. 变异操作：学习率扰动、噪声参数重采样、奖励权重微调

#### 3.4.4 Intrinsic Curiosity Module (ICM)（可选）

对于纯原始的稀疏奖励环境（禁用塑性奖励时），使用 ICM 增强探索：

\[
r_t^{\text{intrinsic}} = \frac{\eta}{2} \|\hat{\phi}(s_{t+1}) - \phi(s_{t+1})\|_2^2
\]

其中 \(\phi\) 是学习到的状态编码（逆动力学模型的前向特征），\(\hat{\phi}(s_{t+1})\) 是从 \(s_t, a_t\) 预测的下一状态编码。ICM 奖励模型预测误差大（即出乎意料）的转移。

> **注意**：有了塑性奖励后 ICM 通常不是必需的。仅在消融实验中验证其作用。

---

## 第四章　训练系统与流程

### 4.1 训练流程总览

```
┌────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                        │
│                                                                │
│  ┌──────────┐   ┌───────────┐   ┌───────────┐   ┌──────────┐ │
│  │ 64 Actor │──▶│ Replay    │──▶│  Learner  │──▶│ Target   │ │
│  │ Processes│   │ Buffer    │   │  (GPU)    │   │ Network  │ │
│  │ (C++ env)│   │ (1M cap)  │   │  Train    │   │ Sync     │ │
│  └──────────┘   └───────────┘   └───────────┘   └──────────┘ │
│       │                                │                       │
│       │      ┌───────────┐             │                       │
│       └──────│ Parameter │◀────────────┘                       │
│              │ Server    │  (梯度/权重同步)                    │
│              └───────────┘                                     │
│                                                                │
│  ┌──────────┐   ┌───────────┐   ┌───────────┐                │
│  │ Eval     │──▶│ Metrics   │──▶│ Checkpoint│                │
│  │ (100 eps)│   │ (WandB)   │   │ Save      │                │
│  └──────────┘   └───────────┘   └───────────┘                │
└────────────────────────────────────────────────────────────────┘
```

**核心流程**：
1. **Actor**：64 个独立进程，每个运行 C++ 环境 + 当前在线网络的参数副本。使用 Noisy Nets 进行探索，生成转移样本（s, a, r, s', done）。
2. **Replay Buffer**：集中式的 Prioritized Experience Replay 缓冲区，存储所有 Actor 的样本。使用 Socket 接收样本，SumTree 实现优先级采样。
3. **Learner**：GPU 训练进程，每 4 个环境步从回放缓冲区采样一个 batch（32 条转移），计算 n-step TD target，执行梯度更新，并将新参数推送给 Actor。
4. **Evaluator**：每 10K 训练步触发一次评估，使用确定性策略（关闭噪声）玩 100 局，记录平均得分和消行数。

### 4.2 经验回放缓冲区设计

#### 4.2.1 核心数据结构

```python
@dataclass
class Transition:
    state_board: np.ndarray       # (22, 10) bool
    state_features: np.ndarray    # (53,) float32
    action: int                   # 动作索引 (0-40)
    reward: float                 # 即时奖励
    next_state_board: np.ndarray  # (22, 10) bool
    next_state_features: np.ndarray  # (53,) float32
    done: bool
    n_step_return: float          # 预计算的 n-step return
    n_step_gamma: float           # γ^n
```

#### 4.2.2 缓冲区配置

| 参数 | 值 | 说明 |
|------|-----|------|
| capacity | 1,000,000 | 总容量 |
| sampling | Prioritized (rank-based) | 采样策略 |
| α | 0.6 | 优先级指数 |
| β₀ | 0.4 | 初始 IS 修正指数 |
| β_final | 1.0 | 最终 IS 修正指数 |
| β_frames | 10,000,000 | β 线性增长到此帧数 |
| n_step | 5 | N-step return 步数 |

#### 4.2.3 分布式回放（可选扩展）

当单机内存不足以存放 100 万条转移时，切换至 Google Reverb 或自建 gRPC 回放服务：

```
Actor ──gRPC──▶ Reverb Server ──gRPC──▶ Learner
                    │
                    └── (sharded storage, ~1TB NVMe)
```

### 4.3 分布式训练架构

#### 4.3.1 进程拓扑

```
┌──────────────────────────────────────────────────────┐
│                   Trainer Node (GPU)                  │
│                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ Learner     │  │ ReplayBuffer │  │ Watcher     │ │
│  │ (GPU, 1×) │  │ (Memory)     │  │ (Log/Eval)  │ │
│  └──────┬──────┘  └──────▲───────┘  └─────────────┘ │
│         │                │                            │
│         │      params    │ samples                    │
│         │                │                            │
│  ┌──────▼────────────────┴───────────────────────┐   │
│  │           Parameter + Sample Bus              │   │
│  │           (Shared Memory / gRPC)              │   │
│  └──────┬────────────────┬───────────────────────┘   │
└─────────┼────────────────┼───────────────────────────┘
          │                │
    ┌─────▼──┐      ┌─────▼──┐     ┌─────▼──┐
    │Actor 0 │      │Actor 1 │ ... │Actor 63│
    │C++ Env │      │C++ Env │     │C++ Env │
    │ONNX    │      │ONNX    │     │ONNX    │
    │Infer   │      │Infer   │     │Infer   │
    └────────┘      └────────┘     └────────┘
```

#### 4.3.2 同步机制

**参数同步（Learner → Actor）**：
- 频率：每 100 训练步
- 方式：使用 PyTorch 的 `share_memory_()` 将模型参数放入共享内存，Actor 通过 `model.load_state_dict()` 同步

**样本收集（Actor → Buffer）**：
- 频率：每步
- 方式：通过 multiprocessing.Queue 或 ZeroMQ PUSH/PULL 传输

#### 4.3.3 硬件配置建议

| 组件 | 推荐配置 |
|------|----------|
| GPU | 1× NVIDIA RTX 4090 / A100 |
| CPU | 32+ cores (for 64 Actor processes) |
| 内存 | 64 GB+ |
| 存储 | 500 GB NVMe (for replay + checkpoints) |
| 预期训练时长 | 2-4 天达到可用水平, 7-14 天达到顶尖水平 |

### 4.4 超参数管理

#### 4.4.1 核心超参数表

```yaml
# configs/training/dqn_rainbow.yaml
training:
  total_steps: 50_000_000        # 总训练步数
  learning_rate: 6.25e-5         # Adam 学习率
  batch_size: 32                 # 训练批大小
  train_every: 4                 # 每 N 个环境步训练一次
  grad_clip_norm: 10.0           # 梯度裁剪
  target_update_freq: 8000       # 目标网络硬更新频率
  target_update_tau: null        # 软更新系数（与硬更新二选一）

rl:
  gamma: 0.99                    # 折扣因子
  n_step: 5                      # N-step return
  double_dqn: true               # 启用 Double DQN
  dueling: true                  # 启用 Dueling Network
  
per:
  alpha: 0.6                     # PER 优先级指数
  beta_start: 0.4                # IS 修正初始值
  beta_end: 1.0                  # IS 修正最终值
  beta_frames: 10_000_000        # β 衰减帧数

exploration:
  type: "noisy_nets"             # noisy_nets / epsilon_greedy
  epsilon_start: 1.0             # (仅 ε-greedy 模式)
  epsilon_end: 0.01
  epsilon_decay: 1_000_000

network:
  cnn_channels: [32, 64, 64]     # CNN 通道数
  cnn_kernel: 3                  # 卷积核大小
  hidden_dim: 128                # 隐藏层维度
  noisy_sigma_init: 0.017        # NoisyNet σ 初始化值
```

#### 4.4.2 学习率调度

采用 Warmup + Cosine Annealing：

```python
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    [
        LinearLR(optimizer, start_factor=0.01, total_iters=5000),  # Warmup
        CosineAnnealingLR(optimizer, T_max=50_000_000 - 5000)       # Decay
    ],
    milestones=[5000]
)
```

### 4.5 实验追踪与可视化

**使用 Weights & Biases (wandb)** 记录：

| 类别 | 指标 | 用途 |
|------|------|------|
| **奖励** | episode_reward, episode_reward_ema(100) | 监控策略改进 |
| **消行** | lines_cleared, tetris_rate, tspin_rate | 游戏技术指标 |
| **损失** | q_loss, td_error_mean, td_error_std | 训练稳定性 |
| **Q 值** | q_value_mean, q_value_max, q_value_overestimation | 检测价值高估 |
| **梯度** | grad_norm_mean, grad_norm_max | 梯度爆炸检测 |
| **吞吐** | env_fps, train_fps, gpu_utilization | 性能优化 |
| **探索** | action_entropy, visited_state_coverage | 探索充分性 |

### 4.6 训练容错与恢复

```python
class CheckpointManager:
    def save(self, step, model, optimizer, scheduler, replay_buffer, metrics):
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'replay_buffer': replay_buffer.state_dict(),
            'metrics': metrics,
            'timestamp': time.time()
        }
        torch.save(checkpoint, f'checkpoints/step_{step:09d}.pt')
        # 保留最近 10 个 checkpoint
        self._cleanup_old('checkpoints/', keep=10)

    def load(self, path):
        checkpoint = torch.load(path)
        # 恢复所有组件
        ...
```

**故障处理策略**：
- **GPU OOM**：自动减小 batch size 并重试
- **Actor 进程崩溃**：重拉新 Actor 进程，不影响训练主循环
- **磁盘满**：监控 checkpoint 目录大小，自动清除最旧文件

---

## 第五章　推理与部署

### 5.1 推理优化

#### 5.1.1 模型导出链路

```
PyTorch Model (training)
    │
    ▼ torch.onnx.export()
ONNX Model
    │
    ├──▶ ONNX Runtime (C++ inference)
    │
    └──▶ TensorRT (optional, for NVIDIA GPU deployment)
         FP16 / INT8 quantization
```

```python
# 导出脚本
dummy_board = torch.randn(1, 1, 22, 10)
dummy_features = torch.randn(1, 53)
torch.onnx.export(
    model,
    (dummy_board, dummy_features),
    "tetris_dqn.onnx",
    input_names=['board', 'features'],
    output_names=['q_values'],
    dynamic_axes={'board': {0: 'batch'}, 'features': {0: 'batch'}, 'q_values': {0: 'batch'}},
    opset_version=14
)
```

#### 5.1.2 性能优化技术

| 技术 | 效果 | 实现 |
|------|------|------|
| FP16 推理 | 1.5-2× 加速, 内存减半 | ONNX Runtime `session_options.graph_optimization_level` |
| INT8 量化 | 2-3× 加速 | TensorRT calibration + QAT (quantization-aware training) |
| 算子融合 | 5-15% 加速 | Conv+BN+ReLU → fused op |
| Batch Inference | 批量处理多个环境 | 同时输入 64 个环境的状态 |

#### 5.1.3 推理延迟基准

| 配置 | 延迟 (batch=1) | 延迟 (batch=64) |
|------|---------------|-----------------|
| PyTorch FP32 (CPU) | ~2ms | ~8ms |
| PyTorch FP32 (GPU) | ~0.8ms | ~1.5ms |
| ONNX Runtime FP32 (CPU) | ~0.5ms | ~3ms |
| ONNX Runtime FP16 (GPU) | ~0.2ms | ~0.5ms |
| TensorRT INT8 (GPU) | ~0.1ms | ~0.3ms |

**目标**：单次推理 < 0.5ms，配合 C++ Bitboard 环境实现 >50K FPS 的采样速度。

### 5.2 C++ 推理架构

```cpp
class TetrisAgent {
public:
    TetrisAgent(const std::string& model_path);
    ~TetrisAgent();

    // 主推理接口：给定状态，返回最优动作
    Action selectAction(const GameState& state);

private:
    // ONNX Runtime session
    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // 输入/输出张量内存（预分配，避免反复分配）
    std::vector<float> board_input_;      // 1 × 1 × 22 × 10
    std::vector<float> features_input_;   // 1 × 53
    std::vector<float> q_values_output_;  // 1 × 41

    // 模型推理
    void infer(const float* board, const float* features, float* q_values);

    // 动作选择（带掩码）
    Action selectBestAction(const float* q_values,
                            const std::vector<Action>& legal_actions);
};

Action TetrisAgent::selectAction(const GameState& state) {
    // 1. 编码状态为模型输入
    encodeState(state, board_input_.data(), features_input_.data());

    // 2. 模型推理（ONNX Runtime）
    infer(board_input_.data(), features_input_.data(), q_values_output_.data());

    // 3. 获取合法动作列表
    auto legal_actions = state.getLegalActions();

    // 4. 选择 Q 值最高的合法动作
    return selectBestAction(q_values_output_.data(), legal_actions);
}
```

### 5.3 实时性保障

在实时对局中，推理延迟必须在重力滴答（最小 9ms @ Level 20）内完成：

| 层级 | 措施 |
|------|------|
| **推理优先级** | 独立推理线程，最高优先级（SCHED_FIFO） |
| **预计算** | 在当前块锁定后的间隙期提前推理下一块的动作 |
| **Fallback** | 推理超时（>2ms）时自动降级为 Dellacherie 启发式决策 |
| **帧率保证** | 推理线程与渲染线程分离，确保 60FPS 渲染不受 AI 推理影响 |

### 5.4 多场景部署

| 场景 | 方案 | 技术栈 |
|------|------|--------|
| **嵌入式（本地应用）** | C++ DLL / .so 库，集成 ONNX Runtime | CMake + ONNX Runtime C API |
| **浏览器** | ONNX Runtime Web + WebAssembly | TypeScript + Web Workers |
| **云端 API** | FastAPI + ONNX Runtime + GPU | Docker + Kubernetes + Triton |
| **Python SDK** | pip 包，内置 pybind11 绑定 | pybind11 + setuptools |

**浏览器端部署示例**：
```javascript
// tetris-ai-web.js
import * as ort from 'onnxruntime-web';

class TetrisAI {
    async init(modelUrl) {
        this.session = await ort.InferenceSession.create(modelUrl);
    }

    async selectAction(state) {
        const input = this.encodeState(state);
        const output = await this.session.run({ board: input.board, features: input.features });
        const qValues = output.q_values.data;
        return this.argmaxMasked(qValues, state.legalActions);
    }
}
```

---

## 第六章　性能评估指标

### 6.1 游戏表现指标

| 指标 | 定义 | 目标值 |
|------|------|--------|
| **平均得分** | 100 局测试的平均最终得分 | ≥ 10,000,000 |
| **最大得分** | 100 局中的最高得分 | ≥ 50,000,000 |
| **平均消行数** | 每局平均消行数 | ≥ 3,000 |
| **Tetris 率** | 四行消除次数 / 总消行事件 | ≥ 40% |
| **T-Spin 率** | T-Spin 消除次数 / 总消除事件 | ≥ 5% |
| **Perfect Clear 率** | 清空棋盘次数 / 总局数 | ≥ 1% |
| **方块效率** | 消行数 / 放置方块数 | ≥ 0.8 |
| **生存率 (5min)** | 存活超过 5 分钟的概率 | ≥ 80% |
| **Level 15 生存率** | 在 Level 15 重力下存活 200 块的比率 | ≥ 60% |

### 6.2 学习性能指标

| 指标 | 定义 | 目标 |
|------|------|------|
| **样本效率** | 达到 100 万分的训练帧数 | ≤ 2,500 万帧 |
| **收敛时间** | 策略达到稳定表现的 wall-clock 时间 | ≤ 3 天 (单 RTX 4090) |
| **训练稳定性** | 最后 1000 个 episode 奖励的标准差 | 相对值 ≤ 15% |
| **FPS (采样)** | 总环境步/秒（含推理+环境更新） | ≥ 100,000 FPS |
| **GPU 利用率** | 训练期间的 GPU SM 利用率 | ≥ 80% |

### 6.3 泛化能力评估

| 测试场景 | 变化 | 评估方法 |
|----------|------|----------|
| **跨尺寸** | 棋盘 5×10 / 8×8 / 15×30 | 零样本测试（zero-shot）和微调后测试 |
| **跨重力** | 重力曲线整体上移/下移 50% | 测试不同重力下的得分衰减 |
| **跨随机策略** | 纯随机块（非 7-bag） | 测试策略对序列随机性变化的鲁棒性 |
| **噪声干扰** | 每 N 步随机插入一个垃圾方块 | 模拟对战模式的干扰场景 |
| **跨旋转系统** | 从 SRS 切换到 ARS / TGM | 测试策略对物理系统的依赖程度 |

### 6.4 基准对比体系

| Baseline | 预期得分 (Level 1-15) | 说明 |
|----------|----------------------|------|
| **随机策略** | ~100 | 性能下界 |
| **贪心策略** (只考虑消行) | ~5,000 | 简单启发式 |
| **Dellacherie 算法** | ~5,000,000 | 最佳传统启发式, 六大特征加权 |
| **DeepTamer (2018)** | ~15,000,000 | 已有 RL 工作的参考表现 |
| **人类初学者** | ~10,000 | |
| **人类高级玩家** | ~500,000 | |
| **人类顶尖选手** | ~10,000,000+ | CTWC (Classic Tetris World Championship) |
| **AI 理论上界** | ~∞ (取决于 RNG) | 完美 AI 的理论上限 |

### 6.5 消融实验设计

逐模块考察各组件的边际贡献：

| 实验编号 | 移除的组件 | 预期影响 |
|----------|-----------|----------|
| **Full** | (基准完整模型) | 最佳表现 |
| **-PER** | Prioritized Experience Replay → 均匀采样 | 样本效率 ↓ 20-30% |
| **-Dueling** | Dueling Head → 标准 Q Head | 最终得分 ↓ 10-15% |
| **-NoisyNet** | Noisy Nets → ε-greedy (ε=0.1→0.01) | 最终得分 ↓ 5-10% |
| **-NStep** | 5-step → 1-step TD | 样本效率 ↓ 30-40% |
| **-Double** | Double DQN → Vanilla DQN | Q 值高估，后期崩溃 |
| **-RewardShaping** | 移除所有塑性奖励 | 样本效率 ↓ 50%+, 可能不收敛 |
| **-Curriculum** | 直接从 Level 15 开始 | 早期无法学习 |
| **-Pretraining** | 无模仿学习预训练 | 冷启动期延长 2-3× |

---

## 第七章　技术栈与代码模块划分

### 7.1 技术栈总览

| 层级 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **环境核心** | C++17 + Eigen | - | bitboard 棋盘、碰撞检测、行消除 |
| **Python 绑定** | pybind11 | ≥2.10 | C++ → Python 接口暴露 |
| **环境封装** | Gymnasium | ≥0.29 | 标准化 RL 环境接口 |
| **深度学习** | PyTorch | ≥2.0 | 神经网络定义与训练 |
| **数值计算** | NumPy | ≥1.24 | 数组操作与数据预处理 |
| **推理部署** | ONNX Runtime | ≥1.16 | C++ 端高效推理 |
| **配置管理** | Hydra / OmegaConf | ≥2.3 | 超参数和实验配置 |
| **实验追踪** | Weights & Biases | ≥0.15 | 指标记录与可视化 |
| **并行** | Python multiprocessing | - | 多环境并行与分布式训练 |
| **测试** | pytest | ≥7.0 | 单元测试与集成测试 |
| **依赖管理** | Poetry | ≥1.5 | Python 依赖管理 |
| **构建系统** | CMake | ≥3.20 | C++ 编译 |
| **容器化** | Docker + nvidia-docker | - | 训练环境一致性与部署 |

### 7.2 模块结构树

```
tetris-ai/
│
├── env/                              # ──────── 游戏环境 ────────
│   ├── core/                         # C++ 环境核心
│   │   ├── CMakeLists.txt
│   │   ├── tetris_core.h/cpp         # 位图棋盘、碰撞检测、行消除
│   │   ├── tetris_env.h/cpp          # 环境交互逻辑 (step/reset/reward)
│   │   ├── piece_data.h/cpp          # 方块形状定义、SRS 踢墙表
│   │   ├── randomizer.h/cpp          # 7-bag 随机生成器
│   │   └── action_gen.h/cpp          # 合法动作生成器（placement-based）
│   ├── bindings/                     # pybind11 绑定
│   │   ├── CMakeLists.txt
│   │   └── tetris_bindings.cpp       # C++ → Python 接口
│   ├── __init__.py
│   ├── tetris_env.py                 # Gymnasium 环境封装
│   ├── state_encoder.py              # 状态表示编码（bitmap/features）
│   └── reward_calculator.py          # 奖励函数计算
│
├── agent/                            # ──────── AI 算法 ────────
│   ├── __init__.py
│   ├── dqn.py                        # DQN 训练器（完整 Rainbow 实现）
│   ├── ppo.py                        # PPO 训练器（备选）
│   ├── model.py                      # 神经网络定义 (CNN+MLP+DuelingHead)
│   ├── noisy_layers.py               # NoisyLinear / FactorisedNoise
│   ├── memory.py                     # Prioritized Experience Replay (SumTree)
│   ├── nstep_buffer.py               # N-step return 缓冲区
│   ├── action_mask.py                # 动作掩码（过滤非法动作）
│   └── pretrain.py                   # 模仿学习预训练（Dellacherie BC）
│
├── trainer/                          # ──────── 训练系统 ────────
│   ├── __init__.py
│   ├── trainer.py                    # 主训练循环
│   ├── distributed.py                # 分布式训练管理器（Actor ↔ Learner）
│   ├── actor_worker.py               # Actor 进程（独立采样 + 参数同步）
│   ├── evaluator.py                  # 评估循环（确定性策略）
│   ├── checkpoint.py                 # Checkpoint 保存/加载管理
│   └── logger.py                     # WandB / TensorBoard 日志
│
├── inference/                        # ──────── 推理部署 ────────
│   ├── cpp/                          # C++ 推理端
│   │   ├── CMakeLists.txt
│   │   ├── model_loader.h/cpp        # ONNX Runtime 模型加载与推理
│   │   ├── player.h/cpp              # 在线对局 AI 控制器
│   │   └── web/                      # WebAssembly 部署
│   │       ├── CMakeLists.txt
│   │       └── tetris_ai_web.cpp     # Emscripten + ONNX Runtime Web
│   └── python/                       # Python 推理端
│       ├── infer.py                  # ONNX Runtime Python 推理
│       └── export.py                 # PyTorch → ONNX 导出脚本
│
├── scripts/                          # ──────── 入口脚本 ────────
│   ├── train.py                      # 训练入口
│   ├── play.py                       # 人机对弈 / 演示
│   ├── eval.py                       # 批量评估
│   └── export_model.py               # 模型导出
│
├── configs/                          # ──────── 配置文件 ────────
│   ├── env/
│   │   └── tetris_standard.yaml      # 环境参数配置
│   ├── training/
│   │   ├── dqn_rainbow.yaml          # DQN 超参数
│   │   └── ppo.yaml                  # PPO 超参数
│   └── experiment/
│       └── baseline.yaml             # 实验配置（Hydra 组合）
│
├── tests/                            # ──────── 测试 ────────
│   ├── test_env.py                   # 环境逻辑测试
│   ├── test_state_encoder.py         # 状态编码测试
│   ├── test_reward.py                # 奖励函数测试
│   ├── test_model.py                 # 网络结构测试
│   ├── test_memory.py                # 回放缓冲区测试
│   └── test_action_mask.py           # 动作掩码测试
│
├── pyproject.toml                    # Poetry 依赖配置
├── CMakeLists.txt                    # 顶层 CMake
├── Dockerfile                        # 训练容器
├── Dockerfile.inference              # 推理容器
├── .gitignore
└── README.md
```

### 7.3 Python/C++ 配合机制

#### 7.3.1 数据流全景

```
Training Phase:
┌──────────────┐   State (numpy)    ┌────────────┐   Batch (GPU tensor)   ┌─────────────┐
│ C++ Env Core │ ──pybind11────────▶│ Python      │ ──torch──────────────▶│ PyTorch GPU │
│ (bitboard     │                    │ Env Wrapper │                        │ Learner     │
│  collisions)  │◀──pybind11────────│ (Gymnasium) │◀──torch───────────────│             │
└──────────────┘   Action (int)     └────────────┘   Gradients             └─────────────┘

Inference Phase:
┌──────────────┐   State (float*)   ┌─────────────┐   Q-values (float*)   ┌──────────────┐
│ Game Loop    │ ──direct call─────▶│ ONNX Runtime │ ────────────────────▶│ Action       │
│ (C++)        │                    │ (C++ API)    │                       │ Selector     │
└──────────────┘                    └─────────────┘                       └──────────────┘
```

#### 7.3.2 pybind11 绑定示例

```cpp
// tetris_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tetris_env.h"

namespace py = pybind11;

PYBIND11_MODULE(tetris_core, m) {
    py::class_<TetrisEnv>(m, "TetrisEnv")
        .def(py::init<int, int, int, const std::string&>(),
             py::arg("cols")=10, py::arg("rows")=20,
             py::arg("hidden_rows")=2, py::arg("bag_type")="7bag")
        .def("reset", &TetrisEnv::reset)
        .def("step", &TetrisEnv::step)
        .def("get_state_board", [](const TetrisEnv& env) {
            auto board = env.getBoard();
            return py::array_t<uint8_t>(
                {22, 10}, {10, 1}, board.data()  // zero-copy
            );
        })
        .def("get_legal_actions", &TetrisEnv::getLegalActions)
        .def("get_drop_speed", &TetrisEnv::getDropSpeed);
}
```

#### 7.3.3 进程间通信对比

| 方案 | 延迟 | 吞吐 | 复杂度 | 适用场景 |
|------|------|------|--------|----------|
| **共享内存 (SharedMemory)** | ~1μs | 最高 | 中 | 同机 Actor-Learner 参数同步 |
| **ZeroMQ** | ~50μs | 高 | 中 | 样本传输（Actor → Buffer） |
| **gRPC** | ~200μs | 中 | 高 | 跨机分布式训练 |
| **Redis** | ~500μs | 中 | 低 | 小规模分布式训练 |

**推荐方案**：同机集群使用**共享内存** + **ZeroMQ**，跨机扩展使用 **gRPC**。

### 7.4 构建与依赖管理

#### 7.4.1 CMake 配置

```cmake
# CMakeLists.txt (顶层)
cmake_minimum_required(VERSION 3.20)
project(tetris_ai_env LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto")

find_package(pybind11 REQUIRED)
find_package(onnxruntime REQUIRED)

add_subdirectory(env/core)
add_subdirectory(env/bindings)
add_subdirectory(inference/cpp)
```

#### 7.4.2 Python 依赖 (pyproject.toml)

```toml
[tool.poetry]
name = "tetris-ai"
version = "0.1.0"
description = "Reinforcement Learning Tetris AI Engine"

[tool.poetry.dependencies]
python = ">=3.10,<3.12"
torch = ">=2.0.0"
numpy = ">=1.24.0"
gymnasium = ">=0.29.0"
hydra-core = ">=2.3.0"
onnxruntime = ">=1.16.0"
wandb = ">=0.15.0"
pybind11 = ">=2.10.0"

[tool.poetry.group.dev.dependencies]
pytest = ">=7.0.0"
black = ">=23.0.0"
ruff = ">=0.1.0"
mypy = ">=1.0.0"
```

#### 7.4.3 Docker 容器化

```dockerfile
# Dockerfile (训练镜像)
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip cmake build-essential \
    libomp-dev libopenblas-dev

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install

COPY . .
RUN cd env/core && mkdir build && cd build && \
    cmake .. && make -j$(nproc)

CMD ["poetry", "run", "python", "scripts/train.py"]
```

---

## 附录

### 附录 A：标准俄罗斯方块规则与 SRS 踢墙表

本项目严格遵循 Tetris Guideline 标准：

- **棋盘**: 10 列 × 20 行可见区域 + 2 行隐藏区域（用于方块生成） = 22 行总计
- **方块**: 7 种标准 Tetromino（I, O, T, S, Z, J, L），每种 4 个旋转状态
- **旋转系统**: Super Rotation System (SRS)，包含标准踢墙表和 I 块专用踢墙表
- **随机生成**: 7-bag 随机（每个 7 块周期内每种方块恰好出现一次）
- **锁定延迟**: 500ms，最多 15 次移动/旋转重置
- **重力**: Level 1 为 1000ms 间隔，Level 20 为 9ms 间隔
- **NEXT 队列**: 显示 4 个
- **HOLD**: 单次 Hold，换块后需重新落地才能再次 Hold

SRS 踢墙表细节参考 `tetris/code.html` 第 306-327 行。

### 附录 B：Dellacherie 启发式算法

```python
def dellacherie_evaluate(board, piece):
    """
    Dellacherie 的六大特征评估函数。
    对每个合法放置计算加权评分，选择最高分放置。
    """
    best_score = float('-inf')
    best_action = None

    for rotation in range(4):
        for column in range(-2, 12):
            ghost_y = compute_ghost(board, piece, rotation, column)
            if not is_valid(board, piece, rotation, column, ghost_y):
                continue

            # 1. 着陆高度
            landing_height = ghost_y

            # 2. 消除行数
            cleared = count_cleared_lines(board_after_placement)

            # 3. 孔洞数
            holes = count_holes(board_after_placement)

            # 4. 崎岖度
            bumpiness = sum(abs(h[i] - h[i+1]) for i in range(9))

            # 5. 最大井深
            max_well = max(well_depth(h, i) for i in range(10))

            # 6. 行变换数
            row_transitions = count_row_transitions(board_after_placement)

            score = (-4.500 * landing_height
                     + 3.418 * cleared
                     - 7.899 * holes
                     - 3.386 * bumpiness
                     - 3.129 * max_well
                     - 2.000 * row_transitions)

            if score > best_score:
                best_score = score
                best_action = (rotation, column)

    return best_action
```

### 附录 C：参考文献

| 论文 | 内容 | 链接 |
|------|------|------|
| Mnih et al. (2015) | DQN — Human-level control through deep RL | Nature |
| Van Hasselt et al. (2016) | Double DQN | AAAI |
| Wang et al. (2016) | Dueling Network Architectures | ICML |
| Schaul et al. (2016) | Prioritized Experience Replay | ICLR |
| Fortunato et al. (2018) | Noisy Networks for Exploration | ICLR |
| Hessel et al. (2018) | Rainbow: Combining Improvements in DQN | AAAI |
| Schulman et al. (2017) | PPO | arXiv |
| Mnih et al. (2016) | A3C | ICML |
| Dellacherie (2003) | Tetris AI with heuristic evaluation | — |
| Stevens & Pradhan (2021) | DeepTamer — Tetris with Deep RL | — |
| Silver et al. (2017) | AlphaZero — Mastering chess and shogi | Science |
| Burda et al. (2019) | RND — Exploration by Random Network Distillation | ICLR |
| Pathak et al. (2017) | ICM — Curiosity-driven Exploration | ICML |

### 附录 D：术语表

| 中文 | 英文 | 缩写 | 释义 |
|------|------|------|------|
| 马尔可夫决策过程 | Markov Decision Process | MDP | RL 问题的形式化框架 |
| 强化学习 | Reinforcement Learning | RL | 通过与环境交互学习最优策略 |
| 深度 Q 网络 | Deep Q-Network | DQN | 基于 Q-Learning 的深度 RL 算法 |
| 双重 DQN | Double DQN | DDQN | 解耦动作选择与评估的 DQN 变体 |
| 竞争网络 | Dueling Network | — | 分解 Q=V+A 的网络架构 |
| 优先经验回放 | Prioritized Experience Replay | PER | 按 TD-error 优先级采样经验 |
| 广义优势估计 | Generalized Advantage Estimation | GAE | 权衡偏差-方差的优势估计 |
| 近端策略优化 | Proximal Policy Optimization | PPO | on-policy 的稳定策略梯度算法 |
| 超级旋转系统 | Super Rotation System | SRS | 标准俄罗斯方块旋转与踢墙规则 |
| 位图棋盘 | Bitboard | — | 用位运算表示的棋盘数据结构 |
| 模仿学习 | Imitation Learning | IL | 从专家示范中学习策略 |
| 课程学习 | Curriculum Learning | — | 从简单任务渐进到困难任务 |
| 奖励塑形 | Reward Shaping | — | 设计辅助奖励加速训练 |
| 消融实验 | Ablation Study | — | 移除组件以衡量其贡献 |

---

> **文档状态**: 待评审  
> **下一阶段**: 按此设计文档启动环境核心（C++ bitboard）的编码实现
