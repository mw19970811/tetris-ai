# 最终评估设计文档

## 目标

训练完成后，在**无探索噪声**的条件下，将训练好的智能体与 Dellacherie 基线进行**同种子头对头对比**，
判断 RL 策略是否超越了启发式专家。

## 评估条件

| 条件 | 说明 |
|------|------|
| **初始局面** | 每局 `env.reset(seed=i)`，i=0..199，双方使用相同种子 |
| **探索噪声** | 关闭。智能体处于 `eval_mode()`，NoisyNet 仅使用 μ 权重 |
| **动作选择** | `deterministic=True`，始终选择 Q 值最高的动作（贪婪策略） |
| **Dellacherie 基线** | 同样 200 局，相同种子序列，六特征加权 argmax |
| **局数** | 200 局 |

## 评估流程

```
for i in 0..199:
    env_agent = make_env()
    env_dl    = make_env()
    env_agent.reset(seed=i)
    env_dl.reset(seed=i)

    while not done:
        agent: masked_q.argmax()  →  Action  →  env_agent.step()
        dl:    Dellacherie argmax  →  Action  →  env_dl.step()

    记录双方的 score, lines, steps
```

## 输出指标

| 指标 | 含义 |
|------|------|
| Agent Avg Score | 智能体 200 局平均分 |
| Agent Max Score | 智能体最高单局分 |
| DL Avg Score | Dellacherie 200 局平均分 |
| DL Max Score | Dellacherie 最高单局分 |
| Win Rate | 智能体得分 > Dellacherie 得分的局数占比 |
| Mean Gap | Agent − DL 的平均分差 |
| Median Gap | Agent − DL 的分差中位数 |
| 显著性 | 配对 t 检验 / Wilcoxon 检验结果 |

## 判断标准

- **Win Rate > 50%** 且 **Mean Gap > 0**：智能体超越基线
- **Win Rate ≈ 50%** 且 **Gap ≈ 0**：智能体与基线持平
- **Win Rate < 50%** 且 **Mean Gap < 0**：智能体未超越基线

## 代码位置

- `trainer/evaluator.py` — `head_to_head()` 新增方法
- `trainer/trainer.py` — 训练结束调用，替换现有 `_eval_dellacherie` 独立评估
- `agent/dellacherie.py` — `run_episode()` 复用（已存在）

---

## 稳健性压力测试（待实现）

标准 head-to-head 评估（空棋盘、同种子）衡量的是"最优条件下的性能"。但 RL agent 相对于 Dellacherie 的**核心优势**在于全局规划能力和恢复策略——这些只能在恶劣初始条件下体现。

Dellacherie 是贪婪算法，只做局部最优选择。在混乱棋盘上，它没有"先填平、再挖井"的规划能力。如果 RL agent 真正学会了通用策略，应在压力测试中展示显著的恢复优势。

### 测试 A：随机填充初始棋盘

```
for i in 0..199:
    env_agent.reset(seed=i); env_dl.reset(seed=i)

    # 随机放置 N 个方块（双方相同的随机序列）
    for _ in range(N_blocks):
        action = random.choice(legal_actions)
        env_agent.step(action)  # 不记录分数
        env_dl.step(action)

    # 从混乱状态开始正式评估
    while not done:
        agent: argmax → env_agent.step()
        dl:    argmax → env_dl.step()
```

建议测试 `N_blocks ∈ {5, 10, 20}` 三种难度。

### 测试 B：故意制造深井

在棋盘特定列（如 column 0 或 column 9）预先放置方块制造 3-5 格深的井。测试 agent 是否能执行"填井 → Tetris"策略——这是对 Tetris 理解能力的终极测试。

### 测试 C：垃圾行注入

在棋盘底部以上随机位置塞入"垃圾行"（每行含一个随机空格），模拟对战游戏中的被攻击场景。测试 agent 的生存能力和创造性恢复。

### 压力测试指标

| 指标 | 标准测试 | 压力测试 |
|------|---------|---------|
| Win Rate vs DL | 衡量最优性能 | 衡量**鲁棒性**和**恢复能力** |
| 预期结果 | Agent 略优 | **Agent 显著优于 DL**（差距随 N 增大而增大）|
| 如果结果相反 | 策略可能过拟合 | 确认了策略缺乏通用性 |

> **核心理念**：如果 RL agent 在压力测试中无法显著超越 Dellacherie，说明它学到的不是通用恢复策略，而仅仅是在干净棋盘上做最优放置。

### 实现

- `trainer/evaluator.py` — 新增 `stress_test()` 方法
- `scripts/eval.py` — 新增 `--stress` CLI 参数
- `env/tetris_env.py` — 可能需要添加 `load_board()` 或 `inject_garbage()` 方法
