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
