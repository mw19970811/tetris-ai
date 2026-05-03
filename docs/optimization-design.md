# 训练优化设计方案

> **版本**: v1.0
> **日期**: 2026-05-03
> **范围**: Checkpoint 保留策略 / NoisyLayer 噪声衰减 / TensorBoard 日志补全 / 文档去重

---

## 一、Checkpoint 保留策略优化

### 1.1 现状

`trainer/checkpoint.py` 中 `CheckpointManager` 当前策略：

- `keep = 10`：保留最近 10 次周期性 checkpoint
- `mark_best(step)`：标记 1 个最优 checkpoint 免于清理
- `_cleanup()`：保留 `{best} ∪ {最近 keep 个}`，其余删除

`trainer/trainer.py` 中调用逻辑：

- 评估时若 `avg_score > best_avg_score`，调用 `save_full()` + `mark_best()`（第 431-437 行）
- 每 `save_every` 步调用 `save_full()` 做周期性保存（第 440-444 行）

**问题**：

1. `keep=10` 保留过多 checkpoint，占用磁盘空间大（含 replay buffer 时每个 checkpoint 可达数 GB）
2. 仅保护 1 个最优 checkpoint，无法回溯到"次优"的历史版本
3. 周期性 checkpoint 按时间保留而非按质量保留，低质量 checkpoint 挤占高质量历史版本

### 1.2 目标

- 保留**最优 5 次** checkpoint（按评估分数排序）
- **始终保留最近一次** checkpoint（确保可恢复最新训练状态）
- 最优 5 次与最近一次可能有交集（去重后最多保留 6 个）

### 1.3 方案

#### 数据结构变更

`CheckpointManager` 新增内部追踪：

```python
# 记录每个 checkpoint 的评估分数，用于排序淘汰
_checkpoint_scores: Dict[str, float]  # path → avg_score
```

#### 接口变更

新增方法 `record_score(path, score)`，在 checkpoint 保存后由 Trainer 调用：

```python
def record_score(self, path: str, score: float):
    """记录 checkpoint 对应的评估分数，用于最优 N 保留策略"""
    self._checkpoint_scores[path] = score
```

#### 清理策略变更

```
protected = 评分最高的 5 个 checkpoint ∪ {最近 1 个 checkpoint}
```

伪代码：

```python
def _cleanup(self):
    files = all checkpoints sorted by step
    if len(files) <= 6:
        return  # 不足 6 个时不清理

    # 按 score 降序取 top 5
    scored = [(f, self._checkpoint_scores.get(f, 0.0)) for f in files]
    scored.sort(key=lambda x: x[1], reverse=True)
    top5 = {f for f, _ in scored[:5]}

    # 最近一个
    latest = files[-1]

    protected = top5 | {latest}

    for f in files:
        if f not in protected:
            os.remove(f)
```

#### Trainer 调用变更

在周期性 checkpoint 保存后，将当次评估分数传入 checkpoint manager：

```python
# 周期性 checkpoint 保存
path = self.checkpoint.save_full(step, self.agent, ...)
self.checkpoint.record_score(path, cur_eval["avg_score"])
```

最优 checkpoint 保存同理。

#### 参数变更

`config.py` 新增配置项：

```python
checkpoint_keep_best: int = 5   # 保留最优 N 次
checkpoint_keep_latest: int = 1  # 保留最近 N 次（始终 ≥ 1）
```

---

## 二、NoisyLayer 噪声衰减时间延长

### 2.1 现状

`agent/noisy_layers.py`：

- `NoisyLinear` 提供 `scale_sigma(factor)` 方法，对所有权重 σ 乘以 factor
- `NoisyLinear.get_sigma_mean()` 返回当前平均 σ 值

`trainer/config.py` / `NetworkConfig`：

```python
sigma_decay: float = 1.0  # 1.0 = no decay; 0.999999 = gentle
```

`agent/dqn.py` 第 217-221 行：

```python
if self.sigma_decay < 1.0:
    for module in self.online_net.modules():
        if hasattr(module, 'scale_sigma'):
            module.scale_sigma(self.sigma_decay)
```

**当前状态**：`sigma_decay = 1.0`，即完全不衰减。噪声在训练全程保持初始强度（`sigma_init = 0.01`）。

### 2.2 问题分析

不衰减的噪声在训练后期存在问题：

1. **收敛阶段噪声过大**：策略已学到有效行为后，噪声扰动导致动作选择不稳定，评估分数波动大
2. **无法渐进利用**：NoisyNet 的设计初衷是"学习型探索"——训练初期自动探索，后期自动收敛。但 σ 不衰减时网络可以通过增大 μ 来相对压制噪声，效率不如显式衰减
3. **与 PER 的 β 退火不协调**：PER 的 β 从 0.4 线性退火到 1.0（3M 训练步），而噪声完全不退火，两者节奏不匹配

### 2.3 衰减时间设计

#### 总训练步数

以默认配置计算：
- `total_samples = 1,000,000,000`
- `total_steps = samples × train_every / batch_size = 1B × 4 / 256 ≈ 15,625,000`

#### 衰减曲线选择

采用**指数衰减**（已内置 `scale_sigma`），目标：
- 训练结束时 σ 降至初始的 **~5%**（从 0.01 降至 0.0005）
- 即：`factor^15.6M = 0.05 → factor = 0.05^(1/15.6M)`

计算：

| σ 剩余比例 | 对应 factor | 训练中期 (7.8M 步) σ | 训练末期 (15.6M 步) σ |
|-----------|-------------|---------------------|---------------------|
| 50% | 0.999999955 | 0.0050 | 0.0050 |
| 10% | 0.999999852 | 0.0032 | 0.0010 |
| 5% | 0.999999808 | 0.0022 | 0.0005 |

推荐 **factor = 0.99999994**（优先保护早期探索）：

| 步数 | σ 比例 | σ 绝对值 | 阶段 |
|------|--------|---------|------|
| 0 | 100% | 0.01 | 初始探索 |
| 1M | 94.2% | 0.00942 | 大量探索 |
| 3M | 83.5% | 0.00835 | PER β→1.0 同步 |
| 6M | 69.8% | 0.00698 | 探索为主 |
| 8M | 61.9% | 0.00619 | 探索-利用过渡 |
| 12M | 48.7% | 0.00487 | 偏向利用 |
| 15.6M | 39.2% | 0.00392 | 收敛阶段 |

> **设计权衡**：指数衰减无法同时满足"前期 60-80%"和"末期 10-20%"的理想目标。选择优先保护早期探索（6M 处 ~70%），牺牲末期收敛深度（39% vs 理想 15%）。若训练后期需要更低的噪声，届时可手动调整 `sigma_decay`。

#### 为什么优先保护早期探索

1. 训练初期 agent 对状态空间几乎一无所知，过早压制噪声会锁死低质量策略
2. 俄罗斯方块状态空间巨大，7-bag 随机性导致某些方块序列极少出现
3. NoisyNet 的因子化噪声本身会随训练自适应（μ 增长相对压制 σ），显式衰减只是辅助
4. 若 `Avg100R` 长期停滞在低位，说明 agent 仍处于探索平台期，此时降低噪声只会固化次优策略

### 2.4 对训练的影响评估

| 维度 | 预期影响 | 风险 |
|------|---------|------|
| **训练初期 (0-6M 步)** | 噪声保持高位（>70%），充分探索 | 低 |
| **训练中期 (6M-12M 步)** | 策略逐渐偏向利用，噪声降至 ~50% | 低 |
| **训练后期 (12M+ 步)** | 噪声降至 ~39%，评估方差减小但仍保留一定探索 | 可能收敛偏慢 |
| **Final 评估** | 噪声约 39%，评估具有一定的随机性 | 可接受 |

**关键监控指标**：

- `sigma_mean`：在训练日志中观察 σ 均值变化，确认衰减按预期进行
- `eval/avg_score` 方差：衰减后期望评估分数更稳定
- `train/q_loss`：不应因噪声衰减出现 loss 突增（说明网络在"记忆"而非"学习"）

### 2.5 实施

修改 `trainer/config.py`：

```python
sigma_decay: float = 0.99999994  # σ → ~70% at 6M, ~39% at 15.6M steps
```

同时在训练日志中增加 `sigma_mean` 的记录频率（当前已在 `dqn.py` 中记录，通过 `log_train` 输出）。

---

## 三、TensorBoard 日志补全

### 3.1 现状

`trainer/logger.py` 中 `log_train_step()` 当前记录的指标：

```python
train/avg_reward, train/avg_lines, train/fps,
train/buffer_size, train/elapsed_h,
train/stale_dead_count, train/stale_dead_rate,
train/avg_pieces, train/avg_score
```

`log_eval()` 记录的指标：

```python
eval/avg_score, eval/max_score, eval/min_score,
eval/std_score, eval/avg_lines, eval/avg_level,
eval/avg_steps, eval/tetris_rate
```

### 3.2 缺失项

| 指标 | log_train_step | log_eval | 问题 |
|------|---------------|----------|------|
| `avg_steps` | **缺失** | 有 | 训练日志中无法追踪每局平均步数 |
| `lines` | — | — | 实际已有 `avg_lines`（消行数均值） |
| `avg_lines` | 有 | 有 | OK |
| `avg_score` | 有 | 有 | OK |

**核心缺失**：训练进度日志中缺少 `avg_steps`（每局平均存活步数），该指标直接反映 agent 的生存能力。

### 3.3 根因

`trainer/trainer.py` 中：

1. 未维护 `self.episode_steps` deque（有 `episode_rewards/lines/pieces/scores`，唯独没有 steps）
2. 日志输出段（第 374-392 行）未计算 `avg_steps`

### 3.4 修复方案

#### trainer.py 变更

1. 新增 `self.episode_steps = deque(maxlen=100)`（第 86 行附近）
2. 在 episode 结束时记录步数（第 354 行附近）：

```python
self.episode_steps.append(info.get("steps", episode_steps[env_id]))
```

3. 在日志输出段计算 `avg_steps`（第 380 行附近）：

```python
avg_steps = np.mean(self.episode_steps) if self.episode_steps else 0
```

4. 传入 `log_train_step()`：

```python
self.logger.log_train_step(
    step, avg_reward=avg_reward, avg_lines=avg_lines,
    fps=fps, buffer_size=buf_size, elapsed=elapsed,
    dead_count=dead_count, dead_rate=dead_rate,
    avg_pieces=avg_pieces, avg_score=avg_score,
    avg_steps=avg_steps,  # 新增
)
```

#### logger.py 变更

`log_train_step()` 签名增加 `avg_steps` 参数，并在日志中记录 `train/avg_steps`。

#### 终端输出同步更新

日志行增加 `Steps` 显示，与评估行格式保持一致。

### 3.5 验证方法

启动训练后执行：

```bash
tensorboard --logdir=logs/tensorboard
```

确认以下标量曲线均出现且有数据点：
- `train/avg_steps`
- `train/avg_lines`
- `train/avg_score`
- `eval/avg_steps`
- `eval/avg_lines`
- `eval/avg_score`

---

## 四、文档环境配置去重

### 4.1 现状

| 文档 | 环境配置内容 |
|------|------------|
| `README.md` 第 14-42 行 | 安装步骤、C++ 编译命令、pybind11 配置 |
| `docs/usage-guide.md` 第 122-268 行 | 环境要求、Python 依赖、Windows/Ubuntu 平台的完整 C++ 编译流程 |

两份文档均包含完整的环境配置方式，内容重复且维护成本高。

### 4.2 方案

**原则**：环境配置方式只在一处详细描述，其余文档引用。

| 文档 | 职责 | 环境配置 |
|------|------|---------|
| `README.md` | 项目概览 + 快速开始 | **删除**详细安装步骤，改为 1 句话 + 链接指向 usage-guide |
| `docs/usage-guide.md` | 完整使用手册 | **保留**完整的环境配置（三、安装），作为唯一权威来源 |
| `docs/tetris-rl-engine-design.md` | 设计文档 | 不包含环境配置（当前已无不必要的配置说明，仅技术栈章节有工具链罗列，属于设计范畴，保留） |

### 4.3 README.md 修改

将当前的"安装"章节从详细步骤改为：

```markdown
## 快速开始

### 安装

```bash
pip install torch numpy
```

详细安装步骤（C++ 加速编译、平台配置等）见 [使用文档 - 安装](docs/usage-guide.md#三安装)。

### 训练

```bash
python scripts/train.py
```
```

**删除**的内容：
- C++ 编译的详细命令
- pybind11 找不到的 FAQ
- 编译产物复制说明

这些内容统一在 usage-guide.md 中维护。

---

## 五、实施顺序

| 步骤 | 内容 | 涉及文件 | 风险 |
|------|------|---------|------|
| 1 | Checkpoint 保留策略 | `checkpoint.py`, `config.py`, `trainer.py` | 中 — 涉及文件删除逻辑 |
| 2 | NoisyLayer 衰减参数 | `config.py` | 低 — 仅改一个默认值 |
| 3 | TensorBoard 日志补全 | `trainer.py`, `logger.py` | 低 — 增加字段 |
| 4 | 文档去重 | `README.md` | 低 — 仅删减内容 |

---

> **审批后进入编码阶段。**
