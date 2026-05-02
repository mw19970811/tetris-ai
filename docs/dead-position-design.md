# 死局判断与重置设计文档

## 问题

训练过程中，智能体可能进入**死局**（无合法动作 / 方块堆积到隐藏行），当前代码的

处理链路分散在 Phase 1 和 Phase 3 两处，逻辑不够集中，且缺少显式的死局统计。

## 当前处理流程（现状）

```
Phase 1 (收集状态):
    legal = env.get_legal_actions()
    if not legal:                          # 死局检测点 A
        env.reset()                        # 重置 → 新 state
        episode_reward = 0
        episode_steps = 0
        legal = env.get_legal_actions()    # 重新获取合法动作
    # → 此 env 的 state/action 已初始化，进入 Phase 2 批量推理

Phase 3 (执行 & 存储):
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if done:                               # 死局检测点 B
        env.reset()                        # 重置 → 为下一步准备新 state
        episode_reward = 0
        episode_steps = 0
```

**问题**：

1. 两处重置逻辑重复，且 A 和 B 对"死局"的定义不同：A 是"无合法动作"，B 是"step 返回 terminated"
2. 死局事件未统计 — 不知道训练过程中发生了多少次死局
3. 死局前的 state→action 转移仍然被存入 replay buffer（带 `done=True`），但死局后的重置
   state 作为 `next_state` 存入了上一个转移。这不一致。

## 设计目标

1. **集中死局判断**：统一在调用 `env.step()` 之后判断，而非分散在 Phase 1 和 Phase 3
2. **显式死局统计**：记录死局次数、死局步数分布
3. **确保 state/action 初始化正确**：死局 → reset → 新的 state 和 legal_actions 进入下一轮
4. **不污染 replay buffer**：死局转移（done=True）存入 buffer 是合法且有价值的（告诉 agent 什么动作
   会导致死亡），但重置后的"空"转移不应被存储

## 设计方案

### 死局判断集中化

将死局检测从 Phase 1 移到 Phase 3 之后、下一轮 Phase 1 之前的统一入口：

```python
# Phase 3 结束后，统一处理所有 env 的死局状态
for env_id in range(self.num_envs):
    if self.obs[env_id] is None or self.envs[env_id].terminated:
        self.obs[env_id] = self.envs[env_id].reset()
        episode_rewards[env_id] = 0.0
        episode_steps[env_id] = 0
        self._dead_count += 1
```

### 死局统计

新增计数器，每个 log 间隔输出：

```
_dead_episodes_this_interval: int   # 本 log 间隔内的死局次数
_dead_total: int                    # 累计死局次数
```

填入 TensorBoard：`train/dead_episodes`、`train/dead_rate`（死局数 / 总 episode 数）

### Phase 1 简化

移除 Phase 1 中的死局检测和重置逻辑，Phase 1 仅负责收集 state 和 legal_actions。死局
env 在进入 Phase 1 之前已经处于已重置状态。

Phase 1 仅保留一个安全检查（assert / skip），不应再有 reset 调用。

### Buffer 处理

死局转移 `(state_before_death, action, reward_death, next_state_after_death, done=True)`
正常存入 replay buffer。这是**有价值的负样本**，告诉 agent 什么动作导致死亡。

死亡后的重置转移 **不** 单独存储。n-step buffer 中未完成的轨迹在 `done=True` 时被
`flush()` 排空，不会与新 episode 的转移混合。

## 编码影响

| 文件 | 修改 |
|------|------|
| `trainer/trainer.py` | Phase 1 移除死局重置 → 移到 Phase 3 后的统一入口；新增死局计数器 |
| `trainer/logger.py` | `log_train_step` 增加 `dead_count`、`dead_rate` 参数 |
| `docs/usage-guide.md` | §10.3 观察指标更新：加入死局率 |
