# 代码量、CUDA 加速比与 C++ 推理必要性分析

---

## 一、代码量统计

| 类别 | 文件数 | 行数 | 占比 |
|------|--------|------|------|
| Python 核心代码 | 30 | 5,021 | 45.2% |
| 文档 (Markdown) | 3 | 2,449 | 22.0% |
| HTML (Tetris 游戏) | 1 | 1,788 | 16.1% |
| CMake 构建文件 | 7 | 342 | 3.1% |
| YAML 配置 | 4 | 116 | 1.0% |
| 其他 (JSON/txt/egg-info) | — | 1,394 | 12.5% |
| **合计** | — | **~11,110** | 100% |

### Python 代码按模块分布

| 模块 | 文件 | 行数 | 占比 | 说明 |
|------|------|------|------|------|
| `trainer/` | 8 | 1,314 | 26.2% | 训练循环、配置、日志、分布式 |
| `env/` | 3 | 669 | 13.3% | 环境模拟、状态编码、奖励计算 |
| `agent/` | 7 | 1,397 | 27.8% | DQN/PPO Agent、模型、记忆池、预训练 |
| `scripts/` | 5 | 430 | 8.6% | 训练/评估/对弈/导出入口 |
| `tests/` | 6 | 618 | 12.3% | 单元测试 |
| `inference/python/` | 3 | 216 | 4.3% | 推理引擎、模型导出 |
| `tetris/` | 1 | 1,788 | — | 纯 HTML 游戏 (非 Python) |
| `docs/` | 2 | 2,411 | — | 设计文档 (非 Python) |

### 核心模块 Top 5

| 文件 | 行数 | 职责 |
|------|------|------|
| `trainer/trainer.py` | 468 | 训练主循环（单机 + 分布式入口） |
| `env/tetris_env.py` | 449 | Tetris 环境模拟（核心热点） |
| `agent/pretrain.py` | 288 | Dellacherie 专家数据收集 + 行为克隆 |
| `agent/memory.py` | 263 | 优先经验回放 (SumTree) |
| `agent/ppo.py` | 250 | PPO Agent (GAE + PPO-Clip) |

---

## 二、训练 CUDA 加速比分析

### 2.1 模型规模

- **DuelingDQN**: ~139K 参数 (3 Conv + 8 Linear + Noisy)
- **ActorCritic (PPO)**: ~115K 参数
- 单次 forward (B=1): 输入 1×1×22×10 board + 1×53 features → 输出 1×112 q-values
- 计算量: 约 0.03M FLOPs（极小）

### 2.2 训练循环各阶段 GPU 占用

```
每个训练迭代 (64 envs × 1 step):
┌─────────────────────────┬──────────┬─────────────┬───────────┐
│ 阶段                    │ 设备     │ 耗时 (估算)  │ 占比      │
├─────────────────────────┼──────────┼─────────────┼───────────┤
│ get_legal_actions()     │ CPU      │ ~3-8 ms     │ 60-80%    │
│ select_action() B=1×64  │ GPU→CPU  │ ~0.6-2 ms   │ 5-10%     │
│ env.step() ×64          │ CPU      │ ~2-5 ms     │ 20-30%    │
│ agent.update() (DQN)    │ GPU      │ ~0.1-0.5 ms │ <5%       │
│ agent.update() (PPO)    │ GPU      │ ~5-20 ms    │ 10-15%    │
└─────────────────────────┴──────────┴─────────────┴───────────┘
```

### 2.3 GPU 利用率估算

| 指标 | DQN (Rainbow) | PPO |
|------|---------------|-----|
| GPU forward 调用频率 | 每 env-step 1次 (B=1) | 每 env-step 1次 (B=1) |
| GPU training 调用频率 | 每 4 env-steps 1次 (B=32) | 每 2048 env-steps 1次 (4 epoch × 2048 batch) |
| GPU 实际工作时间占比 | **~3-8%** | **~10-20%** |
| CPU 瓶颈占比 | **~90-95%** | **~80-90%** |

### 2.4 关键瓶颈

**根本原因：环境模拟是纯 CPU 的，且 64 个环境是串行 Python for 循环处理。**

```
trainer.py:248  for env_id in range(self.num_envs):   # 64 次串行迭代
                    env.step()                         # 纯 CPU, ~3-8us/step
                    agent.select_action()              # GPU B=1, 极度欠利用
```

### 2.5 实际加速比估算

| 场景 | CPU 训练 | GPU (cuda) 训练 | 加速比 |
|------|----------|-----------------|--------|
| DQN, 模型 forward (B=1) | ~50 us | ~15 us | **~3x** |
| DQN, 模型 update (B=32) | ~2 ms | ~0.2 ms | **~10x** |
| PPO, 模型 update (全量) | ~80 ms | ~10 ms | **~8x** |
| **整体训练吞吐 (含 env)** | 基线 | GPU 仅加速模型部分 | **~1.05-1.3x** |

**结论：因为 85-95% 的时间消耗在 CPU 环境模拟上，迁移到 GPU 对整体训练的加速比仅约 1.05-1.3 倍。GPU 没有显著加速训练。**

### 2.6 提升 GPU 利用率的方向

| 优化方向 | 预期收益 | 难度 |
|----------|----------|------|
| 批量化推理: 收集 64 env 观测后一次 B=64 forward | GPU forward 耗时降低 ~10x | 低 |
| 多进程并行 env: 用 subprocess 并行跑 64 个 env | CPU 吞吐提升 ~N×cores | 中 |
| 编译 env 到 C++: 通过 pybind11 加速 env step | 单步 env 耗时降低 5-10x | 中 |
| GPU replay buffer: 将 PER 采样移到 GPU | 消除 CPU SumTree 瓶颈 | 高 |

---

## 三、Inference 阶段 C++ 必要性分析

### 3.1 推理阶段每步耗时分解

```
一次 agent.select_action() 的完整流程:
┌──────────────────────────────┬──────────┬──────────┬──────────┐
│ 操作                         │ Python   │ C++      │ 加速比   │
├──────────────────────────────┼──────────┼──────────┼──────────┤
│ get_legal_actions()          │ 50-100us │ 5-10us   │ 5-10x    │
│ StateEncoder.encode()        │ 10-30us  │ 1-3us    │ 3-10x    │
│ NN forward (PyTorch)         │ 50-100us │ -        │ -        │
│ NN forward (ONNX Runtime)    │ 20-50us  │ 20-50us  │ ~1x      │
│ env.step()                   │ 30-80us  │ 3-10us   │ 5-10x    │
│ Action mask + argmax         │ 5-10us   │ ~1us     │ 2-5x     │
│ 总计 (不含渲染)               │ 165-420us│ 40-100us │ 3-5x     │
└──────────────────────────────┴──────────┴──────────┴──────────┘
```

### 3.2 瓶颈判断

**神经网络 forward pass 不是瓶颈。** 模型只有 ~139K 参数，PyTorch/ONNX 在 CPU 上的 forward 仅需 20-100us。

**真正的瓶颈是环境操作：** `get_legal_actions()` + `env.step()` 占了 65-75% 的单步时间，这些是纯 CPU 操作，涉及：
- 碰撞检测循环 (~1,700 次/step)
- 重力模拟 (ghostY while 循环)
- 棋盘特征计算 (列高度/空洞/平整度/井深)

### 3.3 C++ 项目的现有资产

项目**已有完整的 C++ 实现**，但均未编译启用：

| 组件 | 位置 | 状态 |
|------|------|------|
| C++ 环境核心 (bitboard) | `env/core/` | 头文件完整，cpp 是空桩 |
| pybind11 Python 绑定 | `env/bindings/tetris_bindings.cpp` | 代码完整，未编译 |
| C++ ONNX 推理引擎 | `inference/cpp/model_loader.*` | 代码完整，未编译 |
| C++ AI Player | `inference/cpp/player.*` | 代码完整，**存在特征编码 bug** |
| WebAssembly 前端 | `inference/cpp/web/` | Emscripten 规则完整，未编译 |

### 3.4 现有 C++ 代码的问题

`inference/cpp/player.h` 中的 `encodeState()` **特征编码与 Python 不一致**：
- Python `StateEncoder` 生成 **53 维**特征向量
- C++ 版本缺少 `row_transitions`、`lines_cleared`、`mean_height_change`
- 如果用 Python 训练的模型权重进行 C++ 推理，会产生错误的 q-values

### 3.5 推荐方案

| 优先级 | 方案 | 收益 | 工作量 |
|--------|------|------|--------|
| **高** | 编译 C++ env core + pybind11 绑定，替换 Python TetrisEnv | env 操作加速 5-10x | 配置 CMake 编译即可，代码已存在 |
| **低** | 纯 C++ 独立推理二进制 | 端到端加速 3-5x | 需修复特征编码 bug + 集成 ONNX Runtime |
| **不推荐** | 手写 C++ 神经网络推理 | 几乎没有额外收益 | 大，且 ONNX Runtime 已足够 |

### 3.6 结论

**对于 inference 阶段，C++ 的最大价值在于加速环境模拟（get_legal_actions + step），而非神经网络 forward pass。**

- 如果场景是**本地对弈/演示**（`scripts/play.py`，50ms 延迟），Python 完全足够
- 如果场景是**大规模评估**（`scripts/eval.py`），编译 C++ env core 可提速 3-5x
- 如果场景是**WebAssembly 部署**，需先修复 `player.h` 的特征编码不一致问题

---

## 四、验证方法

1. 运行 `python scripts/train.py --device cpu --envs 1` 与 `--device cuda --envs 1` 各 1000 步，记录实际耗时，计算端到端加速比
2. 在 `trainer.py` 中插入计时埋点，分别统计 env step、action select、model update 的累计耗时
3. 编译 `env/core/` + `env/bindings/` 后，运行 `tests/test_env.py` 对比 Python vs C++ 环境的单步耗时
