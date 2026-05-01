# Tetris AI — 基于强化学习的俄罗斯方块智能引擎

> **算法**: Rainbow DQN (Double + Dueling + PER + Noisy Nets + N-step TD)  
> **备选**: PPO + GAE  
> **环境**: Python Gymnasium 接口  
> **推理**: ONNX Runtime (CPU/GPU) + 浏览器 ONNX Runtime Web

📖 **完整使用文档**: [docs/usage-guide.md](docs/usage-guide.md)（含算法说明、配置参考、常见问题）

---

## 快速开始

### 安装

```bash
# 克隆仓库
git clone <repo-url> && cd tetris-ai

# 安装 Python 依赖
pip install -e ".[train,inference]"

# (可选) 编译 C++ 环境加速核心
cd env/core && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 训练

```bash
# DQN 训练 (默认配置)
python scripts/train.py

# PPO 训练
python scripts/train.py --algo ppo

# 自定义步数 + WandB 日志
python scripts/train.py --algo dqn --steps 10000000 --wandb

# 从 checkpoint 恢复 (自动检测最新)
python scripts/train.py --checkpoint-dir checkpoints/
```

### 评估

```bash
# 评估模型
python scripts/eval.py --model checkpoints/step_005000000.pt --episodes 100

# ONNX 模型评估
python scripts/eval.py --model tetris_ai.onnx --backend onnx
```

### 观战

```bash
# AI 自动游玩
python scripts/play.py --model checkpoints/step_010000000.pt --delay 100
```

### 导出 ONNX

```bash
python scripts/export_model.py checkpoints/step_010000000.pt -o tetris_ai.onnx
```

---

## 项目结构

```
tetris-ai/
├── env/                    # 游戏环境
│   ├── core/               # C++ bitboard 核心 (header-only)
│   ├── bindings/           # pybind11 绑定
│   ├── tetris_env.py       # Gymnasium 环境
│   ├── state_encoder.py    # 状态特征编码 (CNN+手工特征)
│   └── reward_calculator.py # 奖励函数计算
├── agent/                  # RL 算法
│   ├── model.py            # DuelingDQN / ActorCritic 网络
│   ├── dqn.py              # Rainbow DQN 智能体
│   ├── ppo.py              # PPO 智能体
│   ├── memory.py           # 优先经验回放 (PER)
│   ├── nstep_buffer.py     # N-step return 缓冲区
│   ├── noisy_layers.py     # Noisy Networks
│   ├── action_mask.py      # 动作掩码
│   └── pretrain.py         # 模仿学习预训练 (Dellacherie)
├── trainer/                # 训练系统
│   ├── trainer.py          # 主训练循环
│   ├── config.py           # 配置 dataclass
│   ├── evaluator.py        # 评估循环
│   ├── checkpoint.py       # Checkpoint 管理
│   └── logger.py           # WandB/TensorBoard 日志
├── inference/              # 推理部署
│   ├── cpp/                # C++ ONNX Runtime 推理
│   └── python/             # Python 推理 + ONNX 导出
├── scripts/                # 入口脚本
├── configs/                # YAML 配置文件
├── tests/                  # 单元测试
├── CMakeLists.txt          # C++ 构建
├── pyproject.toml          # Python 依赖
├── Dockerfile              # 训练镜像
└── Dockerfile.inference    # 推理镜像
```

## 核心设计

### MDP 建模

| 元素 | 设计 |
|------|------|
| 状态 S | Board (22×10 bitboard) + 当前块 + Hold + Next队列 |
| 动作 A | placement-based: (rotation, column, hold), ~10-40 legal |
| 奖励 R | 消行奖励 + 塑性奖励 (高度/孔洞/崎岖/井深惩罚) |
| γ | 0.99 |

### 算法: Rainbow DQN

- **Double DQN**: 解耦动作选择与评估
- **Dueling Network**: Q = V + A 分解
- **Prioritized Experience Replay**: 按 TD-error 优先采样
- **Noisy Networks**: 状态相关探索
- **N-step TD (n=5)**: 加速信用传播
- **Action Masking**: 仅考虑合法放置

### 性能建议

| 组件 | 推荐配置 |
|------|----------|
| GPU | 1× RTX 4090 / A100 |
| CPU | 32+ cores |
| RAM | 64 GB+ |
| 并行环境 | 64 |
| 训练时长 | 2-4 天 (可达人类水平) |

## 运行测试

```bash
pytest tests/ -v
```

## 许可证

MIT
