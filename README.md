# Tetris AI — 基于强化学习的俄罗斯方块智能引擎

> **算法**: Rainbow DQN (Double + Dueling + PER + Noisy Nets + N-step TD)  
> **备选**: PPO + GAE  
> **环境**: Python Gymnasium 接口 + C++ pybind11 加速  
> **推理**: ONNX Runtime (CPU/GPU) + 浏览器 ONNX Runtime Web

📖 **完整使用文档**: [docs/usage-guide.md](docs/usage-guide.md)

---

## 快速开始

### 安装

```bash
# 克隆仓库
git clone <repo-url> && cd tetris-ai

# 安装 Python 依赖
pip install torch numpy

# (可选但推荐) 编译 C++ 环境加速核心 (3-5x 吞吐提升)
pip install pybind11 cmake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target tetris_core --config Release
```

### 训练

```bash
# DQN 训练 (默认配置, 纯 Python 环境)
python scripts/train.py

# 启用 C++ 加速训练 (需先编译 tetris_core)
# 在 YAML 配置中设 env.use_cpp_env: true, 或传入 CLI
python scripts/train.py --device cuda --envs 64

# PPO 训练
python scripts/train.py --algo ppo

# 从 checkpoint 恢复
python scripts/train.py --resume
```

### 评估

```bash
python scripts/eval.py --model checkpoints/step_05000000.pt --episodes 100
```

### 观战

```bash
python scripts/play.py --model checkpoints/step_10000000.pt --delay 100
```

---

## 项目结构

```
agent-ai/
├── env/                         # 游戏环境
│   ├── core/                    # C++ bitboard 核心 (header-only, 已编译启用)
│   │   ├── tetris_core.h        #   Board 位棋盘 + 碰撞/消行
│   │   ├── tetris_env.h         #   TetrisEnv C++ 实现
│   │   ├── action_gen.h         #   合法动作生成器
│   │   └── state_encoder.h      #   C++ 53 维特征编码器
│   ├── bindings/                # pybind11 Python 绑定
│   │   ├── tetris_bindings.cpp  #   C++ → Python 接口
│   │   └── cpp_env.py           #   CppTetrisEnv Python 包装器
│   ├── tetris_env.py            # 纯 Python TetrisEnv
│   ├── state_encoder.py         # 状态特征编码
│   └── reward_calculator.py     # 奖励函数
├── agent/                       # RL 算法
│   ├── model.py                 # DuelingDQN / ActorCritic
│   ├── dqn.py                   # Rainbow DQN
│   ├── ppo.py                   # PPO
│   ├── memory.py                # PER + Uniform Buffer
│   ├── nstep_buffer.py          # N-step return
│   ├── action_mask.py           # 动作掩码
│   └── pretrain.py              # 模仿学习预训练
├── trainer/                     # 训练系统
│   ├── trainer.py               # 主训练循环
│   ├── config.py                # 超参数配置
│   ├── evaluator.py             # 评估
│   ├── checkpoint.py            # Checkpoint
│   └── logger.py                # 日志
├── inference/                   # 推理部署
│   ├── cpp/                     # C++ ONNX Runtime 推理
│   │   ├── model_loader.cpp     #   ONNX 模型加载 + 推理
│   │   └── player.cpp           #   AI Player (状态编码 + 决策)
│   └── python/                  # Python 推理 + ONNX 导出
├── scripts/                     # 入口脚本
├── configs/                     # YAML 配置
├── tests/                       # 单元测试
├── docs/                        # 文档
├── CMakeLists.txt               # C++ 构建
└── tetris/code.html             # 浏览器游戏
```

---

## C++ 加速

项目提供两层 C++ 加速，均为可选：

| 组件 | 加速对象 | 预期提升 | 启用方式 |
|------|---------|---------|---------|
| `env/core` + pybind11 | 环境模拟 (get_legal_actions, step, encode) | **3-4x** 训练吞吐 | `env.use_cpp_env: true` |
| `inference/cpp` | ONNX 推理 + 裸奔 AI | **1.5-2x** 推理延迟 | `cmake --build . --target tetris_inference` |

详情见 [docs/usage-guide.md#八-C++ 加速方案](docs/usage-guide.md)。

---

## 运行测试

```bash
pytest tests/ -v
```

## 许可证

MIT
