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
│   ├── core/                    # C++ bitboard 实现 (header-only, 已编译启用)
│   │   ├── tetris_core.h        #   Board 位棋盘 (碰撞/消行/特征)
│   │   ├── tetris_env.h         #   TetrisEnv C++ 实现
│   │   ├── action_gen.h         #   合法动作生成器
│   │   └── state_encoder.h      #   C++ 53 维特征编码器
│   ├── bindings/                # pybind11 Python 绑定
│   │   ├── tetris_bindings.cpp  #   C++ → Python 接口
│   │   └── cpp_env.py           #   CppTetrisEnv Python 包装器
│   ├── tetris_env.py            # Gymnasium 标准 RL 环境 (纯 Python + numpy)
│   ├── state_encoder.py         # 状态编码 (bitmap + 手工特征)
│   └── reward_calculator.py     # 奖励函数计算
│
├── agent/                       # ──────── AI 算法 ────────
│   ├── model.py                 # 神经网络 (DuelingDQN / ActorCritic)
│   ├── dqn.py                   # Rainbow DQN 智能体
│   ├── ppo.py                   # PPO 智能体
│   ├── noisy_layers.py          # NoisyLinear 噪声层
│   ├── memory.py                # 优先经验回放 (SumTree + StoredTransition)
│   ├── nstep_buffer.py          # N-step return 缓冲
│   ├── action_mask.py           # 动作掩码 (过滤非法动作)
│   └── pretrain.py              # 模仿学习预训练
│
├── trainer/                     # ──────── 训练系统 ────────
│   ├── trainer.py               # 主训练循环
│   ├── config.py                # 超参数配置 dataclass
│   ├── hardware_probe.py        # 硬件探测 (CPU/GPU/RAM)
│   ├── resource_planner.py      # 资源规划与配置推荐
│   ├── evaluator.py             # 评估循环
│   ├── checkpoint.py            # Checkpoint 管理
│   └── logger.py                # WandB 日志
│
├── inference/                   # ──────── 推理部署 ────────
│   ├── cpp/                     # C++ ONNX Runtime 推理
│   │   ├── model_loader.h/cpp   #   ONNX 模型加载
│   │   ├── player.h/cpp         #   AI Player
│   │   └── web/                 #   Emscripten WASM 构建
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
│   ├── usage-guide.md           # 本文档
│   └── analysis-cuda-cpp.md     # CUDA 加速比分析
│
├── CMakeLists.txt               # C++ 构建入口
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
pip install torch numpy

# 3. 安装推理依赖 (评估/部署)
pip install onnx onnxruntime

# 4. (可选) 训练日志可视化
pip install wandb

# 5. (可选) 安装测试依赖
pip install pytest
```

### Windows 平台配置

#### 编译 C++ 环境加速（Visual Studio）

```powershell
# 1. 安装依赖
pip install pybind11 cmake

# 2. 准备构建目录
mkdir build
cd build

# 3. CMake 配置 (自动检测 Visual Studio)
cmake .. -DCMAKE_BUILD_TYPE=Release

# 如果 cmake 报 "pybind11 not found"，手动指定 pybind11 路径：
# cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$(python -c 'import pybind11;print(pybind11.get_cmake_dir())')"

# 4. 编译 tetris_core 模块
cmake --build . --target tetris_core --config Release

# 5. 将编译产物复制到 Python 可发现路径
# 【关键】编译产物在 build/env/bindings/Release/ 下，需复制到项目根目录的 env/bindings/
copy env\bindings\Release\tetris_core.*.pyd ..\env\bindings\
```

编译产物为 `tetris_core.cp314-win_amd64.pyd`（文件名随 Python 版本变化），位于 `build/env/bindings/Release/`。编译完成后回到项目根目录：

```powershell
cd ..
```

验证加载：

```powershell
python -c "from env.bindings.cpp_env import CppTetrisEnv; print('C++ 模块加载成功')"
```

成功时输出：

```
[CppTetrisEnv] tetris_core C++ module loaded successfully (platform: Windows, module: tetris_core)
```

> **注意**：需要安装 Visual Studio 2022（或 Build Tools）并勾选"使用 C++ 的桌面开发"工作负载。MSVC 编译器 (cl.exe) 必须在 PATH 中（可通过 Developer Command Prompt 或 `vcvarsall.bat` 配置）。

> ⚠️ **关键**：编译产物必须放在 `env/bindings/` 目录下（与 `cpp_env.py` 同级），**不要**创建 `tetris_core/` 子目录或将其放在项目根目录的 `tetris_core/` 文件夹内。如果项目根目录已存在 `tetris_core/` 目录，请先删除它，否则 Python 会将其识别为 namespace package 而导致 `.pyd` 无法加载。

#### 编译 C++ 推理引擎（可选）

```powershell
# 需要先安装 ONNX Runtime
pip install onnxruntime

cd build
cmake .. -Donnxruntime_DIR="$(python -c 'import onnxruntime,os;print(os.path.dirname(onnxruntime.__file__))')"
cmake --build . --target tetris_inference --config Release
```

### Ubuntu 平台配置

#### 编译 C++ 环境加速（GCC）

```bash
# 1. 安装系统依赖
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev

# 2. 安装 Python 依赖
pip install pybind11 cmake

# 3. 准备构建目录
mkdir -p build && cd build

# 4. CMake 配置 (GCC + Release)
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++

# 如果 cmake 报 "pybind11 not found"，手动指定 pybind11 路径：
# cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$(python3 -c 'import pybind11;print(pybind11.get_cmake_dir())')"

# 5. 编译 tetris_core 模块
cmake --build . --target tetris_core -j$(nproc)

# 6. 将编译产物复制到 env/bindings/ (与 cpp_env.py 同级)
# 【关键】编译产物在 build/env/bindings/ 下，需复制到项目根目录的 env/bindings/
cp env/bindings/tetris_core.*.so ../env/bindings/
```

编译产物为 `tetris_core.cpython-3xx-x86_64-linux-gnu.so`（文件名随 Python 版本变化）。编译完成后回到项目根目录：

```bash
cd ..
```

验证加载：

```bash
python -c "from env.bindings.cpp_env import CppTetrisEnv; print('C++ 模块加载成功')"
```

成功时输出：

```
[CppTetrisEnv] tetris_core C++ module loaded successfully (platform: Linux, module: tetris_core)
```

> **注意**：需要 GCC ≥ 9.0（推荐 GCC 11+）。确保 `python3-dev` 版本与当前 Python 版本一致（如 `python3.11-dev` / `python3.12-dev`）。

> ⚠️ **关键**：编译产物必须放在 `env/bindings/` 目录下（与 `cpp_env.py` 同级），**不要**创建 `tetris_core/` 子目录或将其放在项目根目录的 `tetris_core/` 文件夹内。如果项目根目录已存在 `tetris_core/` 目录，请先删除它，否则 Python 会将其识别为 namespace package 而导致 `.so` 无法加载。

#### 编译 C++ 推理引擎（可选）

```bash
# 需要先安装 ONNX Runtime
pip install onnxruntime

cd build
cmake .. -Donnxruntime_DIR="$(python3 -c 'import onnxruntime,os;print(os.path.dirname(onnxruntime.__file__))')"
cmake --build . --target tetris_inference -j$(nproc)
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

训练过程中，checkpoint 会自动保存到 `checkpoints/` 目录，每 `save_every` 步（默认 10,000）保存一次。每次保存包含模型权重、optimizer 状态、训练步数计数器，以及 replay buffer（存为独立 `_buffer.pt` 文件）。

**Checkpoint 保留策略**：保留评估分数最高的 5 次 checkpoint + 最近 1 次 checkpoint（两者可能有交集，最多保留 6 个）。旧 checkpoint 自动清理，避免磁盘占用过大。

**训练日志格式**：

```
[2026-05-03 08:42:30]  Step    50,000/15,625,000 (0.3%)  |  Avg100R:     123.4  |  Pieces:    45/ep  |  Steps:    320/ep  |  Stale:    12  |  FPS:   32,000  |  Elapsed: 1h02m03s  |  ETA: 1h30m00s
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
- 噪声参数 \(\sigma\) 初始化为 0.01，每训练步乘以衰减系数 0.99999994

**σ 衰减曲线**（默认 15.6M 训练步）：

| 步数 | σ 比例 | 阶段 |
|------|--------|------|
| 0 | 100% | 初始探索 |
| 3M | 84% | 强力探索，PER β→1.0 同步 |
| 6M | 70% | 探索为主 |
| 12M | 49% | 偏向利用 |
| 15.6M | 39% | 收敛阶段 |

衰减仅在训练时生效（`forward()` 的 `self.training` 分支），评估时关闭噪声仅用 μ 权重。

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

### 6.0 配置方式

项目支持两种配置方式：直接修改 `trainer/config.py` 中的 dataclass 默认值，或通过 CLI 参数传入。

**方式 A — 修改 `trainer/config.py`**（推荐，持久生效）：

```python
# trainer/config.py

@dataclass
class TrainingConfig:
    total_samples: int = 500_000_000  # 改为 500M 样本

@dataclass
class EnvConfig:
    reward_weights: dict = field(default_factory=lambda: {
        "w_height": 0.0,     # 高度惩罚
        "w_holes": 0.0,      # 孔洞惩罚
        "w_bumpiness": 0.0,  # 崎岖度惩罚
        "w_well": 0.0,       # 井深惩罚
        "w_survival": 0.01,  # 存活奖励
        "w_death": -100.0,   # 死亡惩罚
    })
```

**方式 B — CLI 参数**（一次性覆盖）：

```bash
# 指定训练样本数（自动推导 total_steps）
python scripts/train.py --samples 500000000

# 直接指定环境步数（绕过推导公式）
python scripts/train.py --steps 10000000

# 其他常用参数
python scripts/train.py --device cuda --envs 64 --algo dqn
```

> **`total_samples` vs `total_steps`**：
> - `total_samples` 是**训练样本数**（喂给 optimizer 的 transition 数量），与 batch_size 无关
> - `total_steps` 是**环境交互步数**，由 `total_samples` 自动推导：
>   - DQN: `total_steps = total_samples × train_every / batch_size`
>   - PPO: `total_steps = total_samples / num_envs`
> - 默认 1B samples + batch_size=256 → 约 15.6M env steps
> - CLI `--steps` 优先级最高，会绕过推导直接覆盖

### 6.1 训练配置 (TrainingConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `algorithm` | `"dqn"` | 算法选择：`"dqn"` 或 `"ppo"` |
| `total_samples` | 1,000,000,000 | 总训练样本数（batch-invariant），自动推导 env steps |
| `eval_every` | 10,000 | 评估间隔（步） |
| `eval_episodes` | 100 | 每次评估的局数 |
| `save_every` | 10,000 | Checkpoint 保存间隔 |
| `num_envs` | 64 | 并行环境数 |
| `num_pretrain_envs` | 16 | 预训练数据收集的并行环境数 |
| `pretrain_sample_tag` | `"latest"` | 预训练样本文件标签 |
| `checkpoint_keep_best` | 5 | 按评估分数保留的最优 checkpoint 数量 |
| `checkpoint_keep_latest` | 1 | 始终保留的最近 checkpoint 数量 |
| `device` | `"cuda"` | 训练设备 |
| `seed` | 42 | 随机种子 |

> **`total_samples` 说明**：不再配置 `total_steps`，改为配置总训练样本数。
> 自动推导公式（DQN）：`total_steps = total_samples × train_every / batch_size`
> 默认 1B 样本 + batch_size=256 → 约 15.6M env steps。
> 仍可通过 CLI `--steps` 直接覆盖。

### 6.2 Rainbow DQN 配置 (DQNConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gamma` | 0.99 | 折扣因子 |
| `n_step` | 5 | N-step return 步数 |
| `lr` | 2.5e-5 | 学习率（保守，适配 batch_size=256） |
| `batch_size` | 256 | 训练批大小 |
| `train_every` | 4 | 每 N 环境步训练一次 |
| `target_update_freq` | 8000 | 目标网络硬更新频率（仅 `use_hard_update=true` 时生效） |
| `target_update_tau` | 0.005 | 软更新系数 τ |
| `use_hard_update` | **false** | Polyak 软更新（推荐；避免灾难性遗忘） |
| `replay_capacity` | 1,000,000 | 回放缓冲区容量 |
| `per_alpha` | 0.6 | PER 优先级指数 |
| `per_beta_start` | 0.4 | IS 修正初始值 |
| `per_beta_end` | 1.0 | IS 修正最终值 |
| `per_beta_frames` | 10,000,000 | β 衰减帧数 |
| `grad_clip_norm` | 10.0 | 梯度裁剪阈值 |

### 6.3 Network 配置 (NetworkConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cnn_channels` | 32 | CNN 通道数 |
| `hidden_dim` | 128 | MLP 隐藏维度 |
| `feature_dim` | 53 | 手工特征维度 |
| `num_actions` | 112 | 最大动作数 |
| `use_noisy` | true | 使用 NoisyNet（学习型探索） |
| `sigma_init` | **0.01** | NoisyNet 初始噪声（保守，配合大 batch_size） |
| `sigma_decay` | **0.99999994** | 每训练步 σ 衰减系数；优先保护早期探索，15.6M 步后 σ 约 39% |

### 6.4 PPO 配置 (PPOConfig)

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

### 6.5 环境配置 (EnvConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cols` | 10 | 棋盘列数 |
| `rows` | 20 | 可见行数 |
| `hidden_rows` | 2 | 隐藏行数 |
| `next_queue_size` | 4 | Next 队列可见长度 |
| `max_steps` | 10000 | 单局最大步数（truncation） |
| `use_cpp_env` | **true** | 启用 C++ pybind11 环境加速 |
| `reward_weights.w_height` | 0.0 | 高度惩罚权重（诊断期归零） |
| `reward_weights.w_holes` | 0.0 | 孔洞惩罚权重（诊断期归零） |
| `reward_weights.w_bumpiness` | 0.0 | 崎岖度惩罚权重（诊断期归零） |
| `reward_weights.w_well` | 0.0 | 井深惩罚权重（诊断期归零） |
| `reward_weights.w_survival` | 0.01 | 存活奖励（诊断期保留） |
| `reward_weights.w_death` | -100.0 | 死亡惩罚（诊断期保留） |

> ⚠️ **诊断期说明**：惩罚权重当前全部归零，仅保留消行奖励、存活奖励和死亡惩罚。
> 这是为了排除奖励塑形导致的灾难性遗忘。收敛稳定后可逐步恢复惩罚权重。

---

## 七、硬件建议

| 组件 | 推荐配置 | 最低配置 |
|------|----------|----------|
| GPU | NVIDIA RTX 4090 / A100 | RTX 3060 (12GB) |
| CPU | 32+ cores | 8 cores |
| RAM | 64 GB+ | 16 GB |
| 存储 | 500 GB NVMe | 100 GB |

**训练耗时估算**（50M steps, DQN, 64 并行环境）：

| 硬件 | 纯 Python | C++ 加速 | 加速比 |
|------|----------|---------|--------|
| AMD EPYC 7T83 (64C) | ~11 小时 | **~3-5 小时** | 3-4x |
| Intel Xeon 32C | ~20 小时 | ~6-8 小时 | 3x |
| Desktop 16C (5950X) | ~30 小时 | ~10 小时 | 3x |

> C++ 加速对纯 CPU 训练提升最显著。GPU 训练中环境模拟占比低 (<20%)，加速效果有限 (~1.05-1.3x)。详见 `docs/analysis-cuda-cpp.md`。

---

## 八、C++ 加速方案

项目提供两层可选的 C++ 加速：

### 8.1 环境模拟加速（推荐）

将训练瓶颈最大的三个函数迁移到 C++，通过 pybind11 绑定回 Python：

| 函数 | Python 耗时 | C++ 耗时 | 加速比 | 说明 |
|------|------------|---------|--------|------|
| `get_legal_actions()` | 150-300 μs | 5-10 μs | **15-30x** | 碰撞检测循环 ~1700 次/step |
| `StateEncoder.encode()` | 50-150 μs | 5-10 μs | **10-15x** | 消除与 step() 的重复计算 |
| `agent.observe()` | 20-50 μs | 10-20 μs | **2-3x** | 消除 dict 分配 |
| **训练整体 (含评估)** | 11-14 小时 | **3-5 小时** | **3-4x** | 50M steps, 1×EPYC 7T83 CPU |

**原理**：训练 85-95% 的 CPU 时间消耗在环境模拟（碰撞检测、重力模拟、棋盘特征计算），而非神经网络。C++ bitboard 实现（`uint16_t` 行表示、列高度缓存、位运算碰撞检测）消除 Python 循环和 numpy 查找开销。同时 C++ Board 缓存列高度，避免 `step()` 和 `encode()` 的重复计算。

#### 编译

**Windows (Visual Studio)**：

```powershell
pip install pybind11 cmake
mkdir build; cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
# 如果 cmake 报 "pybind11 not found"：
# cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$(python -c 'import pybind11;print(pybind11.get_cmake_dir())')"
cmake --build . --target tetris_core --config Release
copy env\bindings\Release\tetris_core.*.pyd ..\env\bindings\
```

**Ubuntu (GCC)**：

```bash
sudo apt-get install -y build-essential cmake python3-dev
pip install pybind11 cmake
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
# 如果 cmake 报 "pybind11 not found"：
# cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$(python3 -c 'import pybind11;print(pybind11.get_cmake_dir())')"
cmake --build . --target tetris_core -j$(nproc)
cp env/bindings/tetris_core.*.so ../env/bindings/
```

编译产物：Windows 为 `tetris_core.cp314-win_amd64.pyd`（位于 `build/env/bindings/Release/`），Linux 为 `tetris_core.cpython-3xx-x86_64-linux-gnu.so`（位于 `build/env/bindings/`）。文件名随 Python 版本变化。

> ⚠️ **关键**：编译产物位于 `build/env/bindings/`（Windows MSVC 多配置下为 `build/env/bindings/Release/`），**必须复制到项目根目录的 `env/bindings/`**（与 `cpp_env.py` 同级），`cpp_env.py` 才能通过 `from . import tetris_core` 加载。
> **不要**将编译产物放在项目根目录的 `tetris_core/` 文件夹内。如果存在 `tetris_core/` 目录，Python 会将其识别为 namespace package 而屏蔽同名的 `.pyd`/`.so` 模块。删除 `tetris_core/` 目录后重新运行即可。

#### 启用

在训练配置中设 `use_cpp_env: true`：

```yaml
# configs/training/dqn_rainbow.yaml
env:
  use_cpp_env: true
```

或在 `trainer/config.py` 中修改 `EnvConfig.use_cpp_env` 默认值。

启用后，`trainer._make_env()` 自动创建 `CppTetrisEnv`（`env/bindings/cpp_env.py`）替代纯 Python `TetrisEnv`。接口完全一致，无需修改训练循环。

> **向后兼容**：`use_cpp_env: false`（默认）时行为与修改前完全相同。不编译 C++ 模块也不影响纯 Python 训练。

#### 架构

```
训练循环 (trainer.py)
    │
    ├── use_cpp_env=false ──▶ TetrisEnv (env/tetris_env.py)
    │                           纯 Python + NumPy
    │
    └── use_cpp_env=true  ──▶ CppTetrisEnv (env/bindings/cpp_env.py)
                                │
                                ├── tetris_core.TetrisEnvCpp  (C++ 游戏逻辑)
                                ├── tetris_core.StateEncoder  (C++ 特征编码)
                                └── tetris_core.Action        (C++ 动作结构)
```

### 8.2 C++ 推理后端

编译独立的 C++ ONNX Runtime 推理引擎，可作为俄罗斯方块游戏的 AI 后端。

#### 架构

```
┌──────────────────────┐
│ 游戏 (code.html / 终端) │
└─────────┬────────────┘
          │ 棋盘状态 (Board + 块信息)
          ▼
┌──────────────────────┐
│  AIPlayer (一站式)    │  ← inference/cpp/player.h
│                      │
│  selectAction(board,  │
│    piece, rot, hold,  │
│    queue, can_hold)   │
│    → Action           │
│                      │
│  内部调用:            │
│  ├ StateEncoder       │  特征编码 (53维, 与训练一致)
│  ├ ActionGenerator    │  合法动作生成
│  └ ONNXModel          │  ONNX 推理
└──────────────────────┘
```

**与 `InferenceEngine` (Python) 的区别**：

| 引擎 | 输入 | 自带编码 | 自带动作过滤 | 适用场景 |
|------|------|---------|-------------|---------|
| `InferenceEngine` (Python) | `(board, features)` numpy | 否 (外部编码) | 否 (外部过滤) | Python 训练/评估 |
| `AIPlayer` (C++) | `Board` + 块状态原始值 | 是 (StateEncoder) | 是 (ActionGenerator) | C++ 后端 / WASM |

#### 编译

**Windows (Visual Studio)**：

```powershell
pip install onnxruntime
cd build
cmake .. -Donnxruntime_DIR="$(python -c 'import onnxruntime,os;print(os.path.dirname(onnxruntime.__file__))')"
cmake --build . --target tetris_inference --config Release
```

**Ubuntu (GCC)**：

```bash
pip install onnxruntime
cd build
cmake .. -Donnxruntime_DIR="$(python3 -c 'import onnxruntime,os;print(os.path.dirname(onnxruntime.__file__))')"
cmake --build . --target tetris_inference -j$(nproc)
```

产物：`build/inference/cpp/tetris_inference.lib`（静态库）。

#### 使用方式

**方式 A：作为 C++ 静态库链接**

```cpp
#include "inference/cpp/player.h"
#include "env/core/tetris_core.h"

// 加载模型
tetris::inference::AIPlayer ai("tetris_ai.onnx");

// 每帧调用
tetris::Board board;
// ... 从游戏获取 board 状态 ...
tetris::Action action = ai.selectAction(
    board,
    tetris::PieceName::I,   // 当前块
    0,                       // 旋转
    tetris::PieceName::NONE, // hold
    true,                    // can_hold
    {tetris::PieceName::T, tetris::PieceName::S, 
     tetris::PieceName::Z, tetris::PieceName::J}  // next queue
);

// action.rotation, action.column, action.hold → 传给游戏执行
```

**方式 B：WebAssembly (浏览器游戏)**

编译 WASM 模块，在 `tetris/code.html` 中替代 ONNX Runtime Web：

```bash
# 需要 Emscripten
emcmake cmake .. -DBUILD_WASM=ON
cmake --build .
```

导出的 C 函数：

| 函数 | 说明 |
|------|------|
| `init_ai(model_path)` | 加载 ONNX 模型 + 创建 AIPlayer 和 TetrisEnv |
| `select_action()` | 返回打包的 int: `rotation(2bit) \| column(5bit) \| hold(1bit)` |
| `step_env(action_packed, &reward, &done, &score, &lines)` | 执行动作，更新内部 TetrisEnv |
| `reset_env(seed)` | 重置环境 |
| `get_board_data()` | 返回 `float[22*10]` 棋盘快照 |
| `destroy_ai()` | 释放资源 |

#### 相关文件

| 文件 | 用途 |
|------|------|
| `inference/cpp/model_loader.h` / `.cpp` | ONNX 模型加载 + 推理（支持 CUDA） |
| `inference/cpp/player.h` / `.cpp` | AIPlayer — 一站式状态→动作 |
| `inference/cpp/web/tetris_ai_web.cpp` | WASM C 导出接口 |
| `inference/cpp/web/CMakeLists.txt` | Emscripten 构建规则 |

### 8.3 CMake 参数参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CMAKE_BUILD_TYPE` | — | `Release` 推荐（`-O3 -march=native`） |
| `ORT_ROOT` | — | 自编译 ONNX Runtime 根目录 |
| `onnxruntime_DIR` | — | pip 安装的 ORT cmake 配置目录 |
| `BUILD_WASM` | `OFF` | 是否构建 WebAssembly 推理模块 |

### 8.4 性能测试

编译 C++ 模块后运行基准测试：

```bash
# 需要先编译 tetris_core 模块
pytest tests/test_cpp_env.py -v -k Benchmark
```

测试项：
- `test_legal_actions_speedup`：对比 Python vs C++ 的 `get_legal_actions()` 耗时
- `test_full_step_speedup`：对比完整 step() 循环的吞吐量

### 8.5 C++ 代码结构

```
env/core/
├── piece_data.h        # 方块形状、SRS 踢墙表、分数表
├── randomizer.h        # BagRandomizer (7-bag) + 纯随机
├── tetris_core.h       # Board 类 (uint16_t 位棋盘, 碰撞/消行/特征计算)
├── action_gen.h        # Action 结构 + ActionGenerator (合法动作枚举)
├── tetris_env.h        # TetrisEnv C++ 实现 (step/reset/奖励)
├── state_encoder.h     # C++ 53 维特征编码器
└── *.cpp               # 空桩 (实现在 .h 中, header-only)

env/bindings/
├── tetris_bindings.cpp # pybind11 绑定 (tetris_core Python 模块)
└── cpp_env.py          # CppTetrisEnv Python 包装器

inference/cpp/
├── model_loader.h/cpp  # ONNX Runtime 模型加载
├── player.h/cpp        # AI Player
└── web/                # Emscripten WASM 构建
```

> 所有 `.cpp` 文件均为空桩（仅 `#include` 对应头文件）。完整实现在 `.h` 头文件中，编译后内联到调用方。

---

## 九、常见问题

### Q: 训练不收敛 / 分数崩溃怎么办？

**症状**：预训练后初始分数 ~30K，RL 训练后迅速掉到负数。

**根因**（已验证）：
1. **奖励塑形惩罚过重**：`w_holes=1.5` 等惩罚权重导致 agent 学到"快速死亡"策略来规避惩罚
2. **硬更新 + 大批量**：`use_hard_update=true` 配合 `batch_size=256`，在线网络快速漂移但 target 冻结 8000 步，Double DQN 无法纠正

**修复步骤**（按优先级）：
1. 将 `w_height/w_holes/w_bumpiness/w_well` 全部设为 0.0（诊断期排除奖励问题）
2. 设 `use_hard_update: false`（Polyak 软更新, τ=0.005）
3. 降低学习率到 `2.5e-5`，降低 `sigma_init` 到 `0.01`
4. 确认 `batch_size=256` 配合以上参数稳定后，再逐步恢复惩罚权重

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
2. **仅 DQN 支持 ONNX 导出**：PPO 模型无法通过 `export_model.py` 导出
3. **C++ 环境加速仅 CPU 训练显著**：GPU 训练中环境模拟占比低，C++ 加速效果有限 (~1.05-1.3x)
4. **RNG 不一致**：C++ 和 Python 环境使用不同随机数生成器（`mt19937_64` vs `numpy.random`），相同种子不产生相同对局
5. **根目录 `test_env.py`** 是 `tests/test_env.py` 的历史副本，建议手动删除

### Q: 如何使用内置的 Dellacherie AI（无需 ONNX 模型）？

在 `code.html` 中点击 **"DL"** 按钮即可切换到 Dellacherie 启发式 AI。无需加载 ONNX 模型文件，直接在浏览器内运行六特征评估算法。

特性：
- 零依赖：纯 JavaScript 实现，无需 GPU/ONNX Runtime
- 消融实验：通过 `agent/dellacherie.py` 的 `DellacherieConfig` 可单独启用/禁用每个特征
- 训练对比：训练过程中自动运行 Dellacherie 基准评估（每 50K 步），与学习策略对比

### Q: 如何复用预训练样本？

首次训练后，样本自动保存到 `pretrain_samples/samples_latest.npz`，BC 权重保存到 `pretrained_weights.pt`。

下次训练时自动跳过收集和 BC 训练，直接加载缓存权重（毫秒级）：

```bash
python scripts/train.py --device cuda --envs 64  # 自动检测并加载缓存
```

三级缓存加速：

```
pretrained_weights.pt 存在? → torch.load() → 跳过一切（毫秒）
samples_latest.npz 存在?   → 加载样本 → BC 训练 → 保存权重（~1分钟）
都不存在                    → 收集 + 训练 + 缓存（~80分钟）
```

### Q: C++ 模块编译失败？
```bash
# 确认安装了 pybind11
pip install pybind11

# 确认 CMake 版本 >= 3.20
cmake --version

# 查看详细编译错误
cmake --build . --target tetris_core --config Release --verbose

# 常见问题：
# - "pybind11 not found": pip install pybind11 && rm -rf build && mkdir build && cd build && cmake ..
# - "tetris_inference not found": ONNX Runtime 未安装或 ORT_ROOT 未设置
# - hold_first error: 确保 env/core/action_gen.h 已更新为 hold
```

---

## 十、下一轮训练注意事项

### 10.1 当前诊断配置状态

以下为当前生效的关键参数（`configs/training/dqn_rainbow.yaml`）：

```yaml
training:
  total_samples: 1_000_000_000   # → 约 15.6M env steps
dqn:
  lr: 2.5e-5                     # 保守学习率
  batch_size: 256                # 大批量（8× 旧值）
  use_hard_update: false         # Polyak 软更新（关键！）
  target_update_tau: 0.005
network:
  sigma_init: 0.01               # 保守噪声
  sigma_decay: 0.99999994         # 缓慢衰减，末期 ~39%
env:
  reward_weights:
    w_height: 0.0                # 诊断期归零
    w_holes: 0.0
    w_bumpiness: 0.0
    w_well: 0.0
    w_survival: 0.01             # 保留
    w_death: -100.0              # 保留
```

### 10.2 训练启动

```bash
# 默认配置（推荐）
python scripts/train.py --device cuda --envs 64

# 快速验证（1000 步，约 3 分钟）
python scripts/train.py --device cuda --envs 64 --steps 1000

# 指定样本数
python scripts/train.py --device cuda --envs 64 --samples 500000000

# 启用性能分析
python scripts/train.py --device cuda --envs 64 --profile
```

### 10.3 观察指标

| 指标 | 健康信号 | 警告信号 |
|------|---------|---------|
| **Avg100R** | 正数且稳定/上升 | 持续下降，特别是掉到负数 |
| **Dead** | <5%，逐个下降 | >20%，智能体频繁死亡 |
| **初始评估分** | ≥30,000（预训练基线） | <5,000（预训练权重未加载） |
| **FPS** | ≥300 | <100（性能瓶颈） |
| **checkpoint Δ score** | 正值或小幅波动 | 持续大幅负值 |
| **Dellacherie 对比** | agent 分数接近或超过 DL | agent 分数远低于 DL |

### 10.4 恢复惩罚权重（收敛后）

当训练分数稳定后，逐步恢复惩罚权重：

```yaml
# 阶段 1：诊断（当前）— 分数 ≥30K 且稳定
w_height: 0.0, w_holes: 0.0, w_bumpiness: 0.0, w_well: 0.0

# 阶段 2：轻度惩罚 — 分数持续上升
w_height: 0.05, w_holes: 0.3, w_bumpiness: 0.05, w_well: 0.1

# 阶段 3：标准惩罚 — 分数达到 100K+
w_height: 0.1, w_holes: 0.5, w_bumpiness: 0.1, w_well: 0.2
```

切勿直接跳回旧权重（0.3/1.5/0.2/0.5），那会导致灾难性遗忘重现。

### 10.5 常见陷阱

1. **`--no-pretrain` 会跳过缓存权重加载**：快速路径代码在 `_pretrain()` 内部，该 flag 会跳过整个方法
2. **旧 checkpoint 含崩溃权重**：删除 `checkpoints/` 目录重新训练
3. **C++ env 未编译**：如果 `use_cpp_env: true` 但 `.pyd` 不在 `env/bindings/`，会回退到纯 Python env（慢 3-5x）
4. **旧 `total_steps` 思维**：配置改为 `total_samples`，旧 YAML 中的 `total_steps` 会被忽略（除非 CLI `--steps` 覆盖）

### 10.6 最终评估（Agent vs Dellacherie 头对头）

训练结束后自动运行 200 局头对头对比：

```
============================================================
[2026-05-03 08:00:00]  Final Evaluation — Agent vs Dellacherie
  (200 episodes, same seeds, deterministic, no exploration noise)
============================================================
  Episodes:       200
  ───────────  Agent  ───────────
  Avg Score:          52,300.5
  Max Score:             245,000
  ───────────  Dellacherie  ──────
  Avg Score:          45,000.0
  Max Score:             210,000
  ───────────  Comparison  ────────
  Mean Gap:            +7,300.5
  Win Rate:              62.5%  (125W / 75L / 0T)
  t-statistic:             3.42
  Verdict:        ✓ Agent beats Dellacherie
============================================================
```

**评估条件：**

| 条件 | 说明 |
|------|------|
| 初始局面 | `env.reset(seed=ep)`，ep=0..199，双方同种子 |
| 探索噪声 | **关闭**。`eval_mode()` → NoisyNet 仅用 μ 权重 |
| 动作选择 | 贪婪（始终选 Q 值最高的动作） |
| 比较局数 | 200 局 |

**判断标准：**

| 结果 | 含义 |
|------|------|
| Win Rate > 50% 且 Mean Gap > 0 | ✓ RL 策略超越 Dellacherie |
| Win Rate ≈ 50% 且 Gap ≈ 0 | ≈ 持平 |
| Win Rate < 50% 且 Mean Gap < 0 | ✗ 未超越，需继续训练或调参 |

**设计文档**：`docs/final-eval-design.md`

---

## 十一、参考

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
