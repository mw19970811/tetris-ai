# Replay Buffer Inspector — 设计文档

> 终端交互式工具，用于浏览训练 replay buffer 中的 transition，可视化棋盘状态和方块信息。

---

## 一、目标

训练过程中，buffer 文件 (`checkpoints/step_XXXXXXXXX_buffer.pt`) 保存了智能体的经验数据。
本工具提供终端内交互浏览能力，帮助排查以下问题：

- 样本质量：transition 中的 state/action/reward 是否合理
- 奖励分布：reward 是否集中在少数 transition
- 局面多样性：buffer 中的棋盘状态是否多样
- 动作分布：agent 是否偏向某些 rotation/column

## 二、数据格式

### 2.1 Buffer 文件结构

```python
# checkpoints/step_XXXXXXXXX_buffer.pt — torch.save() 序列化的 dict:
{
    "tree_data":     [StoredTransition | None, ...],  # 长度 = capacity
    "tree_tree":     np.ndarray (2*capacity-1, float64),  # SumTree 内部节点
    "tree_write_pos": int,
    "tree_size":     int,       # 实际存储的 transition 数量
    "max_priority":  float,
}
```

### 2.2 StoredTransition 字段

| 字段 | 类型 | 形状 | 含义 |
|------|------|------|------|
| `board` | float32 | (1, 22, 10) | 动作**前**的棋盘 |
| `features` | float32 | (53,) | 动作前的手工特征向量 |
| `action` | int64 | 标量 | 编码后的动作 (0-111) |
| `reward` | float32 | 标量 | n-step 回报 |
| `next_board` | float32 | (1, 22, 10) | 动作**后**的棋盘 |
| `next_features` | float32 | (53,) | 动作后的手工特征向量 |
| `done` | bool | 标量 | 是否终局 |

### 2.3 特征向量解码 (53 维)

```
索引   维度  含义
0      1     累计高度 Σ column_heights
1      1     消行数占位 (恒为 0)
2      1     孔洞数
3      1     崎岖度 Σ|h_c - h_{c+1}|
4      1     最大井深
5      1     高度变化占位 (恒为 0)
6-12   7     当前块 one-hot: I O T S Z J L
13-16  4     旋转 one-hot: 0 1 2 3
17-24  8     Hold 块 one-hot: I O T S Z J L [empty]
25-52  28    Next 队列 one-hots: 4×7
```

### 2.4 动作解码

```
action ∈ [0, 111]
hold   = action >= 56    # 后半段 = 使用 hold
idx    = action % 56
rotation = idx // 14      # 0-3
column   = (idx % 14) - 2  # -2 ~ 11 (含越界列，用于生成阶段)
```

## 三、功能设计

### 3.1 加载

```
python tools/inspect_buffer.py checkpoints/step_000010000_buffer.pt
```

启动后加载 buffer 文件，显示摘要信息：
- 总容量 / 实际样本数
- 优先级范围 (min / max / mean)
- 操作提示

### 3.2 主界面布局

```
═══════════════════════════════════════════════════════════════
  Buffer: step_000010000_buffer.pt
  Samples: 45,832 / 2,000,000    Priority: [0.12 … 847.3]
  Transition #12458 / 45832       Priority: 23.45
═══════════════════════════════════════════════════════════════
                        BEFORE (board)
  col→ 0 1 2 3 4 5 6 7 8 9
      ┌─────────────────────┐
row 0 │ . . . . . . . . . . │
    2 │ . . . . . . . . . . │
    4 │ . . . . . . . . . . │
    ⋮
   20 │ # # # . . . # # # # │
   21 │ # # # # # . # # # # │
      └─────────────────────┘

  Action: rot=1, col=3, hold=False  |  Reward: +500.0  |  Done: False
  Current: T (rot 1)   Hold: [none]   Next: S Z J L

                        AFTER (next_board)
      ┌─────────────────────┐
    0 │ . . . . . . . . . . │
      │                     │
      │       (T piece       │
      │        placed)       │
      │                     │
   21 │ # # # # # . # # # # │
      └─────────────────────┘
═══════════════════════════════════════════════════════════════
  [←→] prev/next 10   [↑↓] prev/next 1   [g] goto index
  [f] filter: all   [q] quit
```

### 3.3 键盘操作

| 按键 | 功能 |
|------|------|
| `→` / `↓` | 下一个 transition |
| `←` / `↑` | 上一个 transition |
| `PgDn` / `]` | 前进 10 个 |
| `PgUp` / `[` | 后退 10 个 |
| `Home` / `^` | 跳到第一个 |
| `End` / `$` | 跳到最后一个 |
| `g` | 输入索引跳转 |
| `f` | 切换过滤模式 (all / done / not-done / high-reward) |
| `p` | 按优先级排序浏览 (降序) |
| `q` | 退出 |

### 3.4 过滤模式

| 模式 | 说明 |
|------|------|
| `all` | 所有 transition |
| `done` | 仅 ep 结束的 transition |
| `live` | 仅 ep 进行中的 transition |
| `reward+` | reward > 0 的 transition |
| `reward-` | reward < 0 的 transition |

### 3.5 棋盘渲染

使用 ANSI 颜色区分：
- `#` 白色 — 已占据的格子
- `.` 灰色 — 空格
- 新放置的方块格子用绿色高亮 (仅在 AFTER 视图中)
- 消行用黄色行标记

## 四、实现要点

### 4.1 跨平台键盘输入

```python
# Windows: msvcrt.getch()
# Unix:    tty.setraw + sys.stdin.read(1)
# 封装为 get_key() 函数
```

### 4.2 棋盘渲染

```python
def render_board(board_2d, highlight_cells=None):
    """将 (22, 10) 二值数组渲染为 ANSI 字符串。
    highlight_cells: 可选，[(r,c), ...] 用绿色高亮。
    """
```

### 4.3 特征解码

```python
PIECE_NAMES = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']

def decode_features(features):
    """从 53 维特征向量提取可读信息。"""
    piece_idx = np.argmax(features[6:13])
    rotation = np.argmax(features[13:17])
    hold_idx = np.argmax(features[17:25])
    hold_name = PIECE_NAMES[hold_idx] if hold_idx < 7 else 'none'
    next_pieces = []
    for i in range(4):
        oh = features[25+i*7:25+(i+1)*7]
        next_pieces.append(PIECE_NAMES[np.argmax(oh)])
    return piece_idx, rotation, hold_name, next_pieces
```

### 4.4 文件结构

```
tools/
└── inspect_buffer.py    # 单文件，无外部依赖 (仅 numpy + torch)
```

## 五、使用流程

```bash
# 1. 训练生成 buffer 文件（自动保存）
python scripts/train.py --device cuda --envs 64

# 2. 训练过程中（或训练后）查看 buffer
python tools/inspect_buffer.py checkpoints/step_000010000_buffer.pt

# 3. 在交互界面中用方向键浏览
```

## 六、不做的

- 不做 GUI / Web 界面 — 终端足够
- 不做 buffer 编辑/修改 — 只读工具
- 不依赖 curses — 用原始 ANSI 转义序列 + 跨平台键盘读取
