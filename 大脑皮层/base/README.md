# Base 初始素材脚本详解（Case 3）

本目录是题目方提供给玩家的“皮层数字孪生”脚手架，是**对题目场景的建模**。
玩家在阶段一 / 阶段二都必须基于这里的三份脚本继续开发，不要另起炉灶。

```
case3/base/
├── neural_model.py         # 共用场景模型（随机图分支神经网络 + 可选动力学突触）
├── starter_simulation.py   # 亚临界基线脚本（必须先跑）
└── plotting.py             # matplotlib 绘图 + 临界指数 / shape-collapse 工具
```

下文按“先看模型 → 再看怎么跑 → 最后看怎么画图”的顺序逐个讲透。建议
先和 `case2/base/` 对照一下：本案例换了**普适类**（从 OFC 地震断层
换到了 mean-field 分支过程 / 神经雪崩），机制从“阈值耗散 + 边界耗散”
换成了“阈值激发 + 突触资源耗竭”，验证目标也从 Gutenberg–Richter 换
成了三个独立指数 (τ, α, 1/σνz) **加上**它们之间的 **crackling-noise
关系**。

---

## 1. `neural_model.py`：场景核心模型

模型是一张 ``N`` 个皮层神经元构成的**有向稀疏随机图**：每个神经元随机
选 ``k`` 个突触后目标（不含自己），形成一份**淬火**（quenched）的网络
拓扑——拓扑在仿真过程中保持不变。每个神经元自带一个连续的“膜电位”
``h_i ∈ [0, θ]``。系统同时存在三种时间尺度，这是 SOC 的招牌：

- **慢驱动（slow drive）**：每一个宏观仿真步随机挑一个神经元 ``i``，
  把外部输入 ``drive_kick`` 注入它的膜电位。默认 ``drive_kick = θ``，
  所以每个宏观步**必然**触发恰好一个“原始尖峰”，这正是分支过程文献
  里最干净的协议——雪崩规模分布等价于该分支过程的**后代分布**。
- **快传播（fast spike propagation）**：被触发的神经元放电
  ``h_i → 0``，向 ``k`` 个突触后目标递送 ``J · u_ij`` 的兴奋。任何被
  推过阈值的目标都成为下一“代”发放的成员。这是同步离散时间更新——
  一“代”就是一个分支过程世代。
- **中等尺度的突触动力学（可选）**：当 ``dynamical_synapses=True``
  时，每一次发放都会把 ``i`` 的所有出突触**资源**乘以 ``(1 - ε)``（耗
  竭），同时每一个宏观步全网突触按
  ``u ← u + (1 - u) / τ_rec`` 线性恢复。这是 Tsodyks–Markram 机制的
  最简版本，也是 Levina–Herrmann–Geisel (Nat. Phys. 2007) 自组织临界
  机制的核心：自动把**有效**分支比 ``σ_eff = k · J · ⟨u⟩`` 锁定在 1。

> 当 ``dynamical_synapses=False`` 时，``u`` 全部恒等于 1，``σ`` 由
> ``J`` 一锤子定下来，即**调参临界**。临界点
> ``J_c = 1/k = 0.125``（``k=8`` 时）。

### 1.1 `NeuralParams` —— 玩家最该关心的旋钮

```python
@dataclass
class NeuralParams:
    # —— 网络与几何 ——
    N: int = 256
    k: int = 8
    threshold: float = 1.0

    # —— 突触强度 ——
    J: float = 0.10            # 临界点 J_c = 1/k = 0.125
    j_disorder: float = 0.0    # 单边权重的乘性扰动半宽（无害异质）

    # —— 时间 / 驱动 ——
    drive_steps: int = 6000
    warmup: int = 1500
    drive_kick: float = 1.0    # 注入到一个随机神经元的外部输入
    seed: int = 7

    # —— 阶段二动态突触（Tsodyks–Markram） ——
    dynamical_synapses: bool = False
    epsilon: float = 0.05      # 每次发放的资源耗竭比例
    tau_rec: float = 400.0     # 资源回到 1 的特征时间
    u_floor: float = 1e-3      # 数值下限

    avalanche_size_cap: int = 200_000  # 防发散硬上限
```

> 阶段一只调 `J`（连同必要时增大 `drive_steps` 改善统计）；阶段二把
> `dynamical_synapses` 打开，把 `J` 设在临界点之上让动态突触把 σ 拉
> 回 1。

### 1.2 `NeuralRunResult` —— 仿真返回的诊断量

```python
@dataclass
class NeuralRunResult:
    # —— 时间序列（长度 == drive_steps） ——
    mean_potential: list[float]
    branching_ratio: list[float]   # 当前 sigma_eff = k * J * <u>
    mean_resource: list[float]
    J_series: list[float]

    # —— 事件序列（warmup 之后非零雪崩） ——
    sizes: list[int]
    durations: list[int]
    waiting_times: list[int]
    event_steps: list[int]
    profiles: list[list[int]]      # 每场雪崩的每代发放数；shape collapse 用
```

`profiles[k][r]` 记录了第 `k` 场雪崩中第 `r` 代有多少神经元同步发放，
``sum(profiles[k]) == sizes[k]``，``len(profiles[k]) == durations[k]``。
这是阶段二做**雪崩形状塌缩（shape collapse）**所需的关键数据。

### 1.3 `CorticalNetwork` —— 仿真主类

构造函数：
- 用 `random.Random(seed)` 创建独立 RNG（不污染全局）。
- 给每个神经元随机抽 `k` 个不重复的下游邻居，存进 `out_neighbours`。
- 初始化 `weights[i][e]` 为 `J`（可加乘性扰动）、`resources[i][e] = 1`。
- 把所有 `h_i` 抽自 `[0, θ/2]` 的均匀分布。

#### `_drive(self) -> int`
随机挑 `i`，`h[i] += drive_kick`，返回 `i`。如果 `drive_kick == θ` 且
`h[i]` 原先 ≥ 0，则 `i` 必跨阈，触发主尖峰；否则不触发，本宏观步
事件大小为 0。

#### `_relax(self, seed_idx) -> (size, duration, profile)`
两遍同步松弛：
1. **Pass 1**：扫描当前 `firing` 集合的每个神经元 `i`，先把它对每个下
   游 `j` 的贡献 `J_ij · u_ij` 累加到一个**临时** `kicks: dict[int,
   float]`，再把 `i` 的所有出突触乘 `(1 - ε)`（仅当
   `dynamical_synapses=True` 时）。
2. **Pass 2**：先把所有 `firing` 神经元 `h[i] = 0`，**然后**才把
   `kicks[j]` 加到 `h[j]`，保证当前轮内被重置的神经元不会被同轮的输
   入推回去并“双重重置”。
3. 把跨阈的 `j` 收进下一轮 `frontier`，并把当前轮发放数追加到
   `profile`。
4. 返回 `(size, duration, profile)`。雪崩规模超过
   `avalanche_size_cap` 时强制截断（防御超临界发散）。

#### `run(self) -> NeuralRunResult`
主循环每个宏观步串起 `_drive → _relax → 资源恢复 → 诊断采样`：

- 突触恢复 `u ← u + (1 - u) / τ_rec` **每个宏观步只做一次**（不是每代
  一次），保证恢复时间确实比传播时间慢。
- 诊断里特别注意 `branching_ratio[t] = k * J * <u>(t)`：这是当前**有
  效分支比**，阶段二判定 SOC 是否成功的核心指标。

---

## 2. `starter_simulation.py`：亚临界态基线

这是玩家最先要跑的脚本，用一组**故意亚临界**（`J=0.08`，标称
`σ = k · J = 0.64`）的参数把模型跑一遍，并产出三张参考图：

| 图 | 内容 |
| --- | --- |
| `starter_mean_potential.svg` | 平均电位 ⟨h⟩(t) 与 σ_eff(t) 双线，水平虚线 σ=1 显示距离临界还有多远 |
| `starter_avalanche_distribution.svg` | 雪崩规模 PDF（对数分箱）+ CCDF；亚临界两条曲线都呈快速衰减 |
| `starter_duration_distribution.svg` | 雪崩时长 PDF + CCDF；亚临界 |

跑完会打印总事件数、平均规模、最大规模和标称 σ。这一步的目的是让玩家
**先有亚临界对照感**，再去阶段一找“调到了临界”的样子。

---

## 3. `plotting.py`：matplotlib 绘图 + 临界指数 / shape-collapse 工具

延续 case1 / case2 风格，所有图走 matplotlib + Agg + SVG 输出。除了
通用的 `plot_lines`、`plot_dual_axis`，新增 5 个**神经雪崩 SOC 专用**
统计工具：

### 3.1 `log_hist(data, bins=36)` 与 `ccdf_log(data)`
和 case2 完全一致：对数分箱 PDF + 互补累积分布。两者搭配看才不容易把
亚临界态误判成幂律。

### 3.2 `power_law_mle(data, smin)`
离散 Hill / Clauset–Shalizi–Newman 估计幂律指数。返回
`(τ_hat, n_used)`，临界附近用来读 `τ`（规模）和 `α`（时长）。

### 3.3 `mean_size_vs_duration(sizes, durations, ...)` 与 `loglog_slope`
按时长 `T` 把雪崩分桶，输出每个桶里平均规模 ⟨s|T⟩。`loglog_slope`
对该曲线做最小二乘拟合，斜率即 `1/(σνz)`。**Sethna 2001 关系**
``(τ - 1) · σνz = α - 1`` 三个独立指数的最重要交叉验证靠这个工具。

### 3.4 `avalanche_profile(profile, duration, n_bins=40)`
把单场雪崩的“每代发放数”重采样到固定网格 `n_bins=40`，得到
形如 ``s(t/T)`` 的归一化时间剖面。这是 shape collapse 的输入。

### 3.5 `collapsed_shape(profiles_by_duration, sigma_nu_z_inv, n_bins=40)`
对每个 `T` 的剖面做平均，再乘以 ``T^(1 - 1/σνz)``，得到“塌缩”后的
通用尺度函数。SOC 的最佳单图证据：不同 ``T`` 的曲线塌缩到一条线上。

> 不同于 case2 的 Gutenberg–Richter / Omori，本案例验证靠**三**个独立
> 指数 + 1 个标度关系，外加一个 shape collapse。难度上明显更高。

---

## 4. 推荐的开发节奏

1. **先跑** `python case3/base/starter_simulation.py`，确认
   `figures/starter_*.svg` 生成、并感受亚临界态的分布形状。
2. 阶段一新建实验脚本，扫描 `J`、找出 `J_c`，再用三段对比 + Hill 估
   计读 `τ` 和 `α`，最后用 `mean_size_vs_duration` + `loglog_slope`
   估 `1/σνz`，并核对 ``(τ-1)σνz = α-1``。
3. 阶段二把 `dynamical_synapses=True`，从 `J ≥ J_c` 起步，用
   `plot_dual_axis` 同时观察 `σ_eff` 与 `<u>`，再多跑两个 `J0` / seed
   组合并叠加 `σ_eff(t)` 验证鲁棒性，最后用 `collapsed_shape` 出
   shape-collapse 图把临界普适性钉死。
