# Base 初始素材脚本详解（Case 2）

本目录是题目方提供给玩家的“断层数字孪生”脚手架，是**对题目场景的建模**。
玩家的两个阶段都必须基于这里的三份脚本继续开发，不要另起炉灶。

```
case2/base/
├── fault_model.py          # 共用场景模型（OFC 风格连续应力场）
├── starter_simulation.py   # 非临界态基线脚本（必须先跑）
└── plotting.py             # 基于 matplotlib 的轻量绘图与统计工具
```

下文按“先看模型 → 再看怎么跑 → 最后看怎么画图”的顺序逐个讲透。建议先在
心里把它和 `case1/base/` 对照一下：本案例把整数沙堆换成了**连续应力场**，
关键控制参数从 `spill_prob` 换成了**保守度** `alpha`，验证目标也从
“单一幂律”升级到了真实地震学里的 **Gutenberg–Richter** 与 **Omori** 律。

---

## 1. `fault_model.py`：场景核心模型

模型是一块 L×L 的断层网格。每个格点存一份**实数**应力 `s[i][j]`，可以理
解为该处的剪切应力。系统同时存在三个时间尺度，这是 SOC 的招牌：

- **慢驱动（slow drive）**：每一个宏观仿真步，对整张网格做一次**等量**
  加载，加载量恰好等于“当前最满格点距其阈值的差额”。等价于 OFC 文献的
  *extremal driving*——把缓慢板块运动和地震破裂时间尺度无限拉开。
- **快松弛（fast rupture）**：当一个格点的应力达到其当前阈值，就发生
  滑动：自身应力归零，把比例为 `alpha` 的释放应力分给上下左右四个邻居。
  `alpha` 是**保守度参数**：`4 * alpha` 是一次倾倒重新分配出去的总比例。
  `alpha = 0.25` 是体内完全保守，体内 SOC 临界点就在 `alpha` 略小于
  0.25 的地方。
- **可选的中间尺度（摩擦愈合）**：开启 `healing=True` 后，刚滑动过的
  格点暂时变“强”——其阈值会按 `tau_static + heal_amp * exp(-dt / heal_time)`
  的方式从一个突起再缓慢回落到该格点的静态阈值。这是阶段二的“有趣
  料”：和**异质静态阈值**`heterogeneity` 配合，再加上**自适应控制器**，
  系统就能不靠手调 `alpha` 而停在临界点附近。

**开边界耗散**是另一处关键：边界格点邻居数量小于 4，多出来的份额自然消
失在系统外，这是 SOC 在 `alpha < 0.25` 仍然能稳定的根本原因；也是为什么
本题用的是**有限网格 + 开边界**而不是周期边界。

### 1.1 `FaultParams` —— 玩家最该关心的旋钮

```python
@dataclass
class FaultParams:
    L: int = 32
    alpha: float = 0.18           # 保守度 (0..0.25)
    threshold: float = 1.0        # 静态破裂阈值
    drive_steps: int = 6000       # 总仿真宏观步数
    warmup: int = 1500            # 进入统计前的预热步数
    seed: int = 7

    # —— 阶段二可选——
    healing: bool = False         # 是否启用摩擦愈合
    heal_amp: float = 0.45
    heal_time: float = 80.0
    heterogeneity: float = 0.0    # 静态阈值随机扰动的半宽

    # —— 自适应保守度（阶段二 SOC 控制器）——
    adaptive: bool = False
    target_size: float = 4.0      # 目标滑动平均级联规模
    adapt_rate: float = 8e-4
    activity_window: int = 250
    alpha_min: float = 0.10
    alpha_max: float = 0.245

    avalanche_size_cap: int = 200_000  # 防发散的硬上限
```

> 阶段一的核心调参对象是 `alpha`。`heterogeneity`、`drive_steps` 等多用
> 于改善统计稳定性；阶段二的核心是把 `adaptive` 打开。

### 1.2 `FaultRunResult` —— 仿真返回的诊断量

```python
@dataclass
class FaultRunResult:
    mean_stress: list[float]      # 每一步结束时的平均应力
    max_stress: list[float]       # 每一步结束时的最大应力
    alpha_series: list[float]     # 每一步使用的 alpha（自适应模式下会变）
    threshold_mean: list[float]   # 每一步的平均阈值（healing 时会变）

    sizes: list[int]              # 预热后每次非平凡级联的总倾倒次数
    durations: list[int]          # 对应级联的同步松弛轮数
    waiting_times: list[int]      # 相邻事件间隔（宏观步）
    event_steps: list[int]        # 每次事件发生的宏观步索引（用于 Omori）

    final_field: list[list[float]]  # 仿真结束时的应力场快照
```

注意几条统计的差异：
- 时间序列（`mean_stress` / `max_stress` / `alpha_series` / `threshold_mean`）长度都是 `drive_steps`，可以直接画时间序列；
- 事件级序列（`sizes` / `durations` / `event_steps` / `waiting_times`）只收集 `warmup` 之后**且非零**的级联事件，是 SOC 统计图的素材；
- `event_steps` 与 `sizes` 一一对应，`plotting.aftershock_rate` 就靠它做 Omori 余震律的叠加。

### 1.3 `FaultStressSystem` —— 仿真主类

构造函数把网格初始化为 `[0, 0.5*threshold]` 上的均匀随机数，并用
`random.Random(seed)` 创建一个**独立**的随机数发生器（不污染全局
`random`，方便并行/复现）。同时为每个格点抽一份静态阈值
`tau_static[i][j] = threshold ± heterogeneity`，并初始化“上次破裂时刻”
为 `+inf`。

#### `_refresh_thresholds(self, step)`
按愈合公式更新当前阈值数组。如果 `healing=False`，则直接返回（保留每
一步 O(L²) 的开销）。如果一个格点从未破裂过（`last == inf`），其当前
阈值就等于静态阈值（不要被 `inf` 卷进 `exp` 里）。

#### `_drive(self)`
扫描整个网格，找出 `tau[i][j] - s[i][j]` 最小的格点 `(best_i, best_j)`
和差额 `best_gap`，然后把 `best_gap` **均匀**加到所有格点。这样下一步
那个格点必然破裂，是 OFC 的标准 *extremal* 驱动。

#### `_relax(self, seed_cell)`
把一个不稳定格点引发的连锁破裂跑完。它实现的是**两遍同步**松弛：

1. 把不稳定格点放进 `frontier`，进入主循环；
2. 每一轮把当前 `frontier` 整体当作 `unstable`：
   - **Pass 1**：对每个 `unstable` 格点，把它**当前的应力快照保存** 到
     `releases` 列表，再把该格点应力**清零**，并记入 `ruptured` 集合。
     这一步必须先做完所有清零，再做 Pass 2，否则会出现“同轮里被清零的
     格点又被邻居推上、然后再次被清零并误用 inflated 值”的经典 bug。
   - **Pass 2**：从 `releases` 里读出快照应力 `s_old`，把
     `alpha * s_old` 加到每个**界内**邻居；超界份额直接丢失（开边界
     耗散）。新越过阈值的邻居进入下一轮 `frontier`。
3. 返回 `(size, duration, ruptured)`：`size` 是倾倒事件数（同一格点
   多次倾倒分别计数），`duration` 是松弛持续了多少同步轮，`ruptured`
   是真正发生过滑动的格点集合（用于愈合记账）。
4. 安全阀：万一 `alpha → 0.25` 并且没有耗散导致雪崩失控，超过
   `avalanche_size_cap` 后强行截断，避免无限循环。

#### `run(self)`
主循环把上面三件事串起来，并多做两件杂活：

1. 如果 `healing=True`，把刚刚 `ruptured_cells` 里的每个格点的
   `last_rupture` 设为当前步 `t`，下一轮 `_refresh_thresholds` 就能给
   它叠上 `heal_amp` 的“摩擦凸起”。
2. 如果 `adaptive=True`，把当前 `size` 推进一个**滑动窗口**
   `recent_sizes`；窗口满了之后计算窗口均值，与 `target_size` 比较，
   按比例控制器修正 `alpha`。事件越大就越往“小 alpha”推（耗散更多），
   反之亦然，并被夹在 `[alpha_min, alpha_max]` 里。窗口和增益都设得很
   慢，是防止控制器把动力学“奴役”掉的关键。

> 这套实现的好处：直接把“同步轮数”作为雪崩持续时间，与 SOC 文献一致；
> 缺点是同步松弛在大型网格上比异步实现慢一些，但 32×32 的默认配置即使
> 在自适应+愈合模式下也只需 ~1 秒/千步。

---

## 2. `starter_simulation.py`：非临界态基线

这是玩家最先要跑的脚本，用一组**故意非临界**（`alpha=0.10`，远低于
`alpha_c ≈ 0.22`）的参数把模型跑一遍，并产出三张参考图：

| 图 | 内容 |
| --- | --- |
| `starter_mean_stress.svg` | 平均应力随时间的变化，并用红色虚线标出 `warmup` 边界 |
| `starter_avalanche_distribution.svg` | 同时画 PDF（对数分箱）和 CCDF，亚临界态下两条曲线都会快速衰减 |
| `starter_stress_field.svg` | 仿真结束时的应力场热力图，亚临界态会看到“离散小热斑”但没有大尺度结构 |

这一步的目的不是“做对”，而是让玩家先有“没调到临界”的对照感，再去
阶段一找“调到了临界”的样子。

> 玩家在阶段一一般不需要修改本脚本本身，而是**复制其结构**写自己的
> 实验入口。如果想在 base 上快速验证“调高 `alpha` 会怎样”，最方便的
> 做法是新建一个脚本，从 `case2.base` 导入 `FaultStressSystem` /
> `FaultParams`，改几个数字再跑。

---

## 3. `plotting.py`：matplotlib 绘图 + 三类统计工具

延续 case1 的风格，所有图都走 matplotlib + Agg 后端 + SVG 输出，可以
在浏览器或编辑器里直接打开。在 case1 的两个绘图函数之外，新增了三件
**地震 SOC 专用的统计工具**：

### 3.1 通用绘图（与 case1 一致或类似）

- `plot_lines(out, series, ...)`：折线/散点画图函数。比 case1 多了
  `hline=` 参数（画水平参考线）和 `alpha=` 透明度。
- `plot_dual_axis(out, x, left, right, ...)`：双 y 轴折线图。阶段二里
  把“平均应力”画在左轴、“自适应 alpha”画在右轴，量级差很大也能看清。
- `plot_heatmap(out, field, ...)`：把二维场画成热力图，固定 colormap
  和 colorbar。便于直观看应力的空间组织。

### 3.2 `log_hist(data, bins=36)` —— 对数分箱概率密度

和 case1 完全一致：把整数雪崩大小列表变成“对数分箱密度估计”。返回
`(centers, probs)`，可以直接喂给 `plot_lines`。

### 3.3 `ccdf_log(data)` —— 互补累积分布

> 拟合幂律时**首选 CCDF 而不是 PDF**：CCDF 不依赖分箱、噪声小、对尾部
> 形状更敏感。对于 `P(s) ∝ s^{-τ}`，`P(S ≥ s) ∝ s^{-(τ-1)}`，在双对数
> 坐标下仍然是直线，斜率比 PDF 平 1。

返回 `(s, p)`，只保留正样本与不同的 `s` 值，方便后续作图。

### 3.4 `power_law_mle(data, smin=1)` —— 离散幂律 Hill 估计

用经典的 Hill / Clauset–Shalizi–Newman 离散修正估计幂律指数 τ：
`τ ≈ 1 + n / Σᵢ ln(sᵢ / (smin - 0.5))`。返回 `(tau, n_used)`；样本不足
时返回 `(NaN, n)`。阶段一末尾用它打印三段对比的 τ。

### 3.5 `aftershock_rate(steps, sizes, ...)` —— Omori 余震率叠加

输入事件时间序列 + 大小序列，按分位数 `mainshock_quantile`（默认 0.95）
挑出主震，然后对每个主震开一个 `window` 长度的窗口，把窗口内事件按
`Δt = t - t_main` 做对数分箱计数，最后除以主震数和箱宽得到平均余震率
`n(Δt)`。

> 临界附近 OFC 大致满足 `n(Δt) ∝ Δt^{-p}`，`p ≈ 1`；亚临界态则会非常
> 平。这是阶段一“把模型推到临界”的第二条证据线。

返回 `(centers, rates, n_mainshocks)`，可以直接喂给 `plot_lines`。

---

## 4. 推荐的开发节奏

1. **先跑** `python case2/base/starter_simulation.py`，确认 `figures/`
   下三张 starter 图能正常生成，并感受亚临界态的形状。
2. 阶段一新建实验脚本，多次调用 `FaultStressSystem(params).run()`，配合
   `ccdf_log` / `power_law_mle` 找到 `alpha_c` 附近的“直线区间”和
   τ 值；再用 `aftershock_rate` 跑 Omori 律。
3. 阶段二在 `FaultParams` 中开 `adaptive=True` + `heterogeneity > 0`，
   用 `plot_dual_axis` 同时观察应力和 `alpha` 的演化，再多跑两个
   `seed` 与 `alpha` 初值的组合，验证 SOC 的鲁棒性。

> 如果你需要更复杂的统计（误差棒、最大似然+KS 优化、多 seed 平均…），
> 直接在自己的脚本里 `import matplotlib.pyplot as plt` 即可——
> `plotting.py` 只把最常用的画图与统计工具封装好，并不限制你扩展。
