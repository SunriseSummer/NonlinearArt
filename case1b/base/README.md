# Base 初始素材脚本详解（case1b）

本目录是题目方提供给玩家的"早高峰数字孪生"脚手架。
case1b 的所有阶段都基于这里的三份脚本继续开发，**不要另起炉灶**。

```
case1b/base/
├── traffic_model.py        # 共用场景模型（重构后）
├── starter_simulation.py   # 远离临界态基线脚本（必须先跑）
├── plotting.py             # 基于 matplotlib 的轻量绘图与统计工具
└── __init__.py
```

下文按 "先看模型 → 再看怎么跑 → 最后看怎么画图" 的顺序逐个讲透。

---

## 1. `traffic_model.py`：场景核心模型

仍然是一个**二维 L×L 路口网格**上的"慢驱动 + 快松弛"模型，骨架沿用
case1（同源于 BTW 沙堆模型），但围绕"早高峰调度员"这个故事做了**关键重构**：

- 入口车流量 `inflow_rate` 不再固定 1 辆/步，可以是任意正实数（用四舍
  五入 + 伯努利残差实现），方便阶段二把"早高峰强度"作为扫描变量。
- 新增**吞吐量** `throughput`：每一步实际离开网络的车辆数。
- 新增**拥堵传播范围** `congestion_range`：每一步处于阈值之上的路口数
  （在 `_drive` 之后、`_relax` 之前采样，捕捉"峰值瞬间"）。
- 新增**事故扰动** `DisturbanceSpec`：在指定时刻把以某中心格为中心、
  半径 `radius` 的方块路口冻结 `duration` 步（这些路口在冻结期间不会
  倾倒，模拟事故占用车道）。
- 新增**两条局部反馈规则**（默认关闭，由阶段四开启）：
  - `inflow_feedback`：当 `<load> > target_load` 时按比例收紧 `inflow_rate`；
  - `local_relief`：高负载路口的 `dissipation` 临时上调，模拟"出口被加大"。
- 保留 case1 的 `adaptive_spill` 比例控制器（可选），方便对照实验。

### 1.1 `TrafficParams` —— 玩家最该关心的旋钮

```python
@dataclass
class TrafficParams:
    L: int = 20
    threshold: int = 6
    inflow_rate: float = 1.0      # 入口车流量
    spill_prob: float = 0.18      # 路口扩散概率
    dissipation: float = 0.20     # 单次扩散尝试被丢弃的概率
    steps: int = 6000
    warmup: int = 1000
    seed: int = 2026

    disturbance: Optional[DisturbanceSpec] = None

    # 局部反馈规则（阶段4）
    local_relief: bool = False
    relief_extra: float = 0.20
    relief_load_frac: float = 0.8

    inflow_feedback: bool = False
    target_load: float = 2.4
    inflow_min_factor: float = 0.2
    inflow_gain: float = 0.6

    # 强控制比例式自适应（可选）
    adaptive_spill: bool = False
    adapt_rate: float = 0.020
    spill_min: float = 0.05
    spill_max: float = 0.45
```

### 1.2 `DisturbanceSpec` —— 事故事件

```python
@dataclass
class DisturbanceSpec:
    start: int          # 事故发生的仿真步
    duration: int       # 持续步数
    cell: tuple|None    # 事故中心，缺省取网格中心
    radius: int = 0     # 0 表示单个路口；1 表示 3x3 街区
```

事故期间，被冻结的路口**会继续接收车辆**，但**不会倾倒**，因此压力会
向四周积累——这正是真实事故的传导机制。

### 1.3 `_relax`：松弛的关键步骤

每次 `_drive()` 把若干辆车注入随机路口，然后 `_relax(seeds)` 让超过阈值
的路口开始倾倒：

1. `cell -= threshold`；其中 `threshold - 4` 辆车直接计入 `served`
   （视作本路口的快速出口）。
2. 对四个邻居方向各做一次扩散尝试：
   - 以 `local_diss` 概率耗散（车辆离开网络，`served += 1`）；
   - 否则以 `1 - spill_prob` 概率扩散失败（车辆被吸收/不再追踪）；
   - 否则若邻居在网格内则把 1 辆车加给邻居；若越界，`served += 1`。
3. 邻居加车后若 ≥ 阈值则进入新一轮松弛，直到稳定。

`size` 累计本次松弛中所有倾倒事件，`duration` 统计同步轮数——这两个量
正好用于阶段2 的级联规模分布分析。

### 1.4 主循环：`run()`

```python
for t in range(steps):
    if adaptive_spill: spill_prob ← 比例控制器
    if inflow_feedback: inflow_rate ← f(<load>)
    maybe_start_disturbance(t)
    seeds = drive()
    congestion_range[t] = #cells with load >= threshold  ← 在松弛之前采样
    size, duration, served = relax(seeds)
    tick_disturbance()
    densities[t]  = mean_load
    throughput[t] = served
    ...
```

注意：`congestion_range` 在松弛**前**采样——松弛后所有不稳定路口都已倾倒，
"高负载快照"必须发生在松弛之前才有意义。

---

## 2. `starter_simulation.py`：远离临界态基线

这一脚本展示**保守调度**下的网络：

```python
TrafficParams(
    L=20, threshold=6,
    inflow_rate=1.0,
    spill_prob=0.10,
    dissipation=0.25,
    steps=5000, warmup=800,
    seed=2026,
)
```

会输出两张图（写入 `case1b/figures/`）：

- `starter_load_throughput.svg`：双 y 轴展示平均负载与吞吐量的时间序列。
- `starter_congestion_range.svg`：拥堵传播范围的时间序列。

终端会打印稳态指标，便于对照阶段一的扫描结果。

---

## 3. `plotting.py`：绘图工具

只用 matplotlib，**不引入新的依赖**。提供的工具：

- `plot_lines(out, series, …, vlines=…)`：单 y 轴多曲线，支持对数坐标和
  多条事件参考线。
- `plot_dual_axis(out, x, left, right, …)`：双 y 轴，专门用来对比量纲不同
  的两组指标（负载 vs 吞吐量、负载 vs 自适应概率 等）。
- `plot_bars(out, categories, values, …)`：分组柱状图，用于阶段一的入口
  车流量对比、阶段三的拥堵峰值对比、阶段四的策略总览。
- `rolling_mean(values, window)`：用于平滑高频抖动，让时间序列图直观可读。
- `log_hist(data, bins=32)`：对数分箱直方图，用于阶段二的级联规模分布。

所有 `plot_*` 函数都用 ASCII 标题——避免在没有中文字体的环境中出现
"missing glyph"警告，提交后他人复现也更稳定。

---

## 4. 怎么扩展（开发提示）

- **不要直接修改 `traffic_model.py`** —— 阶段一/二/三只调参，阶段四只
  打开局部反馈开关。如果一定要扩展，建议**新增字段并保持向后兼容**，
  默认值要让 `starter_simulation.py` 的输出不变。
- **保持随机性可复现** —— 所有脚本都要传入 `seed`；多种子取平均时记得
  用一组固定种子（参考实现里都列在脚本最上方）。
- **指标采样的位置很关键** —— `congestion_range` 必须在松弛前采样，
  `throughput` 必须在松弛中累加。改动主循环顺序前请先看明白这两点。
- **绘图风格统一** —— 复用 `plotting.py` 中的工具能让所有阶段的图保持
  视觉一致，方便阅卷者对比。
