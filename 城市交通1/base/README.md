# Base 初始素材脚本详解

本目录是题目方提供给玩家的“数字孪生”脚手架，是**对题目场景的建模**。
玩家的两个阶段都必须基于这里的三份脚本继续开发，不要另起炉灶。

```
case1/base/
├── traffic_model.py        # 共用场景模型
├── starter_simulation.py   # 非临界态基线脚本（必须先跑）
└── plotting.py             # 基于 matplotlib 的轻量绘图与统计工具
```

下文按“先看模型 → 再看怎么跑 → 最后看怎么画图”的顺序逐个讲透。

---

## 1. `traffic_model.py`：场景核心模型

这是一个**二维 L×L 路口网格**上的“慢驱动 + 快松弛”模型，结构上和经典的
BTW 沙堆模型同源，但参数语义换成了交通：

- **慢驱动（drive）**：每一个仿真步在某个随机路口加 1 辆车。
- **快松弛（relax）**：一旦路口的负载达到阈值 `threshold`，路口会“倾倒”
  ——负载下降 `threshold`，并尝试把压力传给四个邻居。
- **耗散（dissipation）**：有一定概率车辆直接离开系统，模拟不再进入下游
  路口的能量耗散。

### 1.1 `TrafficParams` —— 玩家最该关心的旋钮

```python
@dataclass
class TrafficParams:
    L: int = 24                 # 路网边长（路口总数 L*L）
    threshold: int = 6          # 触发倾倒的负载阈值
    spill_prob: float = 0.12    # 倾倒时每个邻居被波及的概率
    dissipation: float = 0.15   # 单次外溢尝试被丢弃的概率
    steps: int = 5000           # 总仿真步数
    warmup: int = 1000          # 进入统计前的预热步数
    seed: int = 42              # 随机种子，保证实验可复现

    # —— 自适应控制器（阶段二会用到）——
    adaptive: bool = False      # 是否开启自适应
    target_load: float = 2.7    # 目标平均负载
    adapt_rate: float = 0.015   # 比例控制器的步长
    spill_min: float = 0.05     # spill_prob 的下界
    spill_max: float = 0.45     # spill_prob 的上界
```

> 阶段一的核心调参对象是 `spill_prob`。`threshold`、`dissipation` 也可
> 联合微调；`L`、`steps`、`warmup`、`seed` 多用于对齐实验条件。

### 1.2 `TrafficRunResult` —— 仿真返回的四条时间序列

```python
@dataclass
class TrafficRunResult:
    densities: list[float]          # 每一步结束时的平均负载
    avalanche_sizes: list[int]      # 每次非平凡级联的总倾倒次数（仅 warmup 之后）
    avalanche_durations: list[int]  # 对应级联的同步松弛轮数
    spill_prob_series: list[float]  # 每一步使用的 spill_prob（自适应模式下会变）
```

注意几条统计的差异：
- `densities` / `spill_prob_series` 长度都是 `steps`，可以直接画时间序列；
- `avalanche_sizes` / `avalanche_durations` 只收集 `warmup` 之后**且非零**
  的级联事件，因此长度通常远小于 `steps`，是 SOC 统计图的素材。

### 1.3 `TrafficCascadeSystem` —— 仿真主类

构造函数把网格初始化为全零，并用 `random.Random(seed)` 创建一个**独立**
的随机数发生器（不污染全局 `random`，方便并行/复现）。

#### `_drive(self)`
随机在 `(i, j)` 处加 1 辆车，返回该坐标。这是“慢驱动”的唯一入口。

#### `_relax(self, seed_cell)`
负责把一个不稳定路口引发的连锁反应跑完。它实现的是**同步**松弛：

1. 如果种子路口尚未达到阈值，直接返回 `(0, 0)`，对应“没有级联”。
2. 否则把它放进 `frontier` 集合，进入主循环：
   - 每一轮把当前 `frontier` 整体当作 `unstable`，全部倾倒；
   - 每个倾倒减 `threshold`，然后对四个邻居各做一次外溢尝试：
     - 以 `dissipation` 概率失败（耗散）；
     - 否则再以 `1 - spill_prob` 的概率失败（没有传递）；
     - 成功才把 1 辆车加到邻居上，并在越界检查通过后判断邻居是否也要加入下一轮 `frontier`；
   - 倾倒完成后，如果该路口本身负载仍超阈值（极端情况下一次倾倒不足以
     稳定，例如累积过多），把它继续放进 `frontier`。
3. 返回 `(size, duration)`：`size` 累计的是**倾倒事件数**（一个路口可能
   多次倾倒，都计入），`duration` 是松弛持续了多少同步轮。

> 这套实现的好处：直接把“同步轮数”作为雪崩持续时间，与 SOC 文献中常用
> 定义一致；缺点是同步松弛在大型网格上会比异步实现慢一些，但 24×24 的
> 默认配置完全够用。

#### `run(self)`
把 `_drive` 和 `_relax` 串起来：

1. 若开启 `adaptive`，先用比例控制器更新 `spill_prob`：
   ```
   err = target_load - mean_load
   spill_prob = clip(spill_prob + adapt_rate * err, spill_min, spill_max)
   ```
   逻辑上是“负载偏低 → 提高传播概率拉高负载；负载偏高 → 降低传播概率
   抑制级联”。这是阶段二最朴素的 SOC 反馈机制。
2. 调 `_drive`，再调 `_relax`，记录 `densities` / `spill_prob_series`；
3. 在 `warmup` 之后，仅当本步真的发生过倾倒（`size > 0`）才记录到雪崩
   统计中——这一点对画幂律分布很关键，否则会被海量“0 大小”事件淹没。

---

## 2. `starter_simulation.py`：非临界态基线

这是玩家最先要跑的脚本，用一组**故意非临界**的参数把模型跑一遍，并产出
两张参考图，供大家直观感受“没调到临界”是什么样子。

```python
params = TrafficParams(
    L=24, threshold=6,
    spill_prob=0.12,        # 偏低，倾向亚临界
    dissipation=0.20,       # 略高耗散，进一步抑制级联
    steps=5000, warmup=1000,
    seed=2026,
    adaptive=False,
)
result = TrafficCascadeSystem(params).run()
```

随后调用两次 `plot_lines`：

1. **`starter_density_timeseries.svg`**：平均负载的时间序列，并用红色虚线
   标出 `warmup` 边界，让玩家直观看到“预热到了哪里、之后是不是稳态”。
2. **`starter_avalanche_distribution.svg`**：用 `log_hist` 对 `avalanche_sizes`
   做对数分箱，再用双对数坐标画出 `P(s)` 分布。亚临界态下分布会快速衰减、
   尾部短，方便玩家在阶段一和“真正的幂律”做对照。

脚本结尾打印当前记录到的非平凡级联条数，便于检查样本量是否足够。

> 玩家在阶段一一般不需要修改本脚本本身，而是**复制其结构**写自己的实验
> 入口。如果想在 base 上快速验证“调高 spill_prob 会怎样”，最方便的做法
> 是新建一个脚本，从 `case1.base` 导入 `TrafficCascadeSystem` / `TrafficParams`，
> 改几个数字再跑。

---

## 3. `plotting.py`：matplotlib 绘图 + 对数分箱直方图

### 3.1 `plot_lines(out, series, title, xlabel, ylabel, *, logx, logy, vline)`

最常用的折线/散点画图函数：
- `series` 是一个列表，每个元素是 `{x, y, label, color?, marker?, linestyle?, linewidth?}`，
  方便一次画多条曲线（阶段一的“三段对比”就靠它）；
- `logx` / `logy` 切换对数轴；
- `vline` 会在指定 x 处画一条红色虚线，常用来标记 `warmup` 或临界点；
- 内部固定使用 `Agg` 后端 + `tight_layout`，无需图形界面即可保存为 SVG。

### 3.2 `plot_dual_axis(out, x, left, right, title, xlabel, *, vline)`

阶段二会用到的双 y 轴折线图。`left` / `right` 各是 `{y, label, ylabel, color}`，
分别画在左右两个独立 y 轴上，并且坐标轴刻度颜色与曲线一致，方便辨认。
平均负载（量级 ~3）和 `spill_prob`（量级 ~0.1）就靠它共用一个时间轴。

### 3.3 `log_hist(data, bins=36)`

把整数雪崩大小列表变成“对数分箱密度”，是画幂律分布的标准做法：

1. 过滤掉非正值（避免 `log10(0)`）。
2. 取 `[log10(min), log10(max)]` 区间，等距切 `bins` 个箱。
3. 每个箱里计数后除以 **(总数 × 箱宽)**，得到密度估计 `P(s)`，
   这样在双对数坐标下不会因为高 `s` 区间天然变宽而失真。
4. 防御边界情况：样本不足两个、或所有正样本相等时返回两个空列表，
   让调用方安全画空图而不会崩。

返回值是 `(centers, probs)`，可以直接喂给 `plot_lines`。

---

## 4. 推荐的开发节奏

1. **先跑** `python case1/base/starter_simulation.py`，确认 `figures/`
   下两张 starter 图能正常生成，并感受非临界态的形状。
2. 阶段一新建实验脚本，多次调用 `TrafficCascadeSystem(params).run()`，
   配合 `plot_lines` 画“mean cascade vs p”和分布对比图。
3. 阶段二在 `TrafficParams` 中开 `adaptive=True`，或在模型外再加一层
   反馈逻辑；用 `plot_dual_axis` 同时观察负载和控制参数的演化。

> 如果你需要更复杂的统计（指数拟合、误差棒、多 seed 平均…），
> 直接在自己的脚本中导入 `matplotlib.pyplot` 即可——`plotting.py` 只是
> 把最常用的两类图封装好，不会限制你扩展。
