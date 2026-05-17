# 阶段一参考实现详解（`phase1_solution.py`）

## 1. 阶段目标回顾

阶段一要求**不改动模型机制**，只通过调参把 base 中的同一套
`FaultStressSystem` 推到“临界态附近”，并给出统计学意义上的证据。本
参考脚本在三条独立证据线上发力（题目要求 ≥2 条）：

1. **序参量曲线**：扫描 `alpha`，画“平均级联规模”随控制参数的变化，
   预期在临界点附近显著抬升甚至发散。
2. **三段分布对比**（CCDF）：在亚临界 / 临界附近 / 超临界三组参数下，
   画雪崩规模的双对数互补累积分布。临界点附近应出现一段近似直线
   （**Gutenberg–Richter 幂律**），并用 Hill 估计读出 τ。
3. **Omori 余震律**：把临界与亚临界两套参数的“主震后余震率”叠加，
   临界附近应出现 `n(Δt) ∝ 1/Δt` 的直线段，亚临界则平坦得多。

代码结构上完全复用 `case2/base/`：

```python
CASE2_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = CASE2_DIR / "base"
sys.path.insert(0, str(BASE_DIR))
from fault_model import FaultParams, FaultStressSystem
from plotting import aftershock_rate, ccdf_log, plot_lines, power_law_mle
```

这一段是“把 `case2/base/` 加入 import 路径”的标准写法，玩家自己写实验
脚本时可以照抄。

---

## 2. 实验一：序参量扫描

```python
alphas = [0.10 + 0.01 * i for i in range(15)]   # 0.10 .. 0.24
for a in alphas:
    res = FaultStressSystem(FaultParams(
        L=32, alpha=a, drive_steps=4000, warmup=1000, seed=2026,
    )).run()
    mean_sizes.append(sum(res.sizes) / len(res.sizes))
```

- **为什么用 `mean cascade size` 当序参量？** OFC 普适类里，平均雪崩
  大小 `<s>` 在临界点附近以系统大小发散；在有限 32×32 网格上即使发散
  被截断，也能看到“随 `alpha` 单调拉起”的序参量曲线，足够定位临界点。
- 参考图 `phase1_mean_size_vs_alpha.svg` 中红色虚线 `alpha=0.22` 是
  我们的临界估计；曲线在 0.20~0.23 区间出现明显跳升。
- **常见坑**：`drive_steps` 不要太短，否则 `<s>` 估计噪声过大；
  `seed` 在所有扫描点保持一致，可以让曲线更光滑。

---

## 3. 实验二：三段 CCDF 对比

```python
compare = [
    ("Sub-critical alpha=0.12",   0.12, "#2ca02c"),
    ("Near-critical alpha=0.22",  0.22, "#ff7f0e"),
    ("Super-critical alpha=0.245", 0.245, "#d62728"),
]
```

- **为什么是 CCDF 不是 PDF？** PDF 要分箱，分箱方式（线性/对数）会显著
  影响“看上去是不是直线”，对小样本尤其敏感。CCDF 不依赖分箱，且对
  `P(s) ∝ s^{-τ}` 有 `P(S ≥ s) ∝ s^{-(τ-1)}` 的关系，斜率比 PDF 平 1。
- 阶段一参考脚本同时调用 `power_law_mle(res.sizes, smin=4)` 输出 Hill
  估计的 `τ`：
  - 亚临界 `alpha=0.12`：`τ ≈ 3.1`，分布陡峭、几乎指数衰减；
  - 临界附近 `alpha=0.22`：`τ ≈ 1.75`，正是 2-D OFC 普适类的典型值；
  - 超临界 `alpha=0.245`：`τ ≈ 1.5`，尾巴变缓且出现“凸起”——这是
    系统级（system-spanning）大事件主导的特征。
- 参考图 `phase1_size_dist_compare.svg` 中三条曲线从陡峭到平直再到
  凸起，是非常直观的“相变指纹”。

> **Gutenberg–Richter 关联**：地震学中震级–频率关系
> `log10 N(≥M) = a - b*M`，对应到能量 `E ~ 10^M` 时 b-value
> ≈ τ - 1。临界附近 τ ≈ 1.75 → b ≈ 0.75，与全球地震观测的
> b ≈ 0.9 ~ 1.0 同量级。

---

## 4. 实验三：Omori 余震律

```python
centers, rates, n_main = aftershock_rate(
    res.event_steps, res.sizes,
    mainshock_quantile=0.95, window=400, bins=20,
)
```

- 把所有事件按大小排序，取 95 分位以上的当成“主震”；对每个主震开一个
  长度为 400 宏观步的窗口，把窗口内事件按 `Δt` 对数分箱，最后除以主震
  数和箱宽得到平均余震率 `n(Δt)`。
- **要点**：临界附近因为存在长程关联，主震后会拉出一条 `n(Δt) ∝ 1/Δt`
  的余震尾巴；亚临界态下事件之间近似独立，余震率随 `Δt` 几乎不变。
- 参考图 `phase1_omori.svg` 上把临界与亚临界两条曲线放在一起：临界曲
  线（橙色）在 `Δt ∈ [3, 200]` 表现为漂亮的负斜率直线，亚临界曲线
  （绿色）则呈平台。
- **常见坑**：`drive_steps` 太短会让“主震后窗口”不够覆盖余震；本参考脚
  本特意把 `drive_steps` 增大到 10000 来稳住 Omori 信号。

---

## 5. 验收清单

- [x] 序参量曲线（`phase1_mean_size_vs_alpha.svg`）
- [x] 三段 CCDF + 临界 τ ≈ 1.7~1.8（`phase1_size_dist_compare.svg`）
- [x] 主震叠加余震率呈现 Omori 形（`phase1_omori.svg`）

题目要求至少 2 项，这里给出 3 项是为了让玩家有“怎么算合格”的对照。
玩家自己实现时只需稳定地做出其中两条即可。
