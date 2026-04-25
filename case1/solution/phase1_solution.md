# 阶段一参考实现详解（`phase1_solution.py`）

## 1. 阶段目标回顾

阶段一要求**不改动模型机制**，只通过调参把 base 中的同一套
`TrafficCascadeSystem` 推到“临界态附近”，并给出统计学意义上的证据。
本参考脚本在两条独立证据线上发力：

1. **序参量曲线**：扫描 `spill_prob`，画“平均级联规模”随控制参数的变化，
   预期在临界点附近显著抬升甚至发散。
2. **三段分布对比**：在亚临界 / 临界附近 / 超临界三组参数下，画雪崩
   规模的双对数分布。临界点附近应出现一段近似直线（幂律）。

代码结构上完全复用 `case1/base/`：

```python
CASE1_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = CASE1_DIR / "base"
sys.path.insert(0, str(BASE_DIR))
from plotting import log_hist, plot_lines
from traffic_model import TrafficCascadeSystem, TrafficParams
```

这一段是“把 `case1/base/` 加入 import 路径”的标准写法，玩家自己写实验
脚本时可以照抄，让脚本既能用 `python xxx.py` 直接跑，也能从仓库根目录
执行。

---

## 2. 实验一：序参量扫描

```python
p_values = [0.08 + 0.02 * i for i in range(15)]   # 0.08 ~ 0.36，步长 0.02
mean_sizes = []

for p in p_values:
    params = TrafficParams(
        L=24, threshold=6,
        spill_prob=p,
        dissipation=0.20,
        steps=3500, warmup=800,
        seed=2026,
        adaptive=False,
    )
    res = TrafficCascadeSystem(params).run()
    mean_sizes.append(
        sum(res.avalanche_sizes) / max(len(res.avalanche_sizes), 1)
    )
```

要点：

- **采样区间**覆盖了从亚临界到超临界，确保曲线两端都能看到“低位平台”
  和“快速抬升”，临界点夹在中间。
- 每一个 `p` 都用同样的 `seed`，意味着扫描的是“纯参数效应”，不掺杂
  随机性差异；如果想要平均掉随机起伏，玩家可以自行多 seed 平均。
- `max(len(...), 1)` 是为了在极端亚临界（基本没有非平凡级联）时避免
  除零。
- `steps=3500, warmup=800` 比 starter 的 5000/1000 略短：扫描 15 个点
  时总时长更可控，又不至于让统计样本太少。

随后把 `(p, mean_sizes)` 喂进 `plot_lines`，并通过 `vline=0.26` 在图上
画一条红色虚线，提示玩家“临界点大约就在这里”。这是参考实现给出的经验
值（不同的 `dissipation`、`L`、`threshold` 会让临界点漂移，玩家应当
自己重新定位）。

输出：`case1/figures/phase1_mean_size_vs_spill_prob.svg`。

> **怎么读这张图**：在临界点之前，平均级联规模会维持在低位（很多倾倒
> 还没传开就被耗散）；越过临界点后会快速抬升甚至饱和（系统倾向于产生
> 横跨整张网格的“系统级”级联）。临界点的特征是“开始抬升的拐点”。

---

## 3. 实验二：三段分布对比

```python
compare = [
    ("Subcritical p=0.12",   0.12, "#2ca02c"),
    ("Near-critical p=0.26", 0.26, "#ff7f0e"),
    ("Supercritical p=0.34", 0.34, "#d62728"),
]
```

三组参数对应三种典型态：

| 状态 | `spill_prob` | 预期分布形状 |
| --- | --- | --- |
| 亚临界 | 0.12 | 衰减很快，几乎看不到大型雪崩，分布尾部缺失 |
| 临界附近 | 0.26 | 双对数下大段近似直线（幂律） |
| 超临界 | 0.34 | 出现“鼓包”：系统级雪崩堆积在大尺度处 |

每一组用同样的 `(L, threshold, dissipation, seed)`，只换 `spill_prob`，
最后用 `log_hist(res.avalanche_sizes)` 做对数分箱，再交给 `plot_lines`
在双对数坐标下画。

输出：`case1/figures/phase1_size_dist_compare.svg`。

> **怎么读这张图**：理想的临界曲线会在中间一段近似直线；亚临界的尾部
> 会在双对数图上明显“塌下去”；超临界则容易在大尺度处出现“凸起/截断”
> 的特征。

---

## 4. 任务清单对照

回顾 `task.md` 中阶段一的三条验证要求，本脚本一次性覆盖了其中两条：

- ✅ **级联规模分布**：实验二的三段分布对比图。
- ✅ **三段对比**：同上，亚 / 临界 / 超三种态并列。
- ✅ **序参量变化**：实验一的 `mean size vs p` 曲线。

> 玩家如果想拿满分，可以再补：多 seed 平均、对临界曲线做幂律拟合得到
> 指数 τ、扩展到 `(spill_prob, dissipation)` 的二维相图等。

---

## 5. 复现命令

```bash
python case1/solution/phase1_solution.py
```

完成后会在 `case1/figures/` 下生成：

- `phase1_mean_size_vs_spill_prob.svg`
- `phase1_size_dist_compare.svg`

整套扫描在普通笔记本上几秒到十几秒可以跑完；如果想加速，可以：
减小 `L`（注意太小会让有限尺度效应破坏幂律）、减小 `steps`、或
减少 `p_values` 的密度。
