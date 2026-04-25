# 阶段二参考实现详解（`phase2_solution.py`）

## 1. 阶段目标回顾

阶段一是“**人工**把 `spill_prob` 拨到临界点”。阶段二要进一步：让系统
**自己**靠近临界态——这就是自组织临界（Self-Organized Criticality, SOC）
的核心思想。

参考脚本采用最朴素也最直观的反馈：**比例控制器（P controller）**——根据
当前平均负载和目标负载的偏差，连续微调 `spill_prob`。这一层逻辑已经内
嵌在 `TrafficCascadeSystem` 中，玩家只需要在 `TrafficParams` 中开
`adaptive=True`、给好目标负载和步长即可。

---

## 2. 反馈机制速览

`traffic_model.py` 的 `run()` 在每一步开头会执行：

```python
err = target_load - mean_load
spill_prob = clip(spill_prob + adapt_rate * err, spill_min, spill_max)
```

直觉解释：

- **当前负载偏低（err > 0）**：道路太空，扩大传播概率，让车流更容易
  外溢，从而提高活动度。
- **当前负载偏高（err < 0）**：道路堵了，压低传播概率，让倾倒更可能
  被耗散吸收，避免发散。
- `clip(..., spill_min, spill_max)` 保证 `spill_prob` 不会跑出物理上
  合理的区间，防止控制器“失控”。

注意这是一个**慢驱动 + 慢反馈**的设计：`adapt_rate` 远小于 1，意味着
`spill_prob` 不会因为某一次大雪崩剧烈跳变，而是在很多步上做平滑滑动，
这样的稳定动力学正是 SOC 系统能稳态存在的关键。

---

## 3. 脚本逐段解读

### 3.1 参数选择

```python
params = TrafficParams(
    L=24, threshold=6,
    spill_prob=0.10,           # 故意从亚临界起步
    dissipation=0.20,
    steps=7000, warmup=1000,
    seed=2026,
    adaptive=True,
    target_load=2.8,           # 目标平均负载
    adapt_rate=0.020,          # 比例步长
    spill_min=0.05, spill_max=0.45,
)
```

几个值得注意的点：

- **故意从亚临界起步**（`spill_prob=0.10`）：用来检验“即使初值偏离，
  系统能不能自己漂到临界”，这正是 SOC 的鲁棒性诉求。
- `target_load=2.8` 接近阶段一估出的临界点附近的稳态负载——选择稍高
  于阶段一“纯亚临界”态的负载即可，不必精确等于临界点；只要选在临界
  邻域内，控制器就会把系统拉到临界曲线上。
- `steps=7000, warmup=1000` 比阶段一更长：自适应模式需要给控制器留
  收敛时间，否则 warmup 之后采到的统计仍然带着“瞬态”痕迹。

### 3.2 仿真 + 双 y 轴时间序列

```python
res = TrafficCascadeSystem(params).run()

plot_dual_axis(
    FIG_DIR / "phase2_density_and_spillprob.svg",
    x=list(range(len(res.densities))),
    left ={"y": res.densities,         "label": "Mean load",
           "ylabel": "Mean load per intersection", "color": "#1f77b4"},
    right={"y": res.spill_prob_series, "label": "Adaptive spill probability",
           "ylabel": "Spill probability p(t)",     "color": "#9467bd"},
    title="Phase 2: self-organization of load and control parameter",
    xlabel="Simulation step",
    vline=params.warmup,
)
```

为什么必须用双 y 轴？因为平均负载量级在 ~3 附近，而 `spill_prob` 始终
在 `[0.05, 0.45]` 之间。如果共用一套 y 轴，`spill_prob` 会被压成一条
几乎不动的横线，失去信息量。`plot_dual_axis` 的左右轴刻度颜色与曲线
一致，便于一眼分辨哪条线对应哪个轴。

> **怎么读这张图**：典型的 SOC 表现是——前面一段瞬态，`spill_prob`
> 从 0.10 缓慢上爬，平均负载也跟着抬升；越过 warmup 之后两条曲线
> 都会进入“窄带波动”的稳态（不再单调发散，也不会塌掉）。

输出：`case1/figures/phase2_density_and_spillprob.svg`。

### 3.3 雪崩统计：尺寸 + 持续时间

```python
x_size, y_size = log_hist(res.avalanche_sizes)
x_dur,  y_dur  = log_hist(res.avalanche_durations)
plot_lines(
    FIG_DIR / "phase2_avalanche_dist.svg",
    series=[
        {"x": x_size, "y": y_size, "label": "Avalanche size s",
         "color": "#e377c2", "marker": "o"},
        {"x": x_dur,  "y": y_dur,  "label": "Avalanche duration T",
         "color": "#8c564b", "marker": "s"},
    ],
    title="Phase 2: avalanche statistics after self-organization",
    xlabel="s or T",
    ylabel="P",
    logx=True, logy=True,
)
```

SOC 的另一个标志是**多尺度的雪崩统计**：不仅雪崩规模 `s`，对应的持续
时间 `T` 也会呈幂律。这两条曲线在双对数图上理论上应当共同呈现近似直
线，并且它们的指数 τ、α 之间存在 SOC 标度关系（如 `(τ - 1) = α · (γ - 1) / γ` 这类标度律）。
玩家可以直接用 `numpy.polyfit` 在自己的脚本里拟合一段直线得到指数。

输出：`case1/figures/phase2_avalanche_dist.svg`。

---

## 4. 任务清单对照

`task.md` 中阶段二的验证要求：

- ✅ **稳态存在**：双 y 轴图上能直接看到平均负载在 warmup 之后窄带
  波动，没有发散。
- ✅ **长尾雪崩统计**：尺寸 + 持续时间两条分布都覆盖多个数量级。
- ☐ **鲁棒性**（参考实现未直接画图，但很容易加）：换不同 `seed` 或
  起始 `spill_prob`，再跑同样的脚本，应当看到 `(mean_load, spill_prob)`
  的稳态值几乎不变——玩家可以把多 seed 的稳态曲线叠在同一张图上证明。

---

## 5. 复现命令

```bash
python case1/solution/phase2_solution.py
```

会在 `case1/figures/` 下产出：

- `phase2_density_and_spillprob.svg`
- `phase2_avalanche_dist.svg`

跑完后建议先看时间序列那张图：如果两条曲线没有进入窄带稳态、或者 `spill_prob`
一路撞到 `spill_max` / `spill_min`，说明 `target_load`、`adapt_rate` 还
需要调；只有先确认稳态，再去看分布才有意义。

---

## 6. 进阶玩法建议

参考实现只示范了“最简单的一种 SOC 反馈”。玩家可以在 base 模型上叠
加更有创意的机制，例如：

- **动态阈值**：让局部高峰路口的 `threshold` 随负载变化（高峰时段
  容量临时下降）；
- **局部拥堵税**：高负载路口的 `dissipation` 提升（收费/限行倒逼
  车辆离开）；
- **网络拓扑改造**：把 4 邻居拓展为不规则路网，用真实城市图代替
  正方形栅格。

只要新的机制依然依赖 `TrafficCascadeSystem` 的 drive/relax 框架，
就可以复用 base 中的 `plotting.py` 和 `log_hist`，把对比图直接画出来。
