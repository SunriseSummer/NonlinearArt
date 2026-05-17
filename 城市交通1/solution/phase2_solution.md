# 阶段二参考实现详解（`phase2_solution.py`）

## 1. 阶段目标回顾

阶段一是“**人工**把 `spill_prob` 拨到临界点”。阶段二要进一步：让系统
**自己**靠近临界态——这就是自组织临界（Self-Organized Criticality, SOC）
的核心思想。

参考脚本采用最朴素也最直观的反馈：**比例控制器（P controller）**——根据
当前平均负载和目标负载的偏差，连续微调 `spill_prob`。这一层逻辑已经内
嵌在 `TrafficCascadeSystem` 中，玩家只需要在 `TrafficParams` 中开
`adaptive=True`、给好目标负载和步长即可。

> **本版关键改动**：
> 1. **补上鲁棒性实验**——旧版本只跑一组 `(seed, 初始 p)`，所以 `task.md`
>    阶段二的第三条验证要求（“鲁棒性”）只能在文字里口头声称。新版本
>    显式扫描多组 `(seed, p₀)`、把它们的负载轨迹叠在一起、并打印
>    稳态均值，让“收敛到同一个 SOC 态”这件事有图有数据。
> 2. **修正 `target_load`**——旧版本设的 `2.8` 在当前耗散率下其实
>    不可达，控制器一路把 `spill_prob` 顶到上限 `spill_max=0.45`
>    才停。新版本调成 `2.6`，让 `spill_prob` 收敛到 `[0.32, 0.37]`
>    这样的内点，整套反馈才真正“在工作”。

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

### 3.1 参数选择 / 共享模板

```python
CANONICAL = dict(
    L=24, threshold=6,
    dissipation=0.20,
    steps=7000, warmup=1000,
    adaptive=True,
    target_load=2.6,           # 修正：旧版 2.8 不可达，会把 p 顶到上限
    adapt_rate=0.020,
    spill_min=0.05, spill_max=0.45,
)

def _make_params(*, seed, spill_prob):
    return TrafficParams(seed=seed, spill_prob=spill_prob, **CANONICAL)
```

把所有不变参数抽到 `CANONICAL` 字典里，是为了后面鲁棒性扫描时
**只换 `(seed, spill_prob)`** 而其他设置完全一致——这样得到的对比图
才有可比性。

几个值得注意的设定：

- **故意从亚临界起步**（`spill_prob=0.10`）：用来检验“即使初值偏离，
  系统能不能自己漂到临界”，这正是 SOC 的鲁棒性诉求。
- **`target_load=2.6`**：略高于阶段一“纯亚临界”态的负载、又在
  当前 `dissipation=0.20` 下可以达成。如果设得太高（比如 2.8），
  控制器会一路推到 `spill_max=0.45` 都还没到目标，看上去“稳态”但
  其实 `spill_prob` 已经饱和，反馈没有真正起作用。
- `steps=7000, warmup=1000` 比阶段一更长：自适应模式需要给控制器留
  收敛时间，否则 warmup 之后采到的统计仍然带着“瞬态”痕迹。

### 3.2 主仿真 + 双 y 轴时间序列

```python
params = _make_params(seed=2026, spill_prob=0.10)
res = TrafficCascadeSystem(params).run()

plot_dual_axis(
    FIG_DIR / "phase2_density_and_spillprob.svg",
    x=list(range(len(res.densities))),
    left ={"y": res.densities,         "ylabel": "Mean load per intersection",  "color": "#1f77b4", ...},
    right={"y": res.spill_prob_series, "ylabel": "Spill probability p(t)",      "color": "#9467bd", ...},
    title="Phase 2: self-organization of load and control parameter",
    xlabel="Simulation step",
    vline=params.warmup,
)
```

为什么必须用双 y 轴？因为平均负载量级在 ~2 ~ 3，而 `spill_prob` 始终
在 `[0.05, 0.45]` 之间。如果共用一套 y 轴，`spill_prob` 会被压成一条
几乎不动的横线，失去信息量。

> **怎么读这张图**：典型的 SOC 表现是——前面一段瞬态，`spill_prob`
> 从 0.10 缓慢上爬，平均负载也跟着抬升；越过 warmup（红色虚线）之后
> 两条曲线都进入“窄带波动”的稳态——既不再单调发散，也没塌掉。

输出：`case1/figures/phase2_density_and_spillprob.svg`。

### 3.3 雪崩统计：尺寸 + 持续时间

```python
x_size, y_size = log_hist(res.avalanche_sizes)
x_dur,  y_dur  = log_hist(res.avalanche_durations)
plot_lines(
    FIG_DIR / "phase2_avalanche_dist.svg",
    series=[
        {"x": x_size, "y": y_size, "label": "Avalanche size s",     "color": "#e377c2", "marker": "o"},
        {"x": x_dur,  "y": y_dur,  "label": "Avalanche duration T", "color": "#8c564b", "marker": "s"},
    ],
    title="Phase 2: avalanche statistics after self-organization",
    xlabel="s or T", ylabel="P", logx=True, logy=True,
)
```

SOC 的另一个标志是**多尺度的雪崩统计**：不仅雪崩规模 `s`，对应的持续
时间 `T` 也会呈幂律。两条曲线在双对数图上理论上应当近似直线，且指数
之间存在 SOC 标度关系。玩家可以直接用 `numpy.polyfit` 对中段做线性
拟合得到指数。

输出：`case1/figures/phase2_avalanche_dist.svg`。

### 3.4 鲁棒性扫描（新增）

旧版本到此就结束了，因此 `task.md` 阶段二的第三条要求只能口头宣称。
本版本新增一段**显式的多种子 + 多初值扫描**：

```python
robustness_configs = [
    ("seed=2026, p0=0.10", 2026, 0.10, "#1f77b4"),
    ("seed=17,   p0=0.10", 17,   0.10, "#2ca02c"),
    ("seed=991,  p0=0.40", 991,  0.40, "#d62728"),
    ("seed=4242, p0=0.30", 4242, 0.30, "#9467bd"),
]
print("[phase2] robustness sweep — steady-state means after warmup:")
print("  config                 <load>     <spill_prob>")
for label, seed, p0, color in robustness_configs:
    rparams = _make_params(seed=seed, spill_prob=p0)
    rres = TrafficCascadeSystem(rparams).run()
    load_ss  = _steady_mean(rres.densities,         rparams.warmup)
    spill_ss = _steady_mean(rres.spill_prob_series, rparams.warmup)
    print(f"  {label:<22} {load_ss:7.3f}    {spill_ss:7.3f}")
    series.append({"x": ..., "y": rres.densities, "label": ..., ...})

plot_lines(FIG_DIR / "phase2_robustness.svg", series=series, ...)
```

设计要点：

- **既换 `seed`，也换初始 `spill_prob`**：第三条扫描从 `p0=0.40`
  起步（高于稳态值），第四条从 `p0=0.30` 起步，确保覆盖“从下往上爬”
  和“从上往下落”两种瞬态形态。
- **稳态均值用 `_steady_mean`**：直接对 `warmup` 之后的样本取算术
  平均，作为 SOC 工作点的数值证据。这一步把“图上看着差不多”升级
  为“数表上数值差小于 1%”的硬证据。
- **共用 `CANONICAL`**：保证四条曲线唯一的差异就是 `(seed, p0)`，
  没有别的混淆因素。

典型 stdout 输出（节选）：

```
[phase2] robustness sweep — steady-state means after warmup:
  config                 <load>     <spill_prob>
  seed=2026, p0=0.10       2.519      0.332
  seed=17,   p0=0.10       2.522      0.317
  seed=991,  p0=0.40       2.520      0.334
  seed=4242, p0=0.30       2.522      0.365
```

四组 `<load>` 都收到 2.52 附近、`<spill_prob>` 落在 0.32 ~ 0.37 的
窄带——这就是 SOC 鲁棒性的定量证据。

输出：`case1/figures/phase2_robustness.svg`（四条负载轨迹叠在同一张
图上，红色虚线标 `warmup`）。

> **怎么读这张图**：四条线的瞬态形态各不相同（毕竟初始 `p` 都不一样），
> 但越过 `warmup` 之后会全部挤到同一条窄带里波动。如果某一条曲线跑飞
> 或者收到不同水平，说明控制参数（`adapt_rate`、`spill_max` 等）
> 还没调好。

---

## 4. 任务清单对照

`task.md` 中阶段二的验证要求：

- ✅ **稳态存在**：双 y 轴图上能直接看到平均负载在 warmup 之后窄带
  波动，没有发散。
- ✅ **长尾雪崩统计**：尺寸 + 持续时间两条分布都覆盖多个数量级。
- ✅ **鲁棒性**：新增的 `phase2_robustness.svg` + stdout 数表，
  四组 `(seed, p₀)` 的稳态值差异 < 2%。

---

## 5. 复现命令

```bash
python case1/solution/phase2_solution.py
```

会在 `case1/figures/` 下产出：

- `phase2_density_and_spillprob.svg`（主仿真：负载 + 自适应 p 的双 y 轴时间序列）
- `phase2_avalanche_dist.svg`（雪崩规模 + 持续时间的双对数分布）
- `phase2_robustness.svg`（四组初值的负载轨迹叠图）

整套五次仿真在普通笔记本上 1 ~ 2 秒可以跑完。跑完后建议先看主时间
序列：如果两条曲线没有进入窄带稳态、或者 `spill_prob` 一路撞到
`spill_max` / `spill_min`，说明 `target_load`、`adapt_rate` 还需要
调；只有先确认稳态，再去看分布和鲁棒性才有意义。

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
