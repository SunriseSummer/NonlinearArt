# Base 初始素材脚本详解（Case 4）

本目录是题目方提供给玩家的“磁性薄膜临界实验台”脚手架。玩家在四个阶段都应基于这里的三份脚本继续开发，不要另起炉灶。

```
case4/base/
├── ising_model.py          # 共用 Ising 薄膜模型：Metropolis 动力学 + 场脉冲 + 反馈温控
├── starter_simulation.py   # 高温顺磁基线脚本（必须先跑）
├── plotting.py             # matplotlib 绘图与临界统计工具
└── __init__.py
```

---

## 1. `ising_model.py`：场景核心模型

模型是一张 `L×L` 正方晶格，每个格点是一枚自旋 `s_i ∈ {-1,+1}`，代表磁性薄膜中一个小磁畴的朝向。系统能量为

```
E = -J Σ_<ij> s_i s_j - h Σ_i s_i
```

其中 `J>0` 让相邻磁畴倾向同向排列，`h` 是外加磁场。脚本使用周期边界与单自旋 Metropolis 更新：每个 Monte Carlo sweep 随机尝试 `L²` 次翻转，按 `exp(-ΔE/T)` 接受升能翻转。

### 玩家最该关心的旋钮

- `temperature`：热浴温度，核心控制参数；二维 Ising 在 `J=1` 时 `T_c≈2.269`。
- `field`：外加磁场；阶段三会施加短脉冲测试“临界放大”。
- `L`：系统尺寸；阶段二用它验证有限尺寸效应。
- `initial_state`：`random/up/down`，用来测试是否存在磁滞与多初值收敛。
- `adaptive_temperature`：阶段四打开的反馈温控规则。

### 返回指标

`IsingRunResult` 会记录：

- `magnetisation` 与 `abs_magnetisation`：序参量 `m` 与 `|m|`；
- `energy`：每自旋能量；
- `temperature_series` / `field_series`：外部控制量轨迹；
- `accepted_flips`：每 sweep 接受的翻转数，可作为磁畴活动度；
- `final_spins`：最终自旋场快照。

---

## 2. `starter_simulation.py`：高温顺磁基线

先运行：

```bash
python case4/base/starter_simulation.py
```

默认 `T=3.6 > T_c`，因此系统处于顺磁态：局部小团簇不断闪烁，整体磁化 `m≈0`，没有长程有序。脚本输出：

- `figures/starter_hot_film_timeseries.svg`：`|m|` 与接受翻转数的双轴时间序列；
- `figures/starter_energy.svg`：能量时间序列；
- `figures/starter_final_domains.svg`：最终磁畴快照。

---

## 3. `plotting.py`：绘图与统计工具

只依赖 `matplotlib`。除通用折线、双轴、柱状图、磁畴快照外，还提供：

- `rolling_mean`：平滑热涨落；
- `susceptibility(m, T, N)`：磁化涨落给出的磁化率 `χ`；
- `heat_capacity(e, T, N)`：能量涨落给出的比热代理量 `C`；
- `binder_cumulant(m)`：四阶 Binder cumulant，用不同尺寸曲线交叉严格定位临界区域；
- `log_hist`：阶段四把“翻转雪崩/活动爆发”画成双对数分布。

---

## 4. 推荐开发节奏

1. 先跑 starter，确认高温顺磁基线与三张图。
2. 阶段一只改温度，直观看到低温铁磁 / 高温顺磁的差异。
3. 阶段二扫描 `T` 和 `L`，用 `χ(T)` 与 `C(T)` 峰值定位 `T_c`。
4. 阶段三施加同一弱磁场脉冲，比较远离临界与接近临界时的响应幅度。
5. 阶段四打开反馈温控，让系统从不同初温自动靠近临界附近，并检查活动分布是否比静态温度更长尾。
