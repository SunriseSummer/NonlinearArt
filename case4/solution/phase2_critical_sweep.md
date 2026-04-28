# Phase 2 解读：用涨落峰值定位相变

本阶段扫描温度 `T ∈ [1.60, 3.20]`，并比较 `L=16/24/32` 三种尺寸。统计 warmup 之后的：

- `⟨|m|⟩`：序参量，随温度升高从有序跨到无序；
- `χ=N·Var(m)/T`：磁化率，临界附近峰值最明显；
- `C=N·Var(E/N)/T²`：比热代理量，作为能量涨落证据。
- `U₄=1-⟨m⁴⟩/(3⟨m²⟩²)`：Binder cumulant，不同尺寸的曲线应在临界附近交叉。

输出图：

- `../figures/phase2_temperature_sweep.svg`：`|m|(T)` 下降曲线。
- `../figures/phase2_susceptibility.svg`：`χ(T)` 峰值，竖线标出精确 `T_c≈2.269`。
- `../figures/phase2_finite_size.svg`：不同尺寸的能量涨落对照。
- `../figures/phase2_binder_cumulant.svg`：Binder cumulant 多尺寸交叉，是独立于峰值高度的临界检验。

阅读方式：有限系统不会出现无限尖峰，峰会被圆滑化并随 `L` 改变；如果 `|m|` 跌落、`χ/C` 增强、Binder 曲线交叉都集中在 `T_c` 附近，才更有把握说系统确实处在相变临界区域。
