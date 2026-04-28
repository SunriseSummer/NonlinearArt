# Phase 2 解读：用涨落峰值定位相变

本阶段扫描温度 `T ∈ [1.60, 3.20]`，并比较 `L=16/24/32` 三种尺寸。统计 warmup 之后的：

- `⟨|m|⟩`：序参量，随温度升高从有序跨到无序；
- `χ=N·Var(m)/T`：磁化率，临界附近峰值最明显；
- `C=N·Var(E/N)/T²`：比热代理量，作为能量涨落证据。

输出图：

- `../figures/phase2_temperature_sweep.svg`：`|m|(T)` 下降曲线。
- `../figures/phase2_susceptibility.svg`：`χ(T)` 峰值，竖线标出精确 `T_c≈2.269`。
- `../figures/phase2_finite_size.svg`：不同尺寸的能量涨落对照。

阅读方式：有限系统不会出现无限尖峰，峰会被圆滑化并随 `L` 改变；但峰值集中在 `T_c` 附近，就是相变的可测指纹。
