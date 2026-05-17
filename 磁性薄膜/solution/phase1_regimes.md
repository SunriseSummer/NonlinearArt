# Phase 1 解读：远离临界的两种迟钝

本阶段只改变热浴温度 `T`，不改 Ising 动力学规则：

- `T=1.60`：低温铁磁相，磁畴迅速锁成同向，`|m|` 高、翻转少。
- `T=2.30`：接近 `T_c≈2.269`，大团簇反复重排，`|m|` 与活动度都有明显波动。
- `T=3.60`：高温顺磁相，翻转频繁但互相抵消，整体 `m≈0`。

输出图：

- `../figures/phase1_magnetisation_regimes.svg`：三段 `|m|(t)` 对比。
- `../figures/phase1_activity_regimes.svg`：接受翻转数对比。
- `../figures/phase1_domains_cold.svg`、`phase1_domains_near.svg`、`phase1_domains_hot.svg`：最终磁畴快照。
- `../figures/phase1_steady_abs_m.svg`：稳态序参量柱状图。

核心结论：低温相稳定但不敏感，高温相活跃但没有长程记忆；真正有趣的多尺度涨落出现在两者之间。
