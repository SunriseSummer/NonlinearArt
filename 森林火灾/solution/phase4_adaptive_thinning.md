# Phase 4：自适应疏伐——保产量，砍尾部

## 目标

阶段三告诉我们：放任 SOC 系统会出现**幂律巨灾**。林业局不接受"反正长期看是临界态就好"——他们关心 P95 / P99 / max 这种尾部指标。本阶段问：

> 用一个**只看本地**、**只能慢慢清理**的策略，能不能在不冻结森林的前提下把巨型火灾尾部压低？

## 实验设计

控制变量：完全相同的驱动 `p_grow=0.02, p_lightning=1e-4`，相同 seed 集合。
唯一区别：是否打开 `adaptive_thinning=True`。

策略参数：

```python
thinning_radius = 3        # 7×7 邻域
thinning_threshold = 0.62  # 邻域树密度高于该值视为高风险
thinning_rate = 0.05       # 高风险格每 sweep 5% 概率被清掉
```

3 个 seed × 6000 sweep，统计每次火灾的尺寸分布、总损失、总疏伐收获。

## 关键观察

典型输出：

```
[phase4] # fires: laissez=2720, managed=2695
[phase4] burnt    : laissez=866825, managed=758620
[phase4] harvested: managed=115831
[phase4] P95: laissez=1030.7, managed=884.6
[phase4] P99: laissez=1669.9, managed=1321.2
[phase4] max: laissez=2761,   managed=2685
```

主要结论：

1. **尾部明显被砍**：P95 下降 14%、P99 下降 21%；
2. **总烧损下降** 12%，同时疏伐**多收获 11.6 万棵树**（变成可用木材，而不是被烧掉的炭）；
3. **没有冻结系统**：管理策略下仍有几千次火灾、最大火灾仍上千棵——系统依然在临界附近，只是巨灾尾部被切掉。

## 产出图

- `phase4_density_compare.svg`：放任 vs 管理两条树密度时间序列；
- `phase4_burning_compare.svg`：燃烧细胞时间序列（管理策略下高峰更扁）；
- `phase4_size_distribution_compare.svg`：火灾尺寸 PDF（对数对数），管理曲线在尾部更陡；
- `phase4_tail_p95.svg` / `phase4_tail_p99.svg` / `phase4_max_fire.svg`：尾部指标对比；
- `phase4_loss_vs_yield.svg`：被烧损失 vs 疏伐收获的总账本。

## 为什么这一阶段重要

- **只用本地信息就能改变全局尾部**——这是 SOC 反馈控制最迷人的地方。林业局不需要装满全省的传感器、也不需要中央计算谁该烧谁不该烧。
- 体现"自组织 + 工程干预"的范式：保留自组织的鲁棒性（不靠精调全局密度），用一个本地策略**塑形**长尾。
- 引出局长会问的下一个问题：尾部被多砍 1% 值不值得多疏伐 10% 木材？这就是真实工程里的 risk-yield trade-off，已经超出了"教学版"任务，但路径已经铺好。
