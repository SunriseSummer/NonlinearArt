# Phase 3：自组织临界（Drossel–Schwabl）

## 目标

不再扫描参数，把森林交给"自然规则"：每一步空地以 `p_grow` 长树、每棵树以 `p_lightning` 被雷击。当时间尺度分离 `p_lightning << p_grow << 1` 时，**系统自己**演化到一个临界稳态，火灾尺寸分布是幂律。这就是 Drossel–Schwabl SOC——不需要任何外部调参员。

## 实验设计

主 SOC 实验：

```python
ForestParams(
    L=64, mode="soc", steps=4000,
    p_grow=0.02, p_lightning=1e-4,
    seed=2026,
)
```

`f/p` 比扫描：

| 比值 | `p_grow` | `p_lightning` | 预期 |
| --- | --- | --- | --- |
| 5e-2（差） | 0.02 | 1e-3 | 时间尺度分离不够，火太频繁，森林永远长不密 |
| 5e-3（典型 SOC） | 0.02 | 1e-4 | 干净幂律 |
| 5e-4（极慢） | 0.02 | 1e-5 | 几次罕见但极大的火 |

## 关键观察

典型输出：

```
[phase3] SOC run: 610 fires, <size>=316.1, max=2212
[phase3] steady tree density ~= 0.398
[phase3] f/p=5e-2: n_fires=4408, max=617,  <s>=72
[phase3] f/p=5e-3: n_fires=888,  max=2761, <s>=326
[phase3] f/p=5e-4: n_fires=127,  max=7197, <s>=1968
```

注意几件事：

1. SOC 稳态密度 ~0.40，**低于** phase 2 测出的 `p_c=0.5928`。这是因为 Drossel–Schwabl 模型的 SOC 临界点和静态渗流临界点**不是同一个**（属于不同普适类），但都属于"长程关联恰好建立"的物理图景。
2. `f/p` 越小，单次火灾越大、总火数越少，但**所有规模上**呈幂律。
3. 时间序列上有大量"低活动"时段间或被巨型火头打断——经典的 SOC 长尾活动。

## 产出图

- `phase3_density_timeseries.svg`：树密度自组织到稳态带；
- `phase3_burning_timeseries.svg`：燃烧细胞数的爆发模式（间歇 + 巨峰）；
- `phase3_size_distribution.svg`：三个 `f/p` 比的火灾尺寸 PDF（对数对数），附 `s^{-1.15}` 参考斜线；
- `phase3_duration_distribution.svg`：火灾持续时间的幂律。

## 为什么这一阶段重要

- 同一物理（树长 + 火烧）能展现**两种相变**：阶段二的渗流相变（外部调密度）与本阶段的 SOC（内部自适应到临界）。
- 体现 SOC 的关键条件：**时间尺度分离**。"f/p 不能太大"在工程上对应"防火响应必须比林木生长快得多"。
- 给阶段四埋伏笔：既然系统自己就能跑到长尾，能不能用本地策略**只压尾部、不破坏整体**？
