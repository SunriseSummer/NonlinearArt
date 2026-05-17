# Case 5：森林火灾治理（Percolation × Drossel–Schwabl SOC）

> 你刚加入某省林业局，被分到"森林防火数字孪生组"。一线护林员吵了很多年同一个问题：
>
> 1. **稀疏森林**几乎不烧大火，但木材产量低、生态价值差；
> 2. **稠密森林**碳汇高、产量高，但一次雷击就可能烧穿整个山头；
> 3. 历史上"零火灾年"反而是巨灾的前兆——火没烧的时候树越长越密，等真烧起来就停不下来。
>
> 局长的问题很直接：
>
> **能不能把这片森林自动维持在"会冒烟、不会成灾"的临界边缘？**

本任务以二维森林火灾模型为物理核心，参考 `case4` 的递进结构：先建立"远离临界"的直觉，再定位渗流相变点，再体验自组织临界（SOC）下的幂律巨火，最后在维持木材产量的前提下用一个本地反馈策略压低尾部风险。

---

## 1. 故事设定

你面对的是一片 `L×L` 的森林（周期边界）。每个格子有三态：

| 状态 | 数值 | 含义 |
| --- | --- | --- |
| `EMPTY` | 0 | 空地 / 防火带 |
| `TREE`  | 1 | 一棵活树 |
| `FIRE`  | 2 | 一棵正在燃烧的树 |

两种相变物理交织在一起：

- **静态渗流相变**：在密度 `p` 的随机森林上点一把火，火按邻接传染。二维方格点渗流的临界密度 `p_c ≈ 0.5928`：低于它火只能烧一小片，高于它会出现跨越整片森林的"无穷簇"。
- **Drossel–Schwabl SOC**：每步空地以 `p_grow` 概率长树、每棵树以 `p_lightning` 概率被雷击。当 `p_lightning << p_grow << 1`，系统**不需要任何人调密度**，会自己反复"长 → 烧 → 再长"，把火灾大小推到幂律分布。

可观测指标：

- **火灾大小** `s`：单次火灾烧掉的树数；
- **火灾持续时间** `T`：火头从点燃到熄灭经历的 sweep 数；
- **最大簇 / 序参量** `P_∞`：渗流相变标准序参量；
- **磁化率类涨落 χ**：最大簇大小在 seed 间的方差（峰值 ≈ `p_c`）；
- **稳态树密度**：SOC 模式下自组织收敛到的密度；
- **燃烧前线**：每步同时着火的格子数（活动度）；
- **森林快照**：直接看树簇形态、火头形状、防火带分布。

---

## 2. 起始素材（Base，必须先跑）

`case5/base/` 是题目方提供的"森林防火实验台"脚手架：

| 文件 | 作用 |
| --- | --- |
| `forest_model.py` | 共用模型 `ForestFire` 与参数 `ForestParams`（静态 / SOC / 自适应疏伐三合一） |
| `starter_simulation.py` | 稀疏森林单次点火基线（必须先跑） |
| `plotting.py` | matplotlib 绘图 + 对数分箱直方图 + 尾部分位数工具 |
| `README.md` | base 三份脚本的中文逐段解读 |

先运行：

```bash
python case5/base/starter_simulation.py
```

它会在 `case5/figures/` 下生成：

- `starter_initial_forest.svg`：初始稀疏森林快照；
- `starter_post_fire.svg`：单次点火后的森林（小范围烧痕）；
- `starter_size_per_seed.svg`：30 个不同 seed 的火灾大小分布。

你会看到一个明显的亚临界基线：火灾大小普遍只有几十棵，没有跨越式巨火，森林大部分仍然是树。这是"安全但低产"的极端。

---

## 3. 四阶段挑战目标

参考实现位于 `case5/solution/`，每个脚本配同名 `.md` 解读：

```bash
python case5/solution/phase1_regimes.py
python case5/solution/phase2_critical_sweep.py
python case5/solution/phase3_soc.py
python case5/solution/phase4_adaptive_thinning.py
```

### 阶段一：远离临界（两种"安全"其实都不可取）

固定 `mode="static"`、中心点火，比较：

- **亚临界** `p = 0.40`：随手一点火只烧几十棵，毫无传播；
- **近临界** `p ≈ 0.59`：火灾大小方差极大，同一参数不同 seed 烧掉的可能从几十到几千；
- **超临界** `p = 0.75`：几乎每次都把森林烧个精光。

#### 阶段一验证要求（至少完成 2 项）

1. 三种密度的**单次火灾后快照**对比（直观看烧痕形状）。
2. 60 个 seed 的**火灾大小分布**，对数 y 轴展示三个数量级差异。
3. **平均** + **P95 火灾尺寸**柱状图，说明"稠密森林是黑天鹅工厂"。
4. 给出每个 regime 的 `<s>`、`max(s)`、`P95(s)`，与 starter 比较。

参考产物：

- `figures/phase1_post_fire_*.svg`
- `figures/phase1_size_per_seed.svg`
- `figures/phase1_mean_fire_size.svg`
- `figures/phase1_p95_fire_size.svg`

---

### 阶段二：定位渗流临界（涨落峰值 = 相变指纹）

扫描 `p ∈ [0.30, 0.80]`（步长 `0.02`），每个密度跑 `≥ 50` 个 seed，统计：

- 平均 / P99 火灾尺寸；
- 平均最大簇大小 / N（**渗流序参量** `P_∞(p)`）；
- 最大簇大小方差 / N（类磁化率 χ）。

然后用至少两种系统尺寸 `L=48` 与 `L=80` 做有限尺寸对照：χ 峰值会随 `L` 增大变得更陡、向 `p_c ≈ 0.5928` 靠拢；这是比单条曲线更严格的临界诊断。

#### 阶段二验证要求（至少完成 2 项）

1. **序参量曲线**：`P_∞(p)` 从低密度的 ~0 跨到高密度的 ~1。
2. **磁化率峰值**：χ(p) 在 `p ≈ 0.59` 附近出现明显峰，定位临界。
3. **有限尺寸效应**：至少比较 `L=48` 与 `L=80`，说明峰值随尺寸变化。
4. **火灾尺寸跃迁**：`<s>(p)` 与 `P99(s)(p)` 在 `p_c` 附近呈现指数式跃升。
5. 报告**经验 `p*`**（χ 峰位置）与 `p_c=0.5928` 的差距。

参考产物：

- `figures/phase2_order_parameter.svg`
- `figures/phase2_susceptibility.svg`
- `figures/phase2_fire_size.svg`

---

### 阶段三：自组织临界（SOC，谁都没调，怎么就到了边缘？）

切到 `mode="soc"`：每个 sweep 空地以 `p_grow` 概率长树、每棵树以 `p_lightning` 概率被雷击。看时间尺度分离 `f/p` 比值的影响：

```python
ForestParams(mode="soc", p_grow=0.02, p_lightning=1e-4, steps=4000)
```

#### 阶段三验证要求（至少完成 2 项）

1. **稳态密度**：`tree density(t)` 经过 warmup 后稳定在某条带上（不发散、不归零）。
2. **活动爆发**：`burning(t)` 长期低活动 + 偶发巨型火头，体现"长尾"。
3. **火灾尺寸幂律**：`P(s)` 在双对数坐标下有 1.5–2 个数量级的幂律段，斜率接近 `-1.15`（理论值，可作参考线）。
4. **持续时间幂律**：`P(T)` 也呈幂律。
5. **f/p 扫描**：至少跑三档 `p_lightning/p_grow`（如 `5e-2 / 5e-3 / 5e-4`），说明只有时间尺度分离够大时尾部才会"开放"成幂律。

参考产物：

- `figures/phase3_density_timeseries.svg`
- `figures/phase3_burning_timeseries.svg`
- `figures/phase3_size_distribution.svg`
- `figures/phase3_duration_distribution.svg`

---

### 阶段四：自适应防火（保产量，砍尾部）

把森林交给一个**只看本地**的护林员：

- 半径 `r=3` 的窗口内树密度若超过 `0.62`，就以 `5%/sweep` 的概率清掉这个格子（疏伐 / 防火带）；
- 不去精调全局密度，也不去预测哪里会被雷击。

把同一组 SOC 参数（`p_grow=0.02, p_lightning=1e-4`）下"放任不管"与"自适应疏伐"两种策略各跑 `≥ 3` 个 seed × `6000` 步。

#### 阶段四验证要求（至少完成 2 项）

1. **稳态密度对比**：放任策略密度更高（接近渗流阈值），管理策略密度被压低。
2. **燃烧时间序列**：管理策略下高峰前线明显被压扁。
3. **火灾尺寸 PDF 尾部**：双对数坐标下管理策略的 PDF 在 `s ≳ 1000` 处明显走低。
4. **尾部指标**：管理策略 P95、P99、max 火灾尺寸全部下降（典型 10–20%）。
5. **木材账本**：把"被烧损失" vs "被疏伐收获"画在一张柱状图上，说明在保住大部分林木的同时还多收获了一笔木材。
6. **与 phase 3 SOC 对比**：管理策略是否依然位于"近临界"（仍能见到中等火），还是已经过度安全（火灾消失）？

> 注意：这里的"自组织临界"是教学版反馈策略，不等同于严格证明。要严格断言，必须给出多尺寸标度、长时间稳态分布与对照模型检验。

参考产物：

- `figures/phase4_density_compare.svg`
- `figures/phase4_burning_compare.svg`
- `figures/phase4_size_distribution_compare.svg`
- `figures/phase4_tail_p95.svg`
- `figures/phase4_tail_p99.svg`
- `figures/phase4_max_fire.svg`
- `figures/phase4_loss_vs_yield.svg`

---

## 4. 提交内容

你需要在 `case5/` 下提交：

1. `task.md`（本任务书）；
2. `base/`（保留题目方提供的初始素材）：
   - `forest_model.py`、`starter_simulation.py`、`plotting.py`、`README.md`、`__init__.py`
3. `solution/phase1_regimes.py` + `.md`
4. `solution/phase2_critical_sweep.py` + `.md`
5. `solution/phase3_soc.py` + `.md`
6. `solution/phase4_adaptive_thinning.py` + `.md`
7. `figures/`（至少包含上文每阶段参考产物）

> 依赖：`matplotlib`。其余仅依赖 Python 标准库。

---

## 5. 评分建议

- **正确性（35%）**：是否清楚复现亚临界 / 渗流临界 / SOC 幂律 / 自适应反馈四种状态。
- **验证质量（30%）**：是否用 `<s>`、`P_∞`、χ、PDF、时间序列、尾部分位数等多证据互相印证。
- **递进体验（20%）**：是否让读者从直观体感走到统计指纹再到工程策略。
- **工程表达（15%）**：代码是否基于 base、可复现、图表清晰、依赖最小。

完成本任务后，你应该能直观回答：

> 为什么"完全不烧的森林"反而是危险的？为什么相同的物理可以同时给出渗流相变（外部调控）和 Drossel–Schwabl SOC（内部自组织）？为什么纯本地的简单策略就能砍掉尾部巨灾？
