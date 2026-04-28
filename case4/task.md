# Case 4：磁性薄膜的临界边缘（Ising × Phase Transition × SOC）

> 你刚加入一家先进材料实验室，任务是给一片二维铁磁薄膜做“数字孪生”。实验组发现：远离相变点时，薄膜要么死死锁成同一磁畴，要么热噪声把磁畴搅成一团；但在某个窄窄的温度窗口里，微小磁场会被成片磁畴放大，响应既敏感又不完全失控。老板的问题很直接：
>
> **我们能不能把这片薄膜自动维持在“临界边缘”，让它像一个高灵敏度传感器，而不是靠人工反复调温？**

本任务以二维 Ising 模型为物理核心，参考 `case1b` 的递进结构：先建立远离临界的直觉，再定位临界点，再体验临界放大，最后构造一个自组织靠近临界的反馈机制。

---

## 1. 故事设定

你面对的是一片 `L×L` 的纳米铁磁薄膜：

- 每个格点是一小块磁畴，自旋 `s_i=+1/-1` 表示磁矩向上/向下。
- 邻近磁畴倾向同向排列，强度由交换耦合 `J` 决定。
- 热浴温度 `T` 不断制造随机翻转。
- 外加磁场 `h` 会偏向某一方向，可作为微弱待测信号。
- 二维 Ising 模型在 `J=1, k_B=1` 时有精确临界温度 `T_c≈2.269`。

你能观测的指标包括：

- **磁化强度** `m`：全局有序程度，是相变序参量；
- **磁化率** `χ`：`m` 的涨落强度，临界附近出现峰值；
- **能量 / 比热代理量**：能量涨落在临界附近增强；
- **接受翻转数**：每个 Monte Carlo sweep 中磁畴实际翻转次数，可视为活动度；
- **磁畴快照**：直接观察短程团簇、长程有序与临界团簇。

---

## 2. 起始素材（Base，必须先跑）

`case4/base/` 是题目方提供的“磁性薄膜实验台”脚手架：

| 文件 | 作用 |
| --- | --- |
| `ising_model.py` | 共用模型 `IsingFilm` 与参数 `IsingParams`、`FieldPulseSpec` |
| `starter_simulation.py` | 高温顺磁基线脚本（必须先跑） |
| `plotting.py` | matplotlib 绘图 + 磁化率 / 比热 / 对数直方图工具 |
| `README.md` | base 三份脚本的中文逐段解读 |

先运行：

```bash
python case4/base/starter_simulation.py
```

它会在 `case4/figures/` 下生成：

- `starter_hot_film_timeseries.svg`
- `starter_energy.svg`
- `starter_final_domains.svg`

你会看到一个高温顺磁基线：`|m|` 长期接近 0，磁畴只有短程小团簇，系统对单次扰动没有记忆。

---

## 3. 四阶段挑战目标

参考实现位于 `case4/solution/`，每个脚本配同名 `.md` 解读：

```bash
python case4/solution/phase1_regimes.py
python case4/solution/phase2_critical_sweep.py
python case4/solution/phase3_field_response.py
python case4/solution/phase4_self_organization.py
```

### 阶段一：远离临界（两种“安全”其实都迟钝）

固定 `J=1`、`h=0`，分别选择：

- 低温 `T<T_c`：铁磁有序，`|m|≈1`，磁畴几乎锁死；
- 高温 `T>T_c`：顺磁无序，`m≈0`，但只有短程噪声；
- 中间温度作为过渡对照。

#### 阶段一验证要求（至少完成 2 项）

1. 画 `|m|(t)` 对比低温 / 中温 / 高温。
2. 给出三种温度的最终磁畴快照。
3. 报告稳态 `|m|` 与接受翻转数，说明“有序但不敏感”和“嘈杂但无长程记忆”的区别。

参考产物：

- `figures/phase1_magnetisation_regimes.svg`
- `figures/phase1_activity_regimes.svg`
- `figures/phase1_domains_*.svg`

---

### 阶段二：定位临界点（涨落峰值 = 相变指纹）

扫描 `T ∈ [1.6, 3.2]`，每个温度跑同一套 Ising 薄膜并统计 warmup 之后的：

- 平均 `|m|`；
- 磁化率 `χ = N·Var(m)/T`；
- 比热代理量 `C = N·Var(E/N)/T²`。

然后用至少两种系统尺寸 `L` 做有限尺寸对照：有限系统中 `χ` 峰值会被截断，并随 `L` 变化向 `T_c≈2.269` 附近靠近。

#### 阶段二验证要求（至少完成 2 项）

1. **序参量曲线**：`|m|(T)` 从低温高值跨到高温低值。
2. **磁化率峰值**：`χ(T)` 在 `T≈2.269` 附近出现明显峰。
3. **有限尺寸效应**：至少比较 `L=16` 与 `L=24`（或更大），说明峰值位置 / 高度随尺寸变化。
4. **比热辅助证据**：`C(T)` 在相同区域增强。

参考产物：

- `figures/phase2_temperature_sweep.svg`
- `figures/phase2_susceptibility.svg`
- `figures/phase2_finite_size.svg`

---

### 阶段三：临界放大（同一微弱磁场，三种命运）

在亚临界有序区、临界附近、超临界无序区分别施加同一个弱磁场脉冲：

```python
FieldPulseSpec(start=500, duration=120, delta_h=0.035)
```

比较脉冲期间和脉冲结束后的响应：

- `m(t)` 的峰值变化；
- 脉冲结束后的恢复时间；
- 接受翻转数是否出现“活动爆发”。

#### 阶段三验证要求（至少完成 2 项）

1. **同一脉冲，三段对照**：叠图展示 `m(t)`，标出脉冲起止。
2. **放大系数**：用柱状图比较 `max(m)-baseline`，临界附近应最大。
3. **恢复时间**：说明临界慢化（critical slowing down）如何让响应拖尾。
4. **活动爆发**：展示接受翻转数在脉冲附近的变化。

参考产物：

- `figures/phase3_field_pulse_magnetisation.svg`
- `figures/phase3_pulse_response_summary.svg`
- `figures/phase3_activity_burst.svg`

---

### 阶段四：自组织靠近临界（反馈温控，不再人工扫温）

现在把薄膜交给一个只看局部宏观指标的“实验员”：

- 如果最近窗口内 `|m|` 太高，说明薄膜过冷、太有序，就升温；
- 如果 `|m|` 太低，说明薄膜过热、太无序，就降温；
- 目标不是固定某个外部温度，而是让系统自己维持在“有序与无序的边缘”。

在 base 模型中打开：

```python
adaptive_temperature=True
```

并从多个初始温度（例如 `T0=1.4, 3.2, 4.0`）出发，观察它们是否收敛到相近的温度窗口。

#### 阶段四验证要求（至少完成 2 项）

1. **多初值收敛**：多条 `T(t)` 轨迹最终落入 `T_c` 附近同一窗口。
2. **稳态质量**：late-time `|m|` 接近目标值且不过度振荡。
3. **活动长尾**：反馈态的接受翻转数分布比固定低温 / 高温更宽、更长尾。
4. **扰动韧性**：在反馈态下重复阶段三的弱脉冲，响应大但不长期锁死。

参考产物：

- `figures/phase4_temperature_convergence.svg`
- `figures/phase4_magnetisation_convergence.svg`
- `figures/phase4_activity_distribution.svg`
- `figures/phase4_summary.svg`

> 注意：这里的“自组织临界”是教学版反馈机制，不等同于严格无限系统 SOC 证明。若要严格下结论，必须进一步给出多尺寸标度、长时间稳态分布与对照模型检验。

---

## 4. 提交内容

你需要在 `case4/` 下提交：

1. `task.md`（本任务书）；
2. `base/`（保留题目方提供的初始素材）：
   - `ising_model.py`、`starter_simulation.py`、`plotting.py`、`README.md`、`__init__.py`
3. `solution/phase1_regimes.py` + `.md`
4. `solution/phase2_critical_sweep.py` + `.md`
5. `solution/phase3_field_response.py` + `.md`
6. `solution/phase4_self_organization.py` + `.md`
7. `figures/`（至少包含上文每阶段参考产物）

> 依赖：`matplotlib`。其余仅依赖 Python 标准库。

---

## 5. 评分建议

- **正确性（35%）**：是否清楚复现低温有序 / 高温无序 / 临界涨落 / 反馈靠近临界。
- **验证质量（30%）**：是否用 `|m|`、`χ`、`C`、磁畴快照、脉冲响应等多证据互相印证。
- **递进体验（20%）**：是否让读者从直觉观察逐步走到临界诊断与自组织机制。
- **工程表达（15%）**：代码是否基于 base、可复现、图表清晰、依赖最小。

完成本任务后，你应该能直观回答：

> 为什么相变不是“某条曲线变陡”这么简单？为什么临界点同时意味着高灵敏度和高波动？为什么工程系统有时不该死守固定参数，而该用反馈把自己维持在边缘？
