# Case 1：城市交通拥堵级联挑战（Self-Organized Criticality）

## 1. 背景故事（现实关联）
你要开发一个“城市交通数字孪生”玩法：
- 路口具有**有限通行能力**（容量阈值）。
- 车辆持续缓慢进入系统（**慢驱动**）。
- 路口超载后会向相邻路口传递压力，形成**快松弛 / 连锁反应**。

为了避免“只看文字不知如何下手”，本题已经把**对题目场景的建模**作为初始
素材交给玩家，统一放在 `case1/base/` 目录下。你后续两个阶段都必须在
**这一份模型**的基础上继续开发和研究，不要另起炉灶。

---

## 2. 起始素材（Base，必须先跑）

`case1/base/` 是题目方提供的“数字孪生”脚手架：

| 文件 | 作用 |
| --- | --- |
| `traffic_model.py` | 共用场景模型 `TrafficCascadeSystem` 和参数 `TrafficParams` |
| `starter_simulation.py` | **明确建模好、且当前不在临界态**的基线脚本 |
| `plotting.py` | 基于 matplotlib 的轻量绘图与对数分箱直方图工具 |

先运行：

```bash
python case1/base/starter_simulation.py
```

它会在 `case1/figures/` 下生成：

- `starter_density_timeseries.svg`
- `starter_avalanche_distribution.svg`

你会看到当前系统是“非临界”的：通常偏亚临界，级联较小、衰减快。

> 提示：所有图都是 matplotlib 的 SVG 输出，可直接在浏览器或编辑器中打开。

参考实现拆分为两个可单独运行的脚本（位于 `case1/solution/`）：

```bash
python case1/solution/phase1_solution.py
python case1/solution/phase2_solution.py
```

---

## 3. 挑战目标

### 阶段一：把同一系统调到临界态（Tuned Criticality）
在 `case1/base/` 提供的模型/参数上**改造**，不要替换核心机制。

建议优先调 `spill_prob`（拥堵传播概率），也可联合 `threshold`、
`dissipation` 等参数。

#### 阶段一验证要求（至少完成 2 项）
1. **级联规模分布**：双对数坐标下出现近似幂律区间。
2. **三段对比**：展示亚临界 / 临界附近 / 超临界三种统计差异。
3. **序参量变化**：平均级联规模随控制参数接近临界点显著上升。

参考产物：
- `figures/phase1_mean_size_vs_spill_prob.svg`
- `figures/phase1_size_dist_compare.svg`

---

### 阶段二：构造自组织临界机制（SOC）
同样基于 base 模型，**增加反馈机制**，让系统不靠手工精调也能靠近临界态。

示例方向：
- **自适应信号灯**：根据平均负载调整传播效率（已在 base 模型中以
  `adaptive=True` 提供基线版本，可继续改造）；
- **动态阈值**：高峰 / 平峰路口容量随时间或本地拥堵变化；
- **局部拥堵税**：高负载路口提高耗散率。

#### 阶段二验证要求（至少完成 2 项）
1. **稳态存在**：平均负载进入稳定波动区，而非发散。
2. **长尾雪崩统计**：事件大小 / 持续时间在双对数坐标下呈多尺度分布。
3. **鲁棒性**：不同初值（`seed`、初始 `spill_prob`）下仍收敛到相近统计态。

参考产物：
- `figures/phase2_density_and_spillprob.svg`（双 y 轴：负载 vs 自适应 p）
- `figures/phase2_avalanche_dist.svg`

---

## 4. 提交内容要求
你需要在 `case1/` 下提交：

1. `task.md`（本任务书，可补充你的赛题说明）；
2. `base/`（保留题目方提供的初始素材，不要删除）：
   - `traffic_model.py`
   - `starter_simulation.py`
   - `plotting.py`
3. `solution/phase1_solution.py`（阶段一参考实现，可单独运行）；
4. `solution/phase2_solution.py`（阶段二参考实现，可单独运行）；
5. `figures/`（图表输出，至少包含）：
   - starter：`starter_density_timeseries.svg`、`starter_avalanche_distribution.svg`
   - phase1：`phase1_mean_size_vs_spill_prob.svg`、`phase1_size_dist_compare.svg`
   - phase2：`phase2_density_and_spillprob.svg`、`phase2_avalanche_dist.svg`

> 依赖：`matplotlib`（绘图）。其余仅依赖 Python 标准库。

---

## 5. 评分建议
- **正确性（40%）**：是否复现“调参临界 + 自组织临界”。
- **验证质量（30%）**：图表、统计与对比是否充分。
- **机制创新（20%）**：阶段二反馈机制是否有创意且有效。
- **工程表达（10%）**：代码结构、注释与可复现性。

祝你玩得开心：你会直观看到交通系统如何在局部规则下涌现临界行为。
