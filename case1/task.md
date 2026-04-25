# Case 1：城市交通拥堵级联挑战（Self-Organized Criticality）

## 1. 背景故事（现实关联）
你要开发一个“城市交通数字孪生”玩法：
- 路口有有限通行能力（阈值）。
- 车辆持续缓慢进入系统（慢驱动）。
- 路口超载后会向邻近路口传递压力（快松弛/连锁反应）。

为了避免“只看文字不知如何下手”，本题已经给出**初始素材脚本**：
- `starter_simulation.py`：一个**明确建模好、且当前不在临界态**的交通系统。

你后续两个阶段都必须在这个脚本/模型基础上继续开发和研究。

---

## 2. 起始素材（必须先跑）
先运行：

```bash
python case1/starter_simulation.py
```

它会生成：
- `figures/starter_density_timeseries.svg`
- `figures/starter_avalanche_distribution.svg`

你会看到当前系统是“非临界”的（通常偏亚临界，级联较小、衰减快）。

参考实现拆分为两个可单独运行脚本：

```bash
python case1/solution/phase1_solution.py
python case1/solution/phase2_solution.py
```

---

## 3. 挑战目标

### 阶段一：把同一系统调到临界态（Tuned Criticality）
在 `starter_simulation.py` 的模型参数/机制上改造，不要另起炉灶。

建议优先调 `spill_prob`（拥堵传播概率），也可联合阈值、耗散率等参数。

#### 阶段一验证要求（至少 2 项）
1. **级联规模分布**：对数坐标下出现近似幂律区间。
2. **三段对比**：展示亚临界 / 临界附近 / 超临界三种统计差异。
3. **序参量变化**：平均级联规模随控制参数接近临界点显著上升。

---

### 阶段二：构造自组织临界机制（SOC）
同样基于 starter 模型，增加反馈机制，让系统不靠手工精调也能靠近临界态。

示例方向：
- 自适应信号灯（根据平均负载调整传播效率）；
- 动态阈值（高峰/平峰路口容量变化）；
- 局部拥堵税（高负载路口提高耗散率）。

#### 阶段二验证要求（至少 2 项）
1. **稳态存在**：平均负载进入稳定波动区，而非发散。
2. **长尾雪崩统计**：事件大小/持续时间呈多尺度分布。
3. **鲁棒性**：不同初值下仍收敛到相近统计态。

---

## 4. 提交内容要求
你需要在 `case1/` 下提交：
1. `task.md`（本任务书，可补充你的赛题说明）；
2. `starter_simulation.py`（初始素材脚本，非临界态）；
3. `solution/phase1_solution.py`（阶段一参考实现，可单独运行）；
4. `solution/phase2_solution.py`（阶段二参考实现，可单独运行）；
5. `traffic_model.py`（共用场景模型）；
6. `figures/`（图表输出，至少包含）：
   - starter：`starter_density_timeseries.svg`、`starter_avalanche_distribution.svg`
   - phase1：`phase1_mean_size_vs_spill_prob.svg`、`phase1_size_dist_compare.svg`
   - phase2：`phase2_density_and_spillprob.svg`、`phase2_avalanche_dist.svg`

---

## 5. 评分建议
- **正确性（40%）**：是否复现“调参临界 + 自组织临界”。
- **验证质量（30%）**：图表、统计与对比是否充分。
- **机制创新（20%）**：阶段二反馈机制是否有创意且有效。
- **工程表达（10%）**：代码结构、注释与可复现性。

祝你玩得开心：你会直观看到交通系统如何在局部规则下涌现临界行为。
