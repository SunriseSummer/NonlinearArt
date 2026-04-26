# Case 2：地震断层应力级联挑战（OFC × Self-Organized Criticality）

## 1. 背景故事（现实关联）

> 你刚刚被聘为某地震研究所的“数字孪生工程师”。研究所给你一块约
> 32 公里 × 32 公里、网格化为 32×32 的断层片，要你用一个轻量化的
> 数字孪生回答两个问题：
>
> 1. **临界判据**：这块断层如果只能调一个总参数（块内的应力保守度
>    `alpha`），调到哪附近，事件分布才会和真实地震学的 Gutenberg–Richter
>    震级–频率关系吻合？
> 2. **自组织假说**：现实板块从来没有“调参员”，可断层确实长期保持在
>    临界态附近。能不能给你的孪生加一层反馈机制，让它从任何不合理的
>    初始条件出发，都能**自己**找到那个临界点？
>
> 研究所同时担心两件事：会不会出现“伪幂律”（短尾被你硬当成幂律）？
> 自组织出来的状态是不是只是单一种子的偶然？所以验证目标里要求你拿
> **多条独立证据**说话——不是只画一条好看的对数斜线就交差。

为了避免“只看文字不知如何下手”，本题已经把**对题目场景的建模**作为
初始素材交给玩家，统一放在 `case2/base/` 目录下。你后续两个阶段都必
须在**这一份模型**的基础上继续开发和研究，不要另起炉灶。

> **本题挑战度**：如果你完全没有玩过 SOC，预计需要用 AI 配合、阅读
> 一些 OFC / Gutenberg–Richter / Omori 的资料，做半天到一天才能交出
> 三组令人信服的图。

---

## 2. 起始素材（Base，必须先跑）

`case2/base/` 是题目方提供的“断层数字孪生”脚手架：

| 文件 | 作用 |
| --- | --- |
| `fault_model.py` | 共用场景模型 `FaultStressSystem` 和参数 `FaultParams` |
| `starter_simulation.py` | **明确建模好、且当前不在临界态**的基线脚本 |
| `plotting.py` | 基于 matplotlib 的折线 / 双轴 / 热力图 + 三类 SOC 统计工具 |

先运行：

```bash
python case2/base/starter_simulation.py
```

它会在 `case2/figures/` 下生成：

- `starter_mean_stress.svg`
- `starter_avalanche_distribution.svg`
- `starter_stress_field.svg`

你会看到当前系统是“非临界”的：通常偏亚临界，雪崩很小、衰减极快、应
力场到处是孤立小热斑。

> 提示：所有图都是 matplotlib 的 SVG 输出，可直接在浏览器或编辑器中打开。

base 中的三份脚本（模型 / 起始仿真 / 绘图工具）有一份**逐段中文解读**：
[`case2/base/README.md`](base/README.md)。**强烈建议**开发前先通读一遍——
本案例的模型比 case1 复杂（连续应力、`extremal` 驱动、可选愈合、可选
自适应），不弄清楚控制流和数据流就写实验脚本会踩坑。

参考实现拆分为两个可单独运行的脚本（位于 `case2/solution/`），每个脚
本都配有同名的中文解读文档：

```bash
python case2/solution/phase1_solution.py   # 解读：solution/phase1_solution.md
python case2/solution/phase2_solution.py   # 解读：solution/phase2_solution.md
```

---

## 3. 挑战目标

### 阶段一：把同一系统调到临界态（Tuned Criticality）

在 `case2/base/` 提供的模型/参数上**改造**，**不要替换核心机制**。

主要调参对象：保守度 `alpha`（OFC 普适类的核心控制参数）。可联合微调
`heterogeneity`（不超过 0.20）、`drive_steps`、`warmup` 提升统计稳定
性，但**不许**触碰 `_drive` / `_relax` 的力学规则，也**不许**打开
`adaptive=True`（那是阶段二的事）。

#### 阶段一验证要求（至少完成 2 项）

1. **序参量曲线**：扫描 `alpha`，画平均级联规模随 `alpha` 的变化，并
   标出你估计的临界点 `alpha_c`。
2. **雪崩规模分布（Gutenberg–Richter）**：在亚临界 / 临界附近 / 超临界
   三组参数下分别跑长仿真，画**双对数 CCDF**，临界一组要出现近似直线
   段；并用 Hill / MLE 给出**临界点的 τ**（与 2-D OFC 文献的 1.7~2.0
   一致即可）。
3. **Omori 余震律**：用 `aftershock_rate` 把所有“主震”（默认取
   95 分位以上）后的事件按 `Δt` 叠加，临界附近应出现
   `n(Δt) ∝ Δt^{-p}`、`p ≈ 1` 的余震尾巴，亚临界则平坦。

> 题目要求 ≥2 项；如果你想冲满分，把 3 项都做上。

参考产物（脚本会一次生成 3 张）：

- `figures/phase1_mean_size_vs_alpha.svg`
- `figures/phase1_size_dist_compare.svg`
- `figures/phase1_omori.svg`

---

### 阶段二：构造自组织临界机制（SOC）

同样基于 base 模型，**增加反馈机制**，让系统**不靠手工精调**也能稳定
落在临界态附近。

可以使用的方向（base 已经预留接口）：

- **自适应保守度**：`adaptive=True`，根据滑动平均级联规模反向调整
  `alpha`（已在 base 中以比例控制器形式提供基线版本，可自由改造）；
- **摩擦愈合**：`healing=True`，刚滑动过的格点暂时变强，逼真模拟
  地震断层的“锁定再活化”循环；
- **异质静态阈值**：`heterogeneity > 0`，让“断层强度”空间不均；
- **自定义反馈**：你也可以把上面三件组合，或写一个全新的反馈机制
  （例如局部 alpha、应变率耦合的耗散等）——只要不替换 `_drive` /
  `_relax` 主框架。

#### 阶段二验证要求（至少完成 2 项）

1. **稳态存在**：平均应力进入稳定波动区，而非发散，并且 `alpha(t)`
   也收敛到一个有限范围。
2. **长尾雪崩统计**：事件大小 / 持续时间在双对数坐标下都呈多尺度
   分布；至少给出 `P(s)` 与 `P(T)` 两条 PDF 或一条 PDF + 一条 CCDF。
3. **鲁棒性**：至少 3 个不同 `(seed, 初始 alpha)` 的组合，最终都收敛
   到**统计一致的状态**——把 3 条 `alpha(t)` 轨迹画在同一张图上，并
   报告 late-time `<alpha>` 与 `<s>` 之间的差异。

参考产物：

- `figures/phase2_stress_and_alpha.svg`（双 y 轴：应力 vs 自适应 `alpha`）
- `figures/phase2_avalanche_dist.svg`
- `figures/phase2_robustness.svg`
- `figures/phase2_stress_field.svg`

---

## 4. 提交内容要求

你需要在 `case2/` 下提交：

1. `task.md`（本任务书，可补充你的赛题说明）；
2. `base/`（保留题目方提供的初始素材，不要删除）：
   - `fault_model.py`
   - `starter_simulation.py`
   - `plotting.py`
   - `README.md`（base 脚本的中文详解）
3. `solution/phase1_solution.py` + `solution/phase1_solution.md`（阶段一参考实现及其中文解读）；
4. `solution/phase2_solution.py` + `solution/phase2_solution.md`（阶段二参考实现及其中文解读）；
5. `figures/`（图表输出，至少包含）：
   - starter：`starter_mean_stress.svg`、`starter_avalanche_distribution.svg`、`starter_stress_field.svg`
   - phase1：`phase1_mean_size_vs_alpha.svg`、`phase1_size_dist_compare.svg`、`phase1_omori.svg`
   - phase2：`phase2_stress_and_alpha.svg`、`phase2_avalanche_dist.svg`、`phase2_robustness.svg`、`phase2_stress_field.svg`

> 依赖：`matplotlib`（绘图）。其余仅依赖 Python 标准库。

---

## 5. 评分建议

- **正确性（35%）**：是否复现“调参临界（含 G-R）+ 自组织临界（含鲁
  棒性）”。
- **验证质量（35%）**：图表、统计与对比是否充分；尤其是
  - τ 的估计是否在 OFC 普适类合理区间（≈1.6~2.4）；
  - Omori 律是否真的看出近 `1/Δt` 的直线段；
  - 鲁棒性是否真的覆盖了 ≥3 个差异显著的初值。
- **机制创新（20%）**：阶段二的反馈机制是否有创意且有效；如果你不
  采用参考实现的“自适应 alpha”而自己设计了别的（局部 alpha、动态
  愈合、异质驱动等）并稳住了 SOC，能拿到这一档。
- **工程表达（10%）**：代码结构、注释、可复现性、依赖最小化。

---

## 6. 进阶玩法（可选挑战）

如果你做完三条验证还想加难度，下面任选一条做出来的同学会被认为
“真的把 SOC 玩明白了”：

- **数据驱动 b-value**：找一份真实区域地震目录（USGS 公开数据），把
  你的 τ 值映射成 b-value，看与目标区域的 b≈0.7~1.1 是否吻合，并讨
  论本模型在该区域是“偏弹性”还是“偏塑性”。
- **断层网络拓扑**：把方形 `L×L` 网格替换成一个简单的小世界图
  （仍然遵守 OFC 局部更新规则），观察 τ 与 p 的变化，并解释“断层
  连通性如何改变 SOC 普适类”。
- **逆问题**：把仿真当“真实地震”，从前 80% 的事件序列出发，预测后
  20% 的最大事件出现的概率窗口；用 Brier score / log-loss 评估，并
  和“无信息基线”比较。

祝你玩得开心：你会直观看到地壳如何在板块缓慢加载的“慢驱动”和瞬间滑
动的“快松弛”夹击下，自己长成一个临界系统——而你写下的几百行 Python，
就是这套真实物理的最小可玩沙盘。
