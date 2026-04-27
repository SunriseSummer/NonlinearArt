# Case 1b：早高峰主干道保畅与临界性体验

> 本任务是 case1 的"教学化重构版"。case1 把"达到临界态"作为核心目标；case1b
> 把这个目标包装到一个**早高峰调度员**的故事里，**逐步**带玩家从"远离临界
> 态" 走到"接近临界态"，再到"越过临界后崩溃"，最后体验"用局部规则做
> 自组织调控"的好处。  
> 设计依据见仓库根目录的 [`全新设计.md`](../全新设计.md)。

---

## 1. 故事设定

你是一名城市交通调度员，要在**早高峰**期间管理一个 `L×L` 的小型城区路网：

- 系统会持续往随机路口注入车辆（**慢驱动**）。
- 路口有**有限通行能力**（容量阈值 `threshold`），超载后会向相邻路口
  传递压力，可能引发**连锁反应**。
- 你能观察到的指标包括：
  - **吞吐量**：单位时间通过/离开网络的车辆数；
  - **平均负载**：每个路口的平均排队长度（≈ 平均延误代理量）；
  - **拥堵传播范围**：当步处于阈值之上的路口数；
  - **级联规模分布**：每次连锁反应波及的路口数。
- 你能调控的"政策旋钮"包括：
  - `inflow_rate`：入口车流量（demand 端）；
  - `spill_prob`：路口将压力扩散到邻居的概率（routing 端）；
  - `dissipation`：扩散尝试中车辆离开网络的概率（off-ramp 端）；
  - 一组**局部反馈规则**（`inflow_feedback`、`local_relief`），用于阶段4。

任务目标**不是**单纯把交通跑到某种状态，而是让你**亲手**体验：

1. 高利用率为什么带来高效率；
2. 为什么小扰动会放大成拥堵；
3. 为什么局部调控能改变全局结果；
4. 为什么"接近临界"比"一味追求满载"更合理；
5. 什么叫自组织，以及它为什么在现实交通管理中有价值。

---

## 2. 起始素材（Base，必须先跑）

`case1b/base/` 是题目方提供的"数字孪生"脚手架，所有阶段的代码必须基于
这里的 `RushHourTrafficSystem` 继续开发。

| 文件 | 作用 |
| --- | --- |
| `traffic_model.py` | 共用模型 `RushHourTrafficSystem` 与参数 `TrafficParams`、`DisturbanceSpec` |
| `starter_simulation.py` | "远离临界态"基线脚本（必须先跑） |
| `plotting.py` | matplotlib 轻量绘图与对数分箱直方图工具 |
| `README.md` | base 三份脚本的中文逐段解读 |

先运行：

```bash
python case1b/base/starter_simulation.py
```

它会在 `case1b/figures/` 下生成：

- `starter_load_throughput.svg`（双 y 轴：平均负载 + 吞吐量）
- `starter_congestion_range.svg`（拥堵传播范围）

你会看到一个"稳但低效"的网络：负载远低于阈值，几乎没有连锁反应，吞吐
量被入口车流量限制，大段道路容量没有被利用——这正是阶段1的主旨。

---

## 3. 四阶段挑战目标

四个阶段都在 `case1b/base/` 提供的同一个模型上做研究，参考实现拆分在
`case1b/solution/`，每个脚本配同名 `.md` 详解：

```bash
python case1b/solution/phase1_far_from_critical.py    # solution/phase1_far_from_critical.md
python case1b/solution/phase2_near_critical.py        # solution/phase2_near_critical.md
python case1b/solution/phase3_cascade_collapse.py     # solution/phase3_cascade_collapse.md
python case1b/solution/phase4_self_organization.py    # solution/phase4_self_organization.md
```

### 阶段一：远离临界（保守 = 稳定但低效）

设定较低的 `inflow_rate` 和保守的 `spill_prob`。验证以下三件事：

1. 平均负载远低于阈值；
2. 拥堵传播范围接近 0，几乎没有级联；
3. 吞吐量被入口车流量限制，**增加入口流量**仍能稳住，但容量利用率仍低。

#### 阶段一验证要求（至少完成 2 项）
- 时间序列上平均负载稳定。
- 给出至少 3 种不同 `inflow_rate` 的对比，体现"加车也不堵但效率不高"。
- 给出"入口车流量 → 吞吐量"的可视化（条形图或散点）。

参考产物：
- `figures/phase1_far_from_critical.svg`
- `figures/phase1_throughput_vs_inflow.svg`

---

### 阶段二：接近临界（高吞吐 / 高敏感）

在阶段一的模型上**升高 `inflow_rate`**（demand 上升 = 早高峰强度增加），
保持 `spill_prob` 不变，扫描不同的入口车流量并记录：

- 平均负载、吞吐量随 demand 的变化；
- 拥堵传播范围、平均级联规模随 demand 的变化；
- 由这些指标自动定位"压力边缘 (stress edge)"——网络刚开始出现明显
  拥堵传播、瓶颈频繁出现的入口车流量。

然后选择**亚临界 / 接近临界 / 超临界**三个 demand，分别做长仿真。

#### 阶段二验证要求（至少完成 2 项）
1. **吞吐量曲线**：在双对数或线性坐标下展示吞吐量与负载随 demand 的变化。
2. **三段对比**：亚 / 临界 / 超 三种 demand 下，负载与吞吐量的时间序列叠图。
3. **级联规模分布**：在双对数坐标下展示三段差异（接近临界处分布更长尾）。

参考产物：
- `figures/phase2_demand_sweep.svg`
- `figures/phase2_regime_compare.svg`
- `figures/phase2_avalanche_dist.svg`
- `figures/phase2_susceptibility.svg` — `χ(p)` 易感度峰值定 `p_c`
- `figures/phase2_powerlaw_fit.svg` — 临界点处 `P(s)~s^(-τ)` 拟合 + R²
- `figures/phase2_finite_size_scaling.svg` — `L=16/24/32` 截断随尺寸增长

> 后三张是临界态的**严格诊断**：仅有"亚 / 临界 / 超"的运营对比不足以
> 证明系统真的处于临界，必须同时给出（a）易感度峰值、（b）临界点处可
> 拟合的幂律 + 高 R²、（c）finite-size scaling，才能下结论。

---

### 阶段三：越过临界后的崩溃（小扰动 = 大后果）

引入一次**事故扰动**：在指定时刻把一小片路口冻结若干步（不再倾倒），
分别在阶段二选出的三段 demand 下重放**同一**事故，对比：

- 拥堵传播范围在事故期间的峰值；
- 事故清除后的拥堵恢复时间；
- 事故期间吞吐量的下降幅度。

#### 阶段三验证要求（至少完成 2 项）
1. **同一扰动，三段对照**：负载 / 拥堵范围两条时间序列叠图，标出事故起讫。
2. **崩溃幅度**：用条形图比较各 regime 下的拥堵峰值（或恢复时间），体现
   "越接近超临界，扰动放大越剧烈"。
3. **吞吐量惩罚**：报告事故期间吞吐量相对事故前的相对下降。

参考产物：
- `figures/phase3_disturbance_load.svg`
- `figures/phase3_congestion_spread.svg`
- `figures/phase3_recovery_summary.svg`

---

### 阶段四：自组织 vs 强控制

**同样**的 demand、**同样**的事故扰动、**同样**的随机种子，对比两种
策略：

- **模式 A（强控制）**：手工把 `spill_prob` 调到一个对稳态吞吐量比较友好
  的值，没有任何反馈机制——所有变化只能事先布置。
- **模式 B（局部规则自组织）**：在 base 模型里把以下两条**局部规则**打开：
  - `inflow_feedback=True`：当网络平均负载超过 `target_load` 时，自动
    收紧入口流量（≈ 上匝道控制 / 出行需求引导）；
  - `local_relief=True`：高负载路口提高耗散概率（≈ 自适应绿灯延长 /
    可变信息板分流）。

不要去手动调 `spill_prob`，让局部反馈规则自己接管系统。

#### 阶段四验证要求（至少完成 2 项）
1. **峰值对比**：B 模式下事故期间的拥堵传播范围峰值显著低于 A。
2. **吞吐量韧性**：B 模式下事故期间的平均吞吐量不弱于（最好优于）A。
3. **稳态质量**：在事故前的 1000 步窗口里，B 模式的负载更接近 `target_load`，
   且波动可控。

参考产物：
- `figures/phase4_compare_load.svg`
- `figures/phase4_compare_congestion.svg`
- `figures/phase4_summary.svg`
- `figures/phase4_robustness.svg` — 9 组初值的 `spill_prob` 轨迹全部收敛
- `figures/phase4_robustness_load.svg` — 同 9 组的 `<load>` 轨迹收敛到 `target_load`
- `figures/phase4_soc_avalanche_dist.svg` — SOC 模式下的雪崩幂律 + 静态对照

> 后三张是 SOC 的**严格诊断**：仅展示"事故下表现更好"不能证明 SOC，必须
> 同时给出（a）多初值收敛到同一吸引子、（b）稳态雪崩幂律 + 高 R²、
> （c）静态对照下分布显著不同，才能严格地说"这是自组织临界"。

---

## 4. 提交内容

你需要在 `case1b/` 下提交：

1. `task.md`（本任务书，可补充你的赛题说明）；
2. `base/`（保留题目方提供的初始素材，不要删除）：
   - `traffic_model.py`、`starter_simulation.py`、`plotting.py`、`README.md`、`__init__.py`
3. `solution/phase1_far_from_critical.py` + `.md`
4. `solution/phase2_near_critical.py` + `.md`
5. `solution/phase3_cascade_collapse.py` + `.md`
6. `solution/phase4_self_organization.py` + `.md`
7. `figures/`（图表输出，至少包含上文每个阶段的"参考产物"）

> 依赖：`matplotlib`（绘图）。其余仅依赖 Python 标准库。

---

## 5. 评分建议

- **正确性（35%）**：四个阶段是否分别复现"远离 / 接近 / 越过 / 自组织"。
- **验证质量（30%）**：图表与统计是否充分、可读、可复现。
- **机制创新（20%）**：阶段四局部规则是否有创意、是否真的更稳更省力。
- **工程表达（15%）**：代码结构、注释、命名是否能让后续学习者直接接手。

完成本任务后，你应该能直观回答这些问题：

> 为什么系统高效时也会危险？为什么小的局部干预能改变全局？为什么好的
> 复杂系统常常不是靠全知全能控制，而是靠合理规则与反馈？

这些都是非常实用的复杂系统、风险管理和效率—韧性平衡思维。
