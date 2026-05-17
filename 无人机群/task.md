# Case 6：无人机集群飞行的相变（Vicsek × Phase Transition × Self-Organisation）

> 你刚加入一家无人机送货公司，被分到"集群协同"组。客户的两个需求互相打架：
>
> 1. **飞得齐**：1000 架无人机配送密集订单时必须维持同向通航，不能各飞各的；
> 2. **听得见**：地面调度员发出"避让"指令时，集群必须**整体**跟着改变航向，单个领飞节点的指令要能传遍全队。
>
> 项目经理直接抛出难题：
>
> **能不能让集群自动维持在"既稳又灵"的临界边缘？**

本任务以 1995 年 Vicsek 自驱粒子模型为物理核心，参考 `case4` 的递进结构：先建立"远离临界"的直觉，再定位有序–无序相变点，再体验临界附近的指令传输/扰动响应，最后构造一个只看本地极化度的反馈控制器，自组织到临界边缘。

---

## 1. 故事设定

`N` 个粒子在边长 `L` 的二维周期方盒内运动。每个粒子有固定速率 `v0` 与朝向 `theta`：

```
theta_i(t+1) = arg( sum_{j: |x_j - x_i| < r} e^{i theta_j(t)} ) + xi_i
xi_i ~ Uniform(-eta/2, +eta/2)
x_i(t+1) = x_i(t) + v0 * (cos theta, sin theta)
```

物理图景：

- **核心控制参数**：噪声 `eta`。`eta` 小 → 邻居对齐主导 → 集群同向；`eta` 大 → 随机扰动主导 → 朝向无序。
- **序参量**（极化度 polarisation）：`phi = |(1/N) sum_i (cos theta_i, sin theta_i)|`，范围 `[0, 1]`。
- **相变**：在临界噪声 `eta_c(rho)` 处发生**有序–无序连续相变**。
- **易感性** `chi = N · Var(phi)`：临界处取峰，是定位 `eta_c` 的标准武器。
- **Binder 累积量** `U_4 = 1 - <phi^4>/(3 <phi^2>²)`：不同 N 的曲线在 `eta_c` 附近交叉。

可观测指标：

- 极化度 `phi(t)`、易感性 `chi`、Binder `U_4`；
- 与领飞者的对齐 `(1/N) Σ cos(theta_i - theta_leader)`；
- 集群 quiver 快照（每个粒子一个箭头，颜色编码朝向）；
- 阵风脉冲下的 `phi` 响应曲线。

---

## 2. 起始素材（Base，必须先跑）

`case6/base/` 是题目方提供的"集群协同实验台"脚手架：

| 文件 | 作用 |
| --- | --- |
| `vicsek_model.py` | 共用模型 `VicsekFlock` + `VicsekParams` + 领飞 + 扰动 + 反馈 |
| `starter_simulation.py` | 高噪声混乱基线（必须先跑） |
| `plotting.py` | matplotlib 绘图 + 易感性 / Binder / quiver 工具 |
| `README.md` | base 脚本的中文逐段解读 |

先运行：

```bash
python case6/base/starter_simulation.py
```

它会在 `case6/figures/` 下生成：

- `starter_phi_timeseries.svg`：极化度时间序列；
- `starter_final_swarm.svg`：最终集群 quiver 快照（彩虹色斑点表示朝向无序）。

你会看到一个明显的高噪声无序基线：`phi ≈ 0.11`，集群完全没有共识。这是"散漫但灵敏（其实不灵敏）"的极端。

---

## 3. 四阶段挑战目标

参考实现位于 `case6/solution/`，每个脚本配同名 `.md` 解读：

```bash
python case6/solution/phase1_regimes.py
python case6/solution/phase2_critical_sweep.py
python case6/solution/phase3_leader_response.py
python case6/solution/phase4_self_organization.py
```

### 阶段一：远离临界（两种"安全"其实都不可取）

固定 `N=400, rho=4`，比较：

- **低噪声** `eta = 1.5`：集群锁成一坨，`phi ≈ 0.88`；
- **中等** `eta = 3.0`：在相变区附近徘徊；
- **高噪声** `eta = 4.5`：朝向接近均匀分布，`phi ≈ 0.11`。

#### 阶段一验证要求（至少完成 2 项）

1. 三种 `eta` 的 `phi(t)` 时间序列对比（平滑过的）。
2. 三种 `eta` 的最终集群 quiver 快照（直观看朝向是否同色）。
3. 稳态 `<phi>` 与 `Var(phi)` 柱状图，说明"齐心但僵硬" vs "嘈杂但无共识"的差别。

参考产物：

- `figures/phase1_phi_timeseries.svg`
- `figures/phase1_swarm_eta*.svg`
- `figures/phase1_steady_phi.svg`
- `figures/phase1_phi_variance.svg`

---

### 阶段二：定位临界点（涨落峰值 = 相变指纹）

扫描 `eta ∈ [0.5, 5.5]`，每个噪声跑一段长仿真并统计 warmup 后：

- 平均 `<phi>`；
- 易感性 `chi = N · Var(phi)`；
- Binder 累积量 `U_4 = 1 - <phi^4>/(3 <phi^2>²)`。

然后用至少两种系统尺寸 `N=200` 与 `N=800`（保持密度 `rho` 不变）做有限尺寸对照：`chi` 峰值会在临界附近出现，并随 `N` 变高变尖；Binder 曲线在 `eta_c` 附近形成交叉。

#### 阶段二验证要求（至少完成 2 项）

1. **序参量曲线**：`<phi>(eta)` 从低噪声的 ~1 跨到高噪声的 ~0。
2. **易感性峰值**：`chi(eta)` 出现明显峰，定位 `eta_c`。
3. **有限尺寸效应**：至少比较 `N=200` 与 `N=800`，说明峰位与峰高随 `N` 变化。
4. **Binder 交叉**：不同尺寸的 `U_4(eta)` 在 `eta_c` 附近形成交叉/汇聚区。

参考产物：

- `figures/phase2_order_parameter.svg`
- `figures/phase2_susceptibility.svg`
- `figures/phase2_binder.svg`

---

### 阶段三：临界放大（同一指令 / 阵风，三种命运）

把粒子 `0` 钉成"领飞"无人机，朝向永远是 `pi/2`（北）。比较三种 `eta` 下整个集群与领飞者的对齐随时间的演化：

```python
VicsekParams(leader_index=0, leader_theta=math.pi / 2)
```

再设计一次"阵风脉冲"：

```python
PerturbationSpec(start=300, duration=40, delta_theta=0.25)
```

在脉冲期间所有粒子角度同时被加 `0.25 rad`。比较三种 `eta` 下 `phi(t)` 的响应。

#### 阶段三验证要求（至少完成 2 项）

1. **领飞对齐时间序列**：三种 `eta` 下 `<cos(theta - theta_leader)>(t)` 的演化。
2. **稳态对齐柱状图**：late-time 平均对齐，比较三个 regime。
3. **阵风脉冲响应**：同一个 `delta_theta=0.25`、不同 `eta` 下 `phi(t)` 的扭曲幅度与恢复速度。
4. **集群 quiver 快照**：领飞者用空心圆圈标出，看周围粒子是否被它"染色"。

注意：单个领飞者要拽动 400 架无人机，本身就是个"小信号 vs 大噪声"问题。低噪声下集群锁成一坨，能慢慢被领飞者拽走；近临界下波动太大，单个领飞者的影响很容易被噪声掩盖；高噪声下完全无对齐。这一阶段帮你建立"指令传输的信噪比"直觉，也就为阶段四埋下伏笔——**单纯的低 eta 或临界 eta 都不是万能解，关键是动态地把系统维持在合适的位置**。

参考产物：

- `figures/phase3_alignment_timeseries.svg`
- `figures/phase3_alignment_summary.svg`
- `figures/phase3_perturbation_response.svg`
- `figures/phase3_swarm_eta*.svg`

---

### 阶段四：自组织靠近临界（反馈噪声，不再人工调）

把集群交给一个**只看 `phi`**的"调度员"：

- 最近窗口 `<phi>` 高于目标 `target_phi=0.55` → 加噪声让集群松一点；
- `<phi>` 低于目标 → 减噪声让集群紧一点；
- 目标不是固定某个 `eta`，而是让系统**自己**维持在"有序与无序的边缘"。

打开：

```python
adaptive_noise=True
```

并从多个初始噪声出发（如 `eta_0 ∈ {1.0, 3.5, 5.0}`），观察是否收敛到相近的 `eta` 带。然后用同一个 `delta_theta=0.25` 阵风对比反馈控制器和三档固定 `eta` 控制器的响应。

#### 阶段四验证要求（至少完成 2 项）

1. **多初值收敛**：三条 `eta(t)` 轨迹最终落入相近的窄带，对应 `phi ≈ target_phi`。
2. **稳态质量**：`<phi>` late-time 接近目标值且涨落不发散。
3. **阵风对比**：与三档固定 `eta`（低 / 临界 / 高）对比，自适应控制器的响应应介于"足够大但不锁死"。
4. **易感性对比**：自适应控制器的 `chi` 接近临界 fixed eta，且远高于过冷或过热 fixed eta。
5. **快照**：自适应集群 quiver——既不是单色一坨、也不是彩虹噪点。

> 注意：这里的"自组织临界"是教学版反馈策略，不等同于无限系统 SOC 的严格证明。要严格断言，需要给出多尺寸标度、长时间稳态分布与对照模型检验。

参考产物：

- `figures/phase4_eta_convergence.svg`
- `figures/phase4_phi_convergence.svg`
- `figures/phase4_final_eta.svg`
- `figures/phase4_final_phi.svg`
- `figures/phase4_gust_response.svg`
- `figures/phase4_susceptibility_compare.svg`
- `figures/phase4_swarm_adaptive.svg`

---

## 4. 提交内容

你需要在 `case6/` 下提交：

1. `task.md`（本任务书）；
2. `base/`（保留题目方提供的初始素材）：
   - `vicsek_model.py`、`starter_simulation.py`、`plotting.py`、`README.md`、`__init__.py`
3. `solution/phase1_regimes.py` + `.md`
4. `solution/phase2_critical_sweep.py` + `.md`
5. `solution/phase3_leader_response.py` + `.md`
6. `solution/phase4_self_organization.py` + `.md`
7. `figures/`（至少包含上文每阶段参考产物）

> 依赖：`matplotlib`。其余仅依赖 Python 标准库。

---

## 5. 评分建议

- **正确性（35%）**：是否清楚复现低噪声有序 / 高噪声无序 / 近临界涨落 / 反馈靠近临界。
- **验证质量（30%）**：是否用 `phi`、`chi`、`U_4`、领飞对齐、阵风响应、quiver 快照等多证据互相印证。
- **递进体验（20%）**：是否让读者从直觉观察 → 临界诊断 → 指令传输 → 自适应控制层层深入。
- **工程表达（15%）**：代码是否基于 base、可复现、图表清晰、依赖最小。

完成本任务后，你应该能直观回答：

> 为什么完全锁齐的集群反而不灵敏？为什么人工固定的"临界 eta"在环境一变就失效，需要反馈控制才稳？为什么"既稳又灵"的工程约束本质上就是把系统驻留在相变边缘？
