# Base 初始素材脚本详解（Case 6：Vicsek 集群）

本目录是题目方提供给玩家的"无人机集群飞行临界实验台"脚手架。后续四个阶段都应基于这里的脚本继续开发，不要另起炉灶。

```
case6/base/
├── vicsek_model.py         # 共用模型 VicsekFlock：自驱粒子 + 邻居对齐 + 噪声 + 领飞 + 反馈
├── starter_simulation.py   # 高噪声混乱基线（必须先跑）
├── plotting.py             # matplotlib 绘图 + 临界统计 + quiver 集群快照工具
└── __init__.py
```

---

## 1. `vicsek_model.py`：场景核心模型

模型是 `N` 个自驱粒子（无人机 / 鱼 / 鸟）在边长 `box_size` 的二维周期性方盒内运动。每个粒子有固定速率 `v0` 与朝向角 `theta`。每一步更新规则：

```
theta_i(t+1) = arg( sum_{j: |x_j - x_i| < r} e^{i theta_j(t)} ) + xi
xi ~ Uniform(-eta/2, +eta/2)
x_i(t+1) = x_i(t) + v0 * (cos theta, sin theta)
```

这就是 1995 年 Vicsek 提出的最小自驱集群模型。它在热力学极限下展现一个**连续的有序–无序相变**：噪声 `eta` 越小，越多粒子能锁到同一方向；噪声 `eta` 越大，朝向越随机。

序参量是**极化度（polarisation）**：

```
phi = | (1/N) * sum_i (cos theta_i, sin theta_i) | ∈ [0, 1]
```

`phi → 1` 表示集群完全同向，`phi → 0` 表示朝向均匀随机分布。临界噪声 `eta_c(rho)` 处的 `phi` 分布最宽、易感性 `chi = N · Var(phi)` 出现峰值。

### 玩家最该关心的旋钮

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `n_agents` | 400 | 粒子数 N |
| `box_size` | 10.0 | 周期方盒边长，配合 N 决定数密度 ρ = N / L² |
| `radius` | 1.0 | 邻居作用半径 r |
| `speed` | 0.4 | 速率 v0（每步移动距离 / 单位） |
| `eta` | 4.5 | 角度噪声幅值（rad），核心控制参数 |
| `steps` | 600 | 总 sweep 数 |
| `warmup` | 150 | 用于丢掉初始瞬态的步数 |

### 阶段三专用：领飞 / 扰动

- `leader_index` + `leader_theta`：把某只粒子的朝向固定为 `leader_theta`（每步重置）。模拟"控制输入"：阶段三测试一架领飞无人机的指令能传播到多远。
- `perturbation = PerturbationSpec(start, duration, delta_theta)`：在指定窗口内给所有粒子加上同一个角度偏移，模拟阵风等外部扰动。

### 阶段四专用：自适应噪声反馈

```python
adaptive_noise = True
target_phi     = 0.55   # 目标极化度（典型在临界附近）
eta_gain       = 0.04   # 反馈增益
feedback_window= 30     # 用最近多少步的均值做估计
```

控制律（每步在更新角度前调用）：

```
phi_obs = mean(recent phi_window steps)
eta <- eta + gain * (phi_obs - target_phi)
```

集群太"团结"（phi 高于目标）→ 加噪声让它松一点；集群太"散漫"（phi 低于目标）→ 减噪声让它紧凑些。结果是 `eta` 自组织到临界附近，无需人工扫描。

---

## 2. `starter_simulation.py`：高噪声混乱基线

默认 `N=400, box_size=10, eta=4.5`：噪声远高于临界值，集群基本无序。脚本会：

1. 跑 600 步、丢掉 warmup；
2. 画极化度时间序列 `starter_phi_timeseries.svg`；
3. 画最终集群 quiver 快照 `starter_final_swarm.svg`（颜色编码朝向角）。

典型输出：

```
[starter] N=400, eta=4.5, density=4.00
[starter] steady phi = 0.112
```

`phi ≈ 0.11` 极小，集群完全没有共识——这是"散漫但灵敏"的极端，也是阶段一的对照基线。

---

## 3. `plotting.py`：临界统计与绘图工具

封装的常用工具：

- `rolling_mean / mean / variance`：常规统计；
- `susceptibility(phi_values, n_agents) = N · Var(phi)`：Vicsek 经典易感性，临界处取峰值；
- `binder_cumulant(phi_values) = 1 - <phi^4>/(3 <phi^2>²)`：四阶 Binder 累积量，不同尺寸曲线在 `eta_c` 附近交叉；
- `plot_lines / plot_dual_axis / plot_bars`：matplotlib 包装；
- `plot_swarm`：把粒子位置 + 朝向画成 quiver（HSV 着色），每个粒子是一个箭头。可选 `leader_index` 用空心圆圈标记领飞者。

绘图全部输出 SVG。

---

## 4. 推荐工作流

1. 先跑 `python case6/base/starter_simulation.py`，看 `case6/figures/` 两张起始图；
2. 阅读本文件 + `vicsek_model.py` 注释，确认你理解周期边界、cell-list 邻居搜索、领飞 / 扰动 / 反馈三种扩展；
3. 进入 `case6/solution/` 按 phase1 → phase4 顺序开发；
4. 每一步用 `figures/` 中的产物做证据。
