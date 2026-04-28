# Phase 4：自组织靠近临界——反馈噪声控制

## 目标

阶段三告诉我们：固定 `eta` 永远在和环境的"信噪比"扭打。本阶段把噪声本身交给一个**只看 `phi`** 的反馈控制器，让它自动维持集群在期望的工作点（典型在临界附近 `target_phi = 0.55`）。

## 控制律

非常简洁，纯本地：

```
phi_obs = mean(phi over last 30 steps)
eta    <- eta + 0.04 * (phi_obs - 0.55)
eta    <- clip(eta, 0.05, 2*pi)
```

集群"太团结"（`phi` 高）→ 加噪声；"太散漫"（`phi` 低）→ 减噪声。

## 实验一：多初值收敛

从 `eta_0 ∈ {1.0, 3.5, 5.0}` 三种截然不同的初始噪声出发：

```
[phase4] eta_0=1.00 -> final eta=3.117, phi=0.541
[phase4] eta_0=3.50 -> final eta=3.105, phi=0.551
[phase4] eta_0=5.00 -> final eta=3.081, phi=0.532
```

三条轨迹**几乎完全收敛**到同一窄带 `eta ≈ 3.10`，对应 `phi ≈ 0.54`，与目标 0.55 非常接近。这就是教学版的"自组织到临界"：不需要知道 `eta_c` 是多少，只要给一个目标 `phi` 与本地观测，系统会自己找到它。

## 实验二：阵风对比

同一个 `delta_theta = 0.25` 阵风，比较自适应控制器与三档固定 `eta`：

```
[phase4] late-time chi (N=400, post-step 300):
   adaptive (eta_0=3.5) : chi=0.665
   fixed eta=1.5        : chi=0.036
   fixed eta=3.5        : chi=1.153
   fixed eta=4.5        : chi=0.885
```

- **固定低 eta**（1.5）：`chi` 极低，集群已锁死；阵风后 `phi` 几乎不动；
- **固定临界 eta**（3.5）：`chi` 最高但**不可控**，对环境变化没有抵抗；
- **固定高 eta**（4.5）：响应大但永远没共识；
- **自适应**：`chi ≈ 0.665`，介于二者之间，**而且无论从哪个初始 `eta_0` 出发都能复现**。这才是工程上的"既稳又灵"。

## 产出图

- `phase4_eta_convergence.svg`：三条 `eta(t)` 轨迹收敛到同一带；
- `phase4_phi_convergence.svg`：`phi(t)` 收敛到目标值（含目标参考水平线）；
- `phase4_final_eta.svg` / `phase4_final_phi.svg`：late-time 数值柱状图；
- `phase4_gust_response.svg`：四档控制器的阵风响应；
- `phase4_susceptibility_compare.svg`：四档控制器的 `chi` 对比；
- `phase4_swarm_adaptive.svg`：自适应集群最终 quiver 快照。

## 为什么这一阶段重要

- 同一物理（Vicsek 集群）能展现两种"到临界"的方式：阶段二的**外部精调**与本阶段的**内部反馈**——后者更符合现实工程：你管不了"今天风多大"，但你管得了"我家无人机要不要再听话一点"。
- 体现 SOC 的核心思想：**不存在零成本的稳定共识**——能稳就不灵敏，灵敏就不稳定。反馈控制把这个 trade-off 暴露在控制器面前，让你**显式设计目标 `phi`** 而不是无意识地撞运气。
- 至此，从远离临界的体感（phase 1）→ 临界精确诊断（phase 2）→ 临界响应的工程意义（phase 3）→ 自组织反馈（phase 4）形成完整闭环。
