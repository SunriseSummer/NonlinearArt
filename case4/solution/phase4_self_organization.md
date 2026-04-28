# Phase 4 解读：反馈温控下的自组织临界边缘

本阶段打开 `adaptive_temperature=True`，让系统不再手工固定温度，而是根据最近窗口的 `|m|` 自动调温：

- `|m|` 太高：薄膜过冷、过有序，升温。
- `|m|` 太低：薄膜过热、过无序，降温。
- 目标：让系统停留在“有序 / 无序边缘”的可响应区域。

脚本从 `T0=1.4/3.2/4.0` 三个初温出发，比较收敛轨迹，并把反馈态的活动分布与固定低温 / 高温对照。

输出图：

- `../figures/phase4_temperature_convergence.svg`：多初值 `T(t)` 收敛。
- `../figures/phase4_magnetisation_convergence.svg`：`|m|` 被维持在目标带附近。
- `../figures/phase4_activity_distribution.svg`：反馈态活动分布与静态温度对照。
- `../figures/phase4_summary.svg`：late-time 温度汇总。

注意：这是教学用“自组织靠近临界”机制。严格 SOC 还需要更长时间、更大尺寸和标度检验；但它已经展示了关键工程思想：不要死守固定参数，而要用反馈把系统维持在高灵敏度边缘。
