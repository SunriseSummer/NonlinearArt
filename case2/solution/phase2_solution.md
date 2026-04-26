# 阶段二参考实现详解（`phase2_solution.py`）

## 1. 阶段目标回顾

阶段二要求：在 base 模型基础上**增加反馈机制**，让系统不靠手动精调
就能停在临界态附近，并满足三条验证目标中的至少两条：

1. **稳态存在**：平均应力进入稳定波动区，而非发散；
2. **长尾雪崩统计**：事件大小 / 持续时间在双对数下都呈多尺度分布；
3. **鲁棒性**：不同初值（`seed`、初始 `alpha`）下仍收敛到相近统计态。

参考实现一次性给出全部三条证据。

---

## 2. SOC 反馈机制：自适应保守度 + 异质静态阈值

`fault_model.py` 已经在 `FaultParams` 中预留了 SOC 入口：

```python
adaptive: bool = False        # 打开后：alpha 受滑动窗口控制器调节
target_size: float = 4.0      # 目标滑动平均级联规模
adapt_rate: float = 8e-4      # 比例增益（小一些更稳）
activity_window: int = 250    # 滑动窗口长度
alpha_min: float = 0.10
alpha_max: float = 0.245
heterogeneity: float = 0.0    # 静态阈值随机扰动半宽，阶段二开到 0.20
```

控制器在每个宏观步追加当前级联规模，窗口满（250 步）后计算窗口均值
`<s>_window` 与 `target_size = 4.0` 的差，按
`alpha ← clip(alpha - rate * (<s>_window - target_size), [α_min, α_max])`
更新。物理上：

- 雪崩偏大 → 控制器减小 `alpha`（一次倾倒重新分配的份额变少 → 更耗散）；
- 雪崩偏小 → 控制器增大 `alpha`（更接近完全保守 → 更易扩展）。

异质阈值 `heterogeneity=0.20` 意味着每个格点静态阈值
`tau_static[i][j] ∈ [0.80, 1.20]`，模拟断层各处“强度不均”，是产生
**真正长尾分布**而不是退化型规则雪崩的关键。

> 与 case1 的对照：case1 的反馈控制器作用于碰撞概率 `spill_prob`，目标
> 是“维持平均负载”；这里反馈作用于**保守度** `alpha`（一个有清晰物理
> 含义的、决定 OFC 普适类临界点位置的参数），目标是“维持平均事件大
> 小”——更接近 SOC 文献里 self-tuning 的写法。

---

## 3. 三个参考实验

### 3.1 主运行：稳态 + alpha 自组织

```python
main_params = _make_adaptive_params(alpha0=0.10, seed=2026)
main_res = FaultStressSystem(main_params).run()

plot_dual_axis(
    out=...,
    left={"y": main_res.mean_stress, ...},
    right={"y": main_res.alpha_series, ...},
)
```

参考图 `phase2_stress_and_alpha.svg`：

- 左轴蓝线：平均应力先快速爬升，越过 `warmup` 后稳定波动在
  `0.55 ± 0.02` 附近——证据 1 “稳态存在”。
- 右轴紫线：`alpha` 从初值 0.10 被控制器逐步抬升到 0.17~0.18，并在那
  里稳定波动；这就是“系统自己找到了临界点附近”。

### 3.2 多尺度雪崩统计

参考图 `phase2_avalanche_dist.svg` 把三条线放在一起：

- 玫红 PDF P(s) 与绿色 CCDF P(S≥s) ：在 `s ∈ [1, 200]` 上呈直线段，
  `power_law_mle(sizes, smin=4)` 给出 `τ ≈ 2.25`；
- 棕色 PDF P(T)：持续时间也跨越多个数量级——证据 2 “长尾分布”。

> 注意：自适应控制器把 `<s>` 钉在 4 附近，**会把分布尾巴往里压一些**，
> 所以 τ 比 phase1 临界点（1.75）略大。这是预期行为，文献也观察到自
> 调节 OFC 的 τ 会随 `target_size` 变化。

### 3.3 鲁棒性：3 个初值收敛到同一终态

```python
robustness_runs = [
    dict(alpha0=0.10, seed=2026, color="#1f77b4"),
    dict(alpha0=0.18, seed=99,   color="#ff7f0e"),
    dict(alpha0=0.235, seed=7,   color="#2ca02c"),
]
```

参考图 `phase2_robustness.svg`：三条 `alpha(t)` 轨迹起点完全不同，
但都在前 4000 步收敛到相同的波动区，late-time 平均：

```
alpha0=0.100 seed=2026 -> <alpha>=0.173, <s>=3.90, tau~2.23
alpha0=0.180 seed=  99 -> <alpha>=0.177, <s>=3.87, tau~2.27
alpha0=0.235 seed=   7 -> <alpha>=0.174, <s>=3.90, tau~2.26
```

`<alpha>` 三者在 ±0.005 内一致，`<s>` 全在 `target_size = 4` 附近，τ
也几乎相同——证据 3 “鲁棒性”。

### 3.4 应力场快照

`phase2_stress_field.svg` 是仿真结束时的应力热力图。和 `starter_*` 那
张比，会更明显地出现“几乎到达阈值的连续大斑块”——SOC 临界态下的
长程关联结构。

---

## 4. 与 case1 的延续与差异

| 维度 | Case 1（交通沙堆） | Case 2（断层应力 / OFC） |
| --- | --- | --- |
| 状态变量 | 整数 load | 连续 stress |
| 控制参数 | `spill_prob` | `alpha`（保守度，OFC 普适类核心） |
| 验证统计 | 单一幂律 P(s) | Gutenberg–Richter + Omori |
| SOC 反馈 | 比例控制 `spill_prob` 锁定平均负载 | 比例控制 `alpha` 锁定平均事件大小，叠加摩擦愈合与异质阈值 |
| 评分难度 | 入门 | 进阶 |

阶段二参考实现没有再用 case1 的“目标平均负载”反馈，因为 OFC 的 *extremal*
驱动天然会把负载压在临界附近（`<s>` 才是真正的相变序参量）。这也是
case2 在物理意味上比 case1 更接近“真实板块力学”的原因。

---

## 5. 验收清单

- [x] 稳态存在（应力稳定在 ~0.55，alpha 收敛到 ~0.175）
- [x] 雪崩 size / duration 多尺度（PDF 与 CCDF 在双对数下直线段）
- [x] 鲁棒性（三组 `alpha0×seed` 都收敛到 ⟨alpha⟩≈0.175 与同一 τ）

题目要求至少 2 项，这里 3 项一并交付，方便玩家做对照。
