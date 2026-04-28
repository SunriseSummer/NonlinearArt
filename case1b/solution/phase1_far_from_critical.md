# 阶段一参考实现详解（`phase1_far_from_critical.py`）

## 1. 阶段目标回顾

阶段一要让玩家**亲自感受**保守调度的代价：网络稳定，但效率没拉起来。
不改动模型机制，只通过给同一个 `RushHourTrafficSystem`
传入不同的 `inflow_rate` 来回答："如果我把入口流量从 0.6 慢慢提到 1.4，
路网会发生什么？"

我们做的是 **inflow 扫描**（不是 case1 风格的 spill_prob 扫描），因为
这一阶段的故事是"早高峰刚开始、车流还不大"，因此把 `inflow_rate` 当
作主变量更直观。

## 2. 核心流程

脚本固定其它参数：

```python
common = dict(L=20, threshold=6, spill_prob=0.10, dissipation=0.25,
              steps=4500, warmup=800, seed=2026)
```

然后跑三组：

```python
configs = [
    ("inflow=0.6 (very conservative)", 0.6, "#2ca02c"),
    ("inflow=1.0 (starter baseline)",  1.0, "#1f77b4"),
    ("inflow=1.4 (mild push)",         1.4, "#ff7f0e"),
]
```

每组运行后从 `RushHourTrafficSystem(...).run()` 拿到结果，统计预热后的
稳态指标：

- `<load>`：平均负载（≈ 平均延误代理）；
- `<thrpt>`：吞吐量；
- `<cong_range>`：拥堵传播范围。

## 3. 关键观察

终端打印（实测，可复现）：

```
inflow=0.6 (very conservative)    2.482   0.235     0.074
inflow=1.0 (starter baseline)     2.513   0.489     0.158
inflow=1.4 (mild push)            2.531   0.744     0.234
```

可以读出：

1. **平均负载几乎不变（~2.5）**——保守 `spill_prob=0.10` 下，路口几乎
   不会被推到阈值以上。
2. **吞吐量随 inflow 单调增加**，倍率约 0.5——这是因为 base 模型保留了
   case1 风格的"扩散失败"机制（车辆被吸收/丢弃，不计入 served）；这
   一阶段的关键在于"加 demand 不会触发崩溃"，吞吐量曲线是平滑、可预
   测的线性区。
3. **拥堵传播范围从 0.07 升到 0.23**——还远没到阶段二定义的 `0.25`
   stress edge，仍在保守区间内。

这正对应任务书阶段一的关键判断："稳但低效"。

## 4. 输出图

- `phase1_far_from_critical.svg`：三组负载时间序列叠图，确认稳态都
  停在 ~2.5，且抖动幅度都很小。
- `phase1_throughput_vs_inflow.svg`：稳态吞吐量条形图，明确"加 demand
  ⇒ 吞吐量线性上涨"——这是没有触发临界的"线性区"。

## 5. 复用与扩展

- 想看更激进的保守区间？把 `spill_prob` 调到 0.05，会发现连"线性区"都
  开始漏车（吞吐量 / inflow 显著 < 1）——这是模型对"道路太窄"的反映。
- 想直接把这个脚本的工具函数搬到自己的实验？`_steady` 的窗口约定（去
  掉前 `warmup` 步）整个 case1b 都通用。
