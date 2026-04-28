# Phase 3 解读：临界放大与临界慢化

本阶段对三种温度施加同一个弱磁场脉冲：

```python
FieldPulseSpec(start=500, duration=120, delta_h=0.035)
```

比较对象：

- 低温有序相：本来就强烈有序，弱场很难改变整体状态。
- 临界附近：大团簇处在重排边缘，弱场最容易诱发宏观响应。
- 高温无序相：热噪声强，响应快但难以保留。

输出图：

- `../figures/phase3_field_pulse_magnetisation.svg`：三段 `m(t)` 脉冲响应。
- `../figures/phase3_activity_burst.svg`：脉冲期间接受翻转数。
- `../figures/phase3_pulse_response_summary.svg`：最大响应幅度柱状图。

核心结论：临界点附近的高磁化率不是抽象定义，而是“同样小的外场能造成更大、拖尾更长的宏观变化”。这同时意味着高灵敏度和高风险。
