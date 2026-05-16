# SOC-AI-2：临界态与深层 MLP 学习效率的理论背景

> 本文件汇总本研究所依赖的物理与机器学习理论，为 `result.md` 中的实验设计与
> 结果解释提供概念基础。文档面向具备本科机器学习与统计物理基础的读者。
>
> SOC-AI-2 与 SOC-AI 共享同一份待检验假说，但把网络架构从 Transformer 换成
> **经典多层前馈神经网络（MLP）**，数据集从合成 Markov 语言换成
> **Fashion-MNIST 服饰图像识别**（28×28 灰度图、10 类）。这样做的目的，
> 是把假说放在它的**理论原产地**——非残差、非归一化的深层网络——上检
> 验：在该最有利的"标准舞台"上，经典 SOC 阈值能否同时给出"临界态"判读
> 与"最佳学习效率"。
>
> 早期版本曾使用 MNIST 手写数字识别作为基准；但 MNIST 过于简单，
> $\sigma\!\in\![0.7,\,1.3]$ 全域都能达到 ~95% 准确率，临界态与非临界态的
> 差距被压缩到仅 4 个百分点。改用 Fashion-MNIST 后，最优区间与两端的
> 差距放大到约 18–19 个百分点，更清晰地暴露出"临界 ↔ 最优"的对应关系
> （详见 `result.md` §2）。

---

## 1. 假说与本节研究的定位

待检验的命题：

> **"大模型处于临界态时，信息处理能力或学习效率最高。"**

SOC-AI（Transformer 实验）已经表明：在带残差和 LayerNorm 的现代架构上，
经典 SOC 阈值（分支比 = 1、Lyapunov 指数 = 0、雪崩分布幂律）**与最佳学
习效率点存在系统偏移**。这一偏移可以从机理上解释（残差通路绕过了信号
衰减/放大）。

但这只能说明"经典阈值不直接适用于残差网络"，**不足以证伪原假说**。为了
公正地检验假说，我们必须把它放在经典理论原本针对的设置里：

| 设置 | 结构特征 | 经典 SOC 是否适用 |
| --- | --- | --- |
| 循环神经网络 / 随机张量网络 | 非残差、无归一化 | ✅ 直接适用（Poole 2016、Bertschinger 2004） |
| 深层前馈网络（本节 SOC-AI-2） | 非残差、无归一化 | ✅ 直接适用 |
| Transformer（SOC-AI） | 残差 + LayerNorm | ❌ 存在系统偏移 |

故 SOC-AI-2 的目标是：在**经典理论适用域**内，看假说是否成立。

---

## 2. 控制参数：权重初始化增益 σ

经典 MLP 的"临界控制旋钮"就是权重的初始化标准差。我们采用 He 法则的
增益形式：

$$
W_{ij} \;\sim\; \mathcal{N}\!\Bigl(0,\;\sigma^{2}\cdot \tfrac{2}{n_{\text{in}}}\Bigr).
$$

其中 $n_{\text{in}}$ 是上层宽度。**$\sigma$ 是 SOC-AI-2 实验的唯一控制旋钮**。

理论上（Poole, Lahiri, Raghu, Sohl-Dickstein, Ganguli 2016）：

- ReLU 激活 + 标准 He 初始化对应 $\sigma=1$；此时每层活动方差守恒，**信号
  既不衰减也不爆炸**——这正是"边缘混沌（edge of chaos）"或 SOC 的临界点。
- $\sigma<1$：**有序相 / 冻结相**，活动方差按 $\sigma^{2L}$ 指数衰减，深
  层网络的前向激活与反向梯度都消失。
- $\sigma>1$：**混沌相 / 发散相**，活动方差按 $\sigma^{2L}$ 指数爆炸，前
  向被噪声主导，反向梯度也爆炸。

由于 MLP 没有残差或归一化通路，这个一维相图理论上应当**清晰、单调、可被
四项独立指标交叉验证**。

---

## 3. 网络结构

为了让信号传播效应能够在深度方向上累积、放大临界态与非临界态之间的差距，
我们采用**深而窄**的 MLP：

$$
x\,\to\,\mathrm{Linear}_{784\to128}\,\to\,\mathrm{ReLU}\,\to\,\underbrace{(\mathrm{Linear}_{128\to128}\,\to\,\mathrm{ReLU})}_{\times 11}\,\to\,\mathrm{Linear}_{128\to10}.
$$

| 超参 | 取值 |
| --- | --- |
| 输入维度 | 784（Fashion-MNIST 28×28 展平） |
| 类别数 | 10 |
| 隐藏宽度 $W$ | 128 |
| 隐藏层数 $L$ | 12 |
| 激活函数 | ReLU |
| **正则化** | **无 BatchNorm / 无 Dropout / 无残差** |
| 总参数量 | **283 402**（约 0.28 M，远低于 20 M 上限） |

刻意保持架构的"裸"——任何归一化层都会把 $\sigma$ 旋钮的影响抵消掉。

---

## 4. 临界性的四项独立诊断指标

四项指标的定义与 SOC-AI 完全相同（见 `criticality.py`）：

1. **分支比 $\sigma_b$**：相邻层归一化激活范数比的几何平均；理论临界值 1。
2. **最大 Lyapunov 指数 $\lambda$**：把输入加微扰，统计 $L$ 层之后扰动的对数
   放大率，再除以 $L$。理论临界值 0。
3. **激活雪崩规模幂律 $(\tau,\,R^2)$**：以每层标准差为阈值定义"活动单元"，
   统计每个样本上的活动单元个数 $s$，在双对数下拟合 $P(s)\propto s^{-\tau}$；
   $R^2$ 越接近 1 表示越接近幂律。
4. **表示有效秩（参与比 PR）**：最终隐藏状态协方差的奇异值参与比，刻画
   表示张成的等效维数。

所有指标都**在权重随机初始化之后、训练开始之前**测量，因此度量的是
"架构 + 初始化"的固有动力学性质。

> **关于雪崩指标在 ReLU MLP 上的一项理论性质**：
> 对于无偏置或偏置可被忽略的 ReLU MLP，把所有权重整体乘 $\sigma$ 会让每层
> 的激活精确地乘 $\sigma^\ell$（其中 $\ell$ 是层索引）。由于阈值
> $\theta=z\cdot \mathrm{std}(h_\ell)$ 也随之同比例放大，**活动单元的集合
> 在 $\sigma$ 下是不变的**。因此 $\tau$、$R^2$、PR 三项指标对 $\sigma$
> 几乎**完全不敏感**，仅在 $\sigma$ 极小（信号被数值下溢吃掉）时才出现
> 变化。这是 ReLU 的尺度对称性 $\mathrm{ReLU}(cx)=c\,\mathrm{ReLU}(x)$
> $(c>0)$ 的直接推论，并不是实现 bug，而是 ReLU 架构本身的一项性质。
>
> 反之，$\sigma_b$ 与 $\lambda$ 直接度量幅值的层间变化率，因此对 $\sigma$
> 非常敏感、不受尺度对称性约束——它们才是这个架构上分辨临界态的"主力指标"。
> 这一点也指导 `result.md` §3 的指标解读。

---

## 5. 学习效率的度量

MNIST 是分类任务，提供**两个互补的学习效率指标**（Fashion-MNIST 与 MNIST 在
类别数、样本数、像素归一化上完全一致，因此下述定义与上界对两者完全相同）：

- **测试交叉熵** $\mathcal{L}_{\text{test}}$（nats/sample）。下界为零（确
  定性正确预测），均匀分类的上界为 $\log 10\approx 2.303$ nats。
- **测试 top-1 准确率**。均匀基线 10%，完美分类 100%。

两个指标都受同一组 $\sigma$ 影响，但准确率对 logits 的精确数值不敏感、
交叉熵则对置信度敏感。两者相互佐证可以排除"是否只是温度校准的副作用"等
解释。

固定训练预算：Adam（lr = 1e-3），batch = 128，800 步，梯度二范数裁剪到 1.0。
800 步 × 128 ≈ 10.2 万样本，约 1.7 个 epoch。

---

## 6. 残差 vs. 非残差网络的对照逻辑

SOC-AI-2 和 SOC-AI 构成一对**控制实验**：

| 维度 | SOC-AI | SOC-AI-2 |
| --- | --- | --- |
| 架构 | 残差 + LayerNorm Transformer | 非残差、无归一化 ReLU MLP |
| 控制旋钮 | 前向信号放大 $g$ | 权重初始化增益 $\sigma$ |
| 数据 | 二阶 Markov 合成语言 | Fashion-MNIST 服饰图像 |
| 经典 SOC 阈值应该适用？ | ❌ | ✅ |

如果"临界 ↔ 最佳学习效率"是个**普适规律**，则两个实验都应给出
肯定结论；如果它是**架构敏感**的，则可能只在其中一个上成立。**我们承诺
按实验真实结果汇报，不预设结论**。结果见 `result.md`。

---

## 7. 代码与运行

| 文件 | 作用 |
| --- | --- |
| `model.py`        | 深层 ReLU MLP，公开 `init_gain` 控制旋钮 |
| `data.py`         | Fashion-MNIST 加载器（从 GitHub 镜像下载 idx 文件并解析） |
| `criticality.py`  | 四项临界性诊断指标 |
| `experiment.py`   | $\sigma$ 扫描主程序，输出 `results.json` 与 `figures/*.svg` |
| `result.md`       | 实验结果、图表与结论 |
| `meta.md`         | 本文件 |

运行：

```bash
cd SOC-AI-2
python data.py          # 触发首次下载并打印 shape 校验
python experiment.py    # 完整扫描，约 1 min（2 核 CPU）
```

依赖：`torch`、`numpy`、`matplotlib`（不要求 `torchvision`，Fashion-MNIST
由 `data.py` 内置的下载器和 idx 解析器处理）。

---

## 8. 主要参考文献

- Bak P., Tang C., Wiesenfeld K. (1987). *Self-Organized Criticality*. **Phys. Rev. Lett.** 59: 381.
- Beggs J. M. & Plenz D. (2003). *Neuronal Avalanches in Neocortical Circuits*. **J. Neurosci.** 23: 11167-11177.
- Bertschinger N. & Natschläger T. (2004). *Real-Time Computation at the Edge of Chaos in Recurrent Neural Networks*. **Neural Computation** 16(7): 1413-1436.
- He K., Zhang X., Ren S., Sun J. (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*. **ICCV**.
- LeCun Y., Bottou L., Bengio Y., Haffner P. (1998). *Gradient-Based Learning Applied to Document Recognition*. **Proc. IEEE** 86(11): 2278-2324. *(原始 MNIST 来源；本研究的早期版本使用此数据集。)*
- Xiao H., Rasul K., Vollgraf R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. **arXiv:1708.07747**. *(本研究当前所用数据集。)*
- Poole B., Lahiri S., Raghu M., Sohl-Dickstein J., Ganguli S. (2016). *Exponential expressivity in deep neural networks through transient chaos*. **NeurIPS**.
- Schoenholz S. S., Gilmer J., Ganguli S., Sohl-Dickstein J. (2017). *Deep Information Propagation*. **ICLR**.
