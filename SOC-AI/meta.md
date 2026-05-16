# SOC-AI：临界态与大模型学习效率假说的理论背景

> 本文件汇总本研究所依赖的物理与机器学习理论，为 `result.md` 中的实验设计与
> 结果解释提供概念基础。文档面向具备本科机器学习与统计物理基础的读者。

---

## 1. 假说的来源与表述

> **"大模型在临界态时，信息处理能力或学习效率最高。"**

这一论断融合了三条相对独立的研究线索：

1. **生物神经科学中的"临界脑"假说**。皮层在大尺度上呈现自组织临界（Self-Organized
   Criticality, SOC）特征：神经雪崩规模 $s$ 的分布服从幂律
   $P(s)\propto s^{-\tau}$，分支比 $\sigma_b\!\approx\!1$。理论与实验都提示，
   临界态时神经群体的**动态范围**、**信息传递能力**和**学习能力**同时达到最大。
2. **递归/混沌网络的"边缘混沌（edge of chaos）"理论**。对随机循环神经网络，
   存在一个控制参数（如权重方差）使得最大 Lyapunov 指数 $\lambda\!=\!0$；
   该相变边界上，网络对扰动的记忆深度最长、计算复杂度最高（Bertschinger &
   Natschläger 2004, Boedecker 等 2012）。
3. **深度前馈网络的"信号传播"理论**。Poole 等（2016）、Schoenholz 等（2017）
   证明：在初始化时，存在使前向激活方差与反向梯度方差同时保持不变的"序参量
   面"。偏离这一面会导致梯度消失（有序相）或爆炸（混沌相），训练效率显著降低。

本研究的目标，是把以上理论叙事**收缩为可被一个小型 Transformer 实证检验的
工作假说**：

> 存在一个控制参数 $g$，使得 (i) 一组多样的临界性诊断指标在 $g^{\!*}$ 附近
> 同时给出"临界"信号；(ii) 同一 $g^{\!*}$ 也最大化在保持训练预算固定时的
> 学习效率（即最小化测试集交叉熵）。

如此表述的好处是：**它是可证伪的**。如果两条结论在数据中并不重合，就应据实
报告。

---

## 2. 控制参数：前向信号增益 $g$

经典 SOC 研究通常用"权重方差"做控制参数。但带 LayerNorm 与残差连接的
Transformer 对权重方差并不敏感：LayerNorm 会把内部信号重新归一化到单位
尺度，残差通路又把信息无条件透传。我们因此采用一个**更直接的可控旋钮**：
对每个残差块的输出乘一个标量 $g$：

$$
x \;\leftarrow\; g\,\bigl(x + \mathrm{Sublayer}(\mathrm{LN}(x))\bigr).
$$

- $g\!<\!1$：每经过一个 block，状态范数被压缩 → 信号在深度方向上衰减
  （**有序相 / 冻结相**）。
- $g\!=\!1$：标准的 pre-norm Transformer。
- $g\!>\!1$：状态范数被放大 → 信号在深度方向上指数膨胀
  （**混沌相 / 发散相**）。

$g$ 仅作用于前向；权重的初始化采用标准 He 法则。这样可以把"动力学相变"和
"参数学习率"两件事干净地分开，便于把损失曲线归因于动力学规整度本身，而
不是初始化方差的大小。

---

## 3. 临界性的四项独立诊断指标

我们使用四个不同物理直觉来源的指标，对同一组网络进行交叉验证。**全部指标
都在权重随机初始化后、训练开始前**采集，因此度量的是**架构-初始化**本身的
动力学性质，与训练耦合解耦。

### 3.1 分支比 $\sigma_b$（信号传播速率）

来自分支过程理论。定义为相邻层激活范数比的几何平均：

$$
\sigma_b \;=\; \exp\!\Bigl[\tfrac{1}{L-1}\!\sum_{\ell=1}^{L-1}\log\!\bigl(\|h_{\ell+1}\|/\|h_\ell\|\bigr)\Bigr].
$$

理论上：$\sigma_b\!\to\!0$ 表示"灭绝"，$\sigma_b\!\to\!\infty$ 表示"爆炸"，
$\sigma_b\!=\!1$ 为临界（信号既不死也不爆）。

### 3.2 最大 Lyapunov 指数 $\lambda$（混沌敏感度）

在嵌入层注入小扰动 $u$（$\|u\|=\varepsilon$），测量经过 $L$ 个 block 之后的
扰动放大率：

$$
\lambda \;=\; \tfrac{1}{L}\,\mathbb{E}_u\!\left[\log\!\bigl(\|f_L(x_0+u)-f_L(x_0)\|/\varepsilon\bigr)\right].
$$

- $\lambda\!<\!0$：有序，扰动指数衰减；
- $\lambda\!=\!0$：临界，扰动多项式增长；
- $\lambda\!>\!0$：混沌，扰动指数放大。

注意：为避免最终 LayerNorm 把幅值信息归一化掉，我们**只对 block 栈做扰动
传播**，不包含 `ln_f`。

### 3.3 雪崩规模分布与幂律指数 $\tau$（SOC 指纹）

定义"活动单元"为 $|h_{\ell,t,i}|>\theta$（$\theta$ 取每层的标准差为单位）。
对每个 $(\ell,t)$ 位置统计活动单元个数 $s$，得到雪崩规模分布 $P(s)$。
在双对数图上做最小二乘拟合 $\log P(s) = -\tau\,\log s + b$，并报告：

- 指数 $\tau$（经典 SOC 沙堆模型给出 $\tau\approx 3/2$，皮层雪崩亦近似如此）；
- 拟合优度 $R^2$（接近 1 才能说"分布像幂律"）。

这是本研究中最严格、**也最容易给出否定证据**的指标。

### 3.4 表示的有效秩（参与比 PR）

最终隐藏状态 $H\in\mathbb{R}^{(B\cdot T)\times C}$ 中心化后做 SVD，定义参与比

$$
\mathrm{PR} \;=\; \frac{(\sum_i s_i)^2}{\sum_i s_i^2}.
$$

PR 度量表示张成的"等效维度"：

- 有序相：表示坍塌到低维（PR 小）；
- 混沌相：表示被噪声搅散、协方差谱平直但因数值不稳定有效维度反而下降；
- 临界相：表示既丰富又结构化，PR 取**极大值**。

PR 极值点在很多场景下是比 $\sigma_b\!=\!1$ 更鲁棒的临界判据。

---

## 4. 学习效率的可校准度量

合成数据使用固定的**二阶 Markov 链**（字母表 20，浓度 0.35）。这样：

- 任意模型在测试集上能达到的最低期望负对数似然等于其**条件熵下界**
  $H_2 = -\sum_{a,b}\pi(a,b)\sum_c P(c\mid a,b)\log P(c\mid a,b)$。
- $H_2$ 可用平稳分布解析地计算（见 `data.py`）。本研究中 $H_2\approx 2.127$
  nats/token，均匀基线 $\log V \approx 2.996$ nats/token。
- 因此，"训练后测试交叉熵 $\mathcal{L}_{\text{test}}$"与"最优熵 $H_2$"
  之差 $\Delta=\mathcal{L}_{\text{test}}-H_2$ 是**绝对刻度**的学习效率度量，
  $\Delta\!=\!0$ 表示已捕获全部可学习结构。

固定训练预算（600 步 × batch 32 × seq 64 ≈ 1.23M tokens、Adam、lr 1.5e-3）后，
不同 $g$ 下的 $\Delta(g)$ 就是该 $g$ 下网络在**单位算力**上的学习效率。

---

## 5. 残差网络的"临界"为何与经典 SOC 不完全等价

文献中的 SOC/edge-of-chaos 结果几乎都建立在**无残差**的循环/前馈网络上。
在残差网络中：

1. 每个 block 输出 $x + \mathrm{Sublayer}(\mathrm{LN}(x))$ 含有一条"恒等通路"，
   信息可以无衰减地透传 → 即使 $\sigma_b>1$，网络仍可正常训练。
2. 反向传播的梯度同样通过残差通路，因此 $\lambda>0$ 不必然导致梯度爆炸——
   只要 Adam 与梯度裁剪能够把每步更新的范数压住。
3. 因此，**"$\sigma_b=1$、$\lambda=0$"等严格 SOC 阈值在残差 Transformer
   中不应被字面套用**。可以预期：经典 SOC 阈值给出的"临界点"与实际
   学习效率最优点之间可能存在**系统偏移**。

这一理论预期在本研究的实验中得到了清晰的实证（见 `result.md` §4）。

---

## 6. 模型与代码

| 文件 | 作用 |
| --- | --- |
| `model.py`        | 微型 pre-norm Transformer，公开 `signal_scale` 控制旋钮 |
| `data.py`         | 二阶 Markov 合成语料，含解析最优熵 |
| `criticality.py`  | 四项临界性诊断指标（分支比 / Lyapunov / 幂律 / 参与比） |
| `experiment.py`   | $g$ 扫描主程序，输出 `results.json` 与 `figures/*.svg` |
| `result.md`       | 实验结果、图表与结论 |
| `meta.md`         | 本文件 |

模型规模：$L\!=\!6$、$d_{\text{model}}\!=\!96$、$n_{\text{head}}\!=\!4$、
$d_{\text{ff}}\!=\!192$，含权重共享后总参数量 **452,928**（远小于 20 M 上限）。

---

## 7. 主要参考文献

- Beggs J. M. & Plenz D. (2003). *Neuronal Avalanches in Neocortical Circuits*. **J. Neurosci.** 23: 11167-11177.
- Bak P., Tang C., Wiesenfeld K. (1987). *Self-Organized Criticality*. **Phys. Rev. Lett.** 59: 381.
- Bertschinger N. & Natschläger T. (2004). *Real-Time Computation at the Edge of Chaos in Recurrent Neural Networks*. **Neural Computation** 16(7): 1413-1436.
- Boedecker J. *et al.* (2012). *Information processing in echo state networks at the edge of chaos*. **Theory in Biosciences** 131: 205-213.
- Poole B., Lahiri S., Raghu M., Sohl-Dickstein J., Ganguli S. (2016). *Exponential expressivity in deep neural networks through transient chaos*. **NeurIPS**.
- Schoenholz S. S., Gilmer J., Ganguli S., Sohl-Dickstein J. (2017). *Deep Information Propagation*. **ICLR**.
- Xiao L., Bahri Y., Sohl-Dickstein J., Schoenholz S. S., Pennington J. (2018). *Dynamical Isometry and a Mean Field Theory of CNNs*. **ICML**.
