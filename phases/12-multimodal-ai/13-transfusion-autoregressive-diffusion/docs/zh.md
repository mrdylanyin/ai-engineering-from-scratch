# Transfusion：一个 Transformer 中的自回归文本 + 扩散图像

> Chameleon 和 Emu3 将一切押在离散 token 上。它们能工作，但量化瓶颈是可见的——图像质量平台低于连续空间扩散模型。Transfusion（Meta, Zhou 等人, 2024年8月）做出相反的赌注：保持图像连续，完全丢弃 VQ-VAE，并用两个损失训练一个 Transformer。文本 token 使用 next-token 预测。图像 patch 使用流匹配/扩散损失。两个目标优化相同权重。Stable Diffusion 3（MMDiT）的底层架构是近亲。本课阅读 Transfusion 的论点，构建一个玩具双损失训练器，并追溯让一个 Transformer 同时做两种工作的注意力掩码。

**Type:** Build
**Languages:** Python (stdlib, two-loss trainer on MNIST-scale toy)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 8 (Generative AI)
**Time:** ~180 minutes

## 学习目标

- 构建一个在单一骨架上运行两个损失（文本上的 NTP、图像 patch 上的扩散 MSE）的 Transformer。
- 解释为什么跨图像 patch 的双向注意力加上跨文本 token 的因果注意力是正确的掩码选择。
- 在算力、质量和代码复杂性上比较 Transfusion 风格（连续图像、扩散损失）和 Chameleon 风格（离散图像、NTP）。
- 说出 MMDiT 的贡献：每个块中模态特定的权重，残差流中的联合注意力。

## 问题

离散 vs 连续图像 token 的辩论比 LLM 更古老。连续表示（原始像素、VAE 潜在变量）保留细节。离散 token（VQ 索引）适配 Transformer 的原生词汇，但在量化步骤丢失细节。

Chameleon / Emu3 选择离散：一个损失、一个架构，但图像保真度受限于分词器质量。

扩散模型选择连续：出色的图像质量，但与 LLM 分离的模型、复杂的噪声调度工程，且与文本生成无法干净整合。

Transfusion 的问题：能否两者兼得？保持图像连续，仍训练一个模型，将两个损失缝合到一个梯度步骤中。

## 概念

### 双损失架构

一个单一的 decoder-only Transformer 处理包含以下内容的序列：

- 文本 token（离散，来自 BPE 词汇）。
- 图像 patch（连续，16x16 像素块通过线性嵌入投影到隐藏维度——与 ViT 编码器的输入相同）。
- `<image>` 和 `</image>` 标签标记连续 patch 的位置。

前向传播运行一次。损失为每个 token 选择两个头之一：

- 文本 token：词汇 logits 头上的标准交叉熵。
- 图像 patch：连续 patch 上的扩散损失——预测每个 patch 上添加的噪声。

梯度流经共享 Transformer 体。两个损失同时改善共享权重。

### 注意力掩码：因果文本 + 双向图像

文本 token 必须是因果的——不能让文本 token 关注未来文本，否则 teacher forcing 会失效。图像 patch 表示一个快照；它们应在同一图像块内双向关注。

掩码：

```
M[i, j] = 1 当：
  (i 是文本且 j 是文本且 j <= i)   # 文本因果
  OR (i 是图像且 j 是图像且 same_image_block(i, j))   # 图像内双向
  OR (i 是文本且 j 是图像且 j < i_image_end)   # 文本关注前面的图像
  OR (i 是图像且 j 是文本且 j < i_image_start)   # 图像关注前面的文本
```

在训练和推理时实现为分块三角掩码。

### Transformer 内的扩散损失

扩散损失是标准的：给图像 patch 加噪声，让模型预测噪声（或等效地预测干净 patch）。Transfusion 的版本使用流匹配——预测从噪声到干净的速度场。

训练期间：
1. 对每个图像 patch x0，采样随机时间步 t。
2. 采样噪声 ε，计算 xt = (1-t) * x0 + t * ε（流匹配的线性插值）。
3. Transformer 预测 v_theta(xt, t)；损失 = MSE(v_theta(xt, t), ε - x0)。
4. 与同一序列的文本 NTP 损失一起反向传播。

推理时，生成：
- 文本 token：标准自回归采样。
- 图像 patch：扩散采样循环（通常 10-30 步），条件化于前面的文本 token。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser 等人, 2024年3月）大约与 Transfusion 同时发布了 MMDiT（多模态扩散 Transformer）。两个架构是兄弟。

MMDiT 的关键差异：

- 每个块中模态特定的权重。每个 Transformer 块对文本 token 和图像 patch 有独立的 Q、K、V 和 MLP 权重。注意力是联合的（跨模态）；其他一切都是模态特定的。
- 整流流训练。一种特定的流匹配变体，已知采样步骤较少，数学比 DDPM 更简单。
- 规模。MMDiT 是 SD3（2B 和 8B 参数变体）的骨干。Transfusion 的论文扩展到 7B。

两者都收敛到同一个核心想法：一个 Transformer 在文本上运行 NTP，在连续图像表示上运行扩散。

### 为什么优于 Chameleon 风格

连续扩散和离散 NTP 在图像生成上的质量差距是可测量的。Transfusion 论文报告：

- 在 7B 参数下，在 FID 上比同规模的 Chameleon 风格模型好 3-5 个点。
- 无需训练分词器——图像编码器更简单（线性投影到隐藏层，与 ViT 输入层相同）。
- 推理可以并行化图像 patch 去噪，不像自回归图像 token。

缺点：Transfusion 是双损失模型，使训练动态更难。损失权重需要调优。NTP 和扩散之间的调度不匹配可能导致一个头主导。

### 下游

Janus-Pro（12.15 课）通过解耦理解和生成的视觉编码器来细化 Transfusion 的想法——理解用 SigLIP，生成用 VQ——同时共享 Transformer 体。Show-o（12.14 课）将扩散替换为离散扩散（掩码预测）。统一生成家族在 Transfusion 后快速分支。

能生成图像的 2026 年生产 VLM——Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成路径——几乎肯定使用了此家族的某种后代。细节是专有的。

## 使用它

`code/main.py` 在一个 MNIST 规模的问题上构建玩具 Transfusion：

- 文本标题是描述数字（0-9）的短整数序列。
- 图像是 4x4 的字节网格。
- 一对共享权重的线性投影充当 Transformer 的替代；文本上的 NTP 损失，噪声 patch 上的 MSE 损失。
- 训练循环交替两个损失，注意力掩码是显式的。
- 生成在一次前向传播中产生文本标题和 4x4 图像。

Transformer 是玩具级的。双损失管道、注意力掩码构建和推理循环是真正的产物。

## 交付

本课产出 `outputs/skill-two-loss-trainer-designer.md`。给定一个新的多模态训练任务（文本 + 图像、文本 + 音频、文本 + 视频），设计双损失调度（损失权重、掩码形状、共享 vs 模态特定块）并标记实现风险。

## 练习

1. Transfusion 风格模型训练 70% 文本 token 和 30% 图像 patch。图像扩散损失幅度约是文本 NTP 损失的 10 倍。什么损失权重能平衡它们？

2. 为序列 `[T, T, <image>, P, P, P, P, </image>, T]` 实现分块三角掩码。将每个条目标记 0 或 1。

3. MMDiT 有模态特定的 QKV 权重。相比 Transfusion 完全共享的 Transformer，这增加多少参数量开销？在 7B 参数下值得吗？

4. 生成：给定文本提示，模型对 50 个 token 运行 NTP，然后命中 `<image>`，然后在 20 个去噪步骤下对 256 个 patch 运行扩散。总共多少次前向传播？

5. 阅读 SD3 论文第 3 节。描述整流流，以及为什么它比 DDPM 用更少推理步骤收敛。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 双损失训练 | "NTP + 扩散" | 单个 Transformer 在同一梯度步骤中优化文本 token 上的交叉熵和连续图像 patch 上的 MSE |
| 流匹配 | "整流流" | 从噪声到干净数据预测速度场的扩散变体；比 DDPM 数学更简单 |
| MMDiT | "多模态 DiT" | Stable Diffusion 3 架构：联合注意力，模态特定 MLP 和 LN |
| 分块三角掩码 | "因果文本 + 双向图像" | 文本跨位置因果但图像区域内双向的注意力掩码 |
| 连续图像表示 | "无 VQ" | 图像 patch 为实值向量，非整数码本索引 |
| 速度预测 | "v-参数化" | 网络输出是噪声和数据之间的速度场，而非噪声本身 |

## 延伸阅读

- [Zhou 等人 — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser 等人 — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao 等人 — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie 等人 — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
