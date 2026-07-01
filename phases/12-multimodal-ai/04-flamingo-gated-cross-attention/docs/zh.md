# Flamingo 与用于少样本 VLM 的门控交叉注意力

> DeepMind 的 Flamingo（2022）率先做了两件事。它展示了单个模型可以处理任意交错序列的图像、视频和文本。它还展示了 VLM 可以进行上下文学习——给出一个包含三个（图像、标题）样本对的少样本提示，模型可以在没有任何梯度步骤的情况下为一张新图像生成标题。其机制是：门控交叉注意力层，插入在冻结 LLM 的现有层之间，带有可学习的 tanh 门，初始为零，以在初始化时保留 LLM 的文本能力。本课通读 Flamingo 的 Perceiver 重采样器和门控交叉注意力架构——这是 Gemini 交错输入和 Idefics2 视觉 token 的祖先。

**Type:** Learn
**Languages:** Python (stdlib, gated cross-attention + Perceiver resampler demo)
**Prerequisites:** Phase 12 · 03 (BLIP-2 Q-Former)
**Time:** ~120 minutes

## 学习目标

- 解释门控交叉注意力如何通过 tanh(gate) = 0 在初始化时保留冻结 LLM 的文本能力。
- 走完一个 Perceiver 重采样器：N 个图像 patch → K 个固定的"潜在"query，通过交叉注意力。
- 描述 Flamingo 如何使用因果掩码处理交错图像-文本序列，并尊重图像位置。
- 复现一个少样本多模态提示结构（3 个图像-标题示例，然后一个查询图像）。

## 问题

BLIP-2 将 32 个视觉 token 喂给冻结 LLM 的输入层。对每个提示一张图像而言是可行的。但如果你想喂入*大量*与文本交错的图像，比如"这是图像 A，请写标题；这是图像 B，请写标题；现在这是图像 C，请写标题"？LLM 的 self-attention 将需要在一个流中处理图像 token 和文本 token，哪些位置可以关注哪些图像变成了复杂问题。

Flamingo 的答案：完全不要改变 LLM 的输入流。在现有的 LLM 块之间插入额外的交叉注意力层。文本 token 仍然像往常一样经过 LLM 的因果 self-attention。每隔几层 LLM 块，文本 token 还通过一层新的门控层交叉关注图像特征。门（初始化为零）意味着在第零步，新层是无操作——模型行为与预训练 LLM 完全一致。随着训练的进行，门打开，视觉信息开始流动。

Flamingo 回答的第二个问题：如何处理每个提示中可变数量的图像（0、1 或多个）？一个 Perceiver 重采样器——一个小型交叉注意力模块，接收任意数量的 patch，产生固定数量的视觉潜在 token。LLM 交叉注意力层看到的形状始终相同，不管提示中有多少图像。

## 概念

### 冻结 LLM

Flamingo 从一个冻结的 Chinchilla 70B LLM 开始。全部 70B 权重不动。现有的文本 self-attention 和 FFN 正常运行。

### Perceiver 重采样器

对于提示中的每张图像，ViT 产生 N 个 patch token。Perceiver 重采样器有 K 个固定的可学习潜在变量（Flamingo 使用 K=64）。每个重采样器块包含两个子步骤：

1. 交叉注意力：K 个潜在变量关注 N 个 patch token（Q 来自潜在变量，K/V 来自 patch）。
2. 潜在变量内部的 self-attention + FFN。

经过 6 个重采样器块后，输出为 K=64 个维度为 1024 的视觉 token，无论 ViT 产生了多少个 patch。一张 224x224 图像（196 个 patch）和一张 480x480 图像（900 个 patch）的输出都是 64 个重采样器 token。

对于视频，重采样器按时间维度应用：每帧的 patch 产生 64 个潜在变量，时间位置编码让模型区分 t=0 和 t=N。完整视频变为 T * 64 个视觉 token。

### 门控交叉注意力

在冻结 LLM 的每 M 层之间（Flamingo 使用 M=4），插入一个新的门控交叉注意力块：

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是一个可学习的标量，初始化为零。
- `tanh(0) = 0`，所以在初始状态下门控分支贡献为零。
- 随着 `alpha` 远离零，交叉注意力贡献平滑增长。
- 残差连接意味着即使门完全打开，也不会覆盖 LLM 的文本表示；只是在上面添加视觉信息。

这是 Flamingo 中最重要的设计选择：视觉条件化是叠加的、门控的，在初始化时为零。一个第 0 步的 Flamingo 在纯文本输入上就是一个完美的 Chinchilla 70B。

### 用于交错输入的掩码交叉注意力

在像 "<image A> caption A <image B> caption B <image C> ?" 这样的提示中，每个文本 token 只能看到它在序列中前面出现的图像。交叉注意力掩码强制执行：位置 `t` 的文本 token 只关注图像索引 `i < i_t` 的重采样器 token，其中 `i_t` 是位置 `t` 之前最近图像。选择"只看到最后一张前面的图像"还是"看到所有前面的图像"都是有效选择；Flamingo 选择了前者。

### 上下文少样本学习

Flamingo 提示看起来像：

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

模型看到补全模式并输出 "bird"（或图像3 实际展示的内容）。没有梯度步骤。冻结 LLM 的上下文学习能力通过门控交叉注意力得以传递——这是论文的核心卖点和它的重要性所在。

### 训练数据

Flamingo 在三个数据集上训练：

1. MultiModal MassiveWeb (M3W)：4300 万交错图像和文本的网页，按阅读顺序重建。
2. 图像-文本对 (ALIGN + LTIP)：44 亿对。
3. 视频-文本对 (VTP)：2700 万个短视频片段。

OBELICS（2023）是交错网页语料的开源复现，Idefics、Idefics2 和大多数开源"Flamingo 风格"模型都在其上训练。

### OpenFlamingo 和 Otter

OpenFlamingo（2023）是开源复现。架构相同（Perceiver 重采样器 + 门控交叉注意力，在冻结 LLaMA 或 MPT 上）。checkpoint 有 3B、4B、9B。质量落后于 Flamingo，因为 LLM 基础较小和数据较少。

Otter（2023）在 OpenFlamingo 的基础上，在 MIMIC-IT（多模态指令数据集）上进行指令微调，展示了门控交叉注意力也适用于指令跟随。

### 后代

- Idefics / Idefics2 / Idefics3：Hugging Face 的门控交叉注意力谱系，逐步简化（Idefics2 去掉了重采样器，改用自适应池化的直接 patch token）。
- Flamingo 到 Chameleon 的过渡：到 2024 年许多团队转向早期融合（12.11 课）；Flamingo 风格的门控交叉注意力在需要骨干冻结的生产中仍在使用。
- Gemini 的交错输入：概念上继承了 Flamingo 的交错格式灵活性，尽管确切机制是专有的。

### 与 BLIP-2 的对比

| | BLIP-2 | Flamingo |
|---|---|---|
| 视觉桥梁 | Q-Former 在输入端一次 | 每 M 层的门控交叉注意力 |
| 视觉 token | 每图像 32 个 | 每图像每交叉注意力层 64 个 |
| 冻结 LLM | 是 | 是 |
| 少样本上下文 | 弱 | 强——论文的核心亮点 |
| 交错输入 | 原生不支持 | 是，设计目标 |
| 训练数据 | 1.3 亿对 | 13 亿对 + 4300 万交错页面 |
| 参数量 | 1.88 亿可训练 | ~100 亿可训练（交叉注意力层） |
| 算力 | 在 8 块 A100 上几天 | 在数千块 TPUv4 上几周 |

预算内做单图像 VQA 选 BLIP-2。交错、少样本或多图像推理选 Flamingo/Idefics2。

## 使用它

`code/main.py` 演示了：

1. 在 36 个伪 patch token 和 8 个可学习潜在变量上的 Perceiver 重采样器（纯 Python 交叉注意力）。
2. 带有 `alpha = 0` → 输出等于输入（LLM 不变）的门控交叉注意力步骤，然后 `alpha = 2.0` → 视觉贡献混入。
3. 一个交错掩码构建器，生成 "(image 1) (text 1) (image 2) (text 2)" 序列的 2D 注意力掩码。

## 交付

本课产出 `outputs/skill-gated-bridge-diagnostic.md`。给定一个开源 VLM 的配置（有无重采样器、交叉注意力频率、门方案），识别 Flamingo 谱系元素并解释冻结策略。用于调试为什么微调后文本性能下降（答案：门打开得太快太宽）。

## 练习

1. 计算 Flamingo-9B 的视觉参数量：9B LLM + 14 亿门控交叉注意力层 + 6400 万重采样器。总参数中可训练占比多少？

2. 在 PyTorch 中实现门控残差 `y = tanh(alpha) * cross + x`。通过实验展示在 `alpha=0` 初始化时 `y==x` 严格成立。

3. 阅读 OpenFlamingo 第 3.2 节（arXiv:2308.01390）关于如何处理每个提示中有不同图像数量的 batch。描述填充策略。

4. 为什么 Flamingo 的交叉注意力掩码让文本 token 只关注*最近一张*前面的图像，而不是所有前面的图像？阅读 Flamingo 论文第 2.4 节，解释权衡。

5. 上下文少样本：为一个新 Flamingo 变体构建一个包含 4 个"图像 → 主要物体的颜色"示例的提示。描述准确率随示例数量从 0 到 8 变化的预期模式。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Perceiver 重采样器 | "固定潜在交叉注意力" | 从可变数量的输入 patch 产生 K 个固定 token 的模块 |
| 门控交叉注意力 | "Tanh 门控桥梁" | 残差层 `y = tanh(alpha)*cross + x`，可学习 alpha，初始化为 0 |
| 交错输入 | "混合序列" | 图像和文本按阅读顺序自由混合的提示格式 |
| 冻结 LLM | "LLM 无梯度" | 文本 LLM 的权重不更新；只有重采样器 + 交叉注意力层训练 |
| 少样本 | "上下文示例" | 在提示中给出几个（图像，答案）对；模型无需微调即可泛化 |
| OBELICS | "交错网页语料" | 1.41 亿按阅读顺序包含图像和文本的网页的开源数据集 |
| Chinchilla | "70B 冻结基础" | Flamingo 的冻结文本 LLM，来自 DeepMind 的 Chinchilla 论文 |
| 门调度 | "alpha 如何移动" | 交叉注意力门在训练期间打开的速率 |
| 交叉注意力频率 | "每 M 层" | 多长时间插入一次门控交叉注意力块；Flamingo 使用 M=4 |
| OpenFlamingo | "开源复现" | MosaicML/LAION 开源的 3-9B checkpoint；架构与 Flamingo 相同 |

## 延伸阅读

- [Alayrac 等人 — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始论文。
- [Awadalla 等人 — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — 开源复现。
- [Laurençon 等人 — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — 交错网页语料。
- [Jaegle 等人 — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 通用 Perceiver 架构。
- [Li 等人 — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — 指令微调的 Flamingo 后代。
- [Laurençon 等人 — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo 方法的现代简化。
