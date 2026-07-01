# CLIP 与对比式视觉-语言预训练

> OpenAI 的 CLIP（2021）证明了一个足以驱动接下来五年的想法：仅使用嘈杂的网络图像-标题对和对比损失，在同一个向量空间中对齐一个图像编码器和一个文本编码器。零监督标签。4 亿对。由此产生的嵌入空间可以进行零样本分类、图像-文本检索，并作为视觉塔被每个 2026 年的 VLM 接入。SigLIP 2（2025）用 sigmoid 替换了 softmax，以更低成本超越了 CLIP。本课从 InfoNCE 到 sigmoid 逐对损失追溯数学推导，并用标准库 Python 实现训练步骤。

**Type:** Build
**Languages:** Python (stdlib, InfoNCE + sigmoid loss implementations)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 7 (Transformers)
**Time:** ~180 minutes

## 学习目标

- 从互信息推导出 InfoNCE 损失，并实现数值稳定的向量化版本。
- 解释为什么 sigmoid 逐对损失（SigLIP）可以扩展到 batch 32768+，而无需 softmax 所需的 all-gather 开销。
- 通过构建文本模板（`a photo of a {class}`）并对余弦相似度取 argmax，运行 ImageNet 零样本分类。
- 说出 CLIP / SigLIP 预训练提供的四个杠杆：batch 大小、温度、提示模板、数据质量。

## 问题

CLIP 出现之前的视觉是监督式的。收集标注数据集（ImageNet：120 万张图像，1000 类），训练 CNN，发布。标签昂贵，标签偏向标注者能达成共识的内容，标签不经过微调无法迁移到新任务。

图像-标题网络有超过十亿对免费、弱标注的数据。一张金毛犬照片配 alt 文本 "my dog Max in the park" 携带了监督信号——文本描述了图像。问题是：怎样把它变成有用的训练？

CLIP 的答案：将图像-标题对视为匹配任务。给定 N 张图像和 N 个标题的 batch，学习将每张图像与其自己的标题匹配，排除 N-1 个干扰项。监督目标是"这两个东西是一对；这 N-1 个不是。"没有类别标签。没有人工标注。只有一个对比损失。

得到的嵌入空间不仅限于训练目标。ImageNet 零样本能工作是因为 "a photo of a cat" 的嵌入接近那些从未被显式标注为猫的猫图片。这就是孕育了每个 2026 年 VLM 的赌注。

## 概念

### 双编码器

CLIP 有两个塔：

- 图像编码器 `f`：ViT 或 ResNet，每张图像输出一个 D 维向量。
- 文本编码器 `g`：小型 Transformer，每个标题输出一个 D 维向量。

两个塔都将其输出归一化为单位长度。相似度为 `cos(f(x), g(y)) = f(x)^T g(y)`，因为两者都是单位范数。

对于一个包含 N 对（图像，标题）的 batch，构建形状为 `(N, N)` 的相似度矩阵 `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是可学习的温度（CLIP 初始化为 0.07；在 log 空间学习）。

### InfoNCE 损失

CLIP 使用对行和列的对称交叉熵：

```
loss_i2t = CE(S, labels=identity)     # 每张图的正样本是其自己的标题
loss_t2i = CE(S^T, labels=identity)   # 每个标题的正样本是其自己的图像
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 中的 softmax 强制每张图像与其标题的匹配程度高于 batch 中的其他所有标题。"负样本"是所有其他 batch 项。更大的 batch = 更多负样本 = 更强的信号。CLIP 在 batch 32k 下训练；规模很重要。

### 温度

`tau` 控制 softmax 的锐度。低 tau → 尖锐分布，硬负样本挖掘效果。高 tau → 平滑，所有样本都有贡献。CLIP 学习 log(1/tau)，做截断以防坍缩。SigLIP 2 固定初始 tau，改用可学习的偏置。

### 为什么 sigmoid 扩展性更好（SigLIP）

Softmax 需要整个相似度矩阵同步。在分布式训练中，必须将每个嵌入 all-gather 到每个副本，然后做 softmax。通信量与世界规模的平方成正比。

SigLIP 用逐元素 sigmoid 替换了 softmax：对于每对 `(i, j)`，损失是一个二分类——"这对是否是匹配对？"正类标签在对角线上，其余为负。损失为：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 当 `i == j`，否则 0。每对的损失独立。不需要 all-gather。每个 GPU 计算本地块并求和。SigLIP 2 可以廉价地扩展到 32k-512k 的 batch，而 CLIP 需要按比例增加通信成本。

### 零样本分类

给定 N 个类别名称，为每个类别构建一个文本模板：

```
"a photo of a {class}"
```

用文本编码器嵌入每个模板。用图像编码器嵌入你的图像。取 argmax 余弦相似度 = 预测类别。无需在目标类别上训练。

提示模板很重要。CLIP 原始论文每类使用了 80 个模板（普通、艺术、照片、绘画等），取嵌入的均值。ImageNet 上提升 3 个百分点。现代使用通常选一到两个模板。

### 线性探测和微调

零样本是基线。线性探测（在冻结的 CLIP 特征之上为你的目标类别训练一个线性层）在域内任务上优于零样本。全量微调在域内优于线性探测，但可能损害零样本迁移能力。三种范式，三种权衡。

### SigLIP 2：NaFlex 和密集特征

SigLIP 2（2025）新增：
- NaFlex：单个模型处理可变宽高比和分辨率。
- 更好的密集特征用于分割和深度估计，目标是用作 VLM 中的冻结骨干网络。
- 多语言：在 100+ 种语言上训练（CLIP 仅英语）。
- 10 亿参数规模（CLIP 上限 4 亿）。

在 2026 年开源 VLM 中，SigLIP 2 SO400m/14 是默认视觉塔。CLIP 仍是纯图像-文本检索的默认选择，当 LAION-2B 训练分布与你的查询模式匹配时。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google, 2021）：与 CLIP 相同的思路，18 亿对规模，90% 噪声数据。证明了噪声数据可以规模化。OpenCLIP（LAION）：在 LAION-400M / 2B 上开源复现的 CLIP，多规模，首选的开放 checkpoint。EVA-CLIP：从掩码图像建模初始化；VLMs 的强力骨干。BASIC：Google 的 CLIP+ALIGN 混合体。同一家族，不同数据和调优。

### 零样本上限

CLIP 类模型在 ImageNet 零样本上大约封顶 76%（CLIP-G, OpenCLIP-G）。超越需要要么更大的数据（SigLIP 2 达到 80%+），要么架构变化（监督头、更多参数）。基准正在饱和；真正价值是下游 VLM 消费的嵌入空间。

```figure
multimodal-fusion
```

## 使用它

`code/main.py` 实现了：

1. 一个玩具双编码器（基于哈希的图像特征和文本字符特征），你可以在不依赖 numpy 的情况下观察 InfoNCE 的形态。
2. 纯 Python 实现的 InfoNCE 损失（通过 log-sum-exp 保证数值稳定性）。
3. sigmoid 逐对损失作为对比。
4. 零样本分类例程：计算与一组文本提示的余弦相似度，取 argmax 得到预测。

运行它并观察损失曲线。绝对值是玩具级的；但形状与真实 CLIP 训练器的输出一致。

## 交付

本课产出 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和目标类别列表，它用 CLIP 模板构建文本提示，用指定 checkpoint（如 `openai/clip-vit-large-patch14`）嵌入两边，返回 top-1 / top-5 预测及相似度分数。该技能拒绝就不在提示列表中的类别做出断言。

## 练习

1. 手工实现一个 batch 为 4 对的 InfoNCE。构建 4x4 相似度矩阵，运行 softmax，取出对角线，计算交叉熵。用这个手工计算验证你的 Python 实现。

2. SigLIP 除了温度还使用偏置参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当 batch 中有大类不平衡（每行负样本远多于正样本）时，`b` 起什么作用？阅读 SigLIP 第 3 节 (arXiv:2303.15343)。

3. 构建一个猫 vs 狗的零样本分类器。尝试两个提示模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上测量准确率。模板集成是否优于单个模板？

4. 计算在 512-GPU、batch 32k 运行中 softmax InfoNCE vs sigmoid 逐对损失的通信成本。哪个随 O(N) 扩展，哪个随 O(N^2) 扩展？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 扩展定律论文 (arXiv:2212.07143, Cherti 等人)。从其图表中复现关于数据扩展的结论：在固定模型大小下，ImageNet 零样本准确率与训练数据规模之间的对数线性关系是什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| InfoNCE | "对比损失" | 对 batch 相似度矩阵做交叉熵；每项的正样本是其配对项，负样本是其他所有 |
| Sigmoid loss | "SigLIP 损失" | 逐对二分类交叉熵；无 softmax、无 all-gather，在分布式训练中廉价扩展 |
| Temperature | "tau" | 在 softmax/sigmoid 前缩放 logits 的标量；控制分布锐度 |
| Zero-shot | "无微调分类" | 用文本提示构建类别嵌入，通过余弦相似度分类；无需在目标类别上训练 |
| 提示模板 | "a photo of a ..." | 包裹类别名称的文本脚手架；以 1-5 个百分点影响零样本准确率 |
| 双编码器 | "双塔" | 一个图像编码器 + 一个文本编码器，输出在共享 D 维空间中 |
| 硬负样本 | "难干扰项" | 与正样本足够相似的负样本，模型必须努力区分 |
| 线性探测 | "冻结 + 一层" | 仅在冻结特征上训练一个线性分类器；衡量特征质量 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 的能力：以任意宽高比和分辨率摄入图像，无需调整大小 |
| 温度缩放 | "log 参数化 tau" | CLIP 用 log(1/tau) 参数化以保证梯度合理；截断防止坍缩到接近零的 tau |

## 延伸阅读

- [Radford 等人 — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 论文。
- [Zhai 等人 — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen 等人 — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 多语言 + NaFlex。
- [Jia 等人 — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 用噪声网络数据扩展。
- [Cherti 等人 — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP 扩展定律。
