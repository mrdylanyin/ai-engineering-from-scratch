# 视觉 Transformer 与 Patch-Token 原语

> 在谈论多模态之前，一张图像必须先变成 Transformer 可以消费的 token 序列。2020 年的 ViT 论文用 16x16 像素的 patch、一个线性投影和一个位置嵌入回答了这个问题。五年后，每一个 2026 年的前沿模型（Claude Opus 4.7 原生 2576px、Gemini 3.1 Pro、Qwen3.5-Omni）仍然从这一步开始——编码器从 ViT 演变为 DINOv2 再到 SigLIP 2，register token 被加入，位置编码方案变成了 2D-RoPE，但这个原语始终未变。本课从头到尾通读 patch-token 流水线，并用标准库 Python 实现它，为 Phase 12 的后续内容提供一个具体的"视觉 token"心智模型。

**Type:** Learn
**Languages:** Python (stdlib, patch tokenizer + geometry calculator)
**Prerequisites:** Phase 7 (Transformers), Phase 4 (Computer Vision)
**Time:** ~120 minutes

## 学习目标

- 将一张 HxWx3 的图像转换为带有正确位置编码的 patch token 序列。
- 对于给定（patch 大小、分辨率、隐藏维度、深度）的 ViT，计算序列长度、参数量和 FLOPs。
- 说出将 ViT 从 2020 年研究推进到 2026 年生产的三大升级：自监督预训练（DINO / MAE）、register token 和原生分辨率打包。
- 在下游任务中，在 CLS 池化、均值池化和 register token 之间做出选择。

## 问题

Transformer 操作的是向量序列。文本已经是一个序列（字节或 token）。图像是一个带有三个颜色通道的 2D 像素网格——这不是序列。如果展平每个像素，一张 224x224 的 RGB 图像会变成 150,528 个 token，在这种长度上做 self-attention 是不可行的（与序列长度平方成正比）。

2020 年前的方法在输入端加了一个 CNN 特征提取器：ResNet 输出 7x7 的特征图（2048 维向量），将这 49 个 token 喂给 Transformer。这能工作，但继承了 CNN 的偏置（平移等变性、局部感受野），且丧失了 Transformer 对规模的偏好。

Dosovitskiy 等人（2020）问了一个直接的问题：如果跳过 CNN 呢？将图像切分成固定大小的 patch（比如 16x16 像素），将每个 patch 线性投影为向量，加上位置嵌入，把序列喂给标准 Transformer。这在当时是异端——不用卷积的视觉。有了足够的数据（JFT-300M，然后是 LAION），它在 ImageNet 上击败了 ResNet 并持续提升。

到 2026 年，ViT 原语已经是无可争议的基础。每个开源 VLM 的视觉塔都是某种后代（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是"该不该用 patch？"，而是"用什么 patch 大小、什么分辨率调度、什么预训练目标、什么位置编码。"

## 概念

### Patch 作为 Token

给定形状为 `(H, W, 3)` 的图像 `x` 和 patch 大小 `P`，将图像切割为 `(H/P) x (W/P)` 个不重叠 patch 的网格。每个 patch 是一个 `P x P x 3` 的像素立方体。将每个立方体展平为 `3 P^2` 维向量。应用形状为 `(3 P^2, D)` 的共享线性投影 `W_E`，将每个 patch 映射到模型的隐藏维度 `D`。

以 ViT-B/16 标准配置为例：
- 分辨率 224，patch 大小 16 → 网格 14x14 → 196 个 patch token。
- 每个 patch 是 `16 x 16 x 3 = 768` 个像素值，投影到 `D = 768`。
- 加上可学习的 `[CLS]` token → 序列长度 197。

Patch 投影在数学上与核大小为 `P`、步幅为 `P`、`D` 个输出通道的 2D 卷积完全相同。这就是生产代码中的实际实现——`nn.Conv2d(3, D, kernel_size=P, stride=P)`。"线性投影"是概念表述；卷积核的表述是高效的。

### 位置嵌入

Patch 没有固有顺序——Transformer 将它们视为一个集合。早期 ViT 添加了可学习的 1D 位置嵌入（每个位置一个 768 维向量，共 197 个）。可行，但将模型绑定到了训练分辨率上：推理时如果改变网格，必须对位置表做插值。

现代视觉骨干网络使用 2D-RoPE（Qwen2-VL 的 M-RoPE、SigLIP 2 的默认方案）或分解的 2D 位置编码。2D-RoPE 根据 patch 的（行、列）索引旋转 query 和 key 向量，模型从旋转角度推断相对 2D 位置。没有位置表。模型在推理时可以处理任意网格大小。

### CLS Token、池化输出和 Register Token

什么是图像级别的表示？三种选择并存：

1. `[CLS]` token。在 patch 序列前加一个可学习向量。经过所有 Transformer 块后，CLS token 的隐藏状态即为图像表示。继承自 BERT。原始 ViT 和 CLIP 使用。
2. 均值池化。对 patch token 的输出隐藏状态取平均。SigLIP、DINOv2 和大多数现代 VLM 使用。
3. Register token。Darcet 等人（2023）观察到，没有显式 sink token 训练的 ViT 会产生高范数的"伪影"patch，劫持 self-attention。添加 4-16 个可学习的 register token 可以吸收这些负荷，改善密集预测质量（分割、深度估计）。DINOv2 和 SigLIP 2 都带有 register。

这个选择对下游任务影响重大。CLS 适合分类。对于将 patch token 喂给 LLM 的 VLM，完全跳过池化——每个 patch 都成为 LLM 的输入 token。Register 在交接前被丢弃（它们是脚手架，不是内容）。

### 预训练：监督、对比、掩码、自蒸馏

2020 年的 ViT 是用 JFT-300M 上的监督分类预训练的。很快被以下方案替代：

- CLIP（2021）：在 4 亿对图像-文本数据上做对比学习。见 12.02 课。
- MAE（2021, He 等人）：掩码 75% 的 patch，重建像素。自监督，纯图像即可工作。
- DINO（2021）/ DINOv2（2023）：学生-教师自蒸馏，无标签、无字幕。2023 年 DINOv2 ViT-g/14 是最强的纯视觉骨干网络，是"密集特征"用例的默认选择。
- SigLIP / SigLIP 2（2023, 2025）：使用 sigmoid loss 和 NaFlex 支持原生宽高比的 CLIP 变体。2026 年开源 VLM 的主导视觉塔（Qwen、Idefics2、LLaVA-OneVision）。

预训练方式的选择决定了骨干网络适合什么任务：CLIP/SigLIP 适合与文本的语义匹配，DINOv2 适合密集视觉特征，MAE 适合下游微调的起点。

### 扩展定律

ViT 的扩展定律（Zhai 等人 2022）确立了 ViT 的质量在模型大小、数据大小和算力方面遵循可预测的规律。在固定算力下：
- 更大模型 + 更多数据 → 更好的质量。
- Patch 大小是序列长度与保真度之间的杠杆。Patch 14（DINOv2/SigLIP SO400m 的典型设置）比 patch 16 产生更多 token；对 OCR 和密集任务更有利，对速度不利。
- 分辨率是另一个重要杠杆。从 224 到 384 再到 512 几乎总是有帮助，代价是 FLOPs 的二次增长。

ViT-g/14（10 亿参数，patch 14，分辨率 224 → 256 token）和 SigLIP SO400m/14（4 亿参数，patch 14）是 2026 年开源 VLM 的两个主力编码器。

### ViT 的参数量计算

完整计算在 `code/main.py` 中。ViT-B/16 @ 224：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

在加载任何 checkpoint 之前，先用这个公式估算每个 ViT 的参数量。骨干网络的大小决定了任何下游 VLM 的 VRAM 下限。

### 2026 年生产配置

2026 年大多数开源 VLM 使用的编码器是原生分辨率（NaFlex）的 SigLIP 2 SO400m/14。其配置为：
- 4 亿参数。
- Patch 大小 14，默认分辨率 384 → 每张图像 729 个 patch token。
- 图像级任务用均值池化；VQA 任务全部 729 个 patch 流入 LLM。
- 4 个 register token，在交给 LLM 前丢弃。
- 2D-RoPE，带图像级缩放的宽高比支持。

配置中的每一个决定都能追溯到一篇你可以读到的论文。

```figure
image-patch-tokens
```

## 使用它

`code/main.py` 是一个 patch 分词器和几何计算器。输入（图像 H, W, patch P, 隐藏维度 D, 深度 L），输出：

- 切分后的网格形状和序列长度。
- 一个合成 8x8 像素玩具图像的 token 序列（走一遍展平 + 投影路径）。
- 按 patch 嵌入、位置嵌入、Transformer 块和头部分解的参数量。
- 目标分辨率下每次前向传播的 FLOPs。
- ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384 的对比表。

运行它。将参数量与已发表的数据对齐。调整 patch 大小和分辨率，感受 token 数量的代价。

## 交付

本课产出 `outputs/skill-patch-geometry-reader.md`。给定一个 ViT 配置（patch 大小、分辨率、隐藏维度、深度），产出 token 数量、参数量和 VRAM 估算及论证。在为 VLM 选择视觉骨干网络时使用这个技能——它防止"token 爆炸了，我的 LLM 上下文满了"的意外。

## 练习

1. 计算 Qwen2.5-VL 在原生 1280x720 输入、patch 大小 14 下的 patch-token 序列长度。与仅使用 CLS 的表示相比如何？

2. 一个 1080p 帧（1920x1080）在 patch 14 下产生多少个 token？在 30 FPS、5 分钟视频下，总共有多少视觉 token？池化、帧采样和 token 合并哪个节省成本最多？

3. 用纯 Python 实现对 patch token 的均值池化。验证对 DINOv2 输出的 196 个 token 做均值池化后，结果是否与模型 `forward` 请求池化嵌入时返回的一致。

4. 阅读 "Vision Transformers Need Registers"（arXiv:2309.16588）第 3 节。用两句话描述 register 吸收了什么样的伪影，以及这对下游密集预测为何重要。

5. 修改 `code/main.py` 以支持 patch-n'-pack：给定一个不同分辨率图像的列表，产出一个打包序列和分块对角注意力掩码。学习 12.06 课后验证实现。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Patch | "16x16 像素方块" | 输入图像中一个固定大小、不重叠的区域；成为一个 token |
| Patch 嵌入 | "线性投影" | 将展平的 patch 像素映射到 D 维向量的共享学习矩阵（或步幅为 P 的 Conv2d） |
| CLS token | "类别 token" | 加在序列前的可学习向量，其最终隐藏状态代表整张图像；2026 年已可选项 |
| Register token | "Sink token" | 额外的可学习 token，吸收 ViT 在预训练中产生的高范数注意力伪影 |
| 位置嵌入 | "位置信息" | 使序列顺序可感知的逐位置向量或旋转；2D-RoPE 是现代默认方案 |
| 网格 | "Patch 网格" | 给定分辨率和 patch 大小下的 (H/P) x (W/P) 2D patch 数组 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 特性：单个模型支持多种宽高比和分辨率，无需重新训练 |
| 骨干网络 | "视觉塔" | 预训练图像编码器，其 patch-token 输出喂给 VLM 中的 LLM |
| 池化 | "图像级摘要" | 将 patch token 转为单个向量的策略：CLS、均值、注意力池化或基于 register 的方式 |
| Patch 14 vs 16 | "细网格 vs 粗网格" | Patch 14 每张图产生更多 token，OCR 保真度更好但更慢；patch 16 是经典默认值 |

## 延伸阅读

- [Dosovitskiy 等人 — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — 原始 ViT。
- [He 等人 — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE，自监督预训练。
- [Oquab 等人 — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 大规模自蒸馏，无标签。
- [Darcet 等人 — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register token 与伪影分析。
- [Tschannen 等人 — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026 年默认视觉塔。
- [Zhai 等人 — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — 经验扩展定律。
