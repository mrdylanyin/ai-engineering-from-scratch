# 任意分辨率视觉：Patch-n'-Pack 与 NaFlex

> 真实图像不是 224x224 的正方形。收据是 9:16，图表是 16:9，医学扫描可能是 4096x4096，手机截图是 9:19.5。2024 年之前的 VLM 答案——把所有东西缩放到固定正方形——丢弃了使 OCR、文档理解和高分辨率场景解析起作用的信号。NaViT（Google, 2023）展示了可以通过分块对角掩码将变分辨率 patch 打包到一个 Transformer batch 中。Qwen2-VL 的 M-RoPE（2024）彻底抛弃了绝对位置表。LLaVA-NeXT 的 AnyRes 将高分辨率图像切分为基础 + 子图像。SigLIP 2 的 NaFlex 变体（2025）现在是希望用单一 checkpoint 服务所有宽高比的开源 VLM 的默认编码器。本课完整实现 patch-n'-pack。

**Type:** Build
**Languages:** Python (stdlib, patch packer + block-diagonal mask)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 12 · 05 (LLaVA)
**Time:** ~120 minutes

## 学习目标

- 将一批变分辨率图像的 patch 打包为一个序列，并构建分块对角注意力掩码。
- 为给定任务在 AnyRes 切分（LLaVA-NeXT）、NaFlex（SigLIP 2）和 M-RoPE（Qwen2-VL）之间做出选择。
- 不缩放地计算 OCR、图表和摄影的 token 预算。
- 说出正方形缩放的三类失败模式：文字挤压、内容裁剪、填充浪费 token。

## 问题

Transformer 期望一个序列。一个 batch 是等长序列的堆叠。如果你的图像是 224x224，每次得到 196 个 patch token，不需要填充，任务完成。在 224 上训练，在 224 上推理，永远不用考虑分辨率。

现实世界不合作。文档是竖版（8.5x11 英寸，约 2:3）。图表截图是横版（16:9）。收据又高又窄（1:3）。医学影像通常是 2048x2048 或更大。手机截图是 1170x2532（0.46:1）。

2024 年之前的三个选项及其失败原因：

1. 缩放到固定正方形（224x224 或 336x336）。挤压扭曲文字和人脸。下采样破坏图表标签和 OCR 内容。这是 LLaVA-1.5 之前的通用做法。
2. 裁剪到固定宽高比。丢弃了图像的大部分内容，选择裁剪位置本身就是一个视觉问题。
3. 填充到最长边。解决了扭曲问题，但对竖版图像浪费 50%+ 的 token 在填充上。所有填充 token 的二次注意力成本。

2024-2025 的答案：让 Transformer 以图像的原生分辨率吃 patch，并找到方法将异构 batch 打包到一个序列中而不浪费算力。

## 概念

### NaViT 和 Patch-n'-Pack

NaViT（Dehghani 等人, 2023）是展示这在规模上可行的论文。想法是机械的：

1. 对 batch 中的每张图像，在选定的 patch 大小（如 14）下计算其原生 patch 网格。
2. 将每张图像的 patch 展开为它自己的变长序列。
3. 将所有图像的 patch 拼接成一个长序列。
4. 构建分块对角注意力掩码，使图像 A 的 patch 只能关注图像 A 内部。
5. 携带每个 patch 的位置信息（2D RoPE 或分数位置嵌入）。

三张图像：336x336（576 token）、224x224（256 token）、448x336（768 token），变成一个 1600 token 的序列，带 1600x1600 分块对角掩码。无填充。无浪费算力。Transformer 处理任意宽高比。

NaViT 还引入了训练时的分数 patch 丢弃——在 batch 中随机丢弃 50% 的 patch——既起到正则化作用又加速训练。SigLIP 2 继承了这个做法。

### AnyRes（LLaVA-NeXT）

LLaVA-NeXT 的 AnyRes 是务实的替代方案。给定一张高分辨率图像和一个固定编码器（CLIP 或 SigLIP @ 336），切分图像：

1. 从预设集合中选择一个网格布局——(1x1)、(1x2)、(2x1)、(1x3)、(3x1)、(2x2) 等——最匹配图像宽高比的。
2. 将完整图像切分为该网格；每个瓦片成为一个 336x336 的裁剪。
3. 同时生成一个缩略图：整张图缩放到 336x336 作为全局上下文 token。
4. 用冻结的 336 编码器编码每个瓦片。拼接瓦片 token + 缩略图 token。

对于 672x672 图像，2x2 网格 + 缩略图：4 * 576 + 576 = 2880 个视觉 token。昂贵但有效——LLM 既看到局部细节，也看到全局上下文。

AnyRes 是编码器被冻结且只支持一个分辨率时的选择。它对大图会爆炸 token 数（1344x1344 图像在 4x4 网格下是 9216 + 576 ≈ 9800 token，快填满 8k LLM 上下文）。

### M-RoPE（Qwen2-VL）

Qwen2-VL 引入了多模态旋转位置嵌入。每个 patch 携带 3D 位置（时间、高度、宽度），而不是 NaViT 的分数位置或 AnyRes 的瓦片+缩略图。query/key 旋转处理任意 H、W 和时间长度。

M-RoPE 原生支持动态分辨率，无需重新训练。推理时喂入任意 HxW 图像，patch 嵌入产生 H/14 x W/14 个 token，每个 token 获得其 (t=0, r=行, c=列) 位置，RoPE 以正确频率旋转注意力，完成。Qwen2.5-VL 和 Qwen3-VL 延续了这个方案。InternVL3 的 V2PE 是相同的想法，按模态可变编码。

不同于 AnyRes，M-RoPE 在原生分辨率下是 O(H x W / P^2) 个 token——没有乘性的瓦片开销。不同于 NaViT，它仍然期望每次前向传播一张图像。跨分辨率 batch 化仍需要 patch-n'-pack。

### NaFlex（SigLIP 2）

NaFlex 是 SigLIP 2 checkpoint 的原生灵活模式。单个模型在推理时支持多种序列长度（256、729、1024 token）。内部在训练时使用 NaViT 风格的 patch-n'-pack 和每个 patch 的绝对分数位置。卖点：一个 checkpoint，根据任务在推理时选择你的 token 预算。

对语义任务（分类、检索），256 token。对 OCR 或图表理解，1024 token。无需重新训练。

### 打包掩码

分块对角掩码是大多数实现出错之处。对于覆盖图像 `i=0..B-1`、长度 `n_i` 的打包序列 `N_total`，掩码 `M` 形状 `(N_total, N_total)` 在两个索引落在同一图像块内时为 1，否则为 0。可以从累积长度列表构建：

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 当存在 b 使 offsets[b] <= i < offsets[b+1] 且 offsets[b] <= j < offsets[b+1]
```

在 PyTorch 中一行代码用 `torch.block_diag` 或显式 gather。FlashAttention 的变长路径（`cu_seqlens`）完全跳过掩码，直接使用累积长度张量在序列内部做 attention——对典型 batch 比密集掩码快约 10 倍。

### Token 预算

按任务选择策略：

- OCR / 文档：1024-4096 token。SigLIP 2 NaFlex @ 1024，或 AnyRes 3x3 + 缩略图。
- 图表和 UI：原生 384-448 下 729-1024 token。Qwen2.5-VL 动态分辨率带最大像素上限。
- 自然照片：256-576 token 即可。下游 LLM 看到足够信息。在内容密度高的地方为 token 付费。
- 视频：空间池化后每帧 64-128 token，2-8 FPS。12.17 课讲解。

2026 年生产规则：选择一个每任务最大像素上限，在原生长宽比下编码至该上限，打包 batch，跳过填充。Qwen2.5-VL 暴露 `min_pixels` 和 `max_pixels` 正是为此旋钮。

## 使用它

`code/main.py` 为具有整数像素坐标的异构图像 batch 实现 patch-n'-pack。它：

- 接受 (H, W) 图像尺寸列表。
- 在 patch 大小 14 下计算每张图像的 patch 序列长度。
- 将它们打包为总长度 `sum(n_i)` 的一个序列。
- 构建分块对角注意力掩码（密集格式，为清晰起见）。
- 将打包成本与正方形缩放和 AnyRes 切分对比。
- 为混合 batch（收据、图表、截图、照片）打印 token 预算表。

运行它。得出的数字就是每个 2026 年开源 VLM 使用 patch-n'-pack 的原因。

## 交付

本课产出 `outputs/skill-resolution-budget-planner.md`。给定混合宽高比工作负载（OCR、图表、照片、视频帧）和总 token 预算，选择正确策略（NaFlex、AnyRes、M-RoPE 或固定正方形），并输出每请求配置。在为产品评估 VLM 规模时使用这个技能——它防止悄无声息的 10 倍 token 膨胀扼杀延迟预算。

## 练习

1. 一张收据是 600x1500（1:2.5）。在 patch 大小 14 下，多少原生分辨率 token？正方形缩放到 336 后呢？实际中哪个丢失更多 OCR 准确率？

2. 为四张长度分别为 256、576、729、1024 的图像 batch 构建分块对角掩码。验证注意力矩阵为 2585x2585，且恰好有 `256^2 + 576^2 + 729^2 + 1024^2` 个非零项。

3. 对 1792x896 图像在 patch 14 下比较：(a) 正方形缩放到 336 然后编码，(b) AnyRes 2x1 + 缩略图，(c) M-RoPE 原生分辨率。哪个用最少 token？哪个保留最多细节？

4. 实现分数 patch 丢弃：给定一个打包序列，均匀随机丢弃 50% 的 token，并相应更新分块对角掩码。测量掩码的稀疏度变化。

5. 阅读 Qwen2-VL 论文第 3.2 节（arXiv:2409.12191）。用两句话描述 `min_pixels` 和 `max_pixels` 控制什么，以及为什么两个边界都很重要。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Patch-n'-pack | "NaViT 风格打包" | 将不同图像变长 patch 序列拼接为一个 batch 维度 |
| 分块对角掩码 | "打包掩码" | 限制每张图像 patch 只能关注自身、不能关注打包中邻居的注意力掩码 |
| AnyRes | "LLaVA-NeXT 切分" | 将高分辨率图切分为固定大小瓦片网格加上全局缩略图；用固定编码器编码每个瓦片 |
| NaFlex | "SigLIP 2 原生灵活" | 单一 SigLIP 2 checkpoint，推理时无需重训即可服务 256/729/1024 token 预算 |
| M-RoPE | "多模态 RoPE" | 3D 旋转位置编码（时间、行、列），无需位置表即可处理任意 H、W、T |
| cu_seqlens | "FlashAttention 打包" | FlashAttention 变长路径使用的累积长度张量，替代密集分块对角掩码 |
| min_pixels / max_pixels | "分辨率边界" | Qwen2.5-VL 每请求旋钮，限制极小或极大输入上的 token 数 |
| 视觉 token 预算 | "每张图像多少 token" | 每张图像发出的 patch token 粗略数量；设定 LLM 的提示预算和注意力成本 |

## 延伸阅读

- [Dehghani 等人 — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang 等人 — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon 等人 — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen 等人 — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen 团队 — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
