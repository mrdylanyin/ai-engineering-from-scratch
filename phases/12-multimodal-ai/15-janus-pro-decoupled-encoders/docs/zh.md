# Janus-Pro：用于统一多模态模型的解耦编码器

> 统一多模态模型有一个不可回避的张力。理解需要语义特征——SigLIP 或 DINOv2 输出富含概念级信息的向量。生成需要适合重建的编码——VQ token 组合回清晰的像素。这两个目标在单个编码器中是不兼容的。Janus（DeepSeek, 2024年10月）和 Janus-Pro（DeepSeek, 2025年1月）论证解决方案是停止尝试：解耦两个编码器。在任务之间共享 Transformer 体，但将理解通过 SigLIP 路由，生成通过 VQ 分词器路由。在 7B 参数下，Janus-Pro 在 GenEval 上击败 DALL-E 3，同时在 MMMU 上匹配 LLaVA。本课阅读为什么两个编码器在单个编码器失败的地方有效。

**Type:** Build
**Languages:** Python (stdlib, dual-encoder routing + shared-body signal)
**Prerequisites:** Phase 12 · 13 (Transfusion), Phase 12 · 14 (Show-o)
**Time:** ~120 minutes

## 学习目标

- 解释为什么单个共享编码器会损害理解或生成质量（或两者）。
- 描述 Janus-Pro 的路由：理解输入侧用 SigLIP 特征，生成用 VQ token（输入和输出）。
- 追踪使 Janus-Pro 成功而 Janus 没有成功的数据混合扩展。
- 比较解耦（Janus-Pro）、耦连续（Transfusion）和耦离散（Show-o）架构。

## 问题

统一模型在理解和生成之间共享 Transformer 体。之前的尝试（Chameleon、Show-o、Transfusion）在双向都使用一个视觉分词器。分词器是折中：

- 为重建优化（生成）：VQ-VAE 捕获细粒度像素细节，但产生的 token 语义一致性弱。
- 为语义优化（理解）：SigLIP 嵌入将"猫"图像分组到"猫"token 附近，但不允许良好重建。

Show-o 和 Transfusion 为此在其中一个方向上付出可见的质量税。Janus-Pro 问：当任务有不同需求时为什么要求一个分词器？

## 概念

### 解耦视觉编码

Janus-Pro 的架构分离了两个编码器：

- 理解路径。输入图像 → SigLIP-SO400m → 2 层 MLP → Transformer 体。
- 生成路径。输入图像（如果条件化于现有图像）→ VQ 分词器 → token ID → Transformer 体。
- 输出生成。Transformer 预测的图像 token → VQ 解码器 → 像素。

Transformer 体是共享的。体上游和下游的一切是任务特定的。

输入由提示格式消歧：`<understand>` 标签路由通过 SigLIP；`<generate>` 路由通过 VQ。或者路由从任务隐式推断。

### 为什么这有效

理解损失获得 SigLIP 特征，CLIP 风格预训练已为语义相似性调整。模型的感知基准比 Show-o / Transfusion 改善，因为输入特征对任务更优。

生成损失获得 VQ token，分词器已为重建调整。图像质量比 Show-o 改善，因为 VQ 编码干净组合回像素。

共享的 Transformer 体看到两个输入分布（SigLIP 和 VQ）并学会与两者工作。声明：足够的数据 + 足够的参数，体吸收切换。

### 数据扩展——Janus vs Janus-Pro

Janus（原始，arXiv 2410.13848）引入了解耦但在小规模上（13 亿参数，有限数据）。Janus-Pro（arXiv 2501.17811）扩展了：

- 70 亿参数（vs 13 亿）。
- 9000 万图像-文本对用于阶段 1（对齐），从 7200 万提升。
- 7200 万用于阶段 2（统一），从 2600 万提升。
- 为阶段 3 添加了 20 万图像生成指令样本。

结果：Janus-Pro-7B 在 MMMU 上匹配 LLaVA（60.3 vs ~58），在 GenEval 上击败 DALL-E 3（0.80 vs 0.67）。一个开源模型，在统一谱系的两侧都有竞争力。

### JanusFlow——整流流变体

JanusFlow（arXiv 2411.07975）将 VQ 生成路径替换为整流流生成路径（连续）。分割变为 SigLIP 用于理解 + 整流流用于生成。质量天花板进一步提升。架构保持解耦编码器共享体。

### 共享体的工作

Transformer 体处理统一序列但有两个输入分布。它的工作是：

- 理解：消费 SigLIP 特征 + 文本 token → 自回归输出文本。
- 生成：消费文本 token +（可选的图像 VQ token）→ 自回归输出图像 VQ token。

体没有每块模态特定权重。它就是你期望在 Qwen 或 Llama 中找到的文本风格 Transformer，加上两个输入适配器。

有趣的是，这意味着 Janus-Pro 的体可以从预训练 LLM 初始化。Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。这个选择很重要：LLM 贡献了纯从头统一模型难以达到的推理能力。

### 与 InternVL-U 比较

InternVL-U（12.10 课）是 2026 年的后续。它结合了：

- 原生多模态预训练（InternVL3 骨干）。
- 解耦编码器路由（SigLIP 进，VQ + 扩散头出）。
- 统一理解 + 生成 + 编辑。

InternVL-U 将 Janus-Pro 的架构选择纳入更大的框架。解耦编码器想法现在是规模统一模型的默认方案。

### 局限性

解耦编码器增加架构复杂性。需要训练两个分词器，维护两条输入路径，两组失败模式。对于不需要生成的产品，Janus-Pro 被过度设计——选 LLaVA 家族理解模型。

对于不需要理解的产品，Janus-Pro 超规格——选 Stable Diffusion 3 / Flux 模型。

对于需要两者的产品，Janus-Pro 现在是参考开源架构。

## 使用它

`code/main.py` 模拟 Janus-Pro 路由：

- 两个模拟编码器：类 SigLIP（产生 256 维语义向量）和类 VQ（产生整数编码）。
- 一个基于任务标签选择编码器的提示路由器。
- 一个共享体（替代），无论哪个编码器产生 token 序列都处理。
- 从阶段 1（对齐）切换到阶段 3（指令微调）的加权采样调度。

为 3 个示例打印路由路径：图像 QA、T2I、图像编辑。

## 交付

本课产出 `outputs/skill-decoupled-encoder-picker.md`。给定一个想要前沿质量统一生成 + 理解的产品，在 Janus-Pro、JanusFlow 或 InternVL-U 之间选择，附具体数据规模建议。

## 练习

1. Janus-Pro-7B 在 GenEval 上击败 DALL-E 3。解释为什么一个 7B 开源模型能在生成上匹配前沿专有模型但在理解上不能。

2. 实现路由器函数：给定提示文本，分类为 `understand` 或 `generate`。如何处理"先描述然后画草图"这样的模糊提示？

3. JanusFlow 用整流流替换 VQ 路径。Transformer 体现在输出什么，损失中有什么变化？

4. 提出 Janus-Pro 架构可以再多一个解耦编码器处理的第四种任务。示例：图像分割（DINO 风格）、深度估计（MiDaS 风格）。

5. 阅读 Janus-Pro 第 4.2 节关于数据扩展。哪个数据阶段对 T2I 质量提升相比 Janus 贡献最大？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 解耦编码 | "两个视觉编码器" | 每个方向独立的分词器或编码器：理解用语义、生成用重建 |
| 共享体 | "一个 Transformer" | 单个 Transformer 处理任一编码器的输出；无模态特定权重 |
| SigLIP 用于理解 | "语义特征" | CLIP 家族视觉塔提供丰富概念特征但重建差 |
| VQ 用于生成 | "重建编码" | 向量量化 token 干净解码回像素 |
| JanusFlow | "整流流变体" | 用连续流匹配生成头替代 VQ 的 Janus-Pro |
| 路由标签 | "任务标签" | 选择输入编码器的提示标记（`<understand>` / `<generate>`） |

## 延伸阅读

- [Wu 等人 — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen 等人 — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma 等人 — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong 等人 — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
