# Emu3：用 Next-Token 预测生成图像和视频

> BAAI 的 Emu3（Wang 等人, 2024年9月）是应该结束扩散 vs 自回归之争的 2024 年结果。一个单一的 Llama 风格 decoder-only Transformer，仅训练 next-token 预测目标，跨文本 + VQ 图像 token + 3D VQ 视频 token 的统一词汇，在图像生成上击败 SDXL，在感知上击败 LLaVA-1.6。无 CLIP 损失。无扩散调度。在推理时使用无分类器引导来提升质量，但核心训练目标是 teacher forcing 的 next-token 预测。发表于 Nature。本课阅读 Emu3 的论点——为什么更好的分词器加上规模就是你需要的全部——并与扩散方法对比。

**Type:** Learn
**Languages:** Python (stdlib, 3D video tokenizer math + autoregressive sampler skeleton)
**Prerequisites:** Phase 12 · 11 (Chameleon)
**Time:** ~120 minutes

## 学习目标

- 解释为什么 Emu3 的单一损失 next-token 目标有效，尽管长期假设是扩散对图像质量是必需的。
- 描述 3D 视频分词器：时空 VQ 码本长什么样，为什么 patch 跨越时间。
- 在（训练算力、推理成本、质量上限）上比较 Emu3 和 Stable Diffusion XL。
- 说出同一个 Emu3 模型扮演的三种角色：Emu3-Gen（图像生成）、Emu3-Chat（感知）、Emu3-Stage2（视频生成）。

## 问题

贯穿 2024 年的传统智慧：图像生成需要扩散。论证：离散图像 token 丢失太多信息，无法重建细节，自回归采样在数千 token 上积累误差。Stable Diffusion、DALL-E 3、Imagen、Midjourney 都使用某种扩散形式。Chameleon（12.11 课）在小规模上部分否证了这一点，但未在质量上匹配 SDXL。

Emu3 正面攻击该论证。声明：更好的视觉分词器 + 足够规模 + next-token 损失 = 在同一模型中实现超越扩散的图像生成，同时还能做感知。

这个赌注在发表时有争议。两年后，开源统一生成家族（Emu3、Show-o、Janus-Pro、Transfusion）是研究的默认路径；生产前沿模型似乎使用了某种变体。

## 概念

### Emu3 分词器

关键成分是视觉分词器。Emu3 训练了一个自定义的 IBQ 级分词器（逆瓶颈量化器，SBER-MoVQGAN 家族），8x8 每 token 分辨率缩减。一张 512x512 图像变成 64x64 = 4096 个 token，码本大小 32768。

这比 Chameleon 的 1024 token/512x512、K=8192 更大，但每 token 更便宜（更小的码本查找，更简单的编解码器）。关键指标：重建 PSNR 在 30.5 dB，与 Stable Diffusion 的连续潜在空间 32 dB 有竞争力。

对于视频：3D VQ 分词器将时空 patch（4x4x4 像素）编码为一个整数。一段 8 FPS 的 4 秒片段有 32 帧；在 256x256、4 倍空间和 4 倍时间缩减下，token 数为 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32768 个 token。

分词器质量是天花板。Emu3 的贡献部分在于"我们训练了一个非常好的分词器。"

### 单损失训练

Emu3 使用一个目标：在跨文本 token、2D 图像 token 和 3D 视频 token 的共享词汇上的 next-token 预测。训练期间权重乘以模态特定因子以平衡贡献，但损失函数相同。

训练数据混合：
- 图像生成：`<text caption> <image> image_tokens </image>`
- 图像感知：`<image> image_tokens </image> <question> text_tokens`
- 视频生成：`<text caption> <video> video_tokens </video>`
- 视频感知：类似。
- 纯文本：标准 NTP。

模型从数据分布中学会何时应该发出图像 token vs 文本 token。生成能力从模型在 `<image>` 标签后预测图像 token 中自然涌现。

### 无分类器引导和温度

自回归图像生成在推理时使用无分类器引导（CFG）会好得多。Emu3 使用它：生成两次，一次用完整标题，一次用空标题，用引导权重混合 logits（通常 3.0-7.0）。这与扩散使用的相同 CFG 技巧，借用到自回归设置中。

温度很重要：太高，伪影；太低，模式坍缩。Emu3 的推荐温度：感知 1.0，图像生成 0.8。

### 三种角色，一个模型

Emu3 作为三个功能上不同的 API 发布，但底层是一组权重：

- Emu3-Gen。图像生成。输入文本，输出图像 token。
- Emu3-Chat。VQA 和标题生成。输入图像（token），输出文本。
- Emu3-Stage2。视频生成和视频 VQA。输入文本或视频，输出文本或视频。

没有任务特定头。只有不同的提示模板。同一个 checkpoint。

### 基准

来自 Emu3 论文（2024年9月）：

- 图像生成：在 MJHQ-30K FID（5.4 vs 5.6）上击败 SDXL，GenEval 整体（0.54 vs 0.55——统计平局），Deep-Eval 综合持平。
- 图像感知：在 VQAv2（75.1 vs 72.4）上击败 LLaVA-1.6，MMMU 大致匹配。
- 视频生成：4秒片段质量在 FVD 上与 Sora 时代的公开基准模型有竞争力。

数字并非总是赢——Emu3 在这里输一点那里赢一点——但声明"next-token 预测就是你需要的全部"跨模态是可辩护的。

### 算力成本

Emu3 在约 3000 亿多模态 token 上以 70 亿参数模型训练。GPU 小时大致相当于 Llama-2-7B 的预训练（A100 级硅上 2k-4k GPU-年）。扩散模型如 Stable Diffusion 3 在类似预算下训练，但需要独立的文本编码器和更复杂的管道。

推理时，Emu3 每张图比 SDXL 慢：4096 个图像 token 在 30 tok/s 下约 2 分钟/512x512 图像，vs SDXL 的 2-5 秒。推测解码和 KV 缓存优化缩小但不消除差距。自回归图像生成计算量大；这是持续的权衡。

### 为什么重要

Emu3 的深层贡献是概念性的。如果 next-token 预测能扩展到匹配扩散在图像生成上的表现，统一模型路径（一个损失、一个骨干、任意模态）是可行的。未来模型不需要独立的文本编码器、独立的扩散调度器、独立的 VAE。一个 Transformer，每种模态一个分词器，规模。

Show-o、Janus-Pro 和 InternVL-U 都在这个论点基础上构建或挑战它。中国实验室（BAAI、DeepSeek）在这个方向上比美国实验室更激进地发表。

## 使用它

`code/main.py` 构建了两个玩具部件：

- 2D vs 3D VQ 分词器 token 数计算器：给定（分辨率、patch、片段长度、FPS），计算图像 vs 视频的 token 数。
- 带无分类器引导和温度的自回归图像 token 采样器。

CFG 实现匹配 Emu3 的配方——用引导权重混合条件和无条件 logits。

## 交付

本课产出 `outputs/skill-token-gen-cost-analyzer.md`。给定生成产品规格（图像或视频、目标分辨率、质量层级、延迟预算），计算 token 数、推理成本，并在 Emu3 家族和扩散之间选择。

## 练习

1. Emu3 在 8x8 缩减下每张 512x512 图像产生 4096 个 token。计算 1024x1024 和 2048x2048 的等效值。推理延迟会怎样？

2. 阅读 Emu3 第 3.3 节关于视频分词器。描述 3D VQ patch 形状以及为什么是 4x4x4 而非 8x8x1。

3. 无分类器引导权重 5.0 vs 3.0：什么视觉效果？在 `code/main.py` 中追溯数学。

4. 计算 Emu3-7B 在 3000 亿 token 上的训练 FLOPs，与 Stable Diffusion 3 比较。哪个训练更贵？

5. Emu3 在 FID 上击败 SDXL，但在 VQAv2 上不敌专门 VLM。解释为什么统一损失方法在不同基准上对专家模型表现出不同的优势。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Next-token 预测 | "NTP" | 标准自回归损失：给定 token[0..i] 预测 token[i+1]；分词后对所有模态有效 |
| IBQ 分词器 | "逆瓶颈量化器" | 一类 VQ-VAE，码本更大（32768+），重建比 Chameleon 的更好 |
| 3D VQ | "时空量化器" | 按（时间、行、列）索引的码本；一个 token 覆盖 4x4x4 像素立方体 |
| 无分类器引导 | "CFG" | 用权重 gamma 混合条件和无条件 logits；推理时提升图像质量 |
| 统一词汇 | "共享 token" | 文本 + 图像 + 视频都从相同整数空间抽取；模型预测下一个模态是什么 |
| MJHQ-30K | "图像生成基准" | 包含 30k 提示的 Midjourney 质量基准；Emu3 在此报告 FID |

## 延伸阅读

- [Wang 等人 — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun 等人 — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu 等人 — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu 等人 — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian 等人 — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
