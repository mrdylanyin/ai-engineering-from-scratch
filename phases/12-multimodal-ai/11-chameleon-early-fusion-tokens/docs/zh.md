# Chameleon 与早期融合纯 Token 多模态模型

> 我们之前看到的每个 VLM 都保持图像和文本分离。视觉 token 来自视觉编码器，流入投影器，然后在 LLM 内部与文本汇合。视觉和文本词汇从不重叠。Chameleon（Meta, 2024年5月）问道：如果能重叠呢？训练一个 VQ-VAE，将图像转换为共享词汇中的离散 token 序列。每个多模态文档现在是一个序列——文本 token 和图像 token 交错，单一自回归损失。副作用：模型可以生成混合模态输出——在单次推理调用中交替输出文本和图像 token。本课阅读早期融合命题并端到端构建一个玩具版本。

**Type:** Build
**Languages:** Python (stdlib, VQ-VAE tokenizer + interleaved decoder)
**Prerequisites:** Phase 12 · 05, Phase 8 (Generative AI)
**Time:** ~180 minutes

## 学习目标

- 解释为什么共享词汇 + 单一损失改变了模型能做什么。
- 描述 VQ-VAE 如何将图像分词为适合 Transformer next-token 目标的离散序列。
- 说出 Chameleon 的训练稳定性技巧：QK-Norm、dropout 位置、LayerNorm 顺序。
- 比较 Chameleon 和 BLIP-2 的 Q-Former 方法，描述何时每种选择正确。

## 问题

基于适配器的 VLM（LLaVA、BLIP-2、Qwen-VL）将文本和图像视为两种不同的东西。文本 token 经过 `embed(text_token)`；图像经过 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有两条输入路径，在中途汇合。

三个后果：

1. LLM 只能消费图像，不能发出图像。输出仅限于文本。
2. 混合模态文档（交替段落和图像，如文章）很别扭——你要么在模型外解析多模态输入，要么链式生成。
3. 分布不匹配。视觉 token 和文本 token 存在于隐藏空间的不同区域，产生微妙的对齐问题。

Chameleon 拒绝了这个前提：图像就是从共享词汇中抽取的离散 token 序列。在交错文档上训练模型，一个损失，一个自回归解码器，你就能免费解锁混合模态生成。

## 概念

### VQ-VAE 作为图像分词器

分词器是一个向量量化变分自编码器。架构：

- 编码器：CNN + ViT，将图像映射到空间特征图，比如 32x32 维度 256 的特征。
- 码本：K 个向量的学习词汇（Chameleon 使用 8192），维度也是 256。
- 量化：对每个空间特征，按 L2 距离查找最近的码本条目。将连续特征替换为整数索引。
- 解码器：CNN 将量化后的特征恢复为像素。

训练：VAE 重建损失 + 承诺损失 + 码本损失。码本索引形成了图像的离散字母表。

对 Chameleon：一张图像变成 32*32 = 1024 个 token，从词汇 8192 中抽取。与文本 token（来自 LLM 的 BPE 词汇，比如 32000）拼接。最终词汇量：40192。Transformer 看到一个序列，一个损失。

### 共享词汇

Chameleon 的词汇包含文本 token、图像 token 和模态分隔符。每个 token 有单一 ID。输入嵌入层将每个 ID 映射到一个 D 维隐藏向量。输出投影将隐藏状态映射回词汇 logits。Softmax 选择下一个 token，无论什么模态。

分隔符很重要：`<image>` 和 `</image>` 标签括起图像 token 序列。在生成时，如果模型发出 `<image>`，下游软件知道接下来的 1024 个 token 是要送到解码器进行像素渲染的 VQ 索引。

### 混合模态生成

推理是在共享词汇上的 next-token 预测。示例提示："Draw a cat and describe it." Chameleon 发出：

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

模型自主选择顺序——它可以先生成图像再文本，先文本再图像，或交错。同一个解码器，同一个损失。

对比于适配器 VLM 中生成仅限于文本。Chameleon 重新打开了模型输出模态的问题。

### 训练稳定性——QK-Norm、dropout、LayerNorm 顺序

早期融合训练在规模上不稳定。Chameleon 论文记录了三个技巧：

- QK-Norm。在注意力内部对 query 和 key 投影应用 LayerNorm，在点积之前。防止深层的 logit 幅度爆炸。多个 2024 年后的大型模型使用。
- Dropout 位置。在每次残差加法后 dropout，而不仅是注意力和 MLP 后。当来自图像 token 的梯度可能主导时需要更多正则化。
- LayerNorm 顺序。残差分支上的 pre-LN（标准），加上最后一个块的跳跃连接上的额外 LN。稳定最后一层的梯度流。

没有这些技巧，340 亿参数的 Chameleon 训练会在多个 checkpoint 发散。有了它们，它收敛。训练配方和架构一样是贡献。

### 分词器的重建上限

VQ-VAE 是有损的。在 8192 码本条、每个 512x512 图像 1024 个 token 下，重建 PSNR 封顶约 26-28 dB。这对可辨认的图像生成足够，但明显比连续空间扩散（Stable Diffusion 3 达到 32+ dB）差。

分词器是瓶颈。更好的分词器（MAGVIT-v2、IBQ、SBER-MoVQGAN）抬高天花板。Emu3（12.12 课）仅通过更好的分词器就实现了 SDXL 质量的生成。

### Chameleon vs BLIP-2 / LLaVA

Chameleon（早期融合，共享词汇）：
- 一个损失，一个解码器。
- 生成混合模态输出。
- 分词器是质量天花板。
- 昂贵：推理每条生成图像需要 VQ-VAE 解码器。

BLIP-2 / LLaVA（晚期融合，分离塔）：
- 视觉进，仅文本出。
- 复用预训练 LLM。
- 理解无分词器瓶颈。
- 便宜：单次前向传播。

按任务选择。如果需要图像生成，Chameleon 家族。如果仅需理解，适配器 VLM 更简单并复用更多预训练算力。

### Fuyu 和 AnyGPT

Fuyu（Adept, 2023）是一种相关方法：完全跳过独立视觉编码器，将原始图像 patch 通过 LLM 的输入投影喂入，如同 token，无分词器。比 Chameleon 简单，但失去共享词汇输出生成。

AnyGPT（Zhan 等人, 2024）将 Chameleon 扩展到四种模态：文本、图像、语音、音乐。每种用相同 VQ-VAE 技巧，共享 Transformer。任意到任意生成。12.16 课更多讲解。

## 使用它

`code/main.py` 构建一个玩具级端到端早期融合模型：

- 一个小型 VQ-VAE 风格量化器，将 8x8 patch 映射到码本索引（K=16）。
- 共享词汇（文本 IDs 0..31）+（图像 IDs 32..47）+（分隔符 48, 49）。
- 一个玩具自回归解码器（二元组表），在合成标题 + 图像 token 序列上训练。
- 采样循环，给定提示发出交替的文本 + 图像 token。

代码有意将 Transformer 保持极小（二元组），让你可以端到端追踪信号流。

## 交付

本课产出 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定产品规格（仅理解 vs 理解+生成、所需图像质量、成本预算），在 Chameleon 家族（早期融合）和 LLaVA 家族（晚期融合）之间选择，并用定量经验法则论证。

## 练习

1. Chameleon 使用 K=8192 码本条，每个 512x512 图像 1024 个 token。估算 vs 24 位 RGB 图像的压缩比。有损吗？损多少？

2. 一张 4K 图像（3840x2160）以相同 VQ-VAE 密度产生多少个图像 token？Chameleon 风格模型能在一次推理调用中生成 4K 图像吗？什么先崩溃——上下文、分词器质量，还是 KV 缓存？

3. 用纯 Python 实现 QK-Norm。给定 64 维 query 和 key，展示 LayerNorm 前后的点积。为什么幅度控制在深层很重要？

4. 阅读 Chameleon 第 2.3 节关于训练稳定性。描述论文在 340 亿参数下没有 QK-Norm 时观察到的确切失败模式。"范数爆炸"的签名是什么？

5. 扩展玩具解码器，在纯文本提示下发出混合模态响应。测量给定训练数据分布 60% 文本优先 / 40% 图像优先下模型选择图像优先 vs 文本优先的频率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 早期融合 | "统一 token" | 图像从步骤一就被转换为离散 token，共享 Transformer 的词汇 |
| VQ-VAE | "图像分词器" | CNN + ViT + 码本，将图像映射到 Transformer 可以预测的整数索引 |
| 共享词汇 | "一个字典" | 覆盖文本 + 图像 + 模态分隔符的单一 token ID 空间 |
| QK-Norm | "注意力稳定器" | 在 query 和 key 点积之前应用 LayerNorm，防止范数爆炸 |
| 混合模态生成 | "文本 + 图像输出" | 推理自主在一次传递中产生交错文本和图像 token |
| 码本大小 | "K 条目" | VQ-VAE 可以量化到的离散向量数量；权衡压缩比和保真度 |
| 分词器天花板 | "重建极限" | 解码 VQ token 可达到的最佳 PSNR；限制模型的图像质量 |

## 延伸阅读

- [Chameleon 团队 — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan 等人 — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu 等人 — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan 等人 — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
