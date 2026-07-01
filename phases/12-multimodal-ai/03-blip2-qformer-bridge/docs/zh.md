# 从 CLIP 到 BLIP-2：Q-Former 作为模态桥梁

> CLIP 对齐了图像和文本，但不能生成标题、回答问题或进行对话。BLIP-2（Salesforce, 2023）用一个小的可训练桥梁解决了这个问题：32 个可学习的 query 向量通过交叉注意力关注冻结 ViT 的特征，然后直接插入到冻结 LLM 的输入流中。一个 1.88 亿参数的桥梁连接了一个 110 亿的 LLM 和一个 ViT-g/14。每一个直到 2026 年的基于适配器的 VLM——MiniGPT-4、InstructBLIP、LLaVA 的变体——都是其后代。本课阅读 Q-Former 的架构，解释其两阶段训练，并构建一个将视觉 token 喂给冻结文本解码器的玩具版本。

**Type:** Build
**Languages:** Python (stdlib, cross-attention + learnable-query demo)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 7 (Transformers)
**Time:** ~180 minutes

## 学习目标

- 解释为什么在冻结的视觉编码器和冻结的 LLM 之间的可训练瓶颈，在成本和稳定性上优于端到端微调。
- 实现一个交叉注意力块，其中一组固定可学习的 query 关注外部图像特征。
- 走完 BLIP-2 的两阶段预训练：表示阶段（ITC + ITM + ITG），然后生成阶段（冻结解码器上的 LM 损失）。
- 比较 Q-Former 与 LLaVA 中使用的更简单 MLP 投影器，论证每种选择在什么场景下胜出。

## 问题

你有一个冻结的 ViT，每张图像产生 256 个维度为 1408 的 patch token。你有一个冻结的 7B LLM，期望维度为 4096 的 token 嵌入。显而易见的桥梁——从 1408 到 4096 的线性层——可行，但将所有 256 个 patch token 喂给 LLM 的上下文需要每张图像 256 个额外 token。在 32 张图像的 batch 中，仅视觉模态就消耗 8192 个 token。

BLIP-2 的问题：能否将 256 token 的图像表示压缩为极少的 token（比如 32 个），同时保留足够信息供 LLM 进行标题生成、问答和图像推理？能否在不动冻结骨干的情况下训练这个桥梁，将训练成本限制在桥梁的参数上？

答案：Q-Former。32 个可学习的"query"向量交叉关注 ViT 的 patch token，产生一个 32 token 的视觉摘要供 LLM 消费。总参数 1.88 亿。在接触 LLM 之前，用对比、匹配和生成目标进行训练。

## 概念

### 可学习的 Query

Q-Former 的核心技巧：不是让 LLM 的文本 token 关注图像 patch，而是引入一组新的 32 个可学习 query 向量 `Q`，让*它们*关注图像 patch。这些 query 是模型的参数——在训练中学习，相同的 32 个 query 用于每张图像。

经过交叉注意力后，每个 query 包含图像的一个压缩摘要——"描述主要物体"、"描述背景"、"数物体数量"等。这些 query 并不会真的专门化到语义标签上；它们学习的是让下游损失下降所需的任意编码。

### 架构

Q-Former 是一个小型 Transformer（12 层，~1 亿参数），有两条路径：

1. Query 路径：32 个 query 向量经历 self-attention（在它们之间），然后交叉关注冻结 ViT 的 patch token，再经过 FFN。
2. 文本路径：一个类似 BERT 的文本编码器与 query 路径共享 self-attention 和 FFN 权重。文本路径不启用交叉注意力。

训练时两条路径都运行。Query 和文本通过共享的 self-attention 交互，这意味着 query 可以对文本条件化以完成需要文本的任务（ITM、ITG）。VLM 交接时的推理阶段，只有 query 流过，产生 32 个视觉 token。

### 两阶段训练

BLIP-2 分两阶段预训练：

第 1 阶段：表示学习（无 LLM）。三种损失：
- ITC（图像-文本对比）：在池化 query token 和文本 CLS token 之间做 CLIP 风格对比。
- ITM（图像-文本匹配）：二分类器——这个图像-文本对是否匹配？使用硬负样本挖掘。
- ITG（基于图像的文本生成）：在文本上做因果 LM 头，条件化于 query。强制 query 编码可生成文本的内容。

只有 Q-Former 训练。ViT 冻结。LLM 不参与。

第 2 阶段：生成学习。附加一个冻结的 LLM（OPT-2.7B 或 Flan-T5-XL 等）。通过一个小线性层将 32 个 query 输出投影到 LLM 嵌入维度。将其前置到文本提示前。仅在投影层和 Q-Former 上训练 LM 损失。

经过第 2 阶段，Q-Former + 投影层构成完整的视觉适配器。推理时：图像 → ViT → Q-Former → 线性投影 → 前置到文本 → 冻结 LLM 产生输出。

### 参数经济学

BLIP-2 配置：ViT-g/14（11 亿，冻结）+ OPT-6.7B（67 亿，冻结）+ Q-Former（1.88 亿，可训练）= 总共 80 亿参数，训练 1.88 亿。Q-Former 仅占全栈参数的约 2.4%。训练成本反映了这一点：几块 A100 训练几天 vs 端到端训练几周。

质量：BLIP-2 在零样本 VQA 上匹配或击败了 Flamingo-80B，同时规模小 50 倍。桥梁有效。

### InstructBLIP 和指令感知 Q-Former

InstructBLIP（2023）扩展了 Q-Former，增加了一个额外输入：指令文本本身。在交叉注意力时，query 现在可以同时访问图像 patch 和指令。Query 可以按指令专门化（"数一数汽车"、"描述氛围"），而不是学习一个单一的固定摘要。在未见任务上的基准有提升。

### MiniGPT-4 和仅投影方法

MiniGPT-4 保留了 Q-Former，但仅训练输出线性投影层，其他全部冻结。便宜，但代价是质量——这些 query 是 BLIP-2 的，不是针对你的任务的。适合快速迭代，但不是最优架构。

### 为什么 LLaVA 选择了更简单的方案

LLaVA（2023，12.05 课）用一个简单的 2 层 MLP 替换了 Q-Former，将每个 ViT patch token 投影到 LLM 空间——每张图像 576 个 token（24x24 网格），全部喂给 LLM。压缩更差，但让 LLM 可以关注原始 patch。这在当时有争议；到 2023 年底它成为主流，因为视觉指令数据（LLaVA-Instruct-150k）证明了 MLP 可以训练出足够的信号保留。权衡：LLaVA 的上下文填得更快，但自然扩展到多图像和视频。

到 2026 年，领域出现分化：Q-Former 在 token 预算受限时存活（长视频、多图像）；MLP 投影器在每 token 原始质量为优先时主导。

### 门控交叉注意力：Flamingo，前身

Flamingo（12.04 课）早于 BLIP-2，使用了相同的交叉注意力思想，但在每个冻结 LLM 层插入，而非作为单次桥梁。BLIP-2 证明了可以仅压缩到输入层且仍能工作。Gemini 和 Idefics 结合了两者：交错输入 token 加上可选的门控交叉注意力用于上下文内少样本。

### 2026 年的后代

- Q-Former：BLIP-2、InstructBLIP、MiniGPT-4 以及大多数因 token 预算限制而使用它的视频-语言模型。
- Perceiver 重采样器：Flamingo 的变体（12.04 课）；Idefics 家族、Eagle、OmniMAE。
- MLP 投影器：LLaVA、LLaVA-NeXT、LLaVA-OneVision、Cambrian-1。
- 注意力池化：VILA、PaliGemma。

四种都有效。决定性问题在于你是受 token 预算约束还是受每个 token 的质量约束。

## 使用它

`code/main.py` 构建了一个标准库 Q-Former 风格的交叉注意力：

1. 模拟 256 个图像 patch token（维度 128）。
2. 实例化 32 个可学习 query（维度 128）。
3. 运行缩放点积交叉注意力（Q 来自 query，K/V 来自 patch）。
4. 通过线性层投影到 LLM 维度（512）。
5. 输出 32 个 LLM 就绪的视觉 token。

所有数学用纯 Python 实现（向量嵌套循环）。玩具级但形态正确。注意力权重矩阵被打印出来，你可以看到每个 query 从哪些 patch 拉取了信息。

## 交付

本课产出 `outputs/skill-modality-bridge-picker.md`。给定目标 VLM 配置（视觉编码器 token 数、LLM 上下文预算、部署约束、质量目标），推荐 Q-Former vs MLP vs Perceiver 重采样器，附简短论证和每个桥梁的参数量估算。

## 练习

1. 在 PyTorch 中实现交叉注意力块。验证 32 个 query 和 256 个 key/value 时，注意力权重矩阵为 32 x 256，softmax 后每行和为 1。

2. 在 BLIP-2 第 1 阶段，Q-Former 同时运行三个损失：ITC、ITM、ITG。用伪代码写出每个的前向签名。哪个需要文本编码器路径处于活跃状态？

3. 比较参数量：Q-Former（12 层，768 隐藏维度）vs 2 层 MLP 投影器（1408 → 4096，两层）。在什么 LLM 规模下，1.88 亿参数的 Q-Former 在训练效率上获得回报？

4. 阅读 BLIP-2 论文（arXiv:2301.12597）第 3.2 节关于 Q-Former 如何初始化。解释为什么从 BERT-base（而非随机）初始化加速收敛。

5. 对一个 10 分钟、1 FPS 采样到 60 帧的视频，计算每帧 token 成本：(Q-Former → 32 tokens/帧) vs (MLP 投影器 → 576 tokens/帧)。哪个能装入 128k token 的 LLM 上下文窗口？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Q-Former | "Querying transformer" | 带 32 个可学习 query 向量的小型 Transformer，交叉关注冻结 ViT 特征 |
| 可学习 query | "视觉软提示" | 作为交叉注意力 query 侧的一组固定参数；按模型学习，所有输入共享 |
| 交叉注意力 | "Q 来自这里，K/V 来自那里" | Query、Key 和 Value 来自不同来源的注意力；query 如何从 ViT patch 提取信息 |
| ITC | "图像-文本对比" | 应用于 Q-Former 池化 query 和文本 CLS 的 CLIP 风格损失 |
| ITM | "图像-文本匹配" | 基于硬负样本挖掘对做二分类；强制 query 区分细粒度不匹配 |
| ITG | "基于图像的文本生成" | 在条件化于 query 的文本上做因果 LM 损失；强制 query 编码可文本解码的内容 |
| 两阶段预训练 | "表示然后生成" | 第 1 阶段仅训练 Q-Former（ITC/ITM/ITG）；第 2 阶段附加冻结 LLM，仅训练投影层 + Q-Former |
| 冻结骨干 | "不微调" | 视觉编码器和 LLM 权重固定；只有桥梁可训练 |
| 投影头 | "线性到 LLM 维度" | 将 Q-Former 输出映射到 LLM 嵌入维度的最终线性层 |
| Perceiver 重采样器 | "Flamingo 的版本" | 类似的可学习 query 交叉注意力，被 Flamingo 用在每层而非作为单次桥梁 |

## 延伸阅读

- [Li 等人 — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — 核心论文。
- [Li 等人 — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — 带有 ITC/ITM/ITG 三元组的前身。
- [Li 等人 — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — "先对齐再融合"——第 1 阶段训练的概念祖先。
- [Dai 等人 — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — 指令感知 Q-Former。
- [Zhu 等人 — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — 仅投影方法。
- [Jaegle 等人 — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 可学习 query 交叉注意力的通用架构。
