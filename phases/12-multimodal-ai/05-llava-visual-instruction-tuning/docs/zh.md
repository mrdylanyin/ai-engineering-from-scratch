# LLaVA 与视觉指令微调

> LLaVA（2023 年 4 月）是这个星球上被复制最多的多模态架构。它用 2 层 MLP 替换了 BLIP-2 的 Q-Former，用朴素的 token 拼接替换了 Flamingo 的门控交叉注意力，并在 GPT-4 从纯文本标题生成的 158k 视觉指令轮次上训练。任何在 2023 年到 2026 年间构建过 VLM 的从业者，都构建了 LLaVA 的某种变体。LLaVA-1.5 增加了 AnyRes。LLaVA-NeXT 提升了分辨率。LLaVA-OneVision 在一个配方中统一了单图像、多图像和视频。本课阅读这个配方，实现投影器，并解释为什么"更简单的一方赢了。"

**Type:** Build
**Languages:** Python (stdlib, projector + instruction-template builder)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 11 (LLM Engineering — instruction tuning)
**Time:** ~180 minutes

## 学习目标

- 构建一个 2 层 MLP 投影器，将 ViT patch 嵌入（维度 1024）映射到 LLM 的嵌入维度（维度 4096）。
- 走完 LLaVA 的两阶段配方：（1）在 558k 标题对上做投影器对齐，（2）在 158k GPT-4 生成的轮次上做视觉指令微调。
- 构建一个包含图像 token 占位符、系统提示和用户/助手轮次的 LLaVA 格式提示。
- 解释社区为什么从 Q-Former 转向 MLP，尽管 Q-Former 在 token 预算上有优势。

## 问题

BLIP-2 的 Q-Former（12.03 课）将一张图像压缩为 32 个 token。干净、高效，基准表现好。但它有两个问题。

第一，Q-Former 是可训练的，但其损失不是最终任务。第 1 阶段训练 ITC+ITM+ITG。第 2 阶段训练 LM 损失。query 学习某种中间表示，LLM 必须对其进行解码。信息在瓶颈处丢失。

第二，Q-Former 占用 1.88 亿参数，在 LLaVA 的 2023 年规模下，你必须与目标 LLM 协同设计它。换 LLM，重新训练 Q-Former。换视觉编码器，重新训练。每个组合都是一个独立的研发项目。

LLaVA 的答案简单到尴尬：取 ViT 的 576 个 patch token，每个通过一个 2 层 MLP（`1024 → 4096 → 4096`），将全部 576 个 dump 到 LLM 的输入序列中。没有瓶颈。没有奇怪目标函数的第 1 阶段预训练。只用直接的 LM 损失训练 MLP。

数据从哪里来？LLaVA 的第二个洞见：使用 GPT-4（纯文本）生成指令数据。将 COCO 标题和边界框数据喂给 GPT-4，让它生成对话、描述和复杂推理问题。免费获得 158k 个指令-响应轮次。零人工标注。

结果：一个在 8 块 A100 上跑一天的 VLM，在 MMMU 上击败了 Flamingo，并发布了社区可以扩展的开放 checkpoint。到 2023 年底，它已经产生了 50+ 个分支。

## 概念

### 架构

LLaVA-1.5 @ 13B：
- 视觉编码器：CLIP ViT-L/14 @ 336（第 1 阶段冻结，第 2 阶段可选解冻）。
- 投影器：2 层 MLP，GELU 激活，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后改为 Llama-3.1-8B）。

在图像 + 文本提示上的前向传递：

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

图像占用 LLM 上下文的 576 个 token。在 2048 上下文下，剩余 1472 个 token 给文本。在 32k 上下文下，这只是一个舍入误差。

### 第 1 阶段：投影器对齐

冻结 ViT。冻结 LLM。仅训练 2 层 MLP。数据集：558k 图像-标题对（LAION-CC-SBU）。损失：在标题上的语言建模，条件化于投影的图像 token。

在 batch 128 的单轮训练中，几小时即可完成。投影器学会将 ViT 空间映射到 LLM 空间。没有任务特定监督。

### 第 2 阶段：视觉指令微调

解冻投影器（仍然可训练）。解冻 LLM（通常全量，有时 LoRA）。在 158k 个视觉指令轮次上训练。

指令数据是诀窍。Liu 等人这样生成：
1. 取一张 COCO 图像。
2. 提取文本描述（5 个人工标题 + 边界框列表）。
3. 发送给 GPT-4，附带三种提示模板：
   - 对话："生成关于这张图像的用户和助手之间的来回对话。"
   - 详细描述："给出图像的丰富、详细描述。"
   - 复杂推理："问一个需要对图像进行推理的问题，然后回答。"
4. 将 GPT-4 输出解析为（指令，响应）对。

这些都不直接涉及图像——只涉及文本描述。GPT-4 会幻觉合理的图像内容。有些噪声，但它奏效了：158k 个轮次足以解锁对话能力。

### 为什么社区复制了这个

- 没有需要调优的第 1 阶段特定损失。全程 LM 损失。
- 投影器几小时即可训练完成，不用几天。
- LLM 可以替换（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），只重新训练投影器即可。
- 视觉指令数据管道使用 GPT-4，为新领域重新生成也很便宜。

### LLaVA-1.5 和 LLaVA-NeXT

LLaVA-1.5（2023 年 10 月）新增：
- 学术任务数据（VQA、OKVQA、RefCOCO）混入指令微调。
- 更好的系统提示。
- 2048 → 32k 上下文。

LLaVA-NeXT（2024 年 1 月）新增：
- AnyRes：将高分辨率图像分割为 2x2 或 1x3 的网格（每个 336x336 裁剪），加上一个全局低分辨率缩略图。每个裁剪 576 个 token；总计约 2880 个视觉 token 每张图像。OCR 和图表任务大幅提升。
- 更好的指令数据混合，使用 ShareGPT4V（高质量 GPT-4V 标题）。
- 更强的 LLM 基础（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

12.08 课深入讲解 OneVision。简短版：相同的投影器，但以课程方式训练，覆盖单图像、多图像和视频，共享视觉 token 预算。

### 与 Q-Former 的对比

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| 每图像视觉 token | 32 | 576（基础）或 2880（AnyRes） |
| 可训练参数 | 1.88 亿 + LM | 4000 万 + LM |
| 第 1 阶段损失 | ITC+ITM+ITG | 仅 LM |
| LLM 替换 | 需要重新训练 | 最小重训即可替换 |
| 多图像 | 别扭 | 自然（拼接） |
| 视频 | 别扭 | 自然（每帧拼接） |
| Token 预算 | 小 | 大 |

MLP 在简单性和 token 灵活性上胜出。Q-Former 在 token 预算上胜出。到 2023 年底，token 预算不再是限制因素（LLM 上下文增长到 32k-128k+），简单性主导。

### 提示格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是一个占位符 token。在分词之前，它被替换为 576 个视觉 token（或 AnyRes 的 2880 个）。tokenizer 看到的序列比训练时稍长，但 LLM 可以处理这个新颖输入，因为第 1 阶段教会了它。

### 参数经济学

LLaVA-1.5-7B 分解：
- CLIP ViT-L/14 @ 336：3.03 亿（第 1 阶段冻结，第 2 阶段通常解冻）。
- 投影器（2 个线性层）：~2200 万可训练。
- Llama-7B：70 亿。
- 总计：73 亿参数。第 2 阶段可训练：完整 70 亿 + 2200 万投影器。

第 2 阶段训练成本：在 8xA100 上约 20 小时。这是关键数字——一天，一个节点，可复现。这就是 LLaVA 传播的原因。

## 使用它

`code/main.py` 实现：

1. 纯 Python 中的 2 层 MLP 投影器（玩具规模 16 → 32 → 32）。
2. 提示构建管道：系统提示 + 被 N 个投影 token 替换的 `<image>` + 用户轮次 + 助手生成占位符。
3. 可视化 576-token 视觉块在 LLM 上下文中的样子（2k / 32k / 128k 上下文的百分比消耗）。

## 交付

本课产出 `outputs/skill-llava-vibes-eval.md`。给定一个 LLaVA 家族 checkpoint，运行一个 10 提示的氛围评估套件（3 个标题生成、3 个 VQA、2 个推理、2 个拒绝回答），并报告人类可读的评分卡。不是一个基准；是一个冒烟测试，确认投影器和 LLM 之间的连接是否良好。

## 练习

1. 计算 `1024 → 4096 → 4096` 的 2 层 MLP 投影器的可训练参数量。包含 GELU 和偏置，它占 LLaVA-13B 的多少比例？

2. 构造一个"拒绝回答"案例的 LLaVA 提示——图像包含一个私密人物。写出期望的助手响应。为什么 LLaVA 应该零样本拒绝这个请求，需要什么训练数据来强化拒绝能力？

3. 阅读 LLaVA-NeXT 博客的 AnyRes 部分。计算 1344x672 图像在 AnyRes 下的视觉 token 数。与基础 336x336 的 576 token 对比。

4. LLaVA 第 1 阶段投影器使用 LM 损失在标题上训练。如果跳过第 1 阶段直接进入第 2 阶段（视觉指令微调）会怎样？引用 Prismatic VLMs 消融实验（arXiv:2402.07865）获得答案。

5. LLaVA-Instruct-150k 使用 GPT-4 和 COCO 标题生成指令。对于新领域（医学 X 光片、卫星图像），描述生成领域指令的四步数据管道。每个步骤可能出什么问题？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 投影器 | "MLP 桥梁" | 带 GELU 的 2 层 MLP，将 ViT 维度映射到 LLM 维度 |
| 图像 token | "<image> 占位符" | 在推理前被 N 个投影视觉 token 替换的提示标记 |
| 视觉指令微调 | "LLaVA 第 2 阶段" | 在 GPT-4 生成的（图像，指令，响应）三元组上训练 |
| 第 1 阶段对齐 | "投影器预训练" | 冻结 ViT 和 LLM，用 LM 损失在标题上训练投影器 |
| AnyRes | "多裁剪切分" | 将高分辨率图像切分为瓦片网格，拼接每个瓦片的视觉 token |
| LLaVA-Instruct | "GPT-4 生成" | 从 COCO 标题 + GPT-4 合成的 158k 指令-响应对 |
| 视觉编码器冻结 | "骨干锁定" | CLIP 权重在第 1 阶段不更新，有时第 2 阶段也不更新 |
| ShareGPT4V | "更好的标题" | GPT-4V 生成的 100 万密集标题，用于更高质量的对齐 |
| VQA | "视觉问答" | 回答关于图像的自由形式问题的任务 |
| Prismatic VLMs | "设计空间论文" | Karamcheti 2024 系统性消融投影器和数据选择的实验 |

## 延伸阅读

- [Liu 等人 — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA 论文。
- [Liu 等人 — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen 等人 — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — 密集标题数据集。
- [Karamcheti 等人 — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — 设计空间消融。
- [Li 等人 — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — 统一单图像、多图像、视频。
