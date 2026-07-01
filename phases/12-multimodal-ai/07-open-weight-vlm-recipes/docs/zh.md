# 开源 VLM 配方：什么真正重要

> 2024-2026 年开源 VLM 文献是一片消融表的森林。Apple 的 MM1 测试了 13 种图像编码器、连接器和数据混合的组合。Allen AI 的 Molmo 证明了详细人工标题优于 GPT-4V 蒸馏。Cambrian-1 进行了 20+ 编码器对比。Idefics2 形式化了五轴设计空间。Prismatic VLMs 在受控基准上比较了 27 个训练配方。在所有这些噪音中，有一小组结果在所有论文中都成立：图像编码器比连接器架构更重要，数据混合比两者都重要，详细人工标题优于蒸馏合成数据。本课阅读这些表格，让你不必亲自去做。

**Type:** Learn + lab
**Languages:** Python (stdlib, ablation table parser + recipe picker)
**Prerequisites:** Phase 12 · 05 (LLaVA baseline)
**Time:** ~180 minutes

## 学习目标

- 说出五轴 VLM 设计空间：图像编码器、连接器、LLM、数据混合、分辨率调度。
- 阅读 MM1 / Idefics2 / Cambrian-1 的消融表，预测哪个旋钮会移动给定基准。
- 在给定算力预算和任务组合的情况下，为新 VLM 选择一个配方（编码器、连接器、数据、分辨率）。
- 解释为什么在相同 token 数量下，详细人工标题优于 GPT-4V 蒸馏。

## 问题

存在数百个开源 VLM。"好"和"最先进"之间的大部分差距不是架构。是数据、分辨率调度和编码器选择。在你的模型表现不佳时知道先调哪个旋钮，可以避免一个 500 万 GPU 小时的错误。

2023 年波次（LLaVA-1.5, InstructBLIP, MiniGPT-4）在标题对预训练 + LLaVA-Instruct-150k 上运行。好的基线。大约在 MMMU 35% 封顶。

2024 年波次（MM1, Idefics2, Molmo, Cambrian-1, Prismatic VLMs）进行了详尽的消融。结果令人惊讶且实用。

## 概念

### 五轴设计空间

Idefics2（Laurençon 等人, 2024）命名了这些轴：

1. 图像编码器。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。编码器在 patch 大小、分辨率和预训练目标上不同。
2. 连接器。MLP（2-4 层）、Q-Former（32 query + 交叉注意力）、Perceiver 重采样器（64 query）、C-Abstractor（卷积 + 双线性池化）。
3. 语言模型。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM 大小是主导参数成本。
4. 训练数据。标题对（CC3M, LAION）、交错（OBELICS, MMC4）、指令（LLaVA-Instruct, ShareGPT4V, PixMo, Cauldron）。
5. 分辨率调度。固定 224/336/448、AnyRes、原生动态。训练期间渐进提升或保持不变。

每个生产 VLM 在每个轴上做出选择。MMMU 分数的大部分方差由轴 1、4 和 5 解释——而非你选择的连接器。

### 轴 1：编码器 > 连接器

MM1 第 3.2 节显示：从 CLIP ViT-L/14 换到 SigLIP SO400m/14 增加 3+ 个 MMMU 点。将连接器从 MLP 换到 Perceiver 重采样器增加不到 1 个点。Idefics2 复现了：SigLIP > CLIP，在相同 token 数下 Q-Former ≈ MLP ≈ Perceiver。

Cambrian-1 的"Cambrian Vision Encoders Match-Up"（Tong 等人, 2024）在视觉中心基准（CV-Bench）上运行了 20+ 编码器。排行榜顶部是 DINOv2 和 SigLIP 的混合；CLIP 处于中游；ImageBind 和 ViT-MAE 较低。从 CLIP ViT-L 到 DINOv2 ViT-g/14 的差距约 5-7 个 CV-Bench 点。

2026 年开源 VLM 的默认编码器是 SigLIP 2 SO400m/14（用于语义 + 密集特征），有时与 DINOv2 ViT-g/14 特征拼接（Cambrian 的 "Spatial Vision Aggregator" 就是这样做的）。

### 轴 2：连接器设计差不多都一样

MM1、Idefics2、Prismatic 和 MM-Interleaved 都达到了相同的结论：在固定视觉 token 数下，连接器架构几乎不重要。在池化 patch 上的 2 层 MLP 在相同 token 预算下与 32-query Q-Former 的性能差距在 1 个点以内。

真正重要的是 token 数量。更多视觉 token = 更多 LLM 算力 = 更好的性能，到某一点后递减。每张图像 64 个 token 对 OCR 来说太少。576-1024 个 token 是大多数开源 VLM 的甜点区。2048+ 仅对文档和图表有帮助。

Q-Former vs MLP 是一个成本问题，不是质量问题：Q-Former 无论图像分辨率大小都将 token 数上限在 32-64；MLP 发出所有 patch token。对高分辨率输入，Q-Former 节省 LLM 上下文；对低分辨率，差异是噪音。

### 轴 3：LLM 大小设定天花板

将 LLM 从 7B 翻倍到 13B，在所有 VLM 论文中可靠地增加 2-4 个 MMMU 点。在 70B 时大多数基准饱和。VLM 的多模态推理上限是 LLM 的文本推理上限——视觉编码器只能输入，不能替它推理。

这就是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 在 MMMU-Pro 和 ScreenSpot-Pro 上碾压：语言大脑巨大。一个 7B VLM 无法通过巧妙的连接器设计替代 70B VLM。

### 轴 4：数据——详细人工标题优于蒸馏

Molmo + PixMo（Deitke 等人, 2024）是每个人都应该阅读的 2024 年结果。Allen AI 让人类标注者用 1-3 分钟的密集语音转文字描述图像，得到 71.2 万张密集标题图像。训练数据中没有任何 GPT-4V 蒸馏。

Molmo-72B 在 11/11 个基准上击败了 Llama-3.2-90B-Vision。差距不是架构——是标题质量。详细人工标题每张图像包含的信息量是短网页标题的 5-10 倍，并且在 GPT-4V 蒸馏会产生幻觉的地方保持事实基准。

ShareGPT4V（Chen 等人, 2023）和 Cauldron（Idefics2）遵循了相同的剧本，混合了人工 + GPT-4V 标题。趋势很明确：对于 2026 年前沿，标题密度 > 标题数量 > 蒸馏便利性。

### 轴 5：分辨率及其调度

Idefics2 的消融：384 -> 448 增加 1-2 点。448 -> 980 带图像切分（AnyRes）在 OCR 基准上再增加 3-5。固定分辨率训练在中精度时平台化；分辨率渐进（从 224 开始，以 448 或原生结束）训练更快、终点更高。

Cambrian-1 运行了分辨率 vs token 的权衡：在固定算力下，你可以有更多低分辨率 token 或更少高分辨率 token。高分辨率对 OCR 有利；低分辨率更多 token 对一般场景理解有利。

2026 年生产配方：第 1 阶段固定 384 训练，第 2 阶段以动态分辨率训练，OCR 重任务达到 1280。

### Prismatic 受控对比

Prismatic VLMs（Karamcheti 等人, 2024）是控制所有轴的论文。相同的 13B LLM、相同的指令数据、相同的评估——每次只变一个轴。结果：

- 每图像视觉 token 数解释约 60% 的方差。
- 编码器选择解释约 20%。
- 连接器架构解释约 5%。
- 其他（数据混合、调度器、LR）剩余约 15%。

这是一个粗略的分解，但它是文献中"我应该先消融什么"的最清晰答案。

### 2026 年选择器

基于证据，2026 年新项目的默认开源 VLM 配方：

- 编码器：SigLIP 2 SO400m/14 原生分辨率（NaFlex），如果需要分割/定位则拼接 DINOv2 ViT-g/14 密集特征。
- 连接器：patch token 上的 2 层 MLP。除非 token 严格受限，跳过 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2，7B 用于成本，70B 用于质量，按目标延迟选择。
- 数据：PixMo + ShareGPT4V + Cauldron，补充任务特定指令数据。
- 分辨率：动态（长边最小 256，最大 1280 像素）。
- 调度：第 1 阶段对齐（仅投影器）、第 2 阶段全量微调、第 3 阶段任务特定微调。

这些默认设置中的每一个都能追溯到本课末尾引用的论文中有测量的消融实验。

## 使用它

`code/main.py` 是一个消融表解析器和配方选择器。它编码了 MM1 和 Idefics2 的消融表（精简版），让你可以查询：

- "给定预算 X 和任务 Y，哪个配方胜出？"
- "如果我把 SigLIP 换成 CLIP 在 7B Llama 上，预期的 MMMU delta 是多少？"
- "我应该先消融哪个轴以获得 80% 信心的答案？"

输出是排名配方列表，含预期基准 delta 和"先消融"建议。

## 交付

本课产出 `outputs/skill-vlm-recipe-picker.md`。给定目标任务组合、算力预算和延迟目标，输出完整配方（编码器、连接器、LLM、数据混合、分辨率调度），并附每个选择的消融引用。防止工程师在每次启动新 VLM 项目时重新发明 Idefics2 的消融表。

## 练习

1. 阅读 MM1 第 3.2 节。对于固定的 2B LLM 和 5000 万图像预算，哪个编码器胜出？答案在 13B LLM 下会翻转吗？为什么？

2. Cambrian-1 发现拼接 DINOv2 + SigLIP 在视觉中心基准上优于单独使用任一，但在 MMMU 上无额外信号。预测哪些基准会提升，哪些保持不变。

3. 你的目标是一个 2B LLM 的移动 UI 智能体。选择编码器、连接器、分辨率和数据混合。用特定消融表论证每个选择。

4. Molmo 发布了 4B 和 72B 模型。4B 与闭源 7B VLM 有竞争力；72B 在 11/11 基准上击败 Llama-3.2-90B-Vision。这说明了 LLM 规模平台化假设的什么？

5. 设计一个消融表，在 7B VLM 上隔离数据混合质量和编码器质量。最少需要多少次训练运行？提出四个轴设置。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 消融 | "转动一个旋钮" | 多次训练运行，只在设计空间的一个轴上不同，其他所有保持不变 |
| 连接器 | "桥梁" / "投影器" | 将视觉编码器输出映射到 LLM token 空间的可训练模块（MLP、Q-Former、Perceiver） |
| 详细人工标题 | "密集标题" | 比网页 alt 文本更丰富的人类编写的多句描述（通常 80-300 token） |
| 蒸馏 | "GPT-4V 标题" | 由更强的专有 VLM 生成的训练数据；方便但容易继承幻觉 |
| AnyRes / 动态分辨率 | "高分辨率路径" | 通过切分或 M-RoPE 喂入比编码器原生分辨率更大图像的策略 |
| 分辨率渐进 | "课程" | 从低分辨率开始逐渐提高的训练调度，加速对齐学习 |
| 视觉中心基准 | "CV-Bench / BLINK" | 强调细粒度视觉感知而非语言重推理的评估 |
| PixMo | "Molmo 的数据" | Allen AI 的 71.2 万密集标题图像数据集；人工语音转文字产生密集标题 |

## 延伸阅读

- [McKinzie 等人 — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon 等人 — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke 等人 — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong 等人 — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti 等人 — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
