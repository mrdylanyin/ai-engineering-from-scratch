# ColPali 与视觉原生文档 RAG

> 传统 RAG 将 PDF 解析为文本，切分为块，嵌入块，存储向量。每一步丢失信号：OCR 丢弃图表数据，切块破坏表格行，文本嵌入忽略图形。ColPali（Faysse 等人，2024年7月）提出了更简单的问题：为什么要提取文本？通过 PaliGemma 直接嵌入页面图像，使用 ColBERT 风格的后期交互进行检索，保留文档携带的所有布局、图形、字体和格式化信号。发表的基准：在视觉丰富文档上比文本 RAG 端到端准确率高 20-40%。ColQwen2、ColSmol 和 VisRAG 扩展了这一模式。本课阅读视觉原生 RAG 命题并构建一个小型 ColPali 风格索引器。

**Type:** Build
**Languages:** Python (stdlib, multi-vector indexer + MaxSim scorer)
**Prerequisites:** Phase 11 (LLM Engineering — RAG basics), Phase 12 · 05 (LLaVA)
**Time:** ~180 minutes

## 学习目标

- 解释双编码器检索（每文档一个向量）和后期交互检索（每文档多个向量）的区别。
- 描述 ColBERT 的 MaxSim 操作以及 ColPali 如何将其从文本 token 泛化到图像 patch。
- 构建一个小型 ColPali 风格索引器：页面 → patch 嵌入 → MaxSim 对 query 术语嵌入 → top-k 页面。
- 在发票/财务报告用例上比较 ColPali + Qwen2.5-VL 生成器 vs 文本 RAG + GPT-4。

## 问题

在 PDF 上做文本 RAG 丢弃了文档的大部分内容。财务报告的 Q3 收入增长通常在图表中；医学报告的发现位于注释图像中；法律合同的签名块是一个布局事实，不是文本事实。

文本 RAG 管道：

1. PDF → 通过 OCR / pdftotext 提取文本。
2. 文本 → 切成 300-500 token 的块。
3. 块 → 双编码器嵌入（一个向量）。
4. 用户查询 → 嵌入 → 余弦相似度 → top-k 块。
5. 块 + 查询 → LLM。

五个有损步骤。图表未被捕获。表格跨块断裂。多列布局展平。图形注释消失。

ColPali 的修复：跳过 OCR，直接嵌入页面图像。使用 ColBERT 风格后期交互进行检索，使模型可以在查询时关注细粒度 patch。

## 概念

### ColBERT（2020）

ColBERT（Khattab & Zaharia, arXiv:2004.12832）是一种文本检索方法。不是每文档一个向量，而是每 token 产生一个向量。查询时：

- Query token 获得自己的嵌入（N_q 个向量）。
- 文档 token 获得嵌入（N_d 个向量，通常缓存）。
- 分数 = 对 query token 求和，取对文档 token 的最大余弦相似度：Σ_i max_j cos(q_i, d_j)。

这就是 MaxSim 操作。每个 query token "挑选"其最佳匹配的文档 token。最终分数是求和。

优点：强召回，处理术语级语义。缺点：每文档 N_d 个向量，存储昂贵。

### ColPali

ColPali（Faysse 等人，arXiv:2407.01449）将 ColBERT 模式应用于图像。

- 每个页面由 PaliGemma（ViT + 语言）编码为 patch 嵌入：每页 N_p 个向量。
- 每个用户查询（文本）编码为 query token 嵌入：N_q 个向量。
- 分数 = Σ_i max_j cos(q_i, p_j)，即对 query 文本 token 和页面图像 patch 做 MaxSim。
- 按总分检索 top-k 页面。

文档摄入时：用 PaliGemma 嵌入每个页面，存储所有 patch 嵌入。查询时：嵌入 query token，对所有存储的页面嵌入计算 MaxSim，返回 top-k 页面。

优点：在视觉丰富文档上端到端比文本 RAG 好 20-40%。每个 patch 向量捕获局部布局和内容。

缺点：N_p 个 patch × 4 字节浮点 × D 维向量每页 = 存储增长快。通过 PQ / OPQ 量化缓解。

### ColQwen2 和 ColSmol

ColQwen2（illuin-tech, 2024-2025）用 Qwen2-VL 替换 PaliGemma。更好的基础编码器，更好的检索。

ColSmol 是用于本地/边缘使用的较小规模变体。约 1B 参数的 ColSmol 检索器在消费级 GPU 上运行。

### VisRAG

VisRAG（Yu 等人，arXiv:2410.10594）是一个不同的变体：不是 patch 上的 MaxSim，而是用 VLM 将每页池化为单个向量然后双编码器检索。更快索引 + 更小存储，较弱召回。

质量 vs 成本权衡：ColPali 为质量，VisRAG 为规模。

### M3DocRAG

M3DocRAG（Cho 等人，arXiv:2411.04952）将多模态检索扩展到多页多文档推理。跨文档检索页面，为 VLM 组合多页上下文。

### ViDoRe——基准

ColPali 的配套基准。视觉文档检索评估。任务包括财务报告、科学论文、行政文档、医疗记录、手册。指标：nDCG@5。

ColPali-v1 在 ViDoRe 上得分约 80% nDCG@5；文本 RAG 在相同文档上得分约 50-60%。

### 端到端 RAG 管道

对于视觉原生 RAG：

1. 摄入：PDF → 页面图像 → PaliGemma 编码 → 存储所有 patch 嵌入。
2. 查询：用户文本 → query token 嵌入 → 对所有索引页面做 MaxSim → top-k 页面。
3. 生成：top-k 页面图像 + 查询 → VLM（Qwen2.5-VL 或 Claude）→ 答案。

全程无 OCR。图形、图表、字体、布局全部流入答案。

### 存储数学

有 729 patch 每页和 128 维嵌入的 50 页财务报告：

- ColPali：50 * 729 * 128 * 4 字节 = 约 18 MB 原始，约 4 MB 经 PQ 后。
- 文本 RAG：50 块 * 768 维 * 4 字节 = 约 150 kB。

ColPali 每文档约 30 倍存储。大规模下 OPQ / PQ 将其降至约 5-10 倍，通常可容忍。

### 文本 RAG 仍然胜出的情况

- 无布局信号的纯文本文档（wiki 文章、聊天日志）。文本 RAG 更简单存储更便宜。
- 存储主导成本的数百万页档案。
- 严格要求需要检索旁侧可提取 OCR 文本的严格监管要求。

对 2026 年其他所有——财务报告、科学论文、法律合同、医疗记录、UX 文档——视觉原生 RAG 胜出。

## 使用它

`code/main.py`：

- 玩具 patch 编码器：将"页面"（特征向量小网格）映射到 patch 嵌入数组。
- MaxSim 评分器：在 query token 嵌入集和页面 patch 集之间计算 ColBERT 风格分数。
- 索引 5 个玩具页面，运行 3 个查询，返回带分数的 top-k。

## 交付

本课产出 `outputs/skill-vision-rag-designer.md`。给定文档 RAG 项目，选择 ColPali / ColQwen2 / VisRAG / text-RAG 并确定存储规模。

## 练习

1. 一份 200 页年度报告，每页 729 patch，128 维嵌入，4 字节浮点。计算原始存储和 PQ 压缩（8 倍）存储。

2. MaxSim 是 Σ_i max_j cos(q_i, p_j)。这个求和捕获了什么简单均值相似度无法捕获的？

3. ColPali 将页面索引为 patch 集合。如果我们改为在词级别索引（如 ColBERT 所做的）会有什么变化？权衡？

4. 为一个 100 万页语料设计端到端管道，每查询延迟预算 500ms。选择 ColQwen2 / VisRAG 并论证。

5. 阅读 M3DocRAG（arXiv:2411.04952）。描述多页注意力模式及其与单页 ColPali 检索的区别。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 后期交互 | "ColBERT 风格" | 使用每 token 或每 patch 嵌入 + MaxSim 的检索，而非单个文档向量 |
| MaxSim | "对 patch 取最大" | 对每个 query token，选择最高相似度的文档 token；跨 query 求和 |
| 双编码器 | "单向量" | 每文档一个向量；更快但丢失粒度 |
| 多向量 | "每文档多向量" | 每文档/页面存储 N_p 个向量；存储成本增长但召回改善 |
| Patch 嵌入 | "页面特征" | VLM 编码器每图像 patch 一个向量，按页面缓存 |
| ViDoRe | "视觉文档基准" | ColPali 的视觉文档检索基准套件 |
| PQ 量化 | "乘积量化" | 保持向量相似度的压缩，同时缩小存储约 8 倍 |

## 延伸阅读

- [Faysse 等人 — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu 等人 — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho 等人 — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
