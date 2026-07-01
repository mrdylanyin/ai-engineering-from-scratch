# 文档与图表理解

> 文档不是照片。一份 PDF、科学论文、发票或手写表单具有纯图像理解无法捕获的布局、表格、图表、脚注、标题和语义结构。VLM 出现之前的技术栈是一个管道：Tesseract OCR + LayoutLMv3 + 表格提取启发式算法。VLM 浪潮用无 OCR 模型替换了它——Donut（2022）、Nougat（2023）、DocLLM（2023）——直接输出结构化标记。到 2026 年，前沿只是"将页面图像以 2576px 原生分辨率喂给 Claude Opus 4.7"，结构化标记输出就免费获得了。本课阅读文档 AI 的三个时代弧线。

**Type:** Build
**Languages:** Python (stdlib, layout-aware document parser skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 5 (NLP)
**Time:** ~180 minutes

## 学习目标

- 解释文档 AI 的三个时代：OCR 管道、无 OCR、VLM 原生。
- 描述 LayoutLMv3 的三条输入流：文本、布局（bbox）、图像 patch，带统一掩码。
- 比较 Donut（无 OCR，图像 → 标记）、Nougat（科学论文 → LaTeX）、DocLLM（布局感知生成）、PaliGemma 2（VLM 原生）。
- 为新任务选择文档模型（发票、科学论文、手写表单、中文收据）。

## 问题

"理解这个 PDF"欺骗性地难。信息分布在：

- 文本内容（90% 的信号）。
- 布局（标题、脚注、侧边栏、双列格式）。
- 表格（行、列、合并单元格）。
- 图形和图表。
- 手写注释。
- 字体和排版（标题 vs 正文）。

原始 OCR 倾倒文本，丢失其余。一个关心发票的系统需要知道 "Total: $1,245" 来自右下角，而不是脚注。

## 概念

### 时代 1——OCR 管道（2021年前）

经典栈：

1. PDF → 每页图像。
2. Tesseract（或商业 OCR）提取文本，带每词边界框。
3. 布局分析器识别块（标题、表格、段落）。
4. 表格结构识别器解析表格。
5. 领域规则 + 正则提取字段。

对干净印刷文本有效。对手写、歪斜扫描、复杂表格、非英语脚本失败。每个失败模式需要自定义异常路径。

### TrOCR（2021）

TrOCR（Li 等人，arXiv:2109.10282）用合成 + 真实文本图像训练的 Transformer 编码器-解码器替换了 Tesseract 的经典 CNN-CTC。在手写和多语言文本上的清晰胜利。仍是一个管道（检测器然后 TrOCR 然后布局），但 OCR 步骤大幅改善。

### 时代 2——无 OCR（2022-2023）

第一个无 OCR 模型说：完全跳过检测，将图像像素直接映射到结构化输出。

Donut（Kim 等人，arXiv:2111.15664）：
- 编码器-解码器 Transformer，编码器是 Swin-B。
- 输出是表单理解的 JSON、摘要的 markdown 或任意任务特定 schema。
- 无 OCR、无布局、无检测。

Nougat（Blecher 等人，arXiv:2308.13418）：
- 专门在科学论文上训练。
- 输出是 LaTeX / markdown。
- 处理方程、多列布局、图形。
- 每个 arXiv 解析器调用的模型。

这些是专家，不是通才。Donut 在科学论文上失败；Nougat 在发票上失败。

### LayoutLMv3（2022）

一条不同的轨道。LayoutLMv3（Huang 等人，arXiv:2204.08387）保留 OCR 但添加布局理解：

- 三条输入流：OCR 文本 token、每 token 2D 边界框、图像 patch。
- 跨所有三种模态的掩码训练目标（掩码文本、掩码 patch、掩码布局）。
- 下游：分类、实体提取、表格 QA。

LayoutLMv3 是基于 OCR 的文档理解顶峰。在表单和发票上强。需要 OCR 上游。标准化文档基准上 VLM 出现前最高准确率。

### DocLLM（2023）

DocLLM（Wang 等人，arXiv:2401.00908）是 LayoutLM 的生成兄弟。在布局 token 条件化下生成自由形式答案。对文档 QA 更好；仍依赖 OCR 输入。

### 时代 3——VLM 原生（2024+）

2024 年 VLM 变得足够好以完全替代管道。以高分辨率将完整页面图像喂给 VLM，提问，获得答案。

- LLaVA-NeXT 336 瓦片 AnyRes 对小文档有效。
- Qwen2.5-VL 动态分辨率原生处理 2048+ 像素。
- Claude Opus 4.7 支持 2576px 文档。
- PaliGemma 2（2025年4月）专门为文档 + 手写训练。

VLM 原生和 OCR 管道之间的差距迅速缩小。到 2026 年，VLM 原生在以下方面胜出：

- 场景文本（手写+印刷，混合脚本）。
- 带合并单元格的复杂表格。
- 嵌入文本中的数学方程。
- 带文本注释的图形。

OCR 管道在以下方面仍胜出：

- 大规模纯扫描工作负载（每页延迟重要）。
- 管道可靠性（确定性失败 vs VLM 幻觉）。
- 需要可审计 OCR 输出的受监管环境。

### Claude 4.7 / GPT-5 前沿

在 2576 像素原生输入下，前沿 VLM 以接近人类准确率进行文档理解。2026 年初基准数据：

- DocVQA：Claude 4.7 ~95.1，PaliGemma 2 ~88.4，Nougat ~77.3，管道 LayoutLMv3 ~83。
- ChartQA：Claude 4.7 ~92.2，GPT-4V ~78。
- VisualMRC：Claude 4.7 ~94。

闭源模型差距主要是分辨率和 LLM 基础规模。开源 7B 模型落后几个点但在追赶。

### 数学方程和 LaTeX 输出

科学论文需要方程的精确 LaTeX 输出。Nougat 在此上训练。用 LaTeX 目标训练的 VLM（Qwen2.5-VL-Math，Nougat 衍生）产生可用的 LaTeX。没有显式 LaTeX 训练，VLM 产生可读但不精确的转录。

对于 2026 年科学论文管道：在 PDF 上链式 Nougat，然后在棘手页面上用 VLM。

### 手写

仍然是最难的子任务。混合印刷 + 手写（医生笔记、填写的表单）是 OCR 管道在成本上仍胜过 VLM 的地方。纯手写 VLM 正在改进（Claude 4.7、PaliGemma 2）。

### 2026 年配方

对于新的文档 AI 项目：

- 大规模纯印刷发票：LayoutLMv3 + 规则，成本高效。
- 混合文档（科学 + 手写 + 表单）：VLM 原生（PaliGemma 2 或 Qwen2.5-VL）。
- 完整 arXiv 摄入：数学用 Nougat，图形用 VLM。
- 监管：OCR 管道 + VLM 验证器交叉检查。

## 使用它

`code/main.py`：

- 玩具布局感知分词器：给定（文本，bbox）对，生成 LayoutLMv3 风格输入。
- Donut 风格任务 schema 生成器：表单 JSON 模板。
- 跨 OCR 管道、Donut、Nougat 和 VLM 原生比较每页 token 预算。

## 交付

本课产出 `outputs/skill-document-ai-stack-picker.md`。给定文档 AI 项目（领域、规模、质量、监管），在 OCR 管道、无 OCR 专家和 VLM 原生之间选择。

## 练习

1. 你的项目是每天 1000 万张发票。哪个栈在不损失准确率的前提下最小化每页成本？

2. 为什么 LayoutLMv3 在表单 QA 上优于纯 CLIP VLM 但在场景文本上表现不佳？bbox 流放弃了什么？

3. Nougat 生成 LaTeX。提出一个 VLM 原生输出在 LaTeX 保真度上击败 Nougat 的测试案例，以及一个 Nougat 胜出的案例。

4. 阅读 PaliGemma 2 论文（Google, 2024）。相比 PaliGemma 1，提升文档准确率的关键训练数据添加是什么？

5. 设计一个监管安全的混合方案：OCR 管道作为主，VLM 作为次要交叉检查。如何解决分歧？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| OCR 管道 | "Tesseract 风格" | 阶段性栈：检测 -> OCR -> 布局 -> 规则；确定性，脆弱 |
| 无 OCR | "Donut 风格" | 跳过显式 OCR 的图像到输出 Transformer；单一模型 |
| 布局感知 | "LayoutLM" | 输入包括每 token bbox 坐标；跨模态统一掩码 |
| VLM 原生 | "前沿 VLM" | 直接将页面图像以高分辨率喂给 Claude/GPT/Qwen VLM；无管道 |
| DocVQA | "文档基准" | 文档 VQA 标准；被引用最多的分数 |
| 标记输出 | "LaTeX / MD" | 结构化输出格式而非自由形式文本；使下游自动化成为可能 |

## 延伸阅读

- [Li 等人 — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher 等人 — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang 等人 — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim 等人 — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang 等人 — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
