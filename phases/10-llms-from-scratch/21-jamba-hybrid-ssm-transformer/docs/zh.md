# Jamba — 混合 SSM-Transformer

> 状态空间模型（State Space Models, SSM）和 Transformer 追求不同的目标。Transformer 通过注意力（Attention）以平方成本换取质量。SSM 通过递归（recurrence）换取线性时间推理和恒定内存，但质量略逊。AI21 的 Jamba（2024 年 3 月）和 Jamba 1.5（2024 年 8 月）将二者合并在同一个模型中：每 7 层 Mamba 搭配 1 层 Transformer，每隔一层使用 MoE，256k 上下文窗口可适配在单张 80GB GPU 上。Mamba-3（ICLR 2026）通过复值状态空间和 MIMO 投影加强了 SSM 侧。本节课从头到尾解读两种架构，并解释为什么这个混合配方在三年 scaling 中存活了下来，而纯 SSM 和纯 Transformer 的长上下文尝试却没有。

**Type:** Learn
**Languages:** Python (stdlib, layer-mix calculator)
**Prerequisites:** Phase 10 · 14（开源模型架构），Phase 10 · 17（native sparse attention）
**Time:** ~60 分钟

## 学习目标

- 解释 Jamba 块（block）中的三个原语——Transformer 层、Mamba 层、MoE——以及 1:7:交替 的交织配方。
- 从高层次陈述 SSM 的递归形式，并解释为什么它能实现恒定内存推理。
- 计算 Jamba 模型在 256k 上下文下的 KV 缓存（KV cache）占用，并与纯 Transformer 模型的需求进行比较。
- 说出 Mamba-3 的三个创新点（指数梯形离散化、复值状态更新、MIMO）以及每个创新点针对的问题。

## 问题

注意力（Attention）的复杂度是序列长度的平方。状态空间模型是线性的。这一差异会叠加：在 256k token 时，一个 Transformer 注意力图每个头是 650 亿个条目；而一个 SSM 的递归状态大小固定，与序列长度无关。

纯 SSM 模型（Mamba、Mamba-2）在小规模下能匹配 Transformer 的困惑度（perplexity），但在状态追踪任务上表现落后，并在某些类别的上下文检索（in-context retrieval）中失败。直觉是：SSM 将历史压缩到一个固定大小的状态中，当历史很长时，信息会泄漏。注意力精确地记住一切，但付出平方成本。

显而易见的修正方案：两者都用。在需要精确召回的地方放置 Transformer 层。其他地方使用 SSM 层。调优比例。Jamba 是第一个在大规模下交付此混合配方的生产级模型（52B 总量/12B 激活，256k 上下文，单张 80GB GPU）。Jamba 1.5 将家族扩展到 398B 总量/94B 激活。Mamba-3（ICLR 2026）是混合模型可以重建的最优纯 SSM 基线。

本节课阅读三篇论文，并产出关于"选择正确比例"的心智模型。

## 概念

### 一页纸讲清 SSM

一个状态空间模型通过一个固定大小的状态 `h` 处理序列 `x_1, ..., x_N`：

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

每一步状态通过线性动力学 `A` 演化，接收输入 `B x_t`，输出 `C h_t`。`A, B, C` 可以被学习。注意关键属性：计算 `y_t` 只需要 `h_{t-1}` 和 `x_t`，不需要任何更早的 `x`。内存是恒定的。推理每 token 为 O(1)。

建模质量的诀窍在于 `A` 的结构。S4（Gu 2021）使用了一个高度结构化的矩阵，可以在训练期间高效地执行长卷积运算。Mamba（Gu, Dao 2023）用数据依赖的（即"选择性 selective"部分）参数替代了固定的 `A, B, C`。Mamba-2（2024）进一步简化了结构。Mamba-3（2026）在特定位置重新增加了复杂度。

关键属性：对于解码器 LLM，一个 SSM 层是注意力层的直接替换（drop-in replacement），使用固定大小的每层状态而不是不断增长的 KV 缓存。

### Jamba 块

一个 Jamba 块根据两个数字来交织层：

- `l`：注意力与 Mamba 的比例。Jamba 使用 `l = 8`，即每 7 层 Mamba 搭配 1 层 Transformer（7 Mamba + 1 Attention = 每组 8 层）。
- `e`：MoE 频率。Jamba 使用 `e = 2`，即每隔一层应用 MoE。

一个块内的层序列：

```
M  M  M  M  M  M  M  A    （7 Mamba + 1 Attention）
|  M  |  M  |  M  |  M    （其中 | 标记应用 MoE 的位置）
```

每个 Jamba 块是 8 层。在 4 个块深度（共 32 层）下，你会得到 28 个 Mamba 层和 4 个 Attention 层。其中 16 层使用 MoE。

### 为什么是 1:7 比例

AI21 进行了消融实验：注意力与 Mamba 的什么比例在困惑度每参数（perplexity-per-parameter）和长上下文评测（in-context recall）上表现最好？

- 注意力过多（1:1）：质量上升但内存和速度变差。
- 注意力过少（1:15）：内存很好但上下文检索失败。
- 甜点：1:7 或 1:8。

直觉：Transformer 层处理精确召回和状态追踪。Mamba 层处理廉价的批量处理。

### 位置编码

Mamba 层本身通过递归感知位置。原始基于 Mamba 的混合模型中，Attention 层不使用 RoPE——由 SSM 层提供位置信息。Jamba 1.5 为 Attention 层添加了 RoPE，以支持更长上下文的泛化，这是基于经验性长上下文评估的事后改进。

### 内存预算

对于一个 Jamba-1 形状（32 层：28 Mamba + 4 Attention，hidden 4096，32 个注意力头）：

- KV 缓存（仅 Attention 层）：在 256k BF16 下为 `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`。只有 4 个 Attention 层贡献。
- SSM 状态：每个 token 前缀为 `28 * hidden * state_size`，但这是每层固定大小，不随序列长度增长。典型 Mamba 状态为每特征 16，hidden 4096：总计 `28 * 4096 * 16 * 2 = 3.7 MB`。

对比同样形状的纯 Transformer（32 层，相同 hidden，32 头 MHA）：在 256k BF16 下为 `2 * 32 * 32 * 128 * 256k * 2 = 128 GB`。KV 缓存减少 8 倍。即使对比大多数 2024 模型使用的 GQA(8) 基线（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`），Jamba 的 1:7 混合方案在 16 GB 下仍然小 2 倍。

这就是 AI21 所说的"单张 80GB GPU 上支持 256k 上下文"的含义。纯 MHA Transformer 的 KV 缓存根本装不下；即使 GQA 基线也不留权重和激活的空间；Jamba 可以。

### Mamba-3：2026 年的纯 SSM 基线

Mamba-3（ICLR 2026，arXiv:2603.15569）在纯 SSM 侧引入了三项创新：

1. **指数梯形离散化（Exponential-trapezoidal discretization）。** 用更具表现力的递归替代 Mamba-2 中的 Euler 方法离散化。类似卷积的运算在核心递归内部应用于状态-输入，而非作为 `x_t` 上的外部卷积。

2. **复值状态更新。** 之前的 Mamba 将状态矩阵从复数（S4）简化为实对角矩阵（Mamba），再简化到缩放恒等矩阵（Mamba-2）。Mamba-3 重新引入复数值——等价于状态上的数据依赖旋转嵌入。这恢复了之前因实值简化而损失的状态追踪能力。

3. **多输入多输出（MIMO）投影。** 使用矩阵值投影替代逐特征的标量投影。在不增加解码延迟的情况下提升建模能力和推理时硬件利用率。

在 1.5B 参数规模下，Mamba-3 相比 Gated DeltaNet 将平均下游准确率提升 0.6 个点；MIMO 变体进一步增加 1.2 个点，总计提升 1.8 个点。在相同状态大小下，Mamba-3 以一半的状态匹配 Mamba-2 的性能。

Mamba-3 尚未大规模部署在生产级混合模型中——但它是下一代 Jamba 类模型 SSM 侧的显然候选。

### 何时选择混合架构

混合架构胜出的情况：

- 上下文足够长，纯 Transformer 的 KV 缓存变得痛苦（64k+）。
- 任务混合了短程结构（适合 SSM）和长程召回（需要 Transformer）。
- 你希望在单 GPU 内存预算内部署，而 Transformer KV 缓存本身装不下。

混合架构失利的情况：

- 上下文较短（低于 16k）。SSM 开销被浪费；纯 Transformer 就够用。
- 任务需要全局注意力（深度推理、多文档交叉引用）。混合模型中注意力层的稀疏性会造成损害。
- 你正在扩展到万亿参数的前沿模型。纯 Transformer + MLA + MoE（DeepSeek-V3 风格）目前在能力竞赛中领先。

### 竞争格局

| 模型 | 家族 | 规模 | 独特卖点 |
|-------|--------|------|-------------|
| Mamba-2 | 纯 SSM | 3B | 线性时间，恒定内存 |
| Jamba | 混合 | 52B/12B | 80GB 上 256k 上下文 |
| Jamba 1.5 Large | 混合 | 398B/94B | 企业级长上下文 |
| Mamba-3 | 纯 SSM | 1.5B（论文） | 状态追踪能力恢复 |
| DeepSeek-V3 | 纯 Transformer + MoE | 671B/37B | 前沿能力 |

2026 年格局：纯 Transformer MoE 主导前沿，但混合架构占据 256k+ 上下文利基。Mamba-3 的状态追踪胜利可能推动下一代混合模型降低注意力比例（更多 SSM，更少注意力）。

```figure
swiglu-ffn
```

## 使用

`code/main.py` 是一个混合架构的内存计算器。给定 SSM-Transformer 比例以及 hidden-size/层数配置，它计算：

- 目标上下文下的 KV 缓存。
- SSM 状态内存。
- 在一系列模型形状下，上下文 N 处的总内存。

计算器支持：

- 纯 Transformer 基线（KV 缓存随 N 增长）。
- Jamba 风格 1:7 混合。
- 纯 SSM（完全没有 KV 缓存）。

数字直接来自 Jamba-1 和 Jamba-1.5 论文中已发布的形状，并对假设变体进行了外推。

实际部署的集成考虑：

- 大多数生产推理服务器（vLLM、SGLang）支持 Jamba 和 Mamba。请检查具体版本。
- 在 256k 上下文下，Jamba 的内存在并发请求吞吐量上展现出优势。在相同 VRAM 下，你可以适配比 Transformer 序列更多的 Jamba 序列。
- Mamba-3 作为独立模型尚未进入生产部署——1.5B 研究预览阶段。

## 产出

本节课产出 `outputs/skill-hybrid-picker.md`。给定一个工作负载规格（上下文长度分布、任务组合、内存预算），它在纯 Transformer、Jamba 风格混合和纯 SSM 之间推荐选择，并附有明确的内存和质量权衡推理。

## 练习

1. 运行 `code/main.py`，计算 256k 上下文下 32 层纯 Transformer（hidden 4096，32 头）和相同形状 Jamba-1 混合模型的 KV 缓存。验证 AI21 论文声称的约 8 倍内存减少。

2. 修改计算器以建模 1:3 混合（4 Mamba : 1 Attention）和 1:15 混合（14 Mamba : 1 Attention）。绘制 KV 缓存与比例的关系图。在什么比例下 KV 缓存等于 SSM 状态内存？

3. 阅读 Jamba 论文第 3 节（arXiv:2403.19887）。解释为什么 AI21 使用 Mamba-1 而非 Mamba-2，尽管 Mamba-2 更快。提示：混合消融部分记录了这一信息。

4. 计算 Jamba 1.5 Large（398B 总量，94B 激活）中每隔一层 MoE 的参数开销。将激活比与 DeepSeek-V3（37B/671B）进行比较，解释为什么 Jamba 的架构将激活比推得更高。

5. 阅读 Mamba-3 论文第 3 节（arXiv:2603.15569）。用三句话解释为什么复值状态更新等价于数据依赖的旋转嵌入。将答案与 Phase 7 · 第 04 课的 RoPE 推导关联起来。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| State space model, SSM（状态空间模型） | "具有固定状态的递归" | 具有学习递归 `h_t = A h_{t-1} + B x_t` 的层；每 token 恒定内存 |
| Selective SSM（选择性 SSM） | "Mamba 的诀窍" | 数据依赖的 A、B、C 参数，在线性时间内为模型提供类似门控的选择性 |
| Attention-to-Mamba ratio（注意力-Mamba 比例） | "有多少注意力层" | Jamba 中 `l = 8` 表示每 7 个 Mamba 层配 1 个注意力层 |
| Jamba block（Jamba 块） | "8 层组" | 一个注意力 + 七个 Mamba + 交替位置上的 MoE |
| SSM state（SSM 状态） | "隐藏缓冲区" | 替代 Mamba 层 KV 缓存的固定大小每层状态 |
| 256k context（256k 上下文） | "Jamba 的标志性数字" | Jamba-1 可在单张 80GB GPU 上适配的序列长度；纯 Transformer 在该规模下无法做到 |
| Mamba-3 | "2026 纯 SSM" | 当前最佳的纯 SSM 架构，具有复值状态 + MIMO；混合模型重建的基线 |
| MIMO（多输入多输出） | "多输入多输出" | Mamba-3 创新，使用矩阵值投影替代逐特征标量 |
| Exponential-trapezoidal discretization（指数梯形离散化） | "Mamba-3 的递归" | 更具表现力的递归，包含了 Mamba-2 的 Euler 方法离散化 |
| Hybrid architecture（混合架构） | "混合注意力和 SSM" | 任何交织 Transformer 和 SSM 层的模型；Jamba 是生产环境的原型 |

## 延伸阅读

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model（arXiv:2403.19887）](https://arxiv.org/abs/2403.19887) — 原始 Jamba 论文，比例消融实验，256k 上下文声明
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale（arXiv:2408.12570）](https://arxiv.org/abs/2408.12570) — 扩展后的家族，398B/94B 和 12B/52B 公开发布
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces（arXiv:2312.00752）](https://arxiv.org/abs/2312.00752) — Jamba 所构建的选择性 SSM 论文
- [Dao, Gu — Mamba-2（arXiv:2405.21060）](https://arxiv.org/abs/2405.21060) — 简化的结构化状态空间后继
- [Lahoti et al. — Mamba-3（arXiv:2603.15569，ICLR 2026）](https://arxiv.org/abs/2603.15569) — 复值状态、MIMO，2026 年纯 SSM 前沿
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces（arXiv:2111.00396）](https://arxiv.org/abs/2111.00396) — S4 论文，SSM 谱系用于 LLM 的起点
