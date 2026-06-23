# 原生稀疏注意力（DeepSeek NSA）

> 在 64k token 下，注意力（Attention）消耗了 70-80% 的解码延迟。每个开源模型实验室都有一个修复计划。DeepSeek 的 NSA（ACL 2025 最佳论文）是那个留下来了的方案：三个并行的注意力分支——压缩后的粗粒度 token、选择性保留的细粒度 token、以及用于局部上下文的滑动窗口——通过一个学习到的门控组合在一起。它是硬件对齐的（hardware-aligned，对 kernel 友好），原生可训练的（在预训练中工作，而非在推理时附加），并且在 64k 解码上比 FlashAttention 更快，同时匹配甚至超过全注意力（full attention）的质量。本节课从头构建三个分支，并展示为什么这种稀疏性是端到端可微的。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 12（KV cache、flash-attention），Phase 7 · 15（注意力变体），Phase 10 · 16（差分注意力）
**Time:** ~60 分钟

## 学习目标

- 陈述 NSA 的三个注意力分支以及每个分支捕获的内容。
- 解释为什么 NSA 是"原生可训练的（natively trainable）"，而先前的稀疏注意力方法仅是推理时的。
- 计算 NSA 相对于全注意力在 64k 上下文下的注意力计算节省，作为压缩块大小和 top-k 选择的函数。
- 在标准库 Python 中对短合成序列实现三分支组合，并验证门控权重的行为。

## 问题

序列长度 N 下的全注意力需要 `O(N^2)` 时间和 `O(N)` 每层 KV 缓存。在 64k token 下，计算和内存带宽数值是灾难性的。来自 NSA 论文的测量理论估计：在 64k 下，注意力占解码总延迟的 70-80%。下游一切——首 token 时间（TTFT）、每秒 token 数、每百万 token 成本——都受注意力成本主导。

稀疏注意力是显而易见的答案。先前的尝试分为两类。固定模式稀疏（滑动窗口、步进式、块局部）丢弃信息，在长程召回任务上失败。推理时稀疏（KV 缓存剪枝、H2O、StreamingLLM）应用于在稠密注意力上预训练的模型，只能回收潜在加速的一小部分，因为模型从未被要求通过稀疏模式来路由信息。

原生稀疏注意力（Native Sparse Attention，Yuan et al.，DeepSeek + 北大 + 华盛顿大学，ACL 2025 最佳论文，arXiv:2502.11089）两者兼备：一种模型在预训练期间学习的稀疏模式，实现为在推理时真正实现计算节省的 kernel 对齐算法。两年后，NSA 或其直接后代将成为每个前沿长上下文模型的默认注意力。

## 概念

### 三个并行分支

对于每个查询（query），NSA 运行三次注意力，分别针对 KV 缓存的三种不同视图：

1. **压缩分支（Compressed branch）。** Token 被分组为大小为 `l`（通常为 32 或 64）的块。每个块通过一个小型学习的 MLP 压缩成单个摘要 token。查询对压缩后的 token 进行注意力计算，获得整个序列的粗粒度视图。

2. **选择分支（Selected branch）。** 利用压缩分支的注意力分数，识别出与当前查询最相关的 top-k 个块。从这些块中读取细粒度（未压缩的）token，查询对所有这些 token 进行注意力计算。可以将压缩分支注意力视为选择的路由信号。

3. **滑动窗口分支（Sliding-window branch）。** 查询对最近的 `W` 个 token（通常为 512）进行注意力计算，用于局部上下文。该分支捕获其他两个可能遗漏的结构密集型短程模式（语法、局部共指）。

三个分支的输出通过每个位置学习的门控进行组合：

```
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` 是来自查询上的小型 MLP 的门控权重。它们的和不要求为 1——它们可以独立加权各分支。

### 为什么这是"原生可训练的"

选择步骤（top-k 块）是离散的。离散操作会破坏梯度流。先前的稀疏注意力工作要么跳过选择步骤的反向传播（限制训练），要么使用连续松弛，在推理时不给出真正的稀疏性。

NSA 回避了这个问题：压缩分支注意力本身是序列上的可微粗粒度注意力。top-k 操作只是复用压缩分支的最高注意力分数来选择要加载哪些细粒度块。梯度流经压缩分支的分数（同时影响压缩输出和选择逻辑），所选块对最终输出的贡献也是可微的。不可微的 `top_k` 操作在前向计算图上是一个 no-op——它只控制从内存中加载哪些块。

这就是为什么 NSA 可以端到端地用于预训练。模型学习通过三个分支联合路由信息，产生一种在推理时真正实现承诺加速的稀疏模式。

### 硬件对齐的 kernel

NSA 的 kernel 针对现代 GPU 内存层次结构设计。kernel 按 GQA 组加载查询（外层循环），为每个组获取相应的稀疏 KV 块（内层循环），并在 SRAM 上运行注意力。由于每个查询组看到相同的选定块（选择是按查询组而非按查询头），KV 加载在组内平摊。算术强度保持较高。

论文报告了 Triton kernel 在 64k 解码上比 FlashAttention 快 9 倍，且加速比随序列长度增长。前向和反向 kernel 均已提供。

### 计算预算

令 `N` 为序列长度，`l` 为压缩块大小，`k` 为 top-k 选择计数，`w` 为滑动窗口，`b` 为选定块大小（通常等于 `l`）。

- 压缩分支：每个查询 `O(N/l)` 个 key，总计 `O(N * N / l)`。
- 选择分支：每个查询 `O(k * b)` 个 key，总计 `O(N * k * b)`。
- 滑动分支：每个查询 `O(w)` 个 key，总计 `O(N * w)`。

总计：`O(N * (N/l + k*b + w))`。

当 `N = 64k, l = 64, k = 16, b = 64, w = 512` 时：每个查询成本为 `1000 + 1024 + 512 = 2536 个 key`。全注意力为 `64000 个 key`。25 倍计算减少。

当 `N = 128k, l = 64, k = 16, b = 64, w = 512` 时：每个查询成本为 `2000 + 1024 + 512 = 3536 个 key`。全注意力为 `128000 个 key`。36 倍减少。收益随序列长度增长，这正是全部意义所在。

### 与其他方法的比较

| 方法 | 可微 | 实际推理加速 | 长程召回 |
|--------|---------------|----------------------|-------------------|
| 仅滑动窗口 | 是 | 是 | 失败 |
| 步进/块稀疏 | 是 | 是 | 部分 |
| KV 剪枝（H2O, StreamingLLM） | 不适用（推理时） | 是 | 部分 |
| MoBA（Moonshot） | 部分 | 是 | 良好 |
| NSA | 是（原生） | 是（64k 下 9 倍） | 匹配全注意力 |

MoBA（Moonshot，arXiv:2502.13189）同期发布，采用类似的三合一方案，将 MoE 原理应用于注意力块。NSA 和 MoBA 是 2026 年长上下文预训练需要了解的两个架构。

```figure
sliding-window-attention
```

## 构建

`code/main.py` 在短合成序列上实现了三个分支，并展示：

- 压缩 MLP（为教学清晰度使用简单均值池化基线；真正的 NSA 使用学习到的 MLP）。
- 由压缩分支分数驱动的 top-k 块选择。
- 对最后 `w` 个 token 的滑动窗口注意力。
- 门控组合。
- 与全注意力对比的计算计数输出。

### 步骤 1：将 token 压缩为块

```python
def compress(K, l):
    # 将 key 分组成大小为 l 的块，每个块压缩为一个摘要向量
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### 步骤 2：压缩分支注意力

对压缩后的 key 运行查询的 softmax 注意力。压缩分支的分数同时作为 top-k 选择的信号。

### 步骤 3：top-k 块选择

选取压缩分支分数最高的 `k` 个块的索引。加载这些块中的原始未压缩 token，并对其运行注意力。

### 步骤 4：滑动窗口注意力

取最后 `w` 个 token 并对其运行标准注意力。

### 步骤 5：门控 + 组合

查询上的小型 MLP 产生三个门控权重。最终输出是三个分支输出的加权和。

### 步骤 6：计算计数

打印每个分支每个查询关注的 key 数量和总计数。与 `N`（全注意力）进行比较。在一个 1024 token 合成序列上，使用 `l = 32, k = 4, w = 128`，NSA 每个查询看到 `32 + 128 + 128 = 288 个 key`，而全注意力为 1024——减少 3.5 倍。

## 使用

NSA 已在 DeepSeek 自身的长上下文预训练流水线中部署。截至 2026 年 4 月，在公开推理栈中的集成状况：

- **DeepSeek 内部**：原生支持，已发布的权重使用 NSA 或其继任者 DSA（Deepseek Sparse Attention）。
- **vLLM**：正在为 DeepSeek-V3.x 权重开发实验性 NSA 支持。
- **SGLang**：NSA 基准测试已发布；生产路径跟随 vLLM。
- **llama.cpp / CPU**：不支持；kernel 分解的开销在 CPU 吞吐量下不值得。

何时选择 NSA：

- 针对 64k+ 上下文、有充足计算预算的预训练或继续训练。
- 推理 DeepSeek 自身的长上下文检查点。权重是 NSA 原生的。

何时不选：

- 服务已有的稠密注意力预训练模型。不经过继续训练无法改装 NSA。
- 上下文低于 16k。三分支开销超过节省。
- Batch-1 交互式聊天。延迟敏感的解码受益，但仅在长上下文下。

## 产出

本节课产出 `outputs/skill-nsa-integrator.md`。给定一个长上下文预训练运行规格，它产出一个 NSA 集成计划：压缩块大小、top-k、滑动窗口、门控 MLP 宽度、kernel 选择，以及证明架构变更合理性的具体长上下文评测。

## 练习

1. 在 1024 token 合成序列上运行 `code/main.py`。在三组预设之间扫描 `(l, k, w)` 并打印计算计数。找出在"大海捞针"（needle-in-haystack）测试中，以最低每个查询 key 数保持 95% 召回率的预设。

2. 将均值池化压缩器替换为一个微小的学习 MLP（2 层，hidden 32）。在一个信号是块均值的合成任务上训练它。测量在留出数据上相对于均值池化基线的困惑度差距。

3. 实现门控 MLP。它以查询为输入，输出三个标量。证明门控的行为是合理的：在随机查询上接近均匀加权，当查询命中一个很靠后的块时，对选择分支赋予高权重。

4. 计算一个启用 NSA 的 70B 模型在 128k 上下文下的 KV 缓存内存预算。KV 头为 8，头维度为 128，BF16。与全注意力和 MLA（Phase 10 · 14 展示了 MLA 的数字）进行比较。找出 NSA 的细粒度分支 KV 缓存等于全注意力的序列长度。

5. 阅读 NSA 论文第 4 节（arXiv:2502.11089），用三句话解释为什么压缩分支的注意力分数被复用于 top-k 选择而非计算独立的路由分数。将答案与梯度流关联起来。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Compressed branch（压缩分支） | "粗视图" | 对块平均 key 的注意力，以每个查询 O(N/l) 个 key 提供全局上下文 |
| Selected branch（选择分支） | "top-k 块" | 对压缩分支分数最高的 `k` 个块的细粒度注意力 |
| Sliding window（滑动窗口） | "局部上下文" | 对最后 `W` 个 token 的注意力，用于短程模式 |
| Native trainability（原生可训练性） | "预训练时开启稀疏" | 稀疏模式在预训练期间学习，而非在推理时附加 |
| Compression block size l（压缩块大小） | "粗视图的分组大小" | 多少 token 合并为一个摘要；典型值 32-64 |
| Top-k | "保留的块数" | 读取其未压缩 token 的压缩块数量；典型值 16 |
| Sliding window W（滑动窗口） | "局部注意力半径" | 通常为 512；更短会损害局部连贯性，更长浪费计算 |
| Branch gate（分支门控） | "如何混合三者" | 每个位置的 MLP 输出，用于加权三个分支的贡献 |
| Hardware alignment（硬件对齐） | "对 kernel 友好的稀疏" | 选择稀疏模式以使实际 GPU kernel 达到理论加速 |
| DSA | "NSA 的继任者" | Deepseek Sparse Attention，在 DeepSeek 谱系中紧随 NSA 的架构 |

## 延伸阅读

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention（arXiv:2502.11089，ACL 2025 最佳论文）](https://arxiv.org/abs/2502.11089) — 原论文
- [DeepSeek-V3 技术报告（arXiv:2412.19437）](https://arxiv.org/abs/2412.19437) — NSA 目标架构家族
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs（arXiv:2502.13189）](https://arxiv.org/abs/2502.13189) — 同期工作，基于块的 MoE 风格注意力
- [Beltagy et al. — Longformer: The Long-Document Transformer（arXiv:2004.05150）](https://arxiv.org/abs/2004.05150) — 滑动窗口的起源
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks（arXiv:2309.17453）](https://arxiv.org/abs/2309.17453) — NSA 改进的推理时稀疏基线
- [Dao et al. — FlashAttention-2（arXiv:2307.08691）](https://arxiv.org/abs/2307.08691) — NSA kernel 在 64k 下超越的全注意力基线
