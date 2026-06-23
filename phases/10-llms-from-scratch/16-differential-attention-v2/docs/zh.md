# 差分注意力（V2）

> Softmax 注意力将少量概率散布在每个不匹配的 token 上。在 10 万个 token 上，这些噪声会累积并淹没信号。差分 Transformer（Differential Transformer，Ye 等，ICLR 2025）通过将注意力计算为两个 softmax 的差值来修复它，减去共享的噪声基底。DIFF V2（Microsoft，2026 年 1 月）是生产栈重写版：解码延迟与基线 Transformer 匹配、无需自定义 CUDA kernel、兼容 FlashAttention。本课是 V1 到 V2 的完整解读，包含一个可用标准库 Python 运行的差分运算玩具实现。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 02（自注意力）、Phase 7 · 15（注意力变体）、Phase 10 · 14（架构走读）
**Time:** ~60 分钟

## Learning Objectives

- 精确陈述为什么 softmax 注意力存在噪声基底（noise floor），以及为什么它随上下文长度增长而增长。
- 推导差分注意力公式，并解释为什么减法在保留信号的同时抵消了共享的噪声分量。
- 走读 V1 到 V2 的差异：什么变快了、什么变简单了、什么变更稳定了，以及为什么每次变更对生产预训练是必要的。
- 在纯 Python 中从零开始实现差分注意力，并在合成的信号加噪声查询上经验性地验证噪声抵消属性。

## 问题

标准 softmax 注意力有一个在大规模下会变成实际麻烦的数学属性。对于查询 `q`，注意力权重是 `softmax(qK^T / sqrt(d))`。Softmax 永远不能产生精确的零——每个不匹配的 token 都获得一些正的质量。这些残差质量就是噪声，且它随上下文长度缩放。在 128k 个 token 时，即使每个不匹配的 token 只获得 0.001% 的概率，其中 127,999 个的总和也会贡献约 12%。模型必须学会绕过这个随上下文增长的噪声基底。

经验上，这表现为注意力头干扰：长上下文 RAG 中的幻觉引用、10 万 token 检索任务中的"中间丢失（lost-in-the-middle）"失败、以及超过 32k 上下文的"大海捞针"基准上的微妙精度退化。差分 Transformer 论文（arXiv:2410.05258，ICLR 2025）测量了差距：DIFF Transformer 在相同大小的基线模型上获得了更低的困惑度、更高的长上下文精度和更少的幻觉。

DIFF V1 有三个问题使其无法进入前沿预训练流水线。其值缓存（value cache）必须在每个解码步骤加载两次，它需要打破 FlashAttention 兼容性的自定义 CUDA kernel，其每头 RMSNorm 在 70B+ 规模上使长程训练不稳定。DIFF V2（Microsoft unilm 博客，2026 年 1 月 20 日）修复了这三个问题。本课走读两个版本，构建差分运算符，并在玩具查询上基准测试噪声抵消。

## 概念

### Softmax 的噪声基底

对于查询 `q` 和键 `K = [k_1, ..., k_N]`，注意力权重是：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有任何 `w_i` 为零。如果 `k_i` 与 `q` 完全不相关，得分 `q . k_i` 不为 0——它在零附近波动，方差为 `||q||^2 / d`。经过 softmax 归一化后，每个不相关的 token 仍然贡献 `O(1/N)` 的加权和。不相关 token 的总贡献是 `O((N-1)/N) = O(1)`——并非小量。

模型真正想要的是某种硬 top-k：在匹配的 token 上高权重，其他地方近零权重。Softmax 太平滑，无法直接做到这一点。

### 差分思路

将每个头的 Q 和 K 投影分成两部分：Q = (Q_1, Q_2) 和 K = (K_1, K_2)。计算两个注意力图：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```
DiffAttn = (A_1 - lambda * A_2) V
```

减法抵消了两个图共享的任何噪声分布。如果两个图在 127k 个不相关 token 上的权重大致均匀（在随机初始化时它们会的），这些就抵消了。信号——在少数实际相关的 token 上的高峰值权重——只有在两个图中以相同幅度出现时才会抵消，而这在模型训练后不会发生。

`lambda` 是每个头的一个可学习标量，参数化为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以是负数。`lambda_init` 默认为一个小正数，如 0.8。

### 为什么这类似于有头噪声抵消

想象两个噪声麦克风录制同一个语音。两者都拾取说话者的声音加上相关的背景噪声。将一个减去另一个，共享噪声被消除。语音存活下来，因为两个信号在相位或幅度上有足够的差异以防止完全抵消。每头的 `lambda` 学习到的正是这种平衡。

### V1 vs V2：差异

V1 将参数数量保持与基线 Transformer 相等。为了得到每个头两个查询，它将头维度减半。这损失了头表达能力，而且——更痛苦的是——将每个头的值缓存减半。解码时每个步骤要加载值缓存两次（每个 softmax 分支一次）。结果：尽管参数数量匹配，解码却慢于基线。

V2 将查询头数量加倍并保持 KV 头不变（从上投影中借用参数）。头维度与基线保持相同。减法之后，额外的维度被投影回来以匹配基线 Transformer 的 O_W 投影。三个效果同时发生：

1. 解码速度与基线匹配（KV 缓存只加载一次）。
2. FlashAttention 无需更改即可运行（无需自定义 kernel）。
3. 解码时的算术强度（arithmetic intensity）提高（从 HBM 每加载一个字节执行更多计算）。

V2 还移除了 V1 用于稳定减法的每头 RMSNorm。在 70B 级别的预训练规模上，该 RMSNorm 会使后期训练不稳定。V2 用一个更简单的初始化方案替代它，无需额外模块即保持训练稳定。

### 何时使用它

| 工作负载 | 收益 |
|----------|---------|
| 长上下文 RAG（64k+） | 更干净的注意力图，更少的幻觉引用 |
| 大海捞针基准 | 超过 32k 时精度大幅提升 |
| 多文档问答 | 更少的跨文档干扰 |
| 8k 代码补全 | 边际收益，不值得架构变更 |
| 短对话（< 4k） | 与基线基本不可区分 |

价值随上下文长度增长。在 4k token 时噪声基底足够小，标准注意力没问题。在 128k 时它在损害你。

### 它与其他 2026 年旋钮的配合

| 特性 | 与 DIFF V2 兼容？ |
|---------|------------------------|
| GQA | 是（V2 增加 Q 头，而非 KV 头） |
| MLA (DeepSeek) | 原则上可行，尚无已发表的组合论文 |
| MoE | 是（注意力独立于 MLP 块） |
| RoPE | 是（不变） |
| YaRN / 长上下文缩放 | 是（这正是 DIFF 帮助最大的地方） |
| FlashAttention | V2 中是（V1 中否） |
| 推测解码 | 是（注意力变化对推测解码循环透明） |

```figure
differential-attention
```

## 构建它

`code/main.py` 在纯 Python 中实现差分注意力。一个具有已知信号加噪声结构的玩具查询让你直接测量噪声抵消比率。

### 步骤 1：标准 softmax 注意力

标准库矩阵运算：列表的列表、手动矩阵乘法、带数值稳定性减最大值的 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 步骤 2：将 Q、K 分成两半

V1 风格：将头维度减半。V2 风格：保持头维度并将头数量加倍。玩具实现使用 V1 以便教学清晰——数学相同，只是记账方式不同。

### 步骤 3：两个 softmax 分支 + 减法

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可以为负。这没问题——值缓存仍然处理有符号的贡献。后续的 V 投影吸收符号。

### 步骤 4：噪声抵消测量

构建一个长度为 1024 的合成序列。将信号 token 放在已知位置，其余填充噪声。计算（a）信号位置上的标准 softmax 注意力权重和（b）差分注意力权重。测量两者中的信噪比。DIFF 注意力可靠地产生高出 3-10 倍的更高信噪比（取决于两个分支被训练得有多大差异）。

### 步骤 5：V1 vs V2 参数核算

给定一个配置（hidden=4096，heads=32，d_head=128），打印：

- 基线 Transformer：Q、K、V 每个大小 `hidden * hidden`，MLP 为 4 * hidden。
- DIFF V1：Q、K 每个大小 `hidden * hidden`，V 大小 `hidden * hidden`（不变），头维度内部减半。添加每头 `lambda` 参数（O(heads * d_head)）。
- DIFF V2：Q 大小 `2 * hidden * hidden`，K 大小 `hidden * hidden`，V 大小 `hidden * hidden`。额外维度在 O_W 之前投影回来。添加相同的 `lambda` 参数。

玩具版测量 V2 的额外参数成本（每个注意力块约 `hidden * hidden` 额外参数）并打印。

## 使用它

截至 2026 年 4 月，DIFF V2 尚未在每个生产推理服务器中发布，但 vLLM 和 SGLang 的集成正在进行中。同时该模式出现在：

- Microsoft 内部的长上下文生产模型中。
- 几个面向 256k+ 上下文的开源模型训练运行中的研究复现。
- 在交替层上组合 DIFF 注意力与滑动窗口注意力的混合架构中。

在 2026 年何时使用它：

- 从零开始训练一个面向 64k+ 有效上下文的新模型。从开始就添加差分注意力；后期重新训练代价高昂。
- 微调一个长上下文模型，其中"中间丢失"失败主导你的评估。对 Q 投影的 LoRA 可以近似 DIFF 结构。

何时不使用：

- 你正在部署一个具有稳定长上下文性能的预训练稠密模型。在现有权重上重新训练的代价很少能回本。
- 你的上下文始终在 16k 以下。噪声基底可忽略不计。

## 交付物

本课产出 `outputs/skill-diff-attention-integrator.md`。给定一个模型架构、目标上下文长度、幻觉画像和训练预算，它生成一个为新的预训练运行或 LoRA 微调添加差分注意力的集成计划。

## 练习

1. 运行 `code/main.py`。验证差分注意力在合成查询上报告的信噪比高于标准 softmax 注意力。改变噪声幅度并展示标准注意力变得不可用的交叉点。

2. 对于一个 7B 级别的模型（hidden=4096，heads=32，d_head=128，32 层），计算从基线到 DIFF V1 以及从基线到 DIFF V2 的参数计数差异。展示哪些组件增加了参数，哪些保持不变。

3. 阅读 DIFF V1 论文（arXiv:2410.05258）的 Section 3 和 DIFF V2 Hugging Face 博客的 Section 2。用两句话解释为什么 V1 的每头 RMSNorm 是必要的，以及为什么 V2 可以在不引起训练发散的情况下移除它。

4. 实现一个消融实验：以 `lambda = 0`（纯第一个 softmax）和 `lambda = 1`（完全减法）计算差分注意力。在合成查询上，测量信噪比在扫描范围内的变化。识别最大化信噪比的 `lambda`。

5. 将玩具扩展到 GQA + DIFF V2。选择 8 个 KV 头和 32 个 Q 头。展示 KV 缓存大小与具有相同（8, 32）配置的基线 GQA 模型匹配。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| 差分注意力 | "两个 softmax 相减" | 将 Q、K 分成两半，计算两个 softmax 图，从第一个中减去第二个（由 lambda 缩放），然后乘以 V |
| 噪声基底 | "softmax 的非零尾部" | Softmax 在每个不相关 token 上施加的 O(1/N) 权重，在长上下文中总和为 O(1) |
| lambda | "减法的缩放系数" | 每个头可学习的标量，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以为负数 |
| DIFF V1 | "ICLR 2025 版本" | 原始差分 Transformer；减半头维度以保持参数数量，需要自定义 kernel，解码更慢 |
| DIFF V2 | "2026 年 1 月的修复" | 将 Q 头加倍同时保持 KV 头；匹配基线解码速度并兼容 FlashAttention |
| 每头 RMSNorm | "V1 的稳定器" | V1 在差分后应用的额外 norm；V2 移除以防止后期训练不稳定 |
| 信噪比 | "多少注意力被浪费了" | 在真实信号位置上的权重与在不相关位置上平均权重的比率 |
| 中间丢失 | "长上下文失败模式" | 检索精度在长上下文中间位置的文档上下降的经验现象——DIFF 注意力减少了此现象 |
| 算术强度 | "每加载一个字节的 FLOP" | V2 通过在每次 KV 加载时加倍查询来提高解码时的比率；对内存受限的解码很重要 |

## 进一步阅读

- [Ye 等 — 差分 Transformer（arXiv:2410.05258，ICLR 2025）](https://arxiv.org/abs/2410.05258) — 原始论文，包含噪声抵消理论和长上下文消融实验
- [Microsoft unilm — 差分 Transformer V2（Hugging Face 博客，2026 年 1 月）](https://huggingface.co/blog/microsoft/diff-attn-v2) — 生产栈重写版，匹配基线解码，FlashAttention 兼容
- [理解差分 Transformer：解锁预训练自注意力（arXiv:2505.16333）](https://arxiv.org/abs/2505.16333) — 关于为什么减法能恢复预训练注意力结构的理论分析
- [共享 DIFF Transformer（arXiv:2501.17900）](https://arxiv.org/html/2501.17900) — 参数共享变体
- [Vaswani 等 — Attention Is All You Need（arXiv:1706.03762）](https://arxiv.org/abs/1706.03762) — DIFF 所减法的基线 Transformer
- [Liu 等 — 中间丢失（arXiv:2307.03172）](https://arxiv.org/abs/2307.03172) — DIFF 注意力所针对的长上下文基准
