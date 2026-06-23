# 推测解码（Speculative Decoding）与 EAGLE

> 一个前沿大语言模型（LLM）生成一个 token 需要对数十亿参数执行一次完整的前向传播（forward pass）。这个前向传播严重过度配置：大多数时候，一个小得多的模型就能正确猜测接下来的 3-5 个 token，而大模型只需要*验证*这个猜测。当猜测正确时，你以一次前向传播的代价获得了 5 个 token。推测解码（Leviathan et al. 2023）精确地实现了这一点，EAGLE-3（2025）将接受率（acceptance rate）推高到每次验证约 4.5 个 token——在输出分布完全匹配的前提下实现 4-5 倍加速。

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10 Lesson 12 (Inference Optimization), Phase 10 Lesson 04 (Pre-training Mini-GPT)
**Time:** ~75 minutes

## 问题

在 H100 上，一个 70B 级别模型的解码吞吐量（decode throughput）通常是每秒 40-80 个 token。每个 token 需要一次从 HBM（高带宽内存）读取所有模型权重的完整前向传播。你不能在不改变其输出的情况下缩小模型，也不能在超出内存的情况下继续增大批次大小（batch size）。你陷入了困境——除非你能让模型每次前向传播输出多个 token。

自回归生成（autoregressive generation）看起来本质上是串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但这里存在一个并发机会。如果你有一个廉价的预测器（predictor）告诉你"接下来的 4 个 token 很可能是 [a, b, c, d]"，你可以在**大模型的一次前向传播中**验证所有 5 个位置，并接受最长的匹配前缀。

Leviathan、Kalai、Matias（2023，"Fast Inference from Transformers via Speculative Decoding"）通过一个巧妙的接受/拒绝规则（accept/reject rule）精确地实现了这一点，该规则保持目标模型（target model）的采样分布不变。相同的输出分布，2-4 倍更快。

## 概念

### 双模型架构

- **目标模型（Target model）** `M_p`：大型、慢速、高质量，你实际想从中采样的模型。分布为：`p(x)`。
- **草稿模型（Draft model）** `M_q`：小型、快速、低质量模型。分布为：`q(x)`。体积小 5-30 倍。

每步流程：

1. 草稿模型自回归地提议 K 个 token：`x_1, x_2, ..., x_K ~ q`。
2. 目标模型对所有 K+1 个位置并行执行**一次**前向传播，为每个提议的 token 生成 `p(x_k)`。
3. 按从左到右的顺序，使用下面的修正拒绝采样规则接受/拒绝每个 token。接受最长的匹配前缀。
4. 如果某个 token 被拒绝，从修正分布中采样替代 token 并停止。否则从 `p(· | x_1...x_K)` 中采样一个额外的 bonus token。

如果草稿模型与目标模型完美匹配，你每次目标前向传播可以获得 K+1 个 token。如果草稿在第一个位置就错了，你只能获得 1 个 token。

### 精确性规则（Exactness Rule）

推测解码**在分布上被证明等价于从 p 采样**。拒绝规则：

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示逐点差的正部（positive part）。当草稿与目标一致（`p ≈ q`）时，接受率接近 1。当它们不一致时，残差分布（residual distribution）被构造为使整体样本仍然精确等于 `p`。

**贪心情况（Greedy case）。** 对于 temperature=0 的采样，只需检查 `argmax(p) == x_t`。如果是，接受；如果否，输出 `argmax(p)` 并停止。

### 期望加速比（Expected Speedup）

如果草稿模型的 token 级接受率（acceptance rate）为 `α`，每次目标前向传播期望产出的 token 数为：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = 草稿长度（draft length）, α 在 [0, 1] 区间
```

在 `α = 0.8, K = 4` 时：`(1 - 0.8^5)/(1 - 0.8) = 3.36` 个每次前向传播产生的 token。一次目标前向传播的代价大致为 `cost_q * K + cost_p`（K 次草稿步骤加一次目标验证）。如果 `cost_p >> cost_q * K`，加速比为 `3.36× / 1 = 3.36×` 的吞吐量提升。

唯一真正的参数是 `α`，它完全取决于草稿与目标的匹配程度。一个好的草稿模型决定一切。

### 训练草稿模型：蒸馏（Distillation）

一个随机的小模型会是一个糟糕的草稿。标准方法是蒸馏（distill）：

1. 选择一个小的架构（对 70B 目标约 1B，对 7B 目标约 500M）。
2. 在一个大型文本语料库上运行目标模型，存储其下一个 token 的分布。
3. 使用与目标分布（而非真实 token）之间的 KL 散度（KL divergence）来训练草稿模型。

结果：`α` 在编程任务上通常为 0.6-0.8，在自然语言对话上为 0.7-0.85。生产环境加速 2-3 倍。

### EAGLE：树形草稿（Tree Drafting）+ 特征复用（Feature Reuse）

Li、Wei、Zhang、Zhang（2024，"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"）观察到标准推测解码中的两个低效问题：

1. 草稿模型执行 K 个串行步骤，每一步都是完整计算。但草稿可以复用目标模型在最近一次验证中的特征（隐藏状态，hidden states）——目标模型已经计算出了丰富的表示（representations），而草稿却在从零重新推导它们。
2. 草稿输出一条线性链（linear chain）。如果草稿能输出一个候选*树*（tree of candidates，每个节点有多个猜测），目标模型的一次前向传播可以通过树形注意力掩码（tree attention mask）并行验证多条候选路径，并选择最长的被接受分支。

EAGLE-1 的改动：
- 草稿输入 = 目标模型在位置 t 的最终隐藏状态，而非原始 token。
- 草稿架构 = 1 个 transformer 解码器层（而不是一个独立的小模型）。
- 输出 = 每深度 K = 4-8 个候选，深度 4-6 的候选树。

EAGLE-2（2024）增加了动态树拓扑（dynamic tree topology）：树在草稿不确定的地方加宽，在自信的地方保持窄。在不增加验证成本的前提下提升 `α_effective`。

EAGLE-3（Li et al. 2025，"EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"）移除了固定的顶层特征依赖（top-layer feature dependency），并使用一种新的"测试时模拟损失"（test-time simulation loss）来训练草稿——草稿在训练时匹配目标模型的测试时分布（test-time distribution），而非教师强制分布（teacher-forced training distribution）。接受率从 0.75（EAGLE-2）提升到 0.82（EAGLE-3），平均每次验证的 token 数从 3.0 提升到 4.5。

### 树形注意力验证（Tree Attention Verification）

当草稿输出一棵树时，目标模型使用**树形注意力掩码**（tree attention mask）在一次前向传播中验证它——这是一个编码树拓扑（tree topology）而非纯线性链的因果掩码（causal mask）。每个 token 只关注其在树中的祖先节点。验证过程仍然是一次前向传播、一次矩阵乘法；拓扑掩码只增加了少量额外的 KV 条目。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是竞争性的第一个 token 候选，`c, d, e, f` 是第二个 token 候选，所有六个位置在一次前向传播中验证。输出是所有被接受路径中最长的前缀。

### 何时有效，何时无效

**有效场景：**
- 可预测文本（代码、普通英语、结构化输出）的对话/补全。`α` 较高。
- 解码过程中 GPU 计算资源空闲的场景（内存瓶颈阶段）。树形草稿可以利用可用的 FLOPs。

**无效/无加速场景：**
- 高度随机的输出（高 temperature 的创意写作）。`α` 降至接近 `1/|vocab|`。
- 极高并发的批量服务（Batch serving）——批处理已经填满了 FLOPs，树形验证的空间很小。
- 非常小的目标模型，草稿模型相较于目标模型优势不明显。

生产环境通常报告聊天场景 2-3 倍 wall-clock 加速，代码生成 3-5 倍加速，创意写作几乎零加速。

```figure
speculative-decoding
```

## 动手构建（Build It）

`code/main.py`：

- 实现一个参考函数 `speculative_decode(target, draft, prompt, K, temperature)`，实现精确的拒绝规则，并验证它保留了目标模型的分布（经验 KL 散度 < 0.01，与普通目标采样对比）。
- 一个 EAGLE 风格的树形草稿器（tree drafter），构建深度为 K、带有 top-p 分支的树。
- 一个树形注意力掩码构建器，为验证器生成正确的因果模式。
- 一个接受率测试工具（acceptance-rate harness），在一个小型语言模型上运行（将一个 GPT-2-small 从 GPT-2-medium 目标中蒸馏出来）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. 草稿 K 个 token
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. 目标模型在每个草稿位置 + 1 个额外位置计算 p
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. 从左到右接受/拒绝
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. 所有 K 个 token 都被接受 → 从目标模型采样 bonus token
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 实际应用（Use It）

- **vLLM** 和 **SGLang** 提供一流的推测解码支持。标志位：`--speculative_model`、`--num_speculative_tokens`。通过 `--spec_decoding_algorithm eagle` 标志支持 EAGLE-2/3。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE 树。
- **参考草稿模型**：`Qwen/Qwen3-0.6B-spec`（为 Qwen3-32B 生成草稿）、`meta-llama/Llama-3.2-1B-Instruct-spec`（为 70B 生成草稿）。
- **Medusa 头**（Cai et al. 2024，"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"）：向目标模型本身添加 K 个并行预测头，而非使用独立的草稿模型。部署更简单，接收率略低于 EAGLE。

## 交付产物（Ship It）

本节课生成 `outputs/skill-speculative-tuning.md`——一个分析目标模型工作负载并选择以下参数的技能：草稿模型、K（草稿长度）、树宽度（tree width）、temperature，以及何时回退到普通解码。

## 练习

1. 实现精确的拒绝规则并经验验证。通过 `speculative_decode` 和普通目标采样各运行 10K 次采样；计算两种输出分布之间的 TV 距离（Total Variation distance）。应小于 0.01。

2. 计算加速公式。给定固定的 `α` 和 `K`，绘制每次目标前向传播的期望 token 数。找出 α ∈ {0.5, 0.7, 0.9} 的最优 K。

3. 训练一个小型草稿模型。以 124M 的 GPT-2 为目标，用 KL 损失在 1 亿 token 上蒸馏一个 30M 的 GPT-2 草稿模型。在留出文本上测量 `α`。期望值：0.6-0.7。

4. 实现 EAGLE 风格的树形草稿。让草稿在每个深度输出 top-3 分支，而非一条线性链。构建树形注意力掩码。验证目标模型接受最长正确分支。

5. 测量失败模式。在 temperature=1.5（高随机性）下运行推测解码。展示 α 急剧下降，且由于草稿开销，算法比普通解码更慢。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| Target model | "大模型" | 慢速、高质量、你实际想采样的模型（p 分布） |
| Draft model | "猜测器（speculator）" | 小型、快速的预测器（q 分布）；小 5-30 倍 |
| K / draft length | "前瞻（look-ahead）" | 每次验证过程中推测的 token 数量 |
| α / acceptance rate | "命中率（hit rate）" | 草稿提议被接受的逐 token 概率 |
| Exact rejection rule | "接受测试（accept test）" | r < p/q 的比较，保持目标模型的分布不变 |
| Residual distribution | "修正后的 p-q" | (p - q)+ / \|\|(p - q)+\|\|_1，拒绝时采样的分布 |
| Tree drafting | "分支推测（branching speculation）" | 草稿输出候选树，使用树形注意力掩码在一次前向传播中验证 |
| Tree attention mask | "拓扑掩码（topological mask）" | 编码树拓扑的因果掩码，使每个节点只关注其祖先节点 |
| Medusa heads | "并行头（parallel heads）" | 在目标模型本身上添加 K 个额外的预测头；无需独立草稿模型 |
| EAGLE feature reuse | "隐藏状态草稿（hidden-state draft）" | 草稿输入是目标模型的最后一个隐藏状态，而非原始 token，从而缩小草稿模型 |
| Test-time simulation loss | "EAGLE-3 训练方法" | 训练草稿时匹配目标模型的测试时分布，而非教师强制分布 |

## 进一步阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 精确拒绝规则和理论加速分析
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind 的同期推测采样论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — 替代草稿模型的并行头方法
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — 特征复用和树形草稿
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — 动态树拓扑
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — 训练时匹配测试时分布
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/lookahead 解码，一种无需推测器的替代方案
