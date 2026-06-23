# 推测解码与 EAGLE-3

> Phase 7 · 第 16 课证明了数学原理：Leviathan 拒绝规则精确地保持了验证器（verifier）的分布。本节课是从训练栈视角来看 2026 年生产环境中的推测解码（Speculative Decoding）。EAGLE-3 将草稿模型（draft model）从一个廉价近似变成了一个在验证器自身隐藏状态上训练的小型专用网络，并添加了一个训练时测试循环来对齐其训练和推理分布。结果是：3× 到 6.5× 的端到端加速，聊天场景下每 token 接受率超过 0.9，无损分布。2026 年每个生产推理栈都默认搭载它。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 16（推测解码数学），Phase 10 · 12（推理优化）
**Time:** ~75 分钟

## 学习目标

- 用一句话陈述 Leviathan 定理，并证明推测循环产生的样本与从验证器直接采样完全同分布。
- 走读从 vanilla Spec-Decoding（Leviathan 2023）到 EAGLE、EAGLE-2、EAGLE-3 的两年演进，并说出每一步消除了哪个确切的局限性。
- 从接受率 `α` 和草稿-验证器成本比 `c` 计算预期加速比，并为每种场景选择最优草稿长度 `N`。
- 从零实现完整的推测循环：草稿生成、验证、从残差分布拒绝采样、拒绝时回滚 KV 缓存、完全接受时发出奖励 token（bonus token）。

## 问题

在 H100 上，70B 模型的自回归解码大约每秒生成 35 个 token。GPU 远未饱和。内存带宽是天花板：生成每个 token 时需要从 HBM 加载 70B 权重，做一步算术，产生一个浮点数。计算单元大部分时间空闲。

推测解码将其转化为一个你实际可以解决的吞吐量问题。一个廉价的草稿模型通过 `N` 次小前向传播生成 `N` 个候选 token。验证器在前缀外加所有 `N` 个候选上运行一次。如果验证器在位置 `i` 的分布与草稿一致（以一种我们将精确阐述的统计意义），则接受；否则拒绝，并从残差分布采样一个修正。一次大模型前向传播最多产生 `N+1` 个被接受的 token，而不是一个。

关键定理是 Leviathan、Kalman、Matias（ICML 2023）：输出分布与从验证器直接采样完全相同。不是近似。完全相同。这就是推测解码在生产中可被接受的唯一原因——它是一个纯粹的延迟优化，没有质量折中。

Phase 7 · 第 16 课给了你数学。本节课给你训练栈。一个好的草稿模型比一个廉价草稿多值 2× 加速。EAGLE、EAGLE-2 和 EAGLE-3（Li et al.，2024–2025）将"草稿 = 同一模型的较小版本"转变为一门精确的工程学科。2026 年生产推理服务器默认使用 EAGLE-3。

## 概念

### 不变式：Leviathan 拒绝采样

令 `p(t)` 为给定某前缀下草稿模型对下一个 token 的分布，`q(t)` 为验证器的分布。采样一个草稿 token `d ~ p`。以概率 `min(1, q(d) / p(d))` 接受。如果拒绝，则从残差分布 `(q - p)_+ / ||(q - p)_+||_1` 采样。结果的分布遵循 `q`。无论 `p` 有多差，这一点都成立——p 越差，拒绝越频繁，但输出保持精确。

将 `N` 次这样的调用依次排列，使用一次验证器前向传播处理 `prefix + d_1 + ... + d_N`。验证器同时返回 `q_1, q_2, ..., q_{N+1}`。从左到右依次检查。在第一个拒绝位置 `j` 处，从 `residual(q_j, p_j)` 采样并停止。如果全部接受，从 `q_{N+1}` 采样一个奖励 token。

### 什么决定加速比

令 `α` 为每个草稿 token 的预期接受率。令 `c = cost(draft) / cost(verifier)` 为成本比。每个验证器前向传播中预期接受的 token 数为：

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

每个被接受 token 的预期总耗时（wall time）为 `(N * c + 1) / E[accepted]`。相对于 `N` 最小化该值即可得到甜点。对于 `α = 0.8, c = 0.05`：最优 `N` 约为 5–7，加速比为 3.2×。对于 `α = 0.95, c = 0.02`：最优 `N` 约为 8–10，加速比可达 5×。

最大的杠杆是 `α`。在固定 `N = 5` 的情况下，将 `α` 从 0.6（vanilla 草稿）提升到 0.9（EAGLE-3），每个验证器前向传播的预期接受 token 数从 2.2 提升到 4.1。相同验证器下近 2× 的吞吐量提升。

### 两年演进

**Vanilla 推测（Leviathan，2023）。** 草稿模型是同一家族中独立训练的较小 LLM。易于接入，`α ≈ 0.6`，加速比最多约 2×。

**EAGLE-1（Li et al.，2024）。** 草稿是一个微型 transformer——通常一到两层——以验证器最后一层的隐藏状态作为输入，直接预测下一个 token。因为草稿能看到验证器的特征表示，其分布更接近验证器。`α` 提升到 0.7–0.8。

**EAGLE-2（Li et al.，2024）。** 增加动态草稿树：不是提出单一 `N` 个 token 序列，而是提出一个小型候选树，用验证器在一次前向传播中评分每个节点（使用 tree attention（树注意力）），并沿着最高概率路径行走。草稿长度每一步自适应。每条被接受路径的 `α` 超过 0.85。

**EAGLE-3（Li et al.，2025，NeurIPS）。** 两个额外变化。第一，完全放弃特征预测损失——EAGLE-1/2 训练草稿去匹配验证器的隐藏状态，这限制了数据能带来的提升上限。EAGLE-3 直接在 token 预测上训练。第二，训练时测试（Training-Time Test，TTT）：在草稿训练期间，多步地将草稿自身的先前预测反馈作为输入，与推理时的运行方式一致。这对齐了训练和测试分布，阻止了误差累积。实测加速比：聊天场景高达 6.5×，在 H100 上使用 SGLang 以 batch 64 运行时吞吐量提升 38%。

### KV 缓存回滚

验证在一次前向传播中将验证器的 KV 缓存扩展 `N` 个条目。如果在位置 `j` 发生拒绝，缓存中位置 `j-1` 之后的内容现在是错误的。两种常见实现：写入临时缓冲区并在接受时提交（vLLM、TensorRT-LLM），或者维护一个物理 KV 缓存加逻辑长度，在拒绝时截断。无论哪种方式，回滚成本是每层每头字节级别，相对于前向传播成本可以忽略不计。

对于 EAGLE-2 的树搜索，验证器运行注意力时使用遵循树拓扑的非因果 mask。工程上比较繁琐，但计算上是标准的 FlashAttention 调用（带自定义 mask）。

### 2026 年的草稿架构

| 策略 | 草稿类型 | `α` | 加速比 | 训练成本 |
|----------|-----------|-----|---------|---------------|
| Vanilla | 独立小 LLM | 0.55-0.70 | 1.8-2.3× | 无（复用现有小模型） |
| Medusa | 验证器上的额外 LM 头 | 0.65-0.75 | 2-3× | ~1B SFT token |
| EAGLE-1 | 基于隐藏状态的单层 transformer | 0.70-0.80 | 2.5-3× | ~60B token |
| EAGLE-2 | EAGLE-1 + 动态草稿树 | 0.80-0.88 | 3-4× | ~60B token |
| EAGLE-3 | 多层特征融合 + TTT | 0.88-0.92 | 3.5-6.5× | ~60-200B token |
| Lookahead | 无草稿（Jacobi 迭代） | N/A | 1.3-1.6× | 无 |

2026 年生产环境：vLLM 和 SGLang 在可用时默认使用 EAGLE-3，否则使用 EAGLE-2。TensorRT-LLM 对 Meta 和 NVIDIA 公开模型有最快的 Medusa 路径。llama.cpp 为 CPU 部署搭载 vanilla 草稿。

## 构建

见 `code/main.py`。这是完整的 Leviathan 推测循环，包含所有部分：生成 N 个草稿、验证器并行传播、逐位置拒绝、残差采样、奖励 token、KV 回滚，以及经验性验证输出分布是否与从 `q` 直接采样一致。

### 步骤 1：拒绝规则

```python
def accept(q_prob, p_prob, u):
    # 如果草稿概率为零则接受；否则按 min(1, q/p) 做 Bernoulli 试验
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### 步骤 2：残差分布

```python
def residual(q, p):
    # 计算 (q - p) 的正部并重新归一化
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### 步骤 3：完整推测步骤

`spec_step` 函数从 `p` 中生成 `N` 个草稿 token，然后在一次并行 `q` 评估中验证所有草稿。对每个草稿 token 应用拒绝规则，在第一次拒绝时从残差分布采样修正。如果全部接受，则从 `q_{N+1}` 发出一个奖励 token。

### 步骤 4：KV 回滚簿记

模拟器跟踪每个 worker 的逻辑 `kv_length`。当接受了 `k` 个草稿时，`kv_length += k`。当在位置 `j` 拒绝时，缓存已经写入到 `j` 之后，但逻辑长度被设为 `prefix_length + j + 1`——即修正 token 之后的位置。后续读取会截断到逻辑长度。

### 步骤 5：Leviathan 检验

运行 50,000 步推测。统计被接受 token 的经验分布。与 50,000 次从 `q` 直接采样对比。卡方统计量应远低于临界值。该定理在实践中成立。

### 步骤 6：加速比与 α 的关系

通过以不同幅度扰动使 `p` 偏离 `q` 来扫描草稿质量。测量 `α`，然后绘制每个验证器调用预期生成的 token 数与 `α` 和 `N` 的函数关系。代码打印表格展示 EAGLE-3 级别的草稿质量（`α ≈ 0.9`）如何解锁每次验证器调用 4–5 个 token。

## 使用

使用 EAGLE-3 的生产级 `vllm serve`：

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

在 H100 上使用 SGLang 的 EAGLE-3（batch 64）：吞吐量比 batch-64 vanilla 解码高约 1.38×，来自 EAGLE-3 论文数据。

何时选择推测解码：

- 任何 P50 延迟比峰值吞吐量更重要的交互式聊天工作负载。
- 代码生成和结构化输出（JSON、SQL）。`α` 超过 0.9，因为目标分布高度可预测。
- 生成长文本（数千 token）。均摊加速持续有效。

何时不选：

- 非常小的模型（< 3B）。草稿模型并不比验证器便宜多少。
- 极小的 batch-1 CPU 部署。草稿模型的内存开销可能不值得。
- 极高温度的创意采样，`α` 会崩溃。

## 产出

本节课产出 `outputs/skill-eagle3-tuner.md`。给定一个推理工作负载（模型、batch 大小、目标延迟、任务画像），它推荐推测解码策略和调优参数（草稿家族、`N`、树深度、温度感知切换）。

## 练习

1. 运行 `code/main.py`。确认在 50,000 个样本上，Leviathan 分布检验的卡方统计量保持在 95% 临界值以下。

2. 在固定 `α = 0.9` 和 `c = 0.04` 的情况下，将 `N` 从 1 扫到 10。绘制每个验证器调用预期生成的 token 数和实际每 token 耗时。找出最小化耗时的 `N`。解释曲线的形状。

3. 修改代码模拟 EAGLE-2 树搜索：每一步草稿提出形状为 `[2, 2, 2]` 的树（八条候选路径）。验证器运行一次，最高概率的接受路径获胜。计算每个叶子节点的 `α` 和每个验证器调用的总 token 数。与等量计算下的线性链推测解码进行比较。

4. 为两个并发序列实现批量 KV 回滚模拟器。序列 A 所有草稿被接受；序列 B 在位置 2 拒绝。证明每个序列的 `kv_length` 被正确更新，且没有工作被浪费。

5. 阅读 EAGLE-3 论文第 4 节（训练时测试）。用两句话解释为什么没有 TTT 的朴素草稿训练会遭受暴露偏差（exposure bias），以及为什么在训练期间将草稿自身的预测反馈作为输入可以解决这个问题。将其与 seq2seq 中的 scheduled-sampling 文献联系起来。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Leviathan 规则 | "min(1, q/p)" | Bernoulli 接受/拒绝，概率为 `min(1, q(d)/p(d))`；当拒绝时从残差分布采样，精确保持验证器分布 |
| Residual distribution（残差分布） | "(q 减 p) 的正部，归一化" | `(q - p)_+` 在零处截断并重新归一化——拒绝时正确采样的分布 |
| Acceptance rate α（接受率） | "草稿正确的频率" | 拒绝规则下每 token 的预期 Bernoulli 成功概率；支配所有加速比计算 |
| EAGLE-1 | "隐藏状态草稿" | 以验证器最后一层隐藏状态为条件的微型 transformer 草稿（Li et al.，2024） |
| EAGLE-2 | "动态草稿树" | EAGLE-1 加上在一次验证器传播中使用树注意力评分的候选延续树 |
| EAGLE-3 | "训练时测试" | 放弃特征预测损失，直接训练 token 预测，训练期间将草稿自身的输出反馈作为输入 |
| Training-time test, TTT（训练时测试） | "暴露偏差修复" | 训练期间让草稿自回归运行，使训练和测试输入分布对齐——scheduled-sampling 的直接类比 |
| KV rollback（KV 回滚） | "撤销被拒绝的草稿" | 在拒绝后将验证器 KV 缓存重置到接受前缀长度的簿记 |
| Bonus token（奖励 token） | "免费的那一个" | 当所有 `N` 个草稿都被接受时，以零额外验证器成本从 `q_{N+1}` 采样一个额外的 token |
| Tree attention（树注意力） | "一次验证多个候选" | 使用遵循草稿树拓扑的非因果 mask 的注意力；在一次前向传播中计算树中每个节点的 `q_i` |

## 延伸阅读

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding（arXiv:2211.17192，ICML 2023）](https://arxiv.org/abs/2211.17192) — 奠基性论文及等价定理
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling（arXiv:2302.01318）](https://arxiv.org/abs/2302.01318) — 同期独立提出，附有简洁证明
- [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty（arXiv:2401.15077）](https://arxiv.org/abs/2401.15077) — EAGLE-1，隐藏状态条件草稿
- [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees（arXiv:2406.16858）](https://arxiv.org/abs/2406.16858) — 动态树搜索
- [Li et al. — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test（arXiv:2503.01840，NeurIPS 2025）](https://arxiv.org/abs/2503.01840) — 2026 年生产环境默认方案
- [Cai et al. — Medusa: Multiple Decoding Heads（arXiv:2401.10774）](https://arxiv.org/abs/2401.10774) — 替代性无草稿方案
- [vLLM 推测解码文档](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 所有策略均已接入的权威生产环境参考
