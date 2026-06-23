# 梯度检查点与激活重计算

> 反向传播会保留每个中间激活值。在 70B 参数和 128K 上下文下，每个 rank 的激活值达 3 TB。检查点（checkpointing）用 FLOPs 换内存：重计算而不是保存。问题在于丢弃哪些段（segment），答案不是"全部"。

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## 问题

训练一个 transformer 时，对每一层都要存储在反向传播中被求导的每个操作的输入：注意力输入、Q/K/V 投影、softmax 输出、FFN 输入、归一化输出和残差流。对于隐藏大小 `d`、序列长度 `L`、批次 `B`，每层大约是 `12 * B * L * d` 个浮点数。

对于 `d=8192, L=8192, B=1`，BF16 下每层 800 MB。一个 64 层的模型是 51 GB 的激活值——这还没有乘以微批次大小（microbatch size），还没加上注意力-softmax 中间值（每头 `L^2`），也还没有计入张量并行（tensor-parallel）的部分拷贝。

两端夹击的账单：BF16 权重加优化器状态可能装入 80GB，但激活值溢出。梯度检查点（gradient checkpointing，也叫激活重计算 activation recomputation）是标准解决方案。丢弃大部分激活值；在反向传播期间重做前向传播来恢复它们。代价：额外的 FLOPs。收益：内存按检查点段与总层数的比例下降。

朴素地使用检查点，每步大约额外消耗 33% 的前向传播 FLOPs。做得好——按 Korthikanti 等人提出的"智能选择"（smart selection）进行选择性检查点（selective checkpointing）——你可以节省 5 倍内存，而 FLOPs 开销不到 5%。结合 FP8 矩阵乘法、FSDP offload 和专家并行 MoE，这确实很重要：你既承受不起额外的内存，也承受不起浪费的计算。

## 概念

### 反向传播到底需要什么

`output = layer(input)`。反向传播需要 `grad_input` 和 `grad_params`。要计算它们，需要：

- `input`（用于计算线性层的 `grad_params = input.T @ grad_output`）
- 一些激活导数的中间值（ReLU/GELU/softmax 的导数依赖于激活值本身）

前向传播将这些自动存储于自动求导图（autograd graph）中。每个 `tensor.retain_grad()` 和每个需要其输入的 op 都会保留一个引用。

### 朴素的全检查点（Full Checkpointing）

将网络分成 N 个段。前向传播期间，只存储每个段的*输入*。当反向传播需要中间值时，重新运行该段的前向传播来物化（materialize）它们，然后求导。

示例：32 层 transformer 分成 32 个段，每段 1 层。

- 内存：32 个层输入（小） vs 32 *（每层激活量）（巨大）。
- 额外计算：每段 1 次额外的前向传播，即总计约 33% 更多的前向 FLOPs（因为反向传播是前向传播的 2 倍，完整步骤变为 1 + 1 + 2 = 4 个单位，而非 1 + 2 = 3）。

这就是 Chen et al. 2016 的原始配方：每 sqrt(L) 层设一个检查点，以平衡内存和计算。对于 L=64，即 8 个检查点。

### 选择性检查点（Selective Checkpointing，Korthikanti 2022）

并非所有激活值的成本相同。注意力 softmax 输出是 `B*L*L*heads`，随序列长度*二次*增长。FFN 隐藏激活值是 `B*L*4d`，线性增长。对于长序列，softmax 占据主导地位。

选择性检查点保留存储成本低廉的激活值（线性投影、残差），仅重计算昂贵的部分（注意力）。你只需付出极少的 FLOPs 用于重计算，但节省了 O(L^2) 的内存。

Megatron-Core 将其实现为"选择性"激活重计算。大多数 2024+ 前沿训练运行都在使用。

### 卸载（Offload）

重计算的替代方案：在前向传播和反向传播之间将激活值发送到 CPU RAM。需要 PCIe 带宽；当空闲带宽超过重物化的成本时是有益的。混合策略很常见：一些层做检查点，另一些做卸载。

FSDP2 将卸载作为一等选项提供。当 GPU 内存瓶颈但 CPU-GPU 传输有余量时，卸载比较有用。

### 重计算成本模型

朴素检查点，每 k 层一个检查点（共 L 层）的逐步 FLOPs：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # 在段中对每层做一次额���的前向传播
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

使用选择性检查点时，你只重计算注意力内核，而非整个层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活量：`A`。对于 L 层，总激活内存：`L * A`。

完整检查点（段大小 1）：只存储 `L * input_volume`（对于标准 transformer 约 `L * 1/10 A`）。节省约 `9 * L * A * 1/10`。

每 k 层一个检查点：存储 `L/k * A` 加上活跃段内的 `k-1` 层激活值。

在 `k = sqrt(L)` 时，内存和重计算成本都随 `sqrt(L)` 缩放——均匀成本层的最优折衷。

### 何时不做检查点

- 已经处于运行中的流水线阶段的最内层。它们无论如何都必须完成。
- 第一层和最后一层如果它们在阶段计算中占主导地位的话（在 transformer 中很少见）。
- 已经使用 FlashAttention 的注意力内核——Flash 已经快速重计算了 softmax，因此额外的层级检查点收效甚微。

### 实现模式

1. **函数包装器：** 将段包装在 `torch.utils.checkpoint.checkpoint(fn, input)` 中。PyTorch 仅存储 `input`，在反向传播时重计算其他一切。

2. **基于装饰器：** 将层标记为可检查点（checkpointable）；训练器在配置时决定哪些段被包装。

3. **手动显式重计算：** 自己写反向传播，调用自定义的 `recompute_forward`，用存储的输入复制前向传播。

三者产生相同的功能结果。包装器是标准惯用法。

### 与 TP / PP / FP8 的交互

- **张量并行（TP）：** 检查点输入在重计算时必须重新 gather 或 rescatter；需处理通信成本。
- **流水线并行（PP）：** 典型模式是对每个流水线阶段的前向传播做检查点，使得反向序列的微批次可以复用激活内存。
- **FP8 重计算：** 在重计算期间更新的 amax 历史必须与原始前向传播匹配，否则 FP8 的 scale 会漂移。大多数框架会快照 scale。

## 动手构建（Build It）

### 步骤 1：一个带段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 步骤 2：需要所有激活值的朴素反向传播

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 步骤 3：每 k 层检查点的内存

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 步骤 4：成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 步骤 5：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 步骤 6：最优段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 步骤 7：选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 实际应用（Use It）

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint`——PyTorch 中的规范包装器。包装一个函数；仅存储输入，反向传播时重计算。
- **Megatron-Core 激活重计算**：支持 `selective`、`full` 和 `block` 模式。2024+ 前沿训练的标准选择。
- **FSDP2 offload**：`module.to_empty(device="cpu")` 配合 FSDP2 中的 `offload_policy`，将激活值分片到 CPU 而非重计算。
- **DeepSpeed ZeRO-Offload**：用于优化器状态和激活值的 CPU 卸载，与检查点互补。

## 交付产物（Ship It）

本节课生成 `outputs/prompt-activation-recompute-policy.md`——一个接受模型配置（层数、隐藏大小、序列长度、批次大小）和可用 GPU 内存作为输入，输出逐层重计算策略（none / selective / full / offload）的提示词。

## 练习

1. 验证正确性。运行 `model_forward` + `model_backward`（完整激活值）与 `model_forward_checkpointed` + `model_backward_checkpointed`（分段）的对比。参数梯度必须达到机器精度一致。

2. 将段大小 `k` 从 1 扫描到 `L`。绘制 FLOP 开销和内存的曲线。找到曲线的拐点。

3. 实现选择性检查点：存储注意力模块输入但不存储其���间值。对于 seq=8192 的 32 层模型，测量 FLOP 开销并与全层检查点对比。

4. 添加卸载功能。将段输入保存到模拟的"CPU 缓冲区"（一个单独的列表）。测量"PCIe 带宽"为字节/时间，并找到卸载和重计算之间的盈亏平衡点。

5. 在真实 PyTorch transformer 上 benchmark，分别测量带和不带 `torch.utils.checkpoint` 的内存（通过 `torch.cuda.max_memory_allocated`）和步骤时间。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| Gradient checkpointing | "通过重做前向传播节省内存" | 仅存储段输入；在反向传播期间重计算中间值以获取梯度所需的张量 |
| Activation recomputation | "与 checkpointing 相同" | 同一技术的 HPC 风格名称 |
| Segment size (k) | "每检查点多少层" | 中间值被一起丢弃和重物化的层数 |
| Selective checkpointing | "Korthikanti 的诀窍" | 仅重计算存储成本高的激活值（注意力 softmax）；保留便宜的部分 |
| Full checkpointing | "朴素版本" | 在每个段中重计算每层的中间值 |
| Block checkpointing | "粗粒度版本" | 整个 transformer 块做检查点；最大粒度 |
| FLOP overhead | "计算税" | 每步额外 FLOPs = (重计算 FLOPs) / (fwd + bwd FLOPs)；朴素方式 33%，选择性方式 5% |
| Activation offload | "发送到 CPU" | 在前向到反向之间将激活值移动到 CPU RAM；重计算的替代方案 |
| sqrt-L rule | "经典最优方案" | 对于均匀成本层，最优检查点间距为 sqrt(L) 层 |
| Attention-softmax volume | "O(L^2) 问题" | L^2 * heads * batch 个浮点数；在长上下文下主导激活内存 |

## 进一步阅读

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 形式化梯度检查点的原始论文
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- 选择性激活重计算及形式化成本分析
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- 通过反向模式重物化的替代恒定内存方法
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- 规模化的激活卸载
- [PyTorch torch.utils.checkpoint 文档](https://pytorch.org/docs/stable/checkpoint.html) -- 标准 API
- [Megatron-Core 激活重计算文档](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- selective、full 和 block 模式
