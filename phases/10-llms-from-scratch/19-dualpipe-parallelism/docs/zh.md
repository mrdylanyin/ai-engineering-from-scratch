# DualPipe 并行

> DeepSeek-V3 在 2,048 块 H800 GPU 上训练，MoE 专家分散在各个节点上。跨节点专家 all-to-all 通信每 1 GPU-小时计算就要消耗 1 GPU-小时通信。GPU 有一半时间处于空闲状态。DualPipe（DeepSeek，2024 年 12 月）是一种双向流水线，将前向和后向计算与它们触发的 all-to-all 通信重叠在一起。气泡（bubble）减少，吞吐量上升，而维护两份模型参数副本（"dual" 的命名来源）的成本很低——因为 Expert Parallelism（专家并行）本来就已经将专家分散到各个 rank 上了。本节课是一次 Learn 类型的走读，讲解 DualPipe 实际做了什么，以及为什么 Sea AI Lab 的 DualPipeV 改进版放弃了 2x 参数成本，代价是气泡略微增大。

**Type:** Learn
**Languages:** Python (stdlib, schedule simulator)
**Prerequisites:** Phase 10 · 05（分布式训练、FSDP、DeepSpeed），Phase 10 · 14（开源模型架构与 MoE）
**Time:** ~60 分钟

## 学习目标

- 说出 DualPipe 前向-后向块（chunk）的四个组成部分，以及每个部分为什么有自己的重叠窗口。
- 解释大规模场景下的流水线气泡（pipeline bubble）问题，以及"无气泡（bubble-free）"在实际中与宣传中的含义区别。
- 手动推导一个 8 PP rank、16 个 micro-batch 的 DualPipe 调度表，确认正向和反向流互相填充对方的空闲槽位。
- 陈述 DualPipeV（Sea AI Lab，2025）所做的权衡：放弃 2x 参数复制，代价是在 Expert Parallelism 不活跃时气泡略微增大。

## 问题

在 2k H800 GPU 上训练一个 671B 的 MoE 模型会遇到三个叠加的瓶颈：

1. **内存压力。** 每个 GPU 保存模型的一个切片。在 61 层、128 个头、序列长度 8k 的情况下，激活内存非常庞大。
2. **流水线气泡。** 传统的流水线并行（GPipe、1F1B）会使 GPU 在等待其阶段的输入或梯度时空闲。在 8 个阶段下，即使使用 1F1B 调度，大约 12% 的 GPU 时间仍可能是气泡。
3. **跨节点 all-to-all。** 使用 Expert Parallelism 的 MoE 将专家分散到节点上。每次前向传播触发一次 all-to-all 来将 token 分发到其对应的专家，再触发另一次来合并结果。在 2k GPU 规模下，这很容易达到 1:1 的计算-通信比。

以上每个问题都有独立的解决方案：梯度检查点（gradient checkpointing）解决内存问题，Zero Bubble（Sea AI Lab，2023）解决流水线气泡问题，专家并行通信 kernel 解决 all-to-all 问题。DualPipe 所做的就是让它们协同工作。该调度方案在单个前向-后向块内重叠计算和通信，从流水线两端同时注入 micro-batch，并利用由此产生的调度模式将 all-to-all 隐藏在计算窗口内。

报告结果：接近消除流水线气泡，在 DeepSeek-V3 的 14.8T token 训练中 GPU 利用率超过 95%。

## 概念

### 流水线并行回顾

将一个 N 层模型拆分到 P 个设备上。设备 `i` 持有第 `i * N/P` 到 `(i+1) * N/P - 1` 层。一个 micro-batch 从设备 0 到 P-1 正向流动，然后从 P-1 到 0 反向流动。每个设备只有在前一个设备发送其输出后才能开始其前向阶段，并且只有在下游设备发送上游梯度后才能开始反向传播。

GPipe（Huang et al., 2019）一次调度一个 micro-batch，浪费了大量 GPU 时间。1F1B（Narayanan et al., 2021）将多个 micro-batch 的前向和反向传播交错调度。Zero Bubble（Qi et al., 2023）将反向传播分为两部分——输入的梯度（B）和权重的梯度（W）——并调度它们来填充气泡。在 Zero Bubble 之后，流水线已经几乎紧凑。

DualPipe 是下一步。它在此基础上增加了两个想法：

### 想法 1：块分解（chunk decomposition）

每个前向块被拆分为四个组件：

- **Attention（注意力）。** Q/K/V 投影、注意力计算、输出投影。
- **All-to-all dispatch（分发）。** 将 token 发送到其对应专家的跨节点通信。
- **MLP。** MoE 专家计算。
- **All-to-all combine（合并）。** 将专家输出返回的跨节点通信。

后向块则增加了上述每个组件的梯度版本。DualPipe 将它们调度为：all-to-all dispatch 与下一个块的 attention 计算并行执行，all-to-all combine 与后续块的 MLP 计算并行执行。

### 想法 2：双向调度

大多数流水线调度方案从阶段 0 注入 micro-batch 并向阶段 P-1 流动。DualPipe 从两端同时注入 micro-batch。阶段 0 看到源自此处的正向 micro-batch；阶段 P-1 也看到源自此处的正向 micro-batch。两股流在中间相遇。

为此，设备 `i` 必须同时持有早期流水线层 `i` 和晚期流水线层 `P - 1 - i`。这就是 DualPipe 中的"dual（双份）"部分：每个设备维护两份模型层副本，服务于两个方向。在 DeepSeek-V3 的规模下，这是 2x 的参数复制成本。这可以承受，因为 Expert Parallelism 已经将 MoE 专家分散得非常薄，复制两次非专家层的开销相对很小。

关键是，一个方向的正向流和另一个方向的反向流恰好重叠在单向调度方案中会出现气泡的位置。气泡消失了。

### 手动推导的调度表

考虑 P = 4 个 rank，8 个 micro-batch，分为 4 个正向 / 4 个反向。时间从左到右推进；行代表设备 rank。

```
           时间 →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

解读 "F4/F5R" 的标记：rank 1 在同一个时间槽内运行 micro-batch 4 的正向传播（在流水线中从左到右）和 micro-batch 5 的正向传播（从右到左）。这就是"双向"在操作上的含义。

在 rank 2 处，交叉流更早重叠；在 rank 0 和 P-1 处，它们最晚重叠。在调度的稳定中间阶段，每个 rank 运行 X 方向的正向与 Y 方向的反向重叠。计算繁忙。前向传播的 all-to-all dispatch 隐藏在反向计算中。all-to-all combine 隐藏在前向计算中。气泡被挤出去了。

### 气泡计算

标准 1F1B 流水线气泡（每个 rank 浪费的时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 的改进将气泡降低但不能降到零。DualPipe 在稳定阶段，如果 micro-batch 数量能被 2 倍的流水线深度整除，气泡为零。在稳定阶段之外（预热和冷却阶段），存在一些气泡，但不会随 micro-batch 数量的增加而增长——这是论文强调的一个关键特性。

在宣传术语中："无气泡"。在技术术语中：气泡不随 micro-batch 数量增加而增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）表明，只有当 Expert Parallelism 不是瓶颈时才真正无气泡；在 EP 驱动的 all-to-all 下，总是存在一定的调度折中。

### DualPipeV——改进版

Sea AI Lab（2025）观察到，当 EP 通信重叠不是重点时，2x 参数复制是一种浪费。他们的 DualPipeV 调度将双向注入折叠为"V 形"调度，在单份参数副本上运行。气泡比 DualPipe 略大，但内存节省显著。DeepSeek 在其开源的 DualPipe 实现中采用了 DualPipeV 作为 EP-off 模式。

权衡：

| 特性 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| 每设备参数副本数 | 2 | 1 | 1 | 1 |
| 气泡与 micro-batch 数量关系 | 恒定 | 轻微增长 | 增长 | 增长 |
| 计算-通信重叠 | 完全 | 部分 | 最小 | 部分 |
| 适用场景 | EP 密集型 MoE | 稠密或 EP 轻量 | 基线 | 任意流水线 |

### 对 14.8T token 训练意味着什么

DeepSeek-V3 的预训练在 2,048 块 H800 GPU 上消耗了 14.8T token，大约用了 280 万 GPU-小时。如果使用朴素的 1F1B，其中 12-15% 将因流水线气泡而损失——即 34-42 万 GPU-小时，足以训练一个完整的 70B 模型。DualPipe 回收了其中大部分。没有内部日志难以直接量化其贡献，但论文声称训练期间 GPU 利用率平均超过 95%。

对于较小的训练（低于 1k GPU），DualPipe 有些过度——相对总成本而言，流水线气泡较小，且稠密模型训练很少达到 all-to-all 瓶颈。而对于数千 GPU 规模的前沿 MoE 训练，它基本上是必需的。

### 在技术栈中的位置

- 与 **FSDP**（Phase 10 · 05）互补。FSDP 在 rank 之间分片模型参数；DualPipe 在 rank 之间调度计算。两者可以结合使用。
- 兼容 **ZeRO-3** 梯度分片。两副本复制的簿记需要与 ZeRO 的分片梯度协同工作。
- 需要针对特定集群拓扑调优的**自定义 all-to-all kernel**。DeepSeek 的开源 kernel 是参考实现。

```figure
expert-capacity
```

## 使用

`code/main.py` 是一个流水线调度模拟器。它接受 `(P, n_micro_batches, schedule)` 作为输入，打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 四种方案的稳定阶段利用率。这是一个教学工具——数字与论文中的定性声明相符，并非对生产环境实测加速比的声明。

模拟器的价值：使用不同的 P 和 micro-batch 数量运行它，观察气泡比例如何随 1F1B 增长而不随 DualPipe 增长。

实际训练运行的集成考虑：

- 选择一个能整除 micro-batch 数量的流水线并行深度。
- 确保你的专家并行网格支持双向 all-to-all。DeepSeek 的 kernel 是参考实现。
- 首次集成时预计需要一周的调试时间用于调度本身。簿记非常繁琐。
- 按 rank 监控 GPU 利用率，而不只是聚合数据。DualPipe 的收益来自收紧落后者。

## 产出

本节课产出 `outputs/skill-dualpipe-planner.md`。给定一个训练集群规格（GPU 数量、拓扑、互联方式、模型形状），它推荐一个流水线并行策略、要使用的调度算法以及目标规模下的预期气泡比例。

## 练习

1. 分别用 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 运行 `code/main.py`。计算 GPU 利用率差异，并将其表示为每百万 token 训练所回收的 GPU-小时。

2. 手动画出 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。用 micro-batch ID 和方向标记每个时间槽。找出第一个没有气泡的时间槽。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的图 5。找出 DualPipe 前向块内 all-to-all dispatch 的重叠窗口。解释计算调度如何隐藏它。

4. 计算 DualPipe 对 70B 稠密模型（P=8 流水线阶段）和 671B MoE 模型（P=16 流水线阶段）的 2x 参数开销。说明为什么 MoE 场景的开销在比例上更小（大多数参数是专家，被分片到一个大型 EP 组中）。

5. 将 DualPipe 与 Chimera（2021 年提出的竞争性双向调度器）进行比较。参考论文第 3.4 节，找出 DualPipe 具有而 Chimera 没有的两个特定属性。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Pipeline bubble（流水线气泡） | "每个 rank 的空闲时间" | 因流水线阶段等待其输入或梯度而浪费的 GPU 周期 |
| 1F1B | "默认流水线调度" | 一次前向/一次反向交错调度；DualPipe 超越的基线 |
| Zero Bubble | "Sea AI Lab 2023" | 将反向传播拆分为 B（输入梯度）和 W（权重梯度）；几乎完全收紧流水线 |
| DualPipe | "DeepSeek-V3 调度" | 双向流水线 + 计算-通信重叠；气泡不随 micro-batch 数量增长 |
| DualPipeV | "Cut-in-half" | V 形改进版，放弃 2x 参数复制，代价是气泡略微增大 |
| Chunk（块） | "流水线工作单元" | 一个 micro-batch 通过一个流水线阶段的一次前向或后向传播 |
| All-to-all dispatch（分发） | "将 token 发送到专家" | 将 token 路由到其分配的 MoE 专家的跨节点通信 |
| All-to-all combine（合并） | "将专家输出返回" | MLP 后收集专家输出的跨节点通信 |
| Expert Parallelism, EP（专家并行） | "专家跨 GPU 分布" | 将 MoE 专家跨 rank 分片，不同 GPU 持有不同专家 |
| Pipeline Parallelism, PP（流水线并行） | "层跨 GPU 分布" | 将模型层跨 rank 分片；DualPipe 调度的维度 |
| Bubble fraction（气泡比例） | "浪费的 GPU 时间" | (bubble_time / total_time)；DualPipe 趋近于零的比例 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 技术报告（arXiv:2412.19437），第 3.3.2 节及图 5](https://arxiv.org/abs/2412.19437) — DualPipe 的主要参考文献
- [DeepSeek — DualPipe GitHub 仓库](https://github.com/deepseek-ai/DualPipe) — 开源参考实现，包括 DualPipeV（Cut-in-half）模式
- [Qi et al. — Zero Bubble Pipeline Parallelism（arXiv:2401.10241，Sea AI Lab 2023）](https://arxiv.org/abs/2401.10241) — Zero Bubble 前身
- [Sea AI Lab — DualPipe 可以更好，不需要 Dual](https://sail.sea.com/blog/articles/63) — 启发了 DeepSeek EP-off 模式的 DualPipeV 分析
- [Narayanan et al. — PipeDream / 1F1B（arXiv:1806.03377，2018-2021）](https://arxiv.org/abs/1806.03377) — DualPipe 作为对比的 1F1B 调度
- [Huang et al. — GPipe（arXiv:1811.06965，2018）](https://arxiv.org/abs/1811.06965) — 原始流水线并行论文及气泡问题
