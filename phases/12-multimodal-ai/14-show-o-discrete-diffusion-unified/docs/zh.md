# Show-o 与离散扩散统一模型

> Transfusion 混合了连续和离散表示。Show-o（Xie 等人, 2024年8月）走了另外一条路：文本 token 使用因果 next-token 预测，图像 token 使用 MaskGIT 精神的掩码离散扩散。两者坐落在带混合注意力掩码的一个 Transformer 中。结果在一个骨干、每种模态一个分词器、一个损失公式（next-token 扩展到掩码预测）上统一了 VQA、文本到图像、图像修补和混合模态生成。本课走读 Show-o 的设计——为什么掩码离散扩散是一个并行的、少步骤的图像生成器——并与 Transfusion 和 Emu3 对比。

**Type:** Learn
**Languages:** Python (stdlib, masked-discrete-diffusion sampler)
**Prerequisites:** Phase 12 · 13 (Transfusion)
**Time:** ~120 minutes

## 学习目标

- 解释掩码离散扩散：统一掩码 token 然后要求 Transformer 恢复它们的调度。
- 在速度和质量上比较并行图像解码（Show-o、MaskGIT）和自回归图像解码（Chameleon、Emu3）。
- 说出 Show-o 在单个 checkpoint 中处理的三种任务：T2I、VQA、图像修补。
- 选择一个掩码调度（余弦、线性、截断）并推理其对样本质量的影响。

## 问题

Transfusion 的双损失训练有效但动态更难——连续扩散损失和离散 NTP 损失处于不同的数值尺度。平衡损失权重是一个超参数搜索。架构有效但复杂。

Show-o 的答案：保持两种模态离散（像 Chameleon），但通过掩码离散扩散并行生成图像，而非顺序生成。训练目标变成单一的掩码 token 预测，自然泛化了 next-token 预测。

## 概念

### 掩码离散扩散（MaskGIT）

原始 Chang 等人（2022）MaskGIT 的技巧是优雅的。从完全掩码的图像开始（每个 token 是特殊 `<MASK>` ID）。在每一步并行预测所有掩码 token，然后保留 top-K 最置信的预测，重新掩码剩下的。经过约 8-16 次迭代，所有 token 被填满。每步揭掩多少 token 的调度是需要调优的——余弦调度效果好。

训练很简单：从 [0, 1] 中均匀采样一个掩码比率，将其应用到图像的 VQ token 上，训练 Transformer 恢复被掩码的。与 BERT 对文本所做的完全相同，扩展到图像生成。

### Show-o：一个 Transformer，混合掩码

Show-o 将 MaskGIT 放入一个因果语言模型 Transformer 中。注意力掩码是：

- 文本 token：因果（标准 LLM）。
- 图像 token：图像块内完全双向（这样掩码 token 可以看其他所有图像 token 进行预测）。
- 文本到图像：文本关注前面的图像，图像关注前面的文本。

训练交替：
1. 文本序列上的标准 NTP。
2. T2I 样本：文本 → 带掩码图像 token 的图像，掩码 token 预测损失。
3. VQA 样本：图像 → 带掩码文本 token 的文本（实际上就是 NTP）。

统一损失是对 `<MASK>` token 的交叉熵，这同时覆盖了文本 NTP（只有最后一个 token 是"掩码的"）和图像掩码扩散（随机子集被掩码）。

### 并行采样

Show-o 以约 16 步生成图像，而非约 1000 步（每 token 自回归）或约 20 步（扩散）。每一步，并行预测所有掩码 token；提交 top-K 置信的；重复。

对比：
- Chameleon / Emu3（自回归逐 token）：N_tokens 次前向传播，通常每图像 1024-4096 次。
- Transfusion（连续扩散）：约 20 步，每步一次完整 Transformer 传递。
- Show-o（掩码离散扩散）：约 16 步，每步一次完整 Transformer 传递。

Show-o 在相似规模模型下比 Chameleon 快，大致匹配 Transfusion 的步骤数，每步成本更低（离散词汇 logits vs 连续 MSE 损失）。

### 一个 Checkpoint 中的任务

Show-o 在推理时通过提示格式选择支持四种任务：

- 文本生成：标准自回归文本输出。
- VQA：图像进，文本出。
- T2I：文本进，通过掩码离散扩散出图像。
- 修补：部分 token 掩码的图像，填充被掩码部分。

修补能力从掩码预测训练中免费获得。掩码 VQ-token 网格的一个区域，喂入其余部分加文本提示，预测掩码 token。

### 掩码调度

每步揭掩多少 token 的调度塑造质量。Show-o 推荐余弦：

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

在步骤 0，所有 token 掩码（比率 1.0）。在步骤 T，无掩码。余弦将质量集中在预测最有信息量的中等比率范围。线性调度也有效但更快平台化。

### Show-o2

Show-o2（2025 年后续，arXiv 2506.15564）扩展了 Show-o：更大的 LLM 基础、更好的分词器、改进的掩码调度。相同的架构模式。

### Show-o 在哪里

在 2026 分类学中：

- 离散 token + NTP：Chameleon、Emu3。简单但推理慢。
- 离散 token + 掩码扩散：Show-o、MaskGIT、LlamaGen、Muse。并行采样，仍受分词器有损限制。
- 连续 + 扩散：Transfusion、MMDiT、DiT。最高质量，训练更复杂。
- VLM 中的连续 + 流匹配：JanusFlow、InternVL-U。最新。

按任务选择：当你想要一个开放模型中以合理速度做 T2I + 修补 + VQA 时选择 Show-o；当质量至上且你能承受双损失管道时选择 Transfusion。

## 使用它

`code/main.py` 模拟 Show-o 采样：

- 一个 16 VQ token 的玩具网格。
- 一个模拟"Transformer"，基于提示和当前未掩码 token 预测 logits。
- 8 步骤带余弦调度的并行掩码采样。
- 打印中间状态（掩码模式演化）和最终 token。

运行它，观察掩码一步步溶解。

## 交付

本课产出 `outputs/skill-unified-gen-model-picker.md`。给定一个同时需要理解（VQA、标题生成）和生成（T2I、修补）且受限于开源权重的产品，在 Show-o 家族、Transfusion/MMDiT 家族和 Emu3/Chameleon 家族之间选择，附具体权衡。

## 练习

1. 掩码离散扩散以约 16 步采样。为什么不是 1 步？如果在步骤 0 就揭掩所有东西会发生什么？

2. 修补在掩码扩散下是免费的。提出一个（真实或假设的）产品用例，其中 Show-o 的修补击败专家模型。

3. 余弦调度 vs 线性调度：追踪 T=8 时每步揭掩 token 数量。哪个更平衡？

4. 一张 512x512 Show-o 图像是 1024 个 token。词汇 K=16384 时，模型发出 1024 * log2(16384) = 14,336 比特（约 1.75 KiB）数据。Stable Diffusion 输出 512*512*24 比特 = 6,291,456 比特（约 768 KiB）原始像素。压缩比是多少，换取了什么质量？

5. 阅读 LlamaGen（arXiv:2406.06525）。LlamaGen 的类别条件自回归图像模型与 Show-o 的掩码方法有何不同？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 掩码离散扩散 | "MaskGIT 风格" | 训练预测掩码 token；推理时迭代揭掩最自信的预测 |
| 余弦调度 | "揭掩调度" | 掩码比率随推理步骤衰减；将置信度增长集中在中等范围 |
| 并行解码 | "所有 token 同时" | 每步在一次前向传播中预测完整掩码 token 序列，然后提交 top-K |
| 混合注意力 | "因果 + 双向" | 文本 token 因果、图像块内双向的掩码 |
| 修补 | "填充生成" | 条件化于部分 token 掩码的图像，预测缺失的；从训练目标中免费获得 |
| 提交率 | "每步 top-K" | 每次迭代声明多少 token 已"完成"；控制推理速度 vs 质量权衡 |

## 延伸阅读

- [Xie 等人 — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang 等人 — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun 等人 — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang 等人 — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
