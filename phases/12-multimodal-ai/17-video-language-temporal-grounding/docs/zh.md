# 视频-语言模型：时序 Token 与定位

> 视频不是一堆照片。一个 5 秒的片段有因果顺序、动作动词和事件时序，这是图像模型无法表示的。Video-LLaMA（Zhang 等人，2023年6月）发布了首个带视听定位的开源视频 LLM。VideoChat 和 Video-LLaVA 扩展了该模式。到 2025 年 Qwen2.5-VL 的 TMRoPE 缩小了与前沿专有模型的差距。每个系统以不同方式解决了时序 token——每片段 Q-former、每帧拼接池化、每 token TMRoPE。本课阅读这些模式，构建一个均匀 vs 动态帧采样器，并在时序定位任务上评估。

**Type:** Build
**Languages:** Python (stdlib, frame sampler + temporal-grounding evaluator)
**Prerequisites:** Phase 12 · 08 (LLaVA-OneVision)
**Time:** ~180 minutes

## 学习目标

- 解释为什么时序位置编码独立于视觉编码器改变视频 VLM 性能。
- 在每秒 token 数 vs 定位准确率上比较均匀、动态 FPS 和事件驱动帧采样。
- 描述每片段 Q-former（Video-LLaMA）vs 每帧池化（Video-LLaVA）vs 每 token M-RoPE（Qwen2.5-VL）设计。
- 说出四个视频基准：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 问题

一个 1 分钟、30 FPS 的视频是 1800 帧。每帧 196 个视觉 token（ViT-B @ 224），即 352k 个 token——大于任何 2024 时代的 LLM 上下文。

存在三种缩减策略：

1. 子采样帧（根据内容 1-8 FPS）。
2. 激进池化每帧的 patch token（3x3 或 4x4 双线性池化）。
3. 通过一个 Q-former 压缩，取 16 帧片段，输出 64 个 token。

每种权衡不同。子采样丢失时序细节。池化丢失空间细节。Q-former 两者丢一点但节省 token。

时序位置编码是另一轴：模型如何知道帧 5 在帧 6 前面？选项包括简单的 1D 时序 RoPE（Video-LLaMA）、可学习的时序嵌入（Video-LLaVA）和 TMRoPE（Qwen2.5-VL，完整 3D）。

## 概念

### Video-LLaMA：每片段 Q-former + 音频分支

Video-LLaMA（2023）是第一个开源视频 LLM。架构：

- 2 FPS 下 16 帧片段（即 8 秒）。
- 每帧 ViT 特征 -> Video Q-former 交叉关注所有 16 帧 -> 32 个可学习 query -> LLM。
- 并行音频分支：波形 -> ImageBind 音频编码器 -> Audio Q-former -> 32 个 query -> LLM。

优势：视听联合推理。劣势：固定片段长度，无任意时间定位。

### VideoChat 和 Video-LLaVA

VideoChat 保留了 Video-LLaMA 的想法但丢弃音频并简化。Video-LLaVA（Lin 等人，2023）在图像和视频帧上训练单一视觉编码器（"先对齐再投影"），给出统一表示。两者都是冻结 CLIP 编码器 + MLP + LLM。

两者都不处理长视频。两者都是 8-16 帧系统。

### Qwen2.5-VL 和 TMRoPE

Qwen2.5-VL 引入 TMRoPE——时域-模态旋转位置嵌入。每个 patch token 携带 (t, h, w) 位置，其中 t 是实际时间戳（非帧索引）。

与简单时序嵌入的关键差异：

- 绝对时间，非索引。模型看到"在 4.2 秒"而非"在帧 15"。
- 每 token 旋转，非每片段。每个视觉 token 按其时间戳独立旋转。
- 与动态 FPS 兼容。如果你这里 2 FPS 采样那里 4 FPS，TMRoPE 原生处理不均匀间距。

TMRoPE 使"In what second does the cat jump?"的查询成为可能。模型可以输出"在 4.2 秒"。Video-LLaMA 只能说"片段早期"。

### 帧采样策略

均匀：在时长上均匀采样 N 帧。简单，丢失运动峰值。

动态 FPS：基于运动强度自适应采样。光流或帧差选择高运动段以更密集采样。Qwen2.5-VL 在此上训练。

事件驱动：运行一个轻量级检测器，在动作发生处采样更多。VideoAgent 使用。

关键帧 + 上下文：在镜头边界采样 + 相邻几帧。用于影视内容。

### 每帧池化

在 1 FPS 和每帧 576 token 下，5 分钟片段是 172,800 个 token。Qwen2.5-VL-72B 的 128k 上下文可行但昂贵。

3x3 双线性池化缩减至每帧 64 token -> 5 分钟 19,200 token。大多数任务的甜点区。

更激进池化（6x6 -> 每帧 16 token）用于空间细节不那么重要的代理工作流。

### 四个视频基准

- VideoMME：综合视频理解，短 + 中 + 长。
- TempCompass：细粒度时序推理，"之前"/"之后"问题。
- EgoSchema：长时域第一人称视频。
- Video-MMMU：多模态多学科视频问题。

完整视频 VLM 评估应涵盖全部四个。它们强调不同轴——TempCompass 全部关于顺序，EgoSchema 关于 3+ 分钟推理，VideoMME 跨时长。

### 定位输出格式

时序定位的输出格式：

- 自由文本："The cat jumps around the 4-second mark." 容易解析但不精确。
- 结构化 JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 训练这个。
- 基于 Token：特殊的 `<time>4.1</time>` token 交错在答案中。Qwen2.5-VL 的内部格式。

基于 token 对下游使用最准确。Qwen2.5-VL 的 JSON 输出格式直接解析。

### 2026 年最佳实践

对 2026 年的视频 VLM：

- 编码器：带 M-RoPE 或 TMRoPE 的 SigLIP 2（Qwen2.5-VL）。
- 帧采样：动态 FPS（根据运动 1-4）带最大帧数上限。
- 每帧池化：3x3 双线性。
- 输出：带时间 + 事件字段的结构化 JSON。
- 基准：VideoMME + TempCompass 用于通用；EgoSchema 用于长时域。

## 使用它

`code/main.py` 包含：

- 均匀和动态 FPS 帧采样器。
- 玩具时序定位评估器：给定"真实"事件时间 T 和模型输出，用容差评分准确率。
- 跨 Video-LLaMA（16 帧，Q-former）、Video-LLaVA（8 帧，MLP）、Qwen2.5-VL（动态 FPS + TMRoPE）的对比。

## 交付

本课产出 `outputs/skill-video-vlm-frame-planner.md`。给定一个视频任务（监控、动作识别、时序定位、摘要），选择帧采样器、池化因子、输出格式和预期准确率层级。

## 练习

1. 对 3 分钟烹饪演示，选择均匀 vs 动态 FPS。用 token 数论证。

2. TMRoPE 具体添加了什么简单的时序嵌入表做不到的？

3. 写一个 VLM 可以学会发出的时序定位 JSON schema。包含错误情况。

4. 阅读 Video-LLaVA 第 3 节关于"先对齐再投影"。为什么这比训练独立的图像和视频编码器更好？

5. 给定 VideoMME 排行榜，2026 年顶级开源模型和顶级专有模型之间的差距是多少？该差距有多少归因于时序编码 vs LLM 基础规模？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 时序定位 | "时间局部化答案" | VLM 输出事件发生的具体时间戳范围 |
| TMRoPE | "时域-多模态 RoPE" | 带绝对时间戳的 3D 旋转位置，Qwen2.5-VL 使用 |
| 动态 FPS | "运动感知采样" | 在高运动段采样更多帧，在静态段采样更少 |
| 帧池化 | "每帧空间压缩" | 在 LLM 之前用双线性插值减少每帧 patch 数 |
| Video Q-former | "片段压缩器" | 将 N 帧映射到 K 个可学习 query 的交叉注意力瓶颈 |
| VideoMME | "视频基准" | 综合短/中/长视频基准，2500+ 样本 |

## 延伸阅读

- [Zhang 等人 — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li 等人 — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin 等人 — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen 团队 — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin 等人 — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
