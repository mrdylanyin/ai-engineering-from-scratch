# Omni 模型：Qwen2.5-Omni 与 Thinker-Talker 分离

> GPT-4o 在 2024年5月的产品演示之所以是颠覆性的，不是因为底层模型，而是因为产品形态——一个语音界面，你说话，模型看到摄像头看到的东西，并且在 250ms 内回应。开源生态用 2024 年剩余时间和 2025 年赛跑来触及这个产品表面。Qwen2.5-Omni（2025年3月）是参考开源设计：一个 Thinker（大型文本生成 Transformer）加上一个 Talker（并行语音生成 Transformer），通过流式语音 token 连接。Mini-Omni 简化了它，Moshi 匹配了它的延迟，GLM-4-Voice 将其扩展到中文。本课阅读 Thinker-Talker 架构和使流式实时对话成为可能的延迟预算。

**Type:** Build
**Languages:** Python (stdlib, streaming pipeline latency simulator + VAD loop)
**Prerequisites:** Phase 12 · 19 (audio-LLMs), Phase 12 · 16 (any-to-any)
**Time:** ~180 minutes

## 学习目标

- 将推理管道拆分为 Thinker（文本推理）和 Talker（语音合成），解释为什么并行流式传输有效。
- 为对话交互计算首字节音频时间（TTFAB）预算，逐个组件。
- 描述 Thinker 内部跨视觉、音频和文本的 TMRoPE 时间对齐位置编码。
- 说出三种实时对话模式：半双工、轮次切换、全双工。

## 问题

一个实时语音助手必须快速做很多事情：

1. 听到用户。实时语音分词、语音活动检测（VAD）以知道用户何时说完。
2. 可选地看到。以 2-4 FPS 的摄像头输入，随音频一起流式输入 Thinker。
3. 思考。在对话历史的条件化下组织回应。
4. 说话。合成音频 token，解码为波形，流式传输到用户扬声器。

每一步增加延迟。对话感要求总往返 < 500ms——低于这个阈值，用户就不再注意到延迟。GPT-4o 声称约 250ms。Moshi 约 160ms。Qwen2.5-Omni 约 350-500ms。

每个组件都必须流式传输。不能"批量处理一切然后解码"。

## 概念

### Thinker 和 Talker

Qwen2.5-Omni 的分解：

- Thinker：7B-80B 文本生成 Transformer。消费交错文本 + 图像 + 音频 token。输出表示"说什么"的文本 token。
- Talker：较小的语音生成 Transformer（2亿-10亿）。消费 Thinker 的文本输出 token 加上最近的语音上下文 token。输出离散语音 token（残差 VQ 索引）。
- 语音解码器：流式波形解码器（SNAC、MoVQGAN 家族），实时将语音 token 转换为音频采样。

分离很重要。Thinker 必须大才能推理好。Talker 可以小，因为其工作是局部的——将文本转换为语音 token。更大的 Talker 不会更有表现力；只会更慢。

并行运行两者：

1. Thinker 发出文本 token t_i。
2. Talker（通过流式传输）消费 t_i 并发出语音 token s_i, s_{i+1}, ..., s_{i+k}。
3. 语音解码器随语音 token 到来消费它们并发出音频采样。
4. 当 Thinker 到文本 token t_{i+3} 时，Talker 已经为 t_0..t_{i+2} 流式传输了音频。

### TMRoPE——时间对齐多模态位置

Thinker 需要整合到达的图像帧（以 4 FPS 到达）、音频帧（以每秒 50 帧到达）和来自对话历史的文本。一个朴素的序列顺序（所有图像，然后所有音频，然后文本）丢失了时间对齐。

TMRoPE 为每个 token 分配绝对时间戳。视觉 token 在 t=2.3s。音频 token 在 t=2.32s。用户说的文本 token "stop" 在 t=2.35s。RoPE 按时间戳旋转注意力；模型将它们视为时间上并发。

这是"他挥手并说你好"得以工作的基础设施——模型看到视频帧和音频在同一个概念时刻。

### 流式语音合成

语音 token 必须流式传输。Mini-Omni（Xie & Wu, 2024）引入了"语言模型可以在流式中听到和说话"：Thinker 输出 token 和 Talker 输出 token 在同一序列中交错。Talker 在 Thinker 提交下一个文本 token 时立刻启动。无批边界。

Moshi（Défossez 等人，2024年10月）是最快的开源实现。单张 A100 上 160ms TTFAB。架构：一个 7B Transformer，在交替位置上发出文本和语音 token，采用"内心独白"将思考流和说话流分离。这实际上是将 Thinker + Talker 融合到一个模型中，通过仔细的训练实现。

### VAD 和轮次切换

语音活动检测运行在输入侧。两种模式：

- 半双工：用户说，模型听。模型说，用户听。通过 VAD 静音检测清晰交接（约 200ms）。
- 全双工：两者可以同时说话。模型可以回音（"嗯嗯"）或打断。难度高得多。Moshi 支持这个。

Qwen2.5-Omni 默认支持半双工，通过静音阈值进行轮次切换。全双工需要应用层处理。

### Qwen3-Omni（2025年11月）

后继者。Qwen3-80B Thinker，更大的 Talker，改进的 TMRoPE-v2。延迟接近 GPT-4o 的 250ms。开源权重。OmniBench 上的基准与 Gemini 2.0 Live 有竞争力。

### 生产延迟预算

对于典型的流式交互：

- 麦克风 -> 音频 token：40-80ms。
- 预填充（提示 + 历史）：7B 上 100-200ms，70B 上更多。
- 首个 Thinker 文本 token：40ms。
- Talker 处理首个文本 token：20ms。
- 首个语音 token 提交：40ms。
- 残差 VQ 解码：30ms。
- 语音波形解码：50-80ms。

总 TTFAB：7B 上 320-510ms，70B 上 600-900ms。前沿质量通常意味着 70B+；因此前沿延迟有差距。

### Token 速率数学

在 16kHz 语音和 50 Hz 基础语音 token 下，每秒需要 50 个语音 token 输出。Talker 必须发出 ≥50 tok/s 才能跟上。在 H100 上典型 LLM 吞吐量 30-80 tok/s 下，一个小的（2亿-3亿）Talker 足够快；7B Talker 会落后。

这就是存在小型专用 Talker 模型而非"只用主模型"的原因。

## 使用它

`code/main.py`：

- 用模拟 token 发出速率模拟 Thinker-Talker 管道。
- 为可配置的模型大小和麦克风采样率计算 TTFAB。
- 演示带 VAD 静音阈值的半双工轮次切换。

## 交付

本课产出 `outputs/skill-omni-streaming-budget.md`。给定实时语音产品的目标 TTFAB 和功能集（视觉输入、双语、全双工），在 Qwen2.5-Omni、Qwen3-Omni、Moshi 或 Mini-Omni 之间选择并确定 Thinker/Talker 规模。

## 练习

1. 你的目标 TTFAB 是 300ms。在 7B Thinker 和 3亿 Talker 上，写出每个组件的延迟。

2. Qwen2.5-Omni 使用 TMRoPE。描述当用户在 t=1s 开始说话、摄像头在 t=1.2s 捕捉到一个手势时，模型看到了什么。

3. 全双工支持要求模型在收听的同时发出音频。提出一个教授这个能力的训练数据格式。

4. 阅读 Moshi 论文第 4 节。描述"内心独白"分离以及为什么它避免了 Thinker-Talker 分离。

5. 计算吞吐量预算：Talker 必须在 16kHz 语音、每秒 50 基础层 token 下发出多快才能跟上？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Thinker | "推理大脑" | 大型文本生成 Transformer，产生"说什么" |
| Talker | "语音生成嘴巴" | 小型 Transformer，从 Thinker 的文本产生离散语音 token |
| TTFAB | "延迟预算" | 首字节音频时间：从用户语音结束到首个音频采样输出 |
| TMRoPE | "时间对齐 RoPE" | 跨视觉、音频、文本使用绝对时间戳的位置编码 |
| 半双工 | "轮次切换" | 用户和模型交替；VAD 静音检测用户完成 |
| 全双工 | "同时" | 模型可以同时说和听；可回音 |
| 内心独白 | "Moshi 分离" | 单模型设计，思考流和说话流交错 |

## 延伸阅读

- [Xu 等人 — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen 团队 — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez 等人 — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng 等人 — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
