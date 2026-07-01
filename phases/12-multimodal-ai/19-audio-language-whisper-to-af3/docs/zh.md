# 音频-语言模型：从 Whisper 到 Audio Flamingo 3 的弧线

> Whisper（Radford 等人，2022年12月）解决了语音识别——68 万小时弱监督多语言语音，一个简单的编码器-解码器 Transformer，一个让每个后续 ASR 发布都必须引用的基准。但识别不是推理。询问"这个录音中有哪些乐器"、"说话人在表达什么情绪"或"第 3 分钟发生了什么"需要音频理解，而非转写。Qwen-Audio、SALMONN、LTU 和 NVIDIA 的 Audio Flamingo 3（AF3，2025年7月）逐步构建了这个栈：保留 Whisper 级编码器，外挂 Q-former，在音频-文本指令数据上训练，添加思维链推理。本课走读这条弧线。

**Type:** Build
**Languages:** Python (stdlib, log-Mel spectrogram + audio Q-former skeleton)
**Prerequisites:** Phase 6 (Speech and Audio), Phase 12 · 03 (Q-Former)
**Time:** ~180 minutes

## 学习目标

- 从波形计算 log-Mel 频谱：加窗、FFT、滤波器组、log 变换。
- 比较编码器选项：Whisper 编码器、BEATs、AF-Whisper 混合。每种何时胜出。
- 构建音频 Q-former：N 个可学习 query 交叉关注频谱 patch。
- 解释级联（Whisper 然后 LLM）vs 端到端音频 LLM 训练：为什么端到端对推理扩展性更好。

## 问题

语音识别被 Whisper 解决了。音频的 OCR 是商品。但"商品"止步于转写。如果模型不能推理它听到的内容——时序、说话人、情绪、音乐结构、环境声音——仅转写无法驱动产品功能。

三条明显的路线：

1. 级联：Whisper 转写，LLM 推理转写文本。对纯语音场景有效。对音乐、环境音频、多说话人重叠、情绪无效。
2. 端到端音频 LLM：音频编码器将音频 token 直接喂入 LLM，跳过转写。保留声学信息（情绪、说话人、环境）。需要新的训练数据。
3. 混合：既能转写又能推理的音频编码器 + 文本解码器。Qwen-Audio 和 Audio Flamingo 选择此路线。

## 概念

### Log-Mel 频谱：输入特征

每个音频编码器从相同的特征开始：log-Mel 频谱。

1. 重采样到 16 kHz。
2. 短时傅里叶变换，25ms 窗口，10ms 跳跃。
3. 取 FFT 结果的幅度。
4. 应用 Mel 滤波器组（通常 80 个对数间隔 0-8000 Hz 的滤波器）以扭曲到感知频率。
5. Log 压缩（log(1 + x)）用于动态范围。

结果：形状为 (T, 80) 的 2D 数组，T 是时间帧数。对于 30 秒片段、100 Hz 帧率：(3000, 80)。

### Whisper 的编码器

Whisper 的编码器是一个 12 层 ViT 风格 Transformer，将 log-Mel 频谱处理为时间帧序列。输出：每个时间帧一个隐藏状态向量。

对于 ASR，Whisper 的解码器是一个交叉注意力 Transformer，在编码器输出的条件化下生成文本 token。标准编码器-解码器。

对于 ALM（音频 LLM），你需要编码器输出作为不同 LLM 的输入。模式：Whisper 编码器冻结，Q-former 可训练，LLM 冻结或微调。

### BEATs 和音频特定编码器

Whisper 在语音主导的数据上训练。它对音乐和环境音频较弱。

BEATs（Chen 等人，2022）是一个在 AudioSet 上训练的自监督 Transformer。在相同参数量下，对音乐和环境声音的捕获优于 Whisper。

AF-Whisper（Audio Flamingo 3 的混合）：拼接 Whisper + BEATs 特征作为音频输入。Whisper 携带语言信号，BEATs 携带声学信号。

### 音频 Q-former

与 BLIP-2 视觉 Q-former 相同的模式。固定数量的可学习 query（通常 32 或 64）交叉关注音频编码器的输出帧。Query 成为 LLM 消费的音频 token。

训练对齐阶段：Q-former 单独，在音频-文本对（AudioCaps、Clotho）上用对比 + 标题损失。指令阶段：端到端，解冻 LLM，在指令数据上训练。

### 弧线——SALMONN、Qwen-Audio、AF3

SALMONN（Tang 等人，2023）：Whisper + BEATs + Q-former + LLaMA。首个有严肃推理能力的开源音频 LLM。MMAU 基准约 0.55 综合分。

Qwen-Audio（Chu 等人，2023）：类似架构，在更丰富数据集上训练，为多轮对话调优。MMAU 约 0.60。

LTU——Listen, Think, Understand（Gong 等人，2023）：显式推理数据，聚焦音频片段的思维链。更小但更专注。

Audio Flamingo 3（Goel 等人，2025年7月）：当前开源 SOTA。8B LLM 骨干（Qwen2 7B），Whisper-large 编码器拼接 BEATs，64 query Q-former，在 100 万+ 音频-文本指令对上训练。MMAU 0.72，匹配专有前沿在某些子任务上。

AF3 还引入了按需思维链用于音频：模型可以可选地在最终答案前发出思考 token（"让我先识别乐器：..."）。启用思考时，复杂推理任务准确率提升 3-5 点。

### 级联 vs 端到端

级联管道：

1. Whisper 转写音频 → 文本。
2. LLM 推理文本。

对"总结这个播客"完美生效。对以下失败：
- "这首歌的情绪是什么？"——情绪在声音中，不在文字中。
- "谁在说话，Alice 还是 Bob？"——需要说话人识别。
- "爆炸发生在第几秒？"——时序定位在文本中丢失。
- "这是真实还是生成的音频？"——深度伪造检测需要声学特征。

端到端保留声学信号。Qwen-Audio 和 AF3 原生处理音乐、环境和情绪。

### 2026 年生产配方

对于新的音频理解产品：

- 级联如果：转写是目标，无音乐，无情绪推理。
- AF3 / Qwen-Audio 家族如果：音乐、情绪、多说话人或复杂音频推理。

级联更便宜更简单。端到端能力更强。

### MMAU——音频推理基准

MMAU（大规模多模态音频理解）是 2024-2025 年音频推理基准：

- 10000 个音频-文本 QA 对，涵盖语音、音乐、环境声音。
- 涵盖分类、时序推理、因果推理、开放式问答。
- 测试级联管道系统性遗漏的内容。

开源 SOTA（AF3）0.72；专有前沿约 0.78（Gemini 2.5 Pro、Claude Opus 4.7）。差距比 VideoMME 的开源 vs 闭源 delta 小，表明音频 LLM 正在成熟。

## 使用它

`code/main.py`：

- 标准库实现 log-Mel 频谱计算：加窗、朴素 DFT、Mel 滤波器组。
- 音频 Q-former 骨架：给定编码器输出帧，计算 Q、K、V、注意力，发出 N 个 token。
- 玩具任务上级联 vs 端到端比较。

## 交付

本课产出 `outputs/skill-audio-llm-pipeline-picker.md`。给定音频任务（转写、音乐标注、情绪推理、多说话人分离、环境分类），选择级联、端到端 AF3 或混合。

## 练习

1. 计算 30 秒片段在 16kHz、25ms 窗口、10ms 跳跃、80 Mel bins 下的 log-Mel 频谱维度。在 48kHz 下会怎样变化？

2. 为什么 Whisper 在音乐上表现不佳？BEATs 捕获了哪些 Whisper 没有捕获的音频特征？

3. 音频 Q-former 64 query vs 32：在什么任务复杂度下 64 值得？32 在什么情况下节省算力？

4. 阅读 AF3 第 4 节关于按需思考。提出三个思维链帮助最大的音频任务。

5. 使用 AF3 的输出实现一个最小分离管道。如何表示说话人变化？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Log-Mel 频谱 | "Mel 特征" | Mel 滤波器组后 log 幅度值的 2D（时间、频率）数组 |
| 音频 Q-former | "音频 Perceiver" | 从音频编码器输出到喂给 LLM 的固定长度 query 的交叉注意力瓶颈 |
| 级联 | "ASR 然后 LLM" | Whisper 转写然后文本 LLM 推理的管道；丢失声学信息 |
| 端到端 | "音频 LLM" | 音频特征通过 Q-former 直接进入 LLM；保留声学信号 |
| BEATs | "音频 AudioSet 编码器" | 在 AudioSet 上训练的 SSL Transformer；在音乐+环境声音上强 |
| MMAU | "音频推理基准" | 10000 个覆盖语音、音乐、环境的 QA 对；2024 年评估标准 |
| 按需思考 | "音频 CoT" | 模型可选地在最终答案前发出推理 token，提升准确率 3-5 点 |

## 延伸阅读

- [Radford 等人 — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu 等人 — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel 等人 — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang 等人 — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong 等人 — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
