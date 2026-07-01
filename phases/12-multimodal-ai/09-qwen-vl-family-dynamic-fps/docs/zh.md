# Qwen-VL 家族与动态 FPS 视频

> Qwen-VL 家族——Qwen-VL（2023）、Qwen2-VL（2024）、Qwen2.5-VL（2025）、Qwen3-VL（2025）——是 2026 年最具影响力的开源视觉语言模型谱系。每一代都做出了一个决定性的架构赌注，开源生态系统在十二个月内纷纷复制：通过 M-RoPE 实现原生动态分辨率、带绝对时间对齐的动态 FPS 采样、ViT 中的窗口注意力以及结构化代理输出格式。到 Qwen3-VL，配方已经稳定：一个原生宽高比输入的 2D-RoPE-ViT 编码器，一个进入大型 Qwen3 语言基础的 MLP 投影器，以及将 OCR、定位和代理行为作为一等目标强调的训练阶段。本课按时间顺序阅读这个家族，让你理解每个旋钮为什么在那个位置。

**Type:** Learn
**Languages:** Python (stdlib, M-RoPE encoder + dynamic-FPS sampler)
**Prerequisites:** Phase 12 · 06 (patch-n'-pack)
**Time:** ~120 minutes

## 学习目标

- 计算 M-RoPE 的三轴旋转（时间、高度、宽度），并解释为什么需要所有三个轴。
- 为视频选择动态 FPS 采样策略，并推理每秒 token 数 vs 事件检测准确率。
- 说出 Qwen-VL 的四代升级及其启用的功能。
- 构建 Qwen2.5-VL 风格的 JSON 代理输出格式，并解析 VLM 响应的结构化工具调用。

## 问题

Qwen-VL 于 2023 年 8 月发布，作为对 LLaVA-1.5 和 BLIP-2 的直接回应。Qwen 团队瞄准的差距是三点：分辨率、视频和结构化输出。

分辨率：LLaVA-1.5 运行在 336x336。对照片足够，对中文发票或密集电子表格截图没用。Qwen-VL 的第一个创新是 448x448 和基于边界框定位的输出，让模型可以指出物体。

视频：Video-LLaMA 堆叠逐帧编码器并喂给 LLM。对短视频片段有效，但对时间轴才是信号的长视频无效。Qwen 团队想要一个理解时间的单一编码器。

结构化输出：LLaVA 输出自由形式文本。代理需要 JSON。Qwen-VL 在包含边界框坐标作为文本的显式 JSON 输出格式上训练。

每一代 Qwen-VL 都在三个轴之一上延伸。

## 概念

### Qwen-VL（2023 年 8 月）

第一代：OpenCLIP ViT-bigG/14 作为编码器（25 亿参数）、LLama 兼容 Q-Former（1 步，256 query）、Qwen-7B 基础。贡献：

- 448x448 分辨率（当时开源 VLM 的 SOTA）。
- 定位：在包含显式坐标 token 输出的图像-文本对上训练。"The cat is at <box>(112, 204), (280, 344)</box>"。
- 从一开始就是中英文双语训练。

当时基准：英文上与 GPT-4V 竞争，中文上主导。定位监督是真正的头条。

### Qwen2-VL（2024 年 9 月）——M-RoPE 和原生分辨率

Qwen2-VL 用原生动态分辨率 ViT 编码器替换了固定分辨率 + Q-Former 堆栈。主要变化：

- 原生动态分辨率。ViT 接受可被 28 整除的任意 HxW（patch 14，2 倍空间合并）。1120x672（40x24 合并 patch）图像产生 960 个视觉 token。无缩放、无切分、无缩略图。
- M-RoPE（多模态 RoPE）。每个 token 携带 3D 位置 (t, h, w) 而非 1D。对图像 t=0，对视频 t = 帧索引。RoPE 按每个轴频率旋转 query/key 向量。无位置嵌入表。
- MLP 投影器。丢弃 Q-Former；在合并的 patch token 上使用 2 层 MLP。
- 动态 FPS 视频。默认 1-2 FPS，但模型接受任意帧数。

结果：Qwen2-VL-7B 在多个多模态基准上匹配 GPT-4o，在 DocVQA 上击败它（94.5 vs 88.4）。架构变化是决定性动作。

### Qwen2.5-VL（2025 年 2 月）——动态 FPS + 绝对时间

Qwen2.5-VL 的重大转变是视频。动态 FPS 不仅仅是"需要时采样更多帧"。论文形式化了：

- 绝对时间 token。不用位置索引（帧 0、1、2...），使用实际时间戳。"At 0:04, the cat jumps." 模型看到 `<time>0.04</time>` token 与帧 token 交错。
- 动态 FPS。对慢动作镜头采样 1 FPS，对动作采样 4+ FPS。用户或训练者选择；M-RoPE 自适应。
- ViT 中的窗口注意力。空间注意力限制在窗口内（局部块内）以获得吞吐量；每几层加全局注意力。
- 显式 JSON 输出格式。在工具调用数据上训练："{\"tool\": \"click\", \"coords\": [380, 220]}"。开箱即用的代理就绪。
- MRoPE-v2 缩放。位置随最大输入大小缩放，使 10 分钟视频不会用完频率范围。

基准：Qwen2.5-VL-72B 在大多数视频基准上击败 GPT-4o，在文档上匹配 Gemini 2.0，在 GUI 定位上设定开源模型 SOTA（ScreenSpot：84% 准确率 vs GPT-4o 的 38%）。

### Qwen3-VL（2025 年 11 月）

Qwen3-VL 是一个巩固而非重新发明的增量升级：更大的 LLM 骨干（Qwen3-72B）、扩展的训练数据、改进的 OCR、通过 Qwen3"思考模式"增强的推理能力。ViT 和 M-RoPE 保持不变。论文专注于数据和训练改进而非架构。

谱系启示：到 2025 年，Qwen-VL 架构已稳定。后续代次扩展的是算力和数据，而非原语。

### M-RoPE 数学

经典 RoPE 通过位置 `m` 使用成对坐标旋转查询 `q`（维度 `d`）：

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE 将隐藏维度分成三个带。假设 `d = 96`。分配 32 维给时间，32 维给高度，32 维给宽度。每个带按其自己的轴位置旋转。位置 (t=5, h=10, w=20) 的 patch 获得旋转 `R_t(5)`, `R_h(10)`, `R_w(20)` 应用于其三个带。

文本 token 使用 `t = text_index, h = 0, w = 0`（或归一化选择），保持兼容性。视频帧使用 `t = frame_time, h = row, w = col`。单图像使用 `t = 0`。

好处：一种位置编码处理文本、图像和视频，无需分支代码或不同的位置表。

### 动态 FPS 采样逻辑

给定时长 `T` 秒的视频和目标 token 预算 `B`：

1. 计算你能承担的最大 FPS：`fps_max = B / (T * tokens_per_frame)`。
2. 从 `{1, 2, 4, 8}` 中选择满足 `fps <= fps_max` 的目标 FPS。
3. 如果运动高（光流启发式或显式用户请求），选更高 FPS。如果运动低，选更低。
4. 在选定 FPS 下均匀采样；在帧之间插入 `<time>t</time>` token。

Qwen2.5-VL 隐式训练此逻辑；推理时用户通过 `fps` 参数控制。在 4 FPS、每帧 81 token 下的 60 秒动作序列 = 19440 token，在 32k 上下文中可管理。

### 结构化代理输出

Qwen2.5-VL 的代理训练明确目标结构化工具调用：

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

解析是确定性的：对模型输出做 JSON.parse。对比于自由形式 "click at (1024, 512)" 需要正则表达式和歧义处理。这个转变是 Qwen2.5-VL 的 ScreenSpot 分数从 Qwen2-VL 的 55% 跃升至 84% 的原因。

## 使用它

`code/main.py` 实现了：

- 混合文本、图像 patch 和视频帧的打包序列的 M-RoPE 位置计算。
- 动态 FPS 采样器：给定（时长、预算、运动级别），选择 FPS 并输出帧时间戳。
- 一个玩具级 Qwen2.5-VL JSON 输出解析器，处理带坐标字段的工具调用响应。

运行它，然后把固定 FPS 换成动态 FPS 处理一个 5 分钟视频，感受其中的差异。

## 交付

本课产出 `outputs/skill-qwen-vl-pipeline-designer.md`。给定一个视频任务（监控、代理、动作识别、无障碍），输出 Qwen2.5-VL 配置（帧预算、FPS 策略、窗口注意力标志、代理输出模式）和延迟估算。在任何部署 Qwen-VL 家族模型用于视频产品时使用。

## 练习

1. 计算隐藏维度 48（每带 16，基础 theta 10000）下位置 (t=3, h=5, w=7) patch 的 M-RoPE 旋转。展示每带前三对的旋转角度。

2. 10 分钟安全摄像头录制在 1 FPS 下产生多少帧？在 384 分辨率、3 倍池化下总共多少 token？Qwen2.5-VL 的默认 32k 上下文能否处理？

3. 为 30 秒网球对打 vs 30 秒食谱演示 vs 30 秒 UI 代理录制选择 FPS。用动态 FPS 逻辑论证每个。

4. Qwen2.5-VL 完全丢弃了 Q-Former。为什么简单的 MLP 在 2025 年有效但在 2023 年不行？（提示：数据规模和编码器质量。）

5. 将三个 Qwen2.5-VL JSON 工具调用输出解析为 Python dict。什么会在畸形 JSON 上失败，Qwen cookbook 推荐什么恢复策略？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| M-RoPE | "多模态 RoPE" | 隐藏维度中分时间、高度、宽度带的 3D 旋转位置嵌入 |
| 动态 FPS | "智能采样" | 基于运动、时长和 token 预算为每个视频选择的帧采样率 |
| 绝对时间 token | "时间戳 token" |在序列中交错的 `<time>t</time>`，让模型看到实际秒数而非帧索引 |
| 窗口注意力 | "局部注意力" | 空间 self-attention 限制在小窗口内以提升速度；定期添加全局注意力 |
| 结构化代理输出 | "JSON 模式" | 教授 VLM 发出带坐标和工具名称的可解析 JSON 的训练数据监督 |
| min_pixels / max_pixels | "分辨率边界" | Qwen2.5-VL 每请求控制，限制总像素数从而限制 token 数 |
| 定位 | "指出物体" | 将边界框坐标作为文本 token 输出；自 Qwen-VL v1 起使用 |

## 延伸阅读

- [Bai 等人 — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang 等人 — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen 团队 — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen 团队 — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu 等人 — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
