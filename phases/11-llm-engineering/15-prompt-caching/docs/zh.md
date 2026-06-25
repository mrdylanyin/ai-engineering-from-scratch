# 提示缓存与上下文缓存

> 你的系统提示是 4,000 个 token。你的 RAG 上下文是 20,000 个 token。你每次请求都发送这两者。你也为每次请求付费——每次都是。提示缓存（Prompt Caching）让服务商在服务端保持该前缀的KV缓存（KV-cache）热度，并在重用时按正常费率的 10% 收费。正确使用时，它可将推理成本降低 50-90%，首 token 延迟降低 40-85%。

**类型：** 构建
**语言：** Python
**前置知识：** 第 11 阶段 · 01（提示工程），第 11 阶段 · 05（上下文工程），第 11 阶段 · 11（缓存与成本）
**时间：** 约 60 分钟

## 问题

一个编程代理在对话的每一轮都向 Claude 发送相同的 15,000 个 token 系统提示。20 轮对话，输入费率为 $3/百万 token，仅输入成本就达到 $0.90——还不包含用户的任何实际消息。乘以每天 10,000 次对话，账单将达到每天 $9,000，用于从未改变的文本。

你不能缩小提示而不影响质量。你不能避免发送——模型每一轮都需要它。唯一的办法是停止为服务商已经看到的前缀支付全价。

这个办法就是提示缓存。Anthropic 于 2024 年 8 月发布了该功能（2025 年推出了 1 小时扩展 TTL 变体），OpenAI 在同年晚些时候将其自动化，Google 在 Gemini 1.5 的同时发布了显式上下文缓存，到 2026 年，三家厂商都在其前沿模型上将其作为一等特性提供。

## 概念

![提示缓存：一次写入，便宜读取](../assets/prompt-caching.svg)

**机制。** 当请求的前缀与最近一个请求的前缀匹配时，服务商从上次运行的 KV 缓存中提供服务，而不是重新编码 token。你第一次支付少量写入溢价，之后每次都享受大幅读取折扣。

**2026 年三种供应商风格。**

| 供应商 | API 风格 | 命中折扣 | 写入溢价 | 默认 TTL | 最小可缓存量 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 内容块上的显式 `cache_control` 标记 | 输入费用打 1 折 | 25% 附加费 | 5 分钟（可延长至 1 小时） | 1,024 token（Sonnet/Opus），2,048（Haiku） |
| OpenAI | 自动前缀检测 | 输入费用打 5 折 | 无 | 最多 1 小时（尽力而为） | 1,024 token |
| Google（Gemini） | 显式 `CachedContent` API | 按存储计费；读取约正常费率的 25% | 每 token·小时的存储费 | 用户设置（默认 1 小时） | 4,096 token（Flash），32,768（Pro） |

**不变量。** 三家都只缓存前缀。如果任意 token 在请求之间不同，第一个不同 token 之后的所有内容都是未命中。把*稳定*的部分放在顶部，*可变*的部分放在底部。

### 缓存友好的布局

```
[系统提示]          <-- 缓存这部分
[工具定义]          <-- 缓存这部分
[少样本示例]        <-- 缓存这部分
[检索到的文档]      <-- 如果重用则缓存，否则不缓存
[对话历史]          <-- 缓存到上一轮
[当前用户消息]      <-- 永远不缓存（每次都不同）
```

违反这个顺序——把用户消息放在系统提示上方，在少样本示例之间穿插动态检索——缓存就永远不会命中。

### 盈亏平衡计算

Anthropic 25% 的写入溢价意味着缓存块至少需要被读取两次才能净节省成本。1 次写入 + 1 次读取平均每次请求成本为 0.675 倍（节省 32%）；1 次写入 + 10 次读取平均为 0.205 倍（节省 80%）。经验法则：缓存任何你期望在 TTL 内至少重用 3 次的内容。

## 构建它

### 步骤 1：使用显式标记的 Anthropic 提示缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "你是一位资深 Python 审查者。请严格按照评分标准执行。\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记告诉 Anthropic 将此块存储 5 分钟。在该时间窗口内重用触发缓存命中；超时后重用则重新写入。

**响应使用量字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # 按 1.25 倍计费
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # 按 0.1 倍计费
```

在 CI 中检查这两个字段——如果 `cache_read_input_tokens` 在多次请求中始终为零，说明你的缓存键正在发生漂移。

### 步骤 2：1 小时扩展 TTL

对于长时间运行的批处理作业，5 分钟默认 TTL 会在作业之间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 的写入溢价是原来的 2 倍（超过基线的 50% 而非 25%），但在任何重用前缀超过 5 次的批处理作业上很快就能回本。

### 步骤 3：OpenAI 自动缓存

OpenAI 不需要你进行任何配置。任何超过 1,024 个 token 且与最近请求匹配的前缀都会自动获得 5 折优惠。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # 长且稳定
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # 享受折扣的部分
```

同样的缓存友好布局规则适用。有两个因素会破坏 OpenAI 的缓存而不会破坏 Anthropic 的：更改 `user` 字段（用作缓存键的一部分）和重新排序工具。

### 步骤 4：Gemini 显式上下文缓存

Gemini 将缓存视为你创建并命名的头等对象：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["请审查这段代码：\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 在缓存存续期间按每 token·小时收取存储费，读取费用约为正常输入费率的 25%。当你需要跨多天重用在多个会话中使用相同的大型提示时，这是合适的方案。

### 步骤 5：测量生产环境中的命中率

参见 `code/main.py` 获取一个模拟三供应商的核算器，它跟踪写入/读取/未命中计数，并计算每 1000 次请求的综合成本。部署前以目标命中率为门——大多数生产环境的 Anthropic 设置在预热后应达到 80% 以上的读取比例。

## 2026 年仍然存在的主要陷阱

- **动态时间戳放在顶部。** 在系统提示顶部放 `"当前时间：2026-04-22 15:30:02"`。每次请求都未命中。将时间戳移到缓存断点之下。
- **工具重新排序。** 按稳定顺序序列化工具——部署之间的字典重排会破坏所有命中。
- **自由文本近似重复。** "你很乐于助人。" 和 "你是一个乐于助人的助手。"——一个字节的不同 = 完全未命中。
- **块太小。** Anthropic 强制 1,024 token 的下限（Haiku 为 2,048）。更小的块会被静默地不缓存。
- **盲目的成本仪表板。** 将"输入 token"拆分为已缓存和未缓存。否则流量下降可能看起来像缓存胜利。

## 使用它

2026 年缓存技术栈：

| 场景 | 选择 |
|-----------|------|
| 具有稳定 10k+ 系统提示的代理，多轮对话 | Anthropic `cache_control`，5 分钟 TTL |
| 重用前缀超过 30 分钟以上的批处理作业 | Anthropic，`ttl: "1h"` |
| GPT-5 上的无服务器端点，无自定义基础设施 | OpenAI 自动缓存（只需确保前缀稳定且长） |
| 跨多天重用大型代码/文档语料库 | Gemini 显式 `CachedContent` |
| 跨供应商回退 | 保持跨供应商可缓存前缀布局一致，使任何厂商都能命中 |

结合语义缓存（第 11 阶段 · 11）处理用户消息层：提示缓存处理*token 完全相同*的重用，语义缓存处理*语义相同*的重用。

## 交付

保存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: 设计缓存友好的提示布局，并选择正确的供应商缓存模式。
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

给定一个提示（系统 + 工具 + 少样本示例 + 检索 + 历史 + 用户）和使用配置（每小时请求数、需要的 TTL、供应商），输出：

1. 布局。重新排序的各节，标记单个缓存断点；解释哪些节是稳定的，哪些是多变的。
2. 供应商模式。Anthropic cache_control、OpenAI 自动缓存，或 Gemini CachedContent。根据 TTL 和重用模式说明理由。
3. 盈亏平衡。TTL 内每次写入的预期读取次数；净成本与无缓存的对比及计算过程。
4. 验证计划。CI 断言在第二次相同请求时 cache_read_input_tokens > 0；按已缓存 vs 未缓存 token 拆分仪表板。
5. 失败模式。列出在此设置中最可能导致缓存未命中的三个原因（动态时间戳、工具重排、近似重复文本）以及你将如何预防每种情况。

拒绝交付将动态字段放在断点之上的缓存方案。拒绝在没有使 2 倍写入溢价回本的重用次数的情况下启用 1 小时 TTL。
```

## 练习

1. **简单。** 用 Claude 运行一个包含 5,000 token 系统提示的 10 轮对话。分别在不使用 `cache_control` 和使用它的情况下运行。报告每种情况下的输入 token 账单。
2. **中等。** 编写一个测试工具，给定提示模板和请求日志，计算每个供应商的预期命中率和节省的金钱（Anthropic 5 分钟、Anthropic 1 小时、OpenAI 自动、Gemini 显式）。
3. **困难。** 构建一个布局优化器：给定一个提示和一组标记为 `stable=True/False` 的字段，重写提示，将单个缓存断点放置在最大缓存友好位置，同时不丢失信息。在真实的 Anthropic 端点上验证。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 提示缓存（Prompt caching） | "让长提示变便宜" | 为匹配前缀重用服务端 KV 缓存；重复输入 token 可享受 50-90% 折扣。 |
| `cache_control` | "Anthropic 的标记" | 内容块属性，声明"此处之前的所有内容都可缓存"；`{"type": "ephemeral"}`。 |
| 缓存写入（Cache write） | "支付溢价" | 首次填充缓存的请求；Anthropic 按输入费率的 1.25 倍计费，OpenAI 免费。 |
| 缓存读取（Cache read） | "折扣" | 匹配前缀的后续请求；按 10%（Anthropic）、50%（OpenAI）、约 25%（Gemini）计费。 |
| TTL | "它存活多久" | 缓存保持热度的秒数；Anthropic 默认 5 分钟（可延长至 1 小时），OpenAI 尽力最多 1 小时，Gemini 用户设置。 |
| 扩展 TTL | "Anthropic 的 1 小时缓存" | `{"type": "ephemeral", "ttl": "1h"}`；2 倍写入溢价，但对批处理重用物有所值。 |
| 前缀匹配 | "为什么我的缓存未命中" | 缓存仅在从开头到断点的每个 token 完全逐字节相同时才命中。 |
| 上下文缓存（Gemini） | "显式的那个" | Google 的命名、按存储计费的缓存对象；最适合跨多天重用大型语料库。 |

## 扩展阅读

- [Anthropic — 提示缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)——`cache_control`、1 小时 TTL、盈亏平衡表。
- [OpenAI — 提示缓存](https://platform.openai.com/docs/guides/prompt-caching)——自动前缀匹配。
- [Google — 上下文缓存](https://ai.google.dev/gemini-api/docs/caching)——`CachedContent` API 和存储定价。
- [Anthropic 工程 — 长上下文工作负载的提示缓存](https://www.anthropic.com/news/prompt-caching)——原始发布文章，包含延迟数据。
- 第 11 阶段 · 05（上下文工程）——在哪里切割提示以便缓存能够落地。
- 第 11 阶段 · 11（缓存与成本）——将提示缓存与用户消息上的语义缓存配对。
- [Pope 等，"Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102)——提示缓存向用户暴露的 KV 缓存内存模型；解释了为什么缓存前缀的重读比重新计算便宜约 10 倍。
- [Agrawal 等，"SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369)——预填充是提示缓存跳过的阶段；这篇论文解释了为什么 TTFT（首 token 时间）在缓存命中时大幅下降而 TPOT（每输出 token 时间）不受影响。
- [Leviathan 等，"Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192)——提示缓存与推测解码、Flash Attention、MQA/GQA 并列为降低推理成本曲线的杠杆；阅读此篇了解其他三种技术。
