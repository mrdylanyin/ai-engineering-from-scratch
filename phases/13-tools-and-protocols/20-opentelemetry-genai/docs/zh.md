# OpenTelemetry GenAI — 端到端追踪工具调用

> 一个代理调用了五个工具、三个 MCP 服务器和两个子代理。你需要一条贯穿所有环节的 trace。OpenTelemetry GenAI 语义约定（v1.37 及以上的稳定属性）是 2026 年的标准，被 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 原生支持。本课列出必需属性，走通 span 层次结构（agent → LLM → tool），并提供一个可插入任何 OTel exporter 的 stdlib span 发射器。

**Type:** Build
**Languages:** Python（stdlib，OTel span 发射器）
**Prerequisites:** Phase 13 · 07（MCP 服务器），Phase 13 · 08（MCP 客户端）
**Time:** ~75 分钟

## Learning Objectives

- 列出 LLM span 和工具执行 span 的必需 OTel GenAI 属性。
- 构建覆盖代理循环、LLM 调用、工具调用和 MCP 客户端分发的 trace 层次结构。
- 决定捕获什么内容（opt-in）vs 脱敏（默认）。
- 向本地收集器（Jaeger、Langfuse）发射 span 而无需重写工具代码。

## 问题所在

2026 年 2 月的一个调试案例：用户报告"我的代理有时需要 30 秒响应；有时 3 秒。"没有 trace。日志显示了 LLM 调用，但没有工具分发，没有 MCP 服务器往返，没有子代理。你在猜测。最终你发现：一个 MCP 服务器偶尔在冷启动时挂起。

没有端到端追踪，你无法找到这个问题。OTel GenAI 解决了它。

这些约定在 2025-2026 年间在 OpenTelemetry 语义约定组下稳定下来。它们定义了稳定的属性名称，使 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都解析相同的 span。一次插桩；发送到任何后端。

## 核心概念

### Span 层次结构

```
agent.invoke_agent  (顶层，INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

所有内容都嵌套在一个 trace id 下。Span id 链接父子关系。

### 必需属性

根据 2025-2026 年的语义约定：

- `gen_ai.operation.name` — `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` — 请求的模型字符串（例如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 实际服务的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 提供者响应 id，用于关联。

对于工具 span：

- `gen_ai.tool.name` — 工具标识符。
- `gen_ai.tool.call.id` — 特定的调用 id。
- `gen_ai.tool.description` — 工具描述（可选）。

对于代理 span：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### Span 类型

- `SpanKind.CLIENT` 用于跨越进程边界的调用（LLM 提供者、MCP 服务器）。
- `SpanKind.INTERNAL` 用于代理自身的循环步骤和工具执行。

### 可选内容捕获

默认情况下，span 携带指标和计时——不包含提示或补全。大载荷和 PII 默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 和特定的内容捕获环境变量来包含内容。在生产环境中启用前请仔细审查。

### Span 上的事件

令牌级事件可以作为 span 事件添加：

- `gen_ai.content.prompt` — 输入消息。
- `gen_ai.content.completion` — 输出消息。
- `gen_ai.content.tool_call` — 记录的工具调用。

事件在 span 内按时间排序，用于详细重放。

### Exporter

OTel span 可导出到：

- **Jaeger / Tempo。** 开源，本地部署。
- **Langfuse。** LLM 可观测性专用；可视化令牌使用。
- **Arize Phoenix。** 评估 + 追踪结合。
- **Datadog。** 商业产品；原生解析 `gen_ai.*` 属性。
- **Honeycomb。** 面向列；查询友好。

都使用 OTLP 线路格式。你的代码不关心具体后端。

### 跨 MCP 传播

当 MCP 客户端调用服务器时，将 W3C traceparent 头注入请求中。Streamable HTTP 支持标准头。Stdio 不原生携带 HTTP 头；规范的 2026 路线图讨论了在 JSON-RPC 调用上添加 `_meta.traceparent` 字段。

在该功能发布之前：在每个请求的 `_meta` 中手动包含 traceparent。服务器记录 trace id。

### 指标

除了 span 之外，GenAI 语义约定还定义了指标：

- `gen_ai.client.token.usage` — 直方图。
- `gen_ai.client.operation.duration` — 直方图。
- `gen_ai.tool.execution.duration` — 直方图。

将这些用于不需要每次调用细节的仪表板。

### AgentOps 层

AgentOps（成立于 2024 年）专注于 GenAI 可观测性。它包装了流行框架（LangGraph、Pydantic AI、CrewAI）以自动发射 OTel span。如果你的技术栈使用受支持的框架则很有用；否则使用手动插桩。

## Use It

`code/main.py` 向 stdout 发射 OTel 格式的 span（OTLP-JSON 类似格式），模拟一个调用 LLM、分发两个工具并进行一次 MCP 往返的代理。没有真实的 exporter——本课聚焦于 span 形态和属性集。将输出粘贴到 OTLP 兼容的查看器中或直接阅读。

需要关注的部分：

- Trace id 在所有 span 间共享。
- 父子链接通过 `parentSpanId` 编码。
- 必需的 `gen_ai.*` 属性已填充。
- 内容捕获默认关闭；一个场景通过环境变量开启。

## Ship It

本课产出 `outputs/skill-otel-genai-instrumentation.md`。给定一个代理代码库，该技能生成一份插桩计划：在哪里添加 span、填充哪些属性、以及目标 exporter。

## Exercises

1. 运行 `code/main.py`。计算 span 数量并识别哪些是 CLIENT vs INTERNAL。

2. 开启内容捕获（环境变量）并确认 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件出现。注意对 PII 的影响。

3. 添加工具执行指标 `gen_ai.tool.execution.duration` 并在每次调用时作为直方图样本发射。

4. 将 traceparent 从父代理 span 传播到 MCP 请求的 `_meta.traceparent` 字段。验证 MCP 服务器会看到相同的 trace id。

5. 阅读 OTel GenAI 语义约定规范。找出语义约定中列出但本课代码未发射的一个属性。添加它。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| OTel | "OpenTelemetry" | 用于 trace、指标、日志的开放标准 |
| GenAI semconv | "GenAI 语义约定" | LLM / 工具 / 代理 span 的稳定属性名称 |
| `gen_ai.*` | "属性命名空间" | 所有 GenAI 属性共享此前缀 |
| Span | "计时操作" | 带有开始、结束和属性的工作单元 |
| Trace | "跨 span 血统" | 共享 trace id 的 span 树 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | 关于 span 方向的提示 |
| OTLP | "OpenTelemetry 线路协议" | exporter 的线路格式 |
| Opt-in content | "提示 / 补全捕获" | 默认关闭；通过环境变量启用 |
| traceparent | "W3C 头" | 跨服务传播 trace 上下文 |
| Exporter | "后端特定的发送器" | 将 span 发送到 Jaeger / Datadog 等的组件 |

## Further Reading

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI span、指标和事件的规范约定
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和工具执行 span 属性列表
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — 代理级 `invoke_agent` span
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub 托管的真实来源
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产集成演练
