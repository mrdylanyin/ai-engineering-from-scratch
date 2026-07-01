# MCP 基础 — 原语、生命周期、JSON-RPC 基础

> MCP 之前的每一次集成都是一次性的。模型上下文协议（Model Context Protocol）最初由 Anthropic 于 2024 年 11 月发布，现由 Linux 基金会的 Agentic AI Foundation 管理，它标准化了发现和调用流程，使任何客户端都能与任何服务器通信。2025-11-25 规范定义了六种原语（三种服务端、三种客户端）、三阶段生命周期和 JSON-RPC 2.0 线路格式。掌握这些内容，本阶段 MCP 章节的其余部分就只是阅读材料了。

**Type:** Learn
**Languages:** Python (stdlib, JSON-RPC parser)
**Prerequisites:** Phase 13 · 01 through 05 (the tool interface and function calling)
**Time:** ~45 minutes

## Learning Objectives

- 列出所有六种 MCP 原语（服务端的 tools、resources、prompts；客户端的 roots、sampling、elicitation）并各举一个用例。
- 走通三阶段生命周期（initialize、operation、shutdown），说明每个阶段由谁发送哪条消息。
- 解析和生成 JSON-RPC 2.0 的请求、响应和通知信封。
- 解释 `initialize` 时的能力协商（capability negotiation）是什么，以及缺少它会出什么问题。

## The Problem

在 MCP 之前，每个使用工具的 agent 都有自己的协议。Cursor 有一个类似 MCP 但不兼容的工具系统。Claude Desktop 用了另一套。VS Code 的 Copilot 扩展又是第三套。一个团队构建了一个"Postgres 查询"工具，却要为三个不同宿主的 API 各写一遍。复用它只能靠复制代码。

结果是一次性集成的寒武纪大爆发，以及生态发展速度的天花板。

MCP 通过标准化线路格式解决了这个问题。一个 MCP 服务器可以在所有 MCP 客户端中运行：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，到 2026 年 4 月已有 300+ 客户端。每月 1.1 亿次 SDK 下载。超过 10,000 个公开服务器。Linux 基金会于 2025 年 12 月在新的 Agentic AI Foundation 下接管了管理权。

本阶段使用的规范版本是 **2025-11-25**。它新增了异步 Tasks（SEP-1686）、URL 模式 elicitation（SEP-1036）、带工具的 sampling（SEP-1577）、增量范围同意（SEP-835）以及 OAuth 2.1 资源指示器语义。Phase 13 · 09 到 16 涵盖这些扩展。本课止步于基础部分。

## The Concept

### 三种服务端原语

1. **Tools（工具）。** 可调用的动作。与 Phase 13 · 01 相同的四步循环。
2. **Resources（资源）。** 暴露的数据。通过 URI 寻址的只读内容：`file:///path`、`db://query/...`、自定义 scheme。
3. **Prompts（提示词模板）。** 可复用的模板。宿主 UI 中的斜杠命令；服务器提供模板，客户端填充参数。

### 三种客户端原语

4. **Roots（根目录）。** 允许服务器访问的 URI 集合。客户端声明，服务器遵守。
5. **Sampling（采样）。** 服务器请求客户端的模型执行一次补全。使服务器端可以在不持有 API 密钥的情况下运行 agent 循环。
6. **Elicitation（信息征询）。** 服务器在运行过程中向客户端用户请求结构化输入。表单或 URL（SEP-1036）。

MCP 中的每个能力都恰好属于这六种原语之一。Phase 13 · 10 到 14 逐一深入讲解。

### 线路格式：JSON-RPC 2.0

每条消息都是包含以下字段的 JSON 对象：

- 请求（Requests）：`{jsonrpc: "2.0", id, method, params}`。
- 响应（Responses）：`{jsonrpc: "2.0", id, result | error}`。
- 通知（Notifications）：`{jsonrpc: "2.0", method, params}` — 没有 `id`，不期望响应。

基础规范约有 15 个方法，按原语分组。重要的方法包括：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务器到客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**阶段 1：initialize（初始化）。**

客户端发送 `initialize`，附带其 `capabilities` 和 `clientInfo`。服务器返回自己的 `capabilities`、`serverInfo` 以及它遵循的规范版本。客户端在消化响应后发送 `notifications/initialized`。从此刻起，任何一方都可以根据协商好的能力发送请求。

**阶段 2：operation（运行）。**

双向通信。客户端调用 `tools/list` 来发现工具，然后调用 `tools/call` 来执行。如果服务器声明了 sampling 能力，它可以发送 `sampling/createMessage`。当服务器的工具集发生变化时，可以发送 `notifications/tools/list_changed`。当用户更改根目录范围时，客户端可以发送 `notifications/roots/list_changed`。

**阶段 3：shutdown（关闭）。**

任一方关闭传输层。MCP 中没有结构化的关闭方法；传输层（stdio 或 Streamable HTTP，见 Phase 13 · 09）承载连接结束信号。

### 能力协商

`initialize` 握手中的 `capabilities` 就是契约。来自服务器的示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务器声明它可以发出 `tools/list_changed` 通知，并支持 `resources/subscribe`。客户端通过声明自己的能力来确认：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端没有声明 `sampling`，服务器就不能调用 `sampling/createMessage`。对称的：如果服务器没有声明 `resources.subscribe`，客户端就不能尝试订阅。

这就是防止生态漂移的机制。不支持 sampling 的客户端仍然是合法的 MCP 客户端；不调用 `sampling` 的服务器仍然是合法的 MCP 服务器。它们只是不一起使用该功能。

### 结构化内容与错误格式

`tools/call` 返回一个类型化块的 `content` 数组：`text`、`image`、`resource`。Phase 13 · 14 将 MCP Apps（`ui://` 交互式 UI）加入该列表。

错误使用 JSON-RPC 错误码。规范定义的附加码：`-32002` "Resource not found"、`-32603` "Internal error"，以及 MCP 特有的错误数据作为 `error.data`。

### 客户端能力 vs 工具调用细节

一个常见的混淆点：`capabilities.tools` 表示客户端是否支持工具列表变更通知。客户端是否会调用特定工具是运行时由模型决定的，不是能力标志。能力标志是规范层面的契约。模型的选择是正交的。

### 为什么用 JSON-RPC 而不是 REST？

JSON-RPC 2.0（2010）是一个轻量级的双向协议。REST 是客户端发起的。MCP 需要服务器发起的消息（sampling、通知），因此 JSON-RPC 的对称请求/响应格式是自然的选择。JSON-RPC 也可以干净地组合在 stdio 和 WebSocket/Streamable HTTP 之上，无需重新发明 HTTP 的请求格式。

```figure
mcp-tool-call
```

## Use It

`code/main.py` 附带一个最小的 JSON-RPC 2.0 解析器和生成器，然后手动走通 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，打印每条消息。没有真实的传输层；只有消息格式。对照"延伸阅读"中链接的规范来验证每个信封。

需要注意的要点：

- `initialize` 双向声明能力；响应中包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回一个 `tools` 数组；每个条目包含 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应的 `content` 是一个 `{type, text}` 块的数组。

## Ship It

本课产出 `outputs/skill-mcp-handshake-tracer.md`。给定一个 MCP 客户端-服务器交互的 pcap 风格记录，该 skill 会标注每条消息属于哪个原语、哪个生命周期阶段，以及依赖哪个能力。

## Exercises

1. 运行 `code/main.py`。找到能力协商发生的那一行，描述如果服务器不声明 `tools.listChanged` 会有什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息格式：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在一个长时间运行的 `tools/call` 执行期间发出该通知，并确认客户端处理器会显示进度条。

3. 从头到尾阅读 MCP 2025-11-25 规范 — 整个文档大约 80 页。找出大多数服务器不需要的那个能力标志。提示：它与资源订阅相关。

4. 在纸上画出一个假设的"定时任务（cron job）"功能属于哪种原语。（提示：服务器希望客户端在预定时间调用它。现有的六种原语都不合适。）MCP 的 2026 路线图中有一个相关的 SEP 草案。

5. 解析 GitHub 上一个公开 MCP 服务器的会话日志。统计请求 vs 响应 vs 通知消息的数量。计算生命周期消息与运行消息的流量占比。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MCP | "模型上下文协议" | 用于模型到工具发现和调用的开放协议 |
| Server primitive | "服务器暴露的内容" | tools（动作）、resources（数据）、prompts（模板） |
| Client primitive | "客户端允许服务器使用的功能" | roots（范围）、sampling（LLM 回调）、elicitation（用户输入） |
| JSON-RPC 2.0 | "线路格式" | 对称的请求/响应/通知信封 |
| `initialize` handshake | "能力协商" | 第一对消息；服务器和客户端声明各自支持的功能 |
| `tools/list` | "发现" | 客户端向服务器请求当前工具集 |
| `tools/call` | "调用" | 客户端请求服务器用参数执行一个工具 |
| `notifications/*_changed` | "变更事件" | 服务器通知客户端其原语列表已变更 |
| Content block | "类型化结果" | 工具结果中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | "规范演进提案" | 命名草案提案（例如 SEP-1686 用于异步 Tasks） |

## Further Reading

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 权威规范文档
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — 六原语心智模型
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月发布文章
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一周年回顾及 2025-11-25 规范变更
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835 和 1724 的摘要
