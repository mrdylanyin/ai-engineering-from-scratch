# MCP Resources 和 Prompts — 超越工具（Tools）的上下文暴露

> Tools 占据了 MCP 90% 的关注度。另外两种服务端原语解决的是不同的问题。Resources 暴露数据供读取；prompts 将可复用的模板暴露为斜杠命令。许多服务器应该使用 resources 而不是把读取操作包装在 tools 中，使用 prompts 而不是在工作流中硬编码客户端提示词。本课给出决策规则，并走通 `resources/*` 和 `prompts/*` 消息。

**Type:** Build
**Languages:** Python (stdlib, resource + prompt handler)
**Prerequisites:** Phase 13 · 07 (MCP server)
**Time:** ~45 minutes

## Learning Objectives

- 针对特定领域，决定将能力暴露为 tool、resource 还是 prompt。
- 实现 `resources/list`、`resources/read`、`resources/subscribe` 并处理 `notifications/resources/updated`。
- 实现 `prompts/list` 和 `prompts/get`，支持参数模板。
- 识别宿主何时将 prompts 作为斜杠命令呈现，何时作为自动注入的上下文。

## The Problem

一个天真的笔记应用 MCP 服务器把所有东西都暴露为 tools：`notes_read`、`notes_list`、`notes_search`。这把每次数据访问都包装在模型驱动的工具调用中。后果：

- 模型必须决定是否对每个可能受益于上下文的查询调用 `notes_read`。
- 只读内容无法被订阅或流式传输到宿主的侧边面板。
- 客户端 UI（Claude Desktop 的资源附件面板、Cursor 的"包含文件"选择器）无法呈现这些数据。

正确的划分：将数据暴露为 resource，将变更或计算操作暴露为 tool，将可复用的多步骤工作流暴露为 prompt。每种原语都有其 UX 便利性和访问模式。

## The Concept

### Tools vs resources vs prompts — 决策规则

| Capability | Primitive |
|------------|-----------|
| 用户想要搜索、过滤或转换数据 | tool |
| 用户想要宿主将此数据作为上下文包含 | resource |
| 用户想要一个可以重复运行的模板化工作流 | prompt |

指南：如果模型在每次相关查询中都会受益于调用它，那就是 tool。如果用户会受益于将其附加到对话中，那就是 resource。如果整个多步骤工作流是用户想要重复使用的单元，那就是 prompt。

### Resources

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接收 `{uri}` 并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的内容：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义 scheme）
- `memory://session-2026-04-22/recent`（服务器特定）

`contents[]` 同时支持文本和二进制。二进制使用 `blob` 作为 base64 编码字符串，加上 `mimeType`。

### 资源订阅

在能力中声明 `{resources: {subscribe: true}}`。客户端调用 `resources/subscribe {uri}`。服务器在资源变更时发送 `notifications/resources/updated {uri}`。客户端重新读取。

用例：一个笔记服务器的资源是磁盘上的文件；文件监视器触发更新通知；Claude Desktop 在文件被宿主外部编辑时重新拉取文件到上下文中。

### 资源模板（2025-11-25 新增）

`resourceTemplates` 允许你暴露一个参数化的 URI 模式：`notes://{id}`，其中 `id` 作为补全目标。客户端可以在资源选择器中自动补全 id。

### Prompts

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接收 `{name, arguments}` 并返回 `{description, messages: [{role, content}]}`。

Prompt 是一个模板，填充后生成一个消息列表，由宿主提供给其模型。例如，一个 `code_review` 提示词接受 `file_path` 参数，返回一个三消息序列：一条系统消息、一条包含文件体的用户消息，以及一条带有推理模板的助手启动消息。

### 宿主与 prompts

Claude Desktop、VS Code 和 Cursor 将 prompts 作为聊天 UI 中的斜杠命令呈现。用户输入 `/code_review` 并从表单中选择参数。服务器的 prompt 是"用户快捷方式"和"发送给模型的完整提示词"之间的契约。

并非每个客户端都支持 prompts — 检查能力协商。声明了 prompt 能力的服务器遇到不支持 prompt 的客户端时，只是不会显示斜杠命令。

### "list changed" 通知

Resources 和 prompts 在集合变更时都会发出 `notifications/list_changed`。一个刚导入 20 条新笔记的笔记服务器会发出 `notifications/resources/list_changed`；客户端重新调用 `resources/list` 来获取新增内容。

### 内容类型约定

文本：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
二进制：`image/png`、`application/pdf`，加上 `blob` 字段。
MCP Apps（第 14 课）：`text/html;profile=mcp-app`，使用 `ui://` URI。

### 动态资源

资源 URI 不必对应静态文件。`notes://recent` 可以在每次读取时返回最新的五条笔记。`db://query/users/active` 可以执行参数化查询。服务器可以自由地动态计算内容。

规则：如果客户端可以按 URI 缓存，URI 必须是稳定的。如果计算是一次性的，URI 应该包含时间戳或 nonce，以免客户端缓存过期。

### 订阅 vs 轮询

支持订阅的客户端通过 `notifications/resources/updated` 获得服务器推送。不支持订阅的客户端或宿主通过重新读取来轮询。两者都符合规范。服务器的能力声明告诉客户端它支持哪种方式。

订阅的成本：服务器上的每会话状态（谁订阅了什么）。保持订阅集合有界；断开的客户端应该超时。

### Prompts vs 系统提示词

MCP 中的 prompts 不是系统提示词。宿主的系统提示词（其自身的操作指令）和 MCP prompts（服务器提供的由用户调用的模板）并存。一个行为良好的客户端永远不会让服务器 prompt 覆盖其自身的系统提示词；它将它们分层处理。

## Use It

`code/main.py` 在第 07 课的笔记服务器基础上扩展了：

- 每条笔记的资源（`notes://note-1` 等），支持 `resources/subscribe`。
- 一个 `review_note` 提示词，渲染为三消息模板。
- 一个文件监视器模拟，在笔记被修改时发出 `notifications/resources/updated`。
- 一个 `notes://recent` 动态资源，始终返回最新的五条笔记。

运行演示以查看完整流程。

## Ship It

本课产出 `outputs/skill-primitive-splitter.md`。给定一个拟议的 MCP 服务器，该 skill 将每个能力分类为 tool / resource / prompt，并给出理由。

## Exercises

1. 运行 `code/main.py`。观察初始资源列表，然后触发一次笔记编辑，验证 `notifications/resources/updated` 事件被触发。

2. 添加一个 `resources/list_changed` 发射器：当新笔记被创建时，发送通知让客户端重新发现。

3. 为 GitHub MCP 服务器设计三个提示词：`summarize_pr`、`triage_issue`、`release_notes`。每个都带参数 schema。提示词体应该无需进一步编辑即可运行。

4. 取第 07 课服务器中一个现有的 tool，分类它应该保持为 tool 还是拆分为 resource 加 tool 的组合。用一句话说明理由。

5. 阅读规范的 `server/resources` 和 `server/prompts` 部分。找出 `resources/read` 中一个很少被填充但规范支持的字段。提示：查看资源内容上的 `_meta`。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Resource | "暴露的数据" | 宿主可读取的 URI 可寻址内容 |
| Resource URI | "数据指针" | 带 scheme 前缀的标识符（`file://`、`notes://` 等） |
| `resources/subscribe` | "监听变更" | 客户端选择加入的特定 URI 的服务器推送更新 |
| `notifications/resources/updated` | "资源已变更" | 通知客户端已订阅的资源有新内容 |
| Resource template | "参数化 URI" | 带补全提示的 URI 模式，供宿主选择器使用 |
| Prompt | "斜杠命令模板" | 带参数槽位的命名多消息模板 |
| Prompt arguments | "模板输入" | 宿主在渲染前收集的类型化参数 |
| `prompts/get` | "渲染模板" | 服务器返回填充后的消息列表 |
| Content block | "类型化块" | `{type: text \| image \| resource \| ui_resource}` |
| Slash-command UX | "用户快捷方式" | 宿主将 prompts 作为以 `/` 开头的命令呈现 |

## Further Reading

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) — 资源 URI、订阅和模板
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) — 提示词模板和斜杠命令集成
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — 完整的 `resources/*` 消息参考
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — 完整的 `prompts/*` 消息参考
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) — 对官方文档进行扩展的社区指南
