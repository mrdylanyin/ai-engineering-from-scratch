# MCP Sampling — 服务器请求的 LLM 补全与 Agent 循环

> 大多数 MCP 服务器只是被动的执行者：接收参数、运行代码、返回内容。Sampling 让服务器可以反转方向：它请求客户端的 LLM 做出决策。这使得服务器可以托管 agent 循环，而无需持有任何模型凭证。SEP-1577 于 2025-11-25 合并，在 sampling 请求中加入了 tools，使循环可以包含更深层的推理。漂移风险说明：SEP-1577 的 tool-in-sampling 形态在 2026 年 Q1 仍处于实验阶段，SDK API 仍在稳定中。

**Type:** Build
**Languages:** Python (stdlib, sampling harness)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources and prompts)
**Time:** ~75 分钟

## Learning Objectives

- 解释 `sampling/createMessage` 解决了什么问题（服务器托管循环而无需服务器端 API 密钥）。
- 实现一个服务器，请求客户端在多轮 prompt 上进行 sampling 并返回补全结果。
- 使用 `modelPreferences`（成本 / 速度 / 智能优先级）引导客户端模型选择。
- 构建一个 `summarize_repo` 工具，内部通过 sampling 迭代而非硬编码行为。

## The Problem

一个用于代码总结工作流的有用 MCP 服务器需要：遍历文件树、选择要读取的文件、综合生成摘要、然后返回。LLM 推理在哪里发生？

选项 A：服务器调用自己的 LLM。需要 API 密钥，服务器端计费，每个用户成本高昂。

选项 B：服务器返回原始内容；客户端的 agent 进行推理。可行，但将服务器逻辑移入客户端 prompt，这很脆弱。

选项 C：服务器通过 `sampling/createMessage` 请求客户端的 LLM。服务器保留算法（读取哪些文件、执行多少轮），而客户端保留计费和模型选择权。服务器完全不持有任何凭证。

Sampling 就是选项 C。它是一种机制，让受信任的服务器可以托管 agent 循环，而自身不必成为完整的 LLM 宿主。

## The Concept

### `sampling/createMessage` 请求

服务器发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

客户端运行其 LLM，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个浮点数，总和为 1.0：

- `costPriority`：偏好更便宜的模型。
- `speedPriority`：偏好更快的模型。
- `intelligencePriority`：偏好更强大的模型。

加上 `hints`：服务器偏好的命名模型。客户端可能遵守也可能不遵守 hints；客户端的用户配置始终优先。

### `includeContext`

三个值：

- `"none"` — 仅使用服务器提供的消息。默认值。
- `"thisServer"` — 包含来自该服务器会话的先前消息。
- `"allServers"` — 包含所有会话上下文。

`includeContext` 自 2025-11-25 起被软弃用（soft-deprecated），因为它会泄露跨服务器上下文，这是一个安全问题。建议使用 `"none"` 并在消息中传递显式上下文。

### Sampling with tools (SEP-1577)

2025-11-25 新增：sampling 请求可以包含 `tools` 数组。客户端使用这些工具运行完整的工具调用循环。这使服务器可以通过客户端的模型托管 ReAct 风格的 agent 循环。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

客户端循环：sampling，如果被调用则执行工具，再次 sampling，返回最终的 assistant 消息。这在 2026 年 Q1 仍处于实验阶段；SDK 签名可能仍在变化。实现时请对照 2025-11-25 规范的 client/sampling 部分进行确认。

### Human-in-the-loop

客户端必须在运行 sampling 之前向用户展示服务器要求模型执行的操作。恶意服务器可以利用 sampling 操纵用户的会话（"对用户说 X 以便他们点击 Y"）。Claude Desktop、VS Code 和 Cursor 将 sampling 请求显示为用户可以拒绝的确认对话框。

2026 年的共识：没有人类确认的 sampling 是一个危险信号。网关（Phase 13 · 17）可以自动批准低风险的 sampling 并自动拒绝任何可疑请求。

### 无需 API 密钥的服务器托管循环

典型用例：一个自身没有 LLM 访问权限的代码总结 MCP 服务器。它执行：

1. 遍历仓库结构。
2. 调用 `sampling/createMessage`，请求"选择最可能描述该仓库用途的五个文件"。
3. 读取这些文件。
4. 调用 `sampling/createMessage`，传入文件内容并请求"用 3 段话总结该仓库"。
5. 将摘要作为 `tools/call` 结果返回。

服务器从不接触 LLM API。客户端用户使用自己的凭证为补全付费。

### 安全风险（Unit 42 披露，2026 Q1）

- **隐蔽 sampling（Covert sampling）。** 一个工具总是调用 sampling 并附带"从会话上下文中回复用户的电子邮件"。Phase 13 · 15 涵盖了攻击向量。
- **通过 sampling 的资源盗用。** 服务器要求客户端总结攻击者的 payload，向用户计费。
- **循环炸弹（Loop bomb）。** 服务器在紧密循环中调用 sampling。客户端必须强制执行每会话速率限制。

## Use It

`code/main.py` 提供了一个模拟的服务器到客户端的 sampling 工具。一个模拟的"summarize_repo"工具调用两轮 sampling（选择文件，然后总结），模拟客户端返回预设响应。该工具展示了：

- 服务器发送带有 `modelPreferences` 的 `sampling/createMessage`。
- 客户端返回补全结果。
- 服务器继续其循环。
- 速率限制器限制每次工具调用的总 sampling 调用数。

需要关注的内容：

- 服务器仅暴露一个工具（`summarize_repo`）；所有推理都发生在 sampling 调用中。
- 模型偏好对客户端的模型选择进行加权；hints 列出首选模型。
- 循环在 `stopReason: "endTurn"` 时终止。
- `max_samples_per_tool = 5` 限制捕获失控循环。

## Ship It

本课产出 `outputs/skill-sampling-loop-designer.md`。给定一个需要 LLM 调用的服务器端算法（研究、总结、规划），该技能设计一个基于 sampling 的实现，包含适当的 modelPreferences、速率限制和安全确认。

## Exercises

1. 运行 `code/main.py`。将 `max_samples_per_tool` 改为 2，观察速率限制截断。

2. 实现 SEP-1577 的 tool-in-sampling 变体：sampling 请求携带 `tools` 数组。验证客户端循环在返回最终补全之前执行这些工具。注意漂移风险：SDK 签名在 2026 年上半年可能仍会变化。

3. 添加 human-in-the-loop 确认：在服务器的第一次 `sampling/createMessage` 之前暂停并等待用户批准。被拒绝的调用返回类型化的拒绝响应。

4. 添加一个以客户端会话为键的每用户速率限制器。同一用户的同一服务器循环应共享预算。

5. 设计一个 `summarize_pdf` 工具，使用 sampling 选择要包含的块。草拟发送的消息。`modelPreferences.intelligencePriority` 在 0.1 和 0.9 时如何改变行为？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Sampling | "服务器到客户端的 LLM 调用" | 服务器请求客户端模型进行补全 |
| `sampling/createMessage` | "该方法" | 用于 sampling 请求的 JSON-RPC 方法 |
| `modelPreferences` | "模型优先级" | 成本 / 速度 / 智能权重加上名称提示 |
| `includeContext` | "跨会话泄露" | 被软弃用的上下文包含模式 |
| SEP-1577 | "Sampling 中的工具" | 允许在 sampling 中使用工具以实现服务器托管的 ReAct |
| Human-in-the-loop | "用户确认" | 客户端在运行前向用户展示 sampling 请求 |
| Loop bomb | "失控的 sampling" | 服务器端无限 sampling 循环；客户端必须进行速率限制 |
| Covert sampling | "隐藏的推理" | 恶意服务器在 sampling prompt 中隐藏意图 |
| Resource theft | "使用用户的 LLM 预算" | 服务器强制客户端在不需要的 sampling 上花费 |
| `stopReason` | "生成停止的原因" | `endTurn`、`stopSequence` 或 `maxTokens` |

## Further Reading

- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — sampling 的高层概述
- [MCP — Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — 规范的 `sampling/createMessage` 形态
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — sampling 中工具的规范演进提案（实验性）
- [Unit 42 — MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — 隐蔽 sampling 和资源盗用模式
- [Speakeasy — MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling) — 带有客户端代码示例的演练
