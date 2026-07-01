# A2A — Agent-to-Agent 协议

> MCP 是 agent-to-tool（代理到工具）。A2A（Agent2Agent）是 agent-to-agent（代理到代理）——一个开放协议，让基于不同框架构建的不透明代理之间进行协作。由 Google 于 2025 年 4 月发布，2025 年 6 月捐赠给 Linux 基金会，2026 年 4 月达到 v1.0，拥有 150+ 支持者，包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow。它吸收了 IBM 的 ACP 并添加了 AP2 支付扩展。本课走通 Agent Card、Task 生命周期和两种传输绑定。

**Type:** Build
**Languages:** Python（stdlib，Agent Card + Task 工具）
**Prerequisites:** Phase 13 · 06（MCP 基础），Phase 13 · 08（MCP 客户端）
**Time:** ~75 分钟

## Learning Objectives

- 区分 agent-to-tool（MCP）和 agent-to-agent（A2A）的使用场景。
- 在 `/.well-known/agent.json` 发布带有技能和端点元数据的 Agent Card。
- 走通 Task 生命周期（submitted → working → input-required → completed / failed / canceled / rejected）。
- 使用带有 Parts（text、file、data）的 Messages 和 Artifacts 作为输出。

## 问题所在

一个客户服务代理需要将报告撰写委托给一个专门的写作代理。A2A 之前的选项：

- 自定义 REST API。可行但每种配对都是一次性的。
- 共享代码库。要求两个代理运行相同的框架。
- MCP。不合适：MCP 用于调用工具，而不是两个代理在保持各自不透明的内部推理的情况下进行协作。

A2A 填补了这一空白。它将交互建模为一个代理向另一个代理发送 Task，带有生命周期、消息和制品（artifacts）。被调用代理的内部状态保持不透明——调用方只能看到任务状态转换和最终输出。

A2A 是"让跨框架的代理互相通信"的协议。它不替代 MCP；两者是互补的。

## 核心概念

### Agent Card

每个符合 A2A 的代理都在 `/.well-known/agent.json` 发布一张卡片：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现是基于 URL 的：获取卡片，学习 A2A 端点的 URL，枚举技能。

### 签名 Agent Card（AP2）

AP2 扩展（2025 年 9 月）为 Agent Card 添加了加密签名。发布者用 JWT 签署自己的卡片；消费者验证。防止冒充。

### Task 生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (通过消息循环)
```

客户端通过 `tasks/send` 发起。被调用代理在状态之间转换；客户端通过 SSE 订阅状态更新或轮询。

### Messages 和 Parts

一条消息携带一个或多个 Parts：

- `text` — 纯文本内容。
- `file` — base64 编码的数据块，带有 mimeType。
- `data` — 类型化的 JSON 载荷（被调用代理的结构化输入）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifacts

输出是 Artifacts，而非原始字符串。Artifact 是一个命名的、类型化的输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts 可以以块的形式流式传输。调用方累积它们。

### 两种传输绑定

1. **基于 HTTP 的 JSON-RPC。** `/a2a` 端点，POST 用于请求，可选 SSE 用于流式传输。默认绑定。
2. **gRPC。** 适用于 gRPC 原生的企业环境。

两种绑定承载相同的逻辑消息形态。

### 不透明性保持

一个关键设计原则：被调用代理的内部状态是不透明的。调用方看到任务状态和制品。被调用代理的思维链、工具调用、子代理委托——全部不可见。这与 MCP 不同，MCP 中工具调用是透明的。

理由：A2A 让竞争对手能够在不暴露内部细节的情况下进行协作。A2A 可以是"调用这个客户服务代理"，而调用方无需了解该代理如何实现服务。

### 时间线

- **2025-04-09。** Google 宣布 A2A。
- **2025-06-23。** 捐赠给 Linux 基金会。
- **2025-08。** 吸收 IBM 的 ACP。
- **2025-09。** AP2 扩展（Agent Payments）发布。
- **2026-04。** v1.0 发布，150+ 支持组织。

### 与 MCP 的关系

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | 透明的工具调用 | 不透明的内部推理 |
| Typical caller | 代理运行时 | 另一个代理 |
| State | 工具调用结果 | 带生命周期的任务 |
| Authorization | OAuth 2.1（Phase 13 · 16） | JWT 签名的 Agent Card（AP2） |
| Transport | Stdio / Streamable HTTP | 基于 HTTP 的 JSON-RPC / gRPC |

当你想调用特定工具时使用 MCP。当你想把整个任务委托给另一个代理时使用 A2A。许多生产系统同时使用两者：代理使用 MCP 作为其工具层，使用 A2A 作为其协作层。

## Use It

`code/main.py` 实现了一个最小 A2A 工具：一个研究代理发布其卡片，一个写作代理接收一个 `tasks/send`，其中包含 PDF 和文本指令的 parts，经历 working → input_required → working → completed 的状态转换，并返回一个文本 artifact。全部使用 stdlib；使用内存传输以聚焦于消息形态。

需要关注的部分：

- Agent Card 的 JSON 结构。
- Task id 分配和状态转换。
- 混合类型 parts 的 Messages。
- 任务中间的 input-required 分支。
- 完成时的 Artifact 返回。

## Ship It

本课产出 `outputs/skill-a2a-agent-spec.md`。给定一个应该可被其他代理调用的新代理，该技能生成 Agent Card JSON、技能 schema 和端点蓝图。

## Exercises

1. 运行 `code/main.py`。追踪完整的 Task 生命周期，包括被调用代理请求澄清时的 input-required 暂停。

2. 添加签名的 Agent Card。使用卡片规范 JSON 的 HMAC 签名。编写验证器并确认它在被篡改的卡片上失败。

3. 实现任务流式传输：写作代理通过 SSE 发出三个增量 artifact 块，调用方累积它们。

4. 设计一个包装 MCP 服务器的 A2A 代理。将每个 MCP 工具映射到一个 A2A 技能。注意权衡——丢失了什么不透明性？

5. 阅读 A2A v1.0 公告并找出截至 2026 年 4 月尚未被任何框架实现的一个功能。（提示：与多跳任务委托有关。）

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| A2A | "Agent-to-Agent 协议" | 用于不透明代理协作的开放协议 |
| Agent Card | "`.well-known/agent.json`" | 描述代理技能和端点的已发布元数据 |
| Skill | "可调用单元" | 代理支持的一个命名操作（类似于 MCP 工具） |
| Task | "委托单元" | 带有生命周期和最终 artifact 的工作项 |
| Message | "任务输入" | 携带 Parts（text、file、data） |
| Part | "类型化块" | 消息中的 `text` / `file` / `data` 元素 |
| Artifact | "任务输出" | 完成时返回的命名、类型化输出 |
| AP2 | "Agent Payments Protocol" | 用于信任和支付的签名 Agent Card 扩展 |
| Opacity | "黑盒协作" | 被调用代理的内部对调用方隐藏 |
| Input-required | "任务暂停" | 代理需要更多信息时的生命周期状态 |

## Further Reading

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — 规范的 A2A 规范
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — 参考实现和 SDK
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025 年 6 月治理移交
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — 路线图和合作伙伴势头
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 发布说明和向后兼容指南
