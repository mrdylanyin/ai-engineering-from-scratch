# 综合实战 — 构建完整的工具生态系统

> Phase 13 教授了每一个组件。本综合实战将它们连接成一个生产级系统：一个包含 tools + resources + prompts + tasks + UI 的 MCP 服务器、边缘侧的 OAuth 2.1、RBAC 网关、多服务器客户端、A2A 子 agent 调用、OTel 追踪到收集器、CI 中的工具投毒检测，以及 AGENTS.md + SKILL.md 打包。完成后你将能为每一个架构决策做出辩护。

**Type:** Build
**Languages:** Python（stdlib，端到端生态系统演示）
**Prerequisites:** Phase 13 · 01 至 21
**Time:** ~120 分钟

## 学习目标

- 组合一个 MCP 服务器，暴露 tools、resources、prompts 以及一个带 `ui://` 应用的 task。
- 在服务器前部署一个 OAuth 2.1 网关，强制执行 RBAC 和固定哈希（pinned hashes）。
- 编写一个多服务器客户端，端到端使用 OTel GenAI 属性进行追踪。
- 将部分工作负载委托给 A2A 子 agent；验证不透明性（opacity）得以保持。
- 使用 AGENTS.md + SKILL.md 打包整个技术栈，使其他 agent 能够驱动它。

## 问题所在

交付"调研与报告"系统：

- 用户提问："总结 2026 年 arXiv 上 agent 协议领域被引用最多的三篇论文。"
- 系统：通过 MCP 搜索 arXiv；通过 A2A 将论文摘要任务委托给专门的写作 agent；汇总结果；以 MCP Apps `ui://` 资源的形式渲染交互式报告；将每个步骤记录到 OTel。

Phase 13 中的所有原语都会出现。这不是玩具——2026 年 Anthropic（Claude Research 产品）、OpenAI（带 Apps SDK 的 GPTs）和第三方交付的生产级研究助手系统正是这种架构。

## 核心概念

### 架构

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### 追踪层次

```
agent.invoke_agent
 ├── llm.chat (启动)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (不透明的内部状态)
 ├── mcp.call -> tools/call generate_report (task 增强)
 │    └── tasks/status 轮询
 │    └── tasks/result (完成，返回 ui:// 资源)
 └── llm.chat (最终综合)
```

一个 trace id。每个 span 都有正确的 `gen_ai.*` 属性。

### 安全态势

- OAuth 2.1 + PKCE，使用 resource indicator 将 audience 固定到网关。
- 网关持有上游凭据；用户永远看不到它们。
- RBAC：`alice` 拥有 `research:read`、`research:write`，可以调用所有工具。`bob` 拥有 `research:read`，不能调用 `generate_report`。
- 固定的描述清单（pinned description manifest）：丢弃任何工具哈希发生变化的服务器。
- 二因素规则审计：没有工具同时结合不受信任的输入、敏感数据和后果性操作。

### 渲染

最终的 `generate_report` task 返回内容块以及一个 `ui://report/current` 资源。客户端的宿主环境（Claude Desktop 等）在沙箱 iframe 中渲染交互式仪表盘。仪表盘包含排序后的论文列表、引用计数，以及一个按钮，当用户点击某篇论文时调用 `host.callTool('summarize_paper', {arxiv_id})`。

### 打包

整个系统以如下结构交付：

```
research-system/
  AGENTS.md                     # 项目约定
  skills/
    run-research/
      SKILL.md                  # 顶层工作流
  servers/
    research-mcp/               # MCP 服务器
      pyproject.toml
      src/
  agents/
    writer/                     # A2A agent
  gateway/
    config.yaml                 # RBAC + 固定清单
```

用户使用 `docker compose up` 部署。Claude Code、Cursor、Codex 和 opencode 用户可以通过调用 `run-research` skill 来驱动系统。

### Phase 13 各课贡献了什么

| 课程 | 综合实战使用的部分 |
|------|-------------------|
| 01-05 | 工具接口、供应商可移植性、并行调用、schema、lint |
| 06-10 | MCP 原语、服务器、客户端、传输层、resources + prompts |
| 11-14 | Sampling、roots + elicitation、异步 tasks、`ui://` 应用 |
| 15-17 | 工具投毒、OAuth 2.1、网关 + 注册表 |
| 18 | A2A 子 agent 委托 |
| 19 | OTel GenAI 追踪 |
| 20 | LLM 层的路由网关 |
| 21 | SKILL.md + AGENTS.md 打包 |

## 动手实践

`code/main.py` 将前面课程中的模式串联成一个可运行的演示。全部使用标准库，全部在进程内运行，便于你从头到尾阅读。它运行调研与报告场景的完整流程：与网关握手、模拟 OAuth 2.1、合并 tools/list、将 generate_report 作为 task、A2A 调用 writer、返回 ui:// 资源、发出 OTel spans。

重点查看：

- 一个 trace id 贯穿每一次跳转。
- 网关策略阻止第二个用户执行写操作。
- Task 生命周期从 working → completed，并返回文本和 ui:// 内容。
- A2A 调用的内部状态对编排器不可见。
- AGENTS.md 和 SKILL.md 是其他 agent 复现工作流所需的唯一文件。

## 交付成果

本课产出 `outputs/skill-ecosystem-blueprint.md`。给定一个产品需求（调研、摘要、自动化），该 skill 生成完整架构：使用哪些 MCP 原语、哪些网关控制、哪些 A2A 调用、哪些遥测、哪些打包方式。

## 练习

1. 运行 `code/main.py`。注意单一的 trace id 以及 spans 如何嵌套。统计演示触及了 Phase 13 中多少个原语。

2. 扩展演示：添加第二个后端 MCP 服务器（例如 `bibliography`），确认网关将其工具合并到同一命名空间。

3. 用运行在子进程中的真实 A2A writer agent 替换虚假的 writer agent。使用 Lesson 19 的演示框架。

4. 在编排器和 LLM 之间的路由网关中添加 PII 脱敏步骤。确认用户查询中的电子邮件被清除。

5. 为一位将维护此系统的队友编写 AGENTS.md。它应该能在五分钟内读完，并为他们提供在 Cursor 或 Codex 中驱动综合实战所需的一切信息。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Capstone | "Phase-13 集成演示" | 使用所有原语的端到端系统 |
| Research and report | "那个场景" | 搜索、总结、渲染模式 |
| Ecosystem | "所有组件在一起" | 服务器 + 客户端 + 网关 + 子 agent + 遥测 + 打包 |
| Trace hierarchy | "单一 trace id" | 每次跳转的 span 共享 trace；通过 span id 建立父子关系 |
| Gateway-issued token | "传递性认证" | 客户端只看到网关的 token；网关持有上游凭据 |
| Merged namespace | "所有工具在一个扁平列表中" | 网关处的多服务器合并；冲突时加前缀 |
| Opacity boundary | "A2A 调用隐藏内部状态" | 子 agent 的推理过程对编排器不可见 |
| Three-layer stack | "AGENTS.md + SKILL.md + MCP" | 项目上下文 + 工作流 + 工具 |
| Defense-in-depth | "多层安全" | 固定哈希、OAuth、RBAC、二因素规则、审计日志 |
| Spec compliance matrix | "我们交付的与规范要求的对照" | 将交付物映射到 2025-11-25 规范要求的清单 |

## 延伸阅读

- [MCP — 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 合并参考
- [MCP 博客 — 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 协议的未来方向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 参考
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范追踪约定
- [Anthropic — Claude Agent SDK 概览](https://code.claude.com/docs/en/agent-sdk/overview) — 生产级 agent 运行时模式
