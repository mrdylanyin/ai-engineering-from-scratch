# Async Tasks (SEP-1686) — 立即调用、稍后获取的长时间运行任务

> 真实的 agent 工作需要数分钟到数小时：CI 运行、深度研究综合、批量导出。同步工具调用会断开连接、超时或阻塞 UI。SEP-1686 于 2025-11-25 合并，增加了 Tasks 原语：任何请求都可以被增强为任务（task），结果可以稍后获取或通过状态通知流式传输。漂移风险说明：Tasks 在 2026 年上半年仍处于实验阶段；SDK 接口仍在围绕规范进行设计。

**Type:** Build
**Languages:** Python (stdlib, async task state machine)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 09 (transports)
**Time:** ~75 分钟

## Learning Objectives

- 识别何时将工具从同步提升为 task 增强（服务器端工作超过 30 秒）。
- 遍历任务生命周期：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化任务状态，使崩溃不会丢失进行中的工作。
- 正确轮询 `tasks/status` 并获取 `tasks/result`。

## The Problem

一个 `generate_report` 工具运行一个多分钟的提取管道。同步模型下的选项：

1. 保持连接打开三分钟。远程传输会断开它；客户端超时；UI 冻结。
2. 立即返回一个占位符；要求客户端轮询自定义端点。破坏了 MCP 的统一性。
3. 即发即忘（fire-and-forget）；没有结果。

都不好。SEP-1686 增加了第四个选项：task 增强。任何请求（通常是 `tools/call`）都可以被标记为任务。服务器立即返回一个任务 id。客户端轮询 `tasks/status` 并在完成时获取 `tasks/result`。服务器端状态在重启后仍然存在。

## The Concept

### Task 增强

通过在请求中设置 `params._meta.task.required: true`（或 `optional: true`，由服务器决定）使其成为任务。服务器立即响应：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` 是服务器保留状态的承诺；超过 ttl 后任务结果将被丢弃。

### 每工具选择加入

工具注解可以声明 task 支持：

- `taskSupport: "forbidden"` — 该工具始终以同步方式运行。适用于快速工具。
- `taskSupport: "optional"` — 客户端可以请求 task 增强。
- `taskSupport: "required"` — 客户端必须使用 task 增强。

`generate_report` 工具应该是 `required`。`notes_search` 工具应该是 `forbidden`。

### 状态

```
working  -> input_required -> working  （通过 elicitation 循环）
working  -> completed
working  -> failed
working  -> cancelled
```

状态机是仅追加的：一旦进入 `completed`、`failed` 或 `cancelled`，任务即终止。

### 方法

- `tasks/status {taskId}` — 返回当前状态和进度提示。
- `tasks/result {taskId}` — 阻塞或如果尚未完成则返回 404。
- `tasks/cancel {taskId}` — 幂等操作；终止状态忽略该请求。
- `tasks/list` — 可选；枚举活跃和最近完成的任务。

### 流式状态变更

当服务器支持时，客户端可以订阅状态通知：

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

使用流式传输而非轮询的客户端获得更好的用户体验。轮询始终作为最小接口被支持。

### 持久化状态

规范要求声明 task 支持的服务器持久化状态。崩溃不应丢失 ttl 内已完成的结果。存储范围从 SQLite 到 Redis 再到文件系统。Lesson 13 的工具使用文件系统。

### 取消语义

`tasks/cancel` 是幂等的。如果任务正在执行中，服务器尝试停止（检查执行器协作式取消）。如果已经终止，该请求是空操作。

### 崩溃恢复

当服务器进程重启时：

1. 加载所有持久化的任务状态。
2. 将任何进程已死亡的 `working` 任务标记为 `failed`，错误为 `CRASH_RECOVERY`。
3. 在 ttl 内保留 `completed` / `failed` / `cancelled` 状态。

### 异步任务加 sampling

任务本身可以调用 `sampling/createMessage`。这就是长时间运行的研究任务的工作方式：服务器的任务线程根据需要 sampling 客户端的模型，而客户端的 UI 将任务显示为 `working` 并定期更新进度。

### 为什么这是实验性的

SEP-1686 于 2025-11-25 发布，但更广泛的路线图指出了三个未解决的问题：持久化订阅原语、子任务（父子任务关系）和结果 TTL 标准化。预计规范将在 2026 年继续演进。生产代码应仅在常见场景中将 Tasks 视为稳定的，并对子任务的未来 SDK 变更进行防护。

## Use It

`code/main.py` 实现了一个持久化任务存储（基于文件系统）和一个在后台线程中运行的 `generate_report` 工具。客户端调用该工具，立即获得任务 id，在工作器更新进度时轮询 `tasks/status`，并在完成时获取 `tasks/result`。取消功能可用；崩溃恢复通过杀死工作线程并重新加载状态来模拟。

需要关注的内容：

- 任务状态 JSON 持久化到 `/tmp/lesson-13-tasks/<id>.json`。
- 工作线程更新 `progress` 字段；轮询显示其推进。
- 客户端的取消设置一个事件；工作线程检查并提前退出。
- "崩溃"时的状态重新加载将进行中的任务标记为 `failed`，错误为 `CRASH_RECOVERY`。

## Ship It

本课产出 `outputs/skill-task-store-designer.md`。给定一个长时间运行的工具（研究、构建、导出），该技能设计任务存储（状态形态、ttl、持久化），选择正确的 taskSupport 标志，并草拟进度通知。

## Exercises

1. 运行 `code/main.py`。启动一个 `generate_report` 任务，轮询状态，然后获取结果。

2. 在运行中途添加 `tasks/cancel` 调用。验证工作线程遵守它并且状态变为 `cancelled`。

3. 模拟崩溃恢复：杀死工作线程，重启加载器，观察 `CRASH_RECOVERY` 故障模式。

4. 将存储扩展到 SQLite。持久化优势相同；查询选项开放（列出会话 X 中的所有任务）。

5. 阅读 MCP 2026 年路线图博文。找出一个最可能在未来一年影响 SDK API 设计的 Tasks 相关未解决问题。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Task | "长时间运行的工具调用" | 使用 `_meta.task` 增强的异步执行请求 |
| SEP-1686 | "Tasks 规范" | 在 2025-11-25 添加 Tasks 的规范演进提案 |
| `_meta.task` | "Task 信封" | 包含 id、state、ttl 的每请求元数据 |
| taskSupport | "工具标志" | 每工具 `forbidden` / `optional` / `required` |
| `tasks/status` | "轮询方法" | 获取当前状态和可选进度提示 |
| `tasks/result` | "获取结果" | 返回完成的载荷或如果尚未完成则返回 404 |
| `tasks/cancel` | "停止它" | 幂等取消请求 |
| ttl | "保留预算" | 服务器承诺保留任务状态的毫秒数 |
| `notifications/tasks/updated` | "状态推送" | 服务器发起的状态变更事件 |
| Durable store | "崩溃安全的状态" | 文件系统 / SQLite / Redis 持久化层 |

## Further Reading

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — 原始提案和完整讨论
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — 带有设计原理的演练
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — 机制和状态机
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) — SDK 级别的任务实现模式
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 未解决问题和 2026 年优先级，包括子任务
