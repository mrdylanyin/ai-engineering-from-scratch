# 构建 MCP 客户端 — 发现、调用、会话管理

> 大多数 MCP 内容都在讲服务器教程，对客户端一笔带过。客户端代码才是困难编排所在：进程生成、能力协商、多服务器工具列表合并、采样回调、重连以及命名空间冲突解决。本课构建一个多服务器客户端，将三个不同的 MCP 服务器提升为模型的一个扁平工具命名空间。

**Type:** Build
**Languages:** Python (stdlib, multi-server MCP client)
**Prerequisites:** Phase 13 · 07 (building an MCP server)
**Time:** ~75 minutes

## Learning Objectives

- 将 MCP 服务器作为子进程生成，完成 `initialize`，并发送 `notifications/initialized`。
- 维护每个服务器的会话状态（能力、工具列表、最后收到的通知 id）。
- 将多个服务器的工具列表合并为一个命名空间，并处理冲突。
- 将工具调用路由到拥有它的服务器，并重组响应。

## The Problem

一个真实的 agent 宿主（Claude Desktop、Cursor、Goose、Gemini CLI）会同时加载多个 MCP 服务器。用户可能同时运行一个文件系统服务器、一个 Postgres 服务器和一个 GitHub 服务器。客户端的工作：

1. 生成每个服务器。
2. 独立与每个服务器握手。
3. 对每个服务器调用 `tools/list` 并扁平化结果。
4. 当模型发出 `notes_search` 时，在合并的命名空间中查找并路由到正确的服务器。
5. 处理来自任何服务器的通知（`tools/list_changed`），不阻塞。
6. 在传输层故障时重连。

手动实现所有这些，就是"玩具"和"可用服务"之间的分水岭。官方 SDK 封装了这些，但心智模型必须是你自己的。

## The Concept

### 子进程生成

使用 `subprocess.Popen`，参数为 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。设置 `bufsize=1` 并使用文本模式进行逐行读取。每个服务器是一个进程；客户端为每个服务器持有一个 `Popen` 句柄。

### 每服务器会话状态

每个服务器一个 `Session` 对象，包含：

- `process` — Popen 句柄。
- `capabilities` — 服务器在 `initialize` 时声明的能力。
- `tools` — 最后一次 `tools/list` 的结果。
- `pending` — 请求 id 到等待响应的 promise/future 的映射。

请求本质上是异步的；向服务器 A 发送 `tools/call` 时，服务器 B 可能正在处理中，不能阻塞。可以使用线程加队列，或者 asyncio。

### 合并命名空间

当客户端看到聚合工具列表时，名称可能冲突。两个服务器可能都暴露了 `search`。客户端有三种选择：

1. **按服务器名加前缀。** `notes/search`、`files/search`。清晰但冗长。
2. **静默先到先得。** 后到的服务器的 `search` 覆盖先到的。有风险；隐藏了冲突。
3. **冲突拒绝。** 拒绝加载第二个服务器；通知用户。对安全敏感的宿主最安全。

Claude Desktop 使用按服务器加前缀。Cursor 使用冲突拒绝并给出清晰的错误提示。VS Code MCP 也采用按服务器加前缀。

### 路由

合并之后，一个分发表将 `tool_name -> session` 映射。模型按名称发出调用；客户端找到会话并向该服务器的 stdin 写入 `tools/call` 消息，然后等待响应。

### 采样回调

如果服务器在 `initialize` 时声明了 `sampling` 能力，它可能发送 `sampling/createMessage`，请求客户端运行其 LLM。客户端必须：

1. 阻塞对该服务器的后续请求，直到采样完成；或者如果实现支持并发，则进行流水线处理。
2. 调用其 LLM 提供商。
3. 将响应发送回服务器。

第 11 课全面讲解 sampling。本课为了完整性做了桩实现。

### 通知处理

`notifications/tools/list_changed` 意味着重新调用 `tools/list`。`notifications/resources/updated` 意味着如果资源正在使用中则重新读取。通知不得产生响应 — 不要尝试确认它们。

一个常见的客户端 bug：在 `tools/call` 时阻塞读取循环，而此时通知正坐在流中。使用一个后台读取线程，将每条消息推入队列；主线程从队列中取出并分发。

### 重连

传输层可能失败：服务器崩溃、操作系统杀死了进程、stdio 管道断开。客户端检测到 stdout 的 EOF 并将会话标记为死亡。选项：

- 静默重启服务器并重新握手。适用于纯只读服务器。
- 将故障告知用户。适用于有用户可见会话的有状态服务器。

Phase 13 · 09 讲解 Streamable HTTP 的重连语义；stdio 更简单。

### 保活和会话 id

Streamable HTTP 使用 `Mcp-Session-Id` 头。stdio 没有会话 id — 进程身份就是会话。保活 ping 是可选的；stdio 管道不会因不活动而断开。

## Use It

`code/main.py` 将三个模拟的 MCP 服务器作为子进程生成，与每个服务器握手，合并它们的工具列表，并将工具调用路由到正确的服务器。这些"服务器"实际上是运行玩具响应器的其他 Python 进程（没有真实的 LLM）。运行它可以看到：

- 三次初始化，各自有不同的能力集。
- 三个 `tools/list` 结果合并为 7 个工具的命名空间。
- 基于工具名称的路由决策。
- 通过命名空间前缀防止的冲突。

需要注意的要点：

- `Session` 数据类干净地持有每服务器状态。
- 后台读取线程从 stdout 逐行取出消息，不阻塞主线程。
- 分发表是一个简单的 `dict[str, Session]`。
- 冲突处理是显式的：当两个服务器声明相同名称时，后者会被加上前缀重命名。

## Ship It

本课产出 `outputs/skill-mcp-client-harness.md`。给定一个声明式的 MCP 服务器列表（名称、命令、参数），该 skill 会生成一个工具框架，负责生成服务器、合并工具列表，并提供带冲突解决的路由函数。

## Exercises

1. 运行 `code/main.py` 并观察服务器生成日志。用 SIGTERM 杀死其中一个模拟服务器进程，观察客户端如何检测 EOF 并将该会话标记为死亡。

2. 实现命名空间前缀。当两个服务器暴露 `search` 时，将第二个重命名为 `<server>/search`。更新分发表并验证工具调用路由正确。

3. 为服务器重启添加连接池风格的退避策略：连续失败时指数退避，上限 30 秒，三次失败后向用户发出通知。

4. 设计一个支持 100 个并发 MCP 服务器的客户端。什么数据结构替代简单的分发表？（提示：用于前缀命名空间的 trie，加上每服务器工具数的指标。）

5. 将客户端移植到官方 MCP Python SDK。SDK 封装了 `stdio_client` 和 `ClientSession`。代码应该从约 200 行缩减到约 40 行，同时保留多服务器路由。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MCP client | "agent 宿主" | 生成服务器并编排工具调用的进程 |
| Session | "每服务器状态" | 能力、工具列表和待处理请求的记录 |
| Merged namespace | "统一工具列表" | 所有活跃服务器的扁平工具名称集 |
| Namespace collision | "两个服务器同名工具" | 客户端必须加前缀、拒绝或先到先得处理重复 |
| Routing | "这个调用归谁？" | 从工具名到所属服务器的分发 |
| Background reader | "非阻塞 stdout" | 将服务器 stdout 排入队列的线程或任务 |
| Sampling callback | "LLM 即服务" | 客户端对服务器 `sampling/createMessage` 的处理 |
| `notifications/*_changed` | "原语已变更" | 信号通知客户端必须重新发现或重新读取 |
| Reconnection policy | "服务器挂了怎么办" | 传输层失败时的重启语义 |
| Stdio session | "进程 = 会话" | 没有会话 id；子进程生命周期就是会话 |

## Further Reading

- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — 权威的客户端行为规范
- [MCP — Quickstart client guide](https://modelcontextprotocol.io/quickstart/client) — 使用 Python SDK 的 hello-world 客户端教程
- [MCP Python SDK — client module](https://github.com/modelcontextprotocol/python-sdk) — 参考 `ClientSession` 和 `stdio_client`
- [MCP TypeScript SDK — Client](https://github.com/modelcontextprotocol/typescript-sdk) — TypeScript 对应实现
- [VS Code — MCP in extensions](https://code.visualstudio.com/api/extension-guides/ai/mcp) — VS Code 如何在单个编辑器宿主中多路复用多个 MCP 服务器
