# 构建 MCP 服务器 — Python + TypeScript SDK

> 大多数 MCP 教程只展示 stdio 的 hello-world。一个真正的服务器需要暴露 tools、resources 和 prompts，处理能力协商，发出结构化错误，并在不同 SDK 之间保持一致。本课从头到尾构建一个笔记服务器：stdlib stdio 传输层、JSON-RPC 分发、三种服务端原语，以及一种纯函数风格，可以在进阶时直接迁移到 Python SDK 的 FastMCP 或 TypeScript SDK。

**Type:** Build
**Languages:** Python (stdlib, stdio MCP server)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals)
**Time:** ~75 minutes

## Learning Objectives

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写一个分发循环（dispatch loop），从 stdin 读取 JSON-RPC 消息并将响应写入 stdout。
- 按照 JSON-RPC 2.0 规范和 MCP 的附加错误码发出结构化错误响应。
- 将 stdlib 实现迁移到 FastMCP（Python SDK）或 TypeScript SDK，无需重写工具逻辑。

## The Problem

在使用远程传输层（Phase 13 · 09）或认证层（Phase 13 · 16）之前，你需要一个干净的本地服务器。本地意味着 stdio：服务器由客户端作为子进程生成，消息通过 stdin/stdout 以换行符分隔的方式传输。

2025-11-25 规范规定 stdio 消息编码为带有显式 `\n` 分隔符的 JSON 对象。这里不使用 SSE；SSE 是旧的远程模式，正在 2026 年中被移除（Atlassian 的 Rovo MCP 服务器于 2026 年 6 月 30 日弃用；Keboola 于 2026 年 4 月 1 日弃用）。对于 stdio，每行一个 JSON 对象就是整个线路格式。

笔记服务器是一个好的示例，因为它涵盖了所有三种服务端原语。Tools 执行变更操作（`notes_create`）。Resources 暴露数据（`notes://{id}`）。Prompts 提供模板（`review_note`）。本课的结构可以推广到任何领域。

## The Concept

### 分发循环

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要向 stdout 打印任何非 JSON-RPC 信封的内容。调试日志输出到 stderr。
- 每个请求必须（MUST）用带有相同 `id` 的响应来匹配。
- 通知不得（MUST NOT）被响应。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明你支持的功能。客户端依赖能力集来控制功能开关。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}` ，每个条目包含 `name`、`description`、`inputSchema`。`tools/call` 接收 `{name, arguments}` 并返回 `{content: [blocks], isError: bool}`。

内容块是类型化的。最常见的类型：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误有两种形式。协议级错误（未知方法、错误参数）是 JSON-RPC 错误。工具级错误（有效调用但工具执行失败）以 `{content: [...], isError: true}` 返回。这样模型可以在其上下文中看到失败信息。

### 实现 resources

Resources 在设计上是只读的。`resources/list` 返回一个清单；`resources/read` 返回内容。URI 可以是 `file://...`、`http://...` 或自定义 scheme 如 `notes://`。

当你把数据暴露为 resource 而不是 tool 时：

- 模型不会"调用"它；客户端可以在用户请求时将其注入上下文。
- 订阅（subscriptions）允许服务器在资源变更时推送更新（Phase 13 · 10）。
- Phase 13 · 14 通过 `ui://` 扩展了这一功能，用于交互式资源。

### 实现 prompts

Prompts 是带有命名参数的模板。宿主将其作为斜杠命令呈现。一个 `review_note` 提示词可能接受 `note_id` 参数，并生成一个多消息提示词模板，由客户端提供给其模型。

### stdio 传输层细节

- 换行符分隔的 JSON。没有长度前缀的帧。
- 不要缓冲。每次写入后调用 `sys.stdout.flush()`。
- 客户端控制生命周期。当 stdin 关闭（EOF）时，干净退出。
- 不要静默处理 SIGPIPE；记录日志并退出。

### 注解（Annotations）

每个工具可以携带 `annotations` 来描述安全属性：

- `readOnlyHint: true` — 纯读取，可安全重试。
- `destructiveHint: true` — 不可逆的副作用；客户端应确认。
- `idempotentHint: true` — 相同输入产生相同输出。
- `openWorldHint: true` — 与外部系统交互。

客户端使用这些来决定 UX（确认对话框、状态指示器）和路由（Phase 13 · 17）。

### 进阶路径

`code/main.py` 中的 stdlib 服务器大约 180 行。FastMCP（Python）将相同逻辑压缩为装饰器风格：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 有等价的写法。当你准备好时，进阶路径是即插即用的；概念（能力、分发、内容块）是相同的。

## Use It

`code/main.py` 是一个完整的基于 stdio 的笔记 MCP 服务器，仅使用 stdlib。它处理 `initialize`、三个工具（`notes_list`、`notes_search`、`notes_create`）的 `tools/list` 和 `tools/call`、每个笔记的 `resources/list` 和 `resources/read`，以及一个 `review_note` 提示词。你可以通过管道传入 JSON-RPC 消息来驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

需要注意的要点：

- 分发器是一个 `dict[str, Callable]`，以方法名为键。
- 每个工具执行器返回内容块列表，而不是裸字符串。
- 当执行器抛出异常时设置 `isError: true`。

## Ship It

本课产出 `outputs/skill-mcp-server-scaffolder.md`。给定一个领域（笔记、工单、文件、数据库），该 skill 会搭建一个 MCP 服务器，包含正确的 tools / resources / prompts 划分和 SDK 进阶路径。

## Exercises

1. 运行 `code/main.py` 并用手工构建的 JSON-RPC 消息驱动它。执行 `notes_create`，然后用 `resources/read` 检索新笔记。

2. 添加一个 `notes_delete` 工具，带有 `annotations: {destructiveHint: true}`。验证客户端会显示确认对话框（这需要一个真实的宿主；Claude Desktop 可以）。

3. 实现 `resources/subscribe`，使服务器在笔记被修改时推送 `notifications/resources/updated`。添加一个保活任务。

4. 将服务器移植到 FastMCP。Python 文件应该缩减到 80 行以下。线路行为必须相同；用相同的 JSON-RPC 测试工具验证。

5. 阅读规范的 `server/tools` 部分，找出本课服务器中未实现的一个工具定义字段。（提示：有好几个；选一个并实现它。）

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MCP server | "暴露工具的东西" | 通过 stdio 或 HTTP 讲 MCP JSON-RPC 的进程 |
| stdio transport | "子进程模型" | 服务器由客户端生成；通过 stdin/stdout 通信 |
| Dispatcher | "方法路由器" | JSON-RPC 方法名到处理函数的映射 |
| Content block | "工具结果片段" | 工具响应 `content` 数组中的类型化元素 |
| `isError` | "工具级失败" | 表示工具失败；与 JSON-RPC 错误区分 |
| Annotations | "安全提示" | readOnly / destructive / idempotent / openWorld 标志 |
| FastMCP | "Python SDK" | 基于装饰器的 MCP 协议上层框架 |
| Resource URI | "可寻址数据" | `file://`、`db://` 或自定义 scheme 标识的资源 |
| Prompt template | "斜杠命令模板" | 服务器提供的带参数槽位的模板，供宿主 UI 使用 |
| Capability declaration | "功能开关" | 在 `initialize` 中声明的每个原语的标志 |

## Further Reading

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 参考 Python 实现
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — 对应的 TypeScript 实现
- [FastMCP — server framework](https://gofastmcp.com/) — 装饰器风格的 MCP 服务器 Python API
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) — 使用任一 SDK 的端到端教程
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tools/* 消息的完整参考
