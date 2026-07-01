# Roots and Elicitation — 作用域界定与运行时用户输入

> 硬编码路径在用户打开不同项目的瞬间就会失效。预填充的工具参数在用户描述不充分时就会出错。Roots 将服务器限定在用户控制的一组 URI 中；elicitation 在工具调用中途暂停，通过表单或 URL 向用户请求结构化输入。两个客户端原语，两个常见 MCP 故障模式的修复方案。SEP-1036（URL 模式 elicitation，2025-11-25）在 2026 年上半年仍处于实验阶段 — 依赖前请检查 SDK 版本。

**Type:** Build
**Languages:** Python (stdlib, roots + elicitation demo)
**Prerequisites:** Phase 13 · 07 (MCP server)
**Time:** ~45 分钟

## Learning Objectives

- 声明 `roots` 并响应 `notifications/roots/list_changed`。
- 将服务器文件操作限制在声明的 root 集合内的 URI。
- 使用 `elicitation/create` 在工具调用中途向用户请求确认或结构化输入。
- 在表单模式（form-mode）和 URL 模式（URL-mode）elicitation 之间进行选择（后者是实验性的；已注明漂移风险）。

## The Problem

一个笔记 MCP 服务器在生产环境中遇到的两个具体故障。

**路径假设失效。** 服务器基于 `~/notes` 编写。另一台机器上笔记位于 `~/Documents/Notes` 的用户会遇到静默失败的工具调用（找不到文件），更糟的是，写入到错误位置。

**模型缺失但用户知道的信息。** 用户说"删除旧的 TPS 报告笔记"。模型调用 `notes_delete(title: "TPS report")`，但有三个匹配的笔记分别来自 2023、2024 和 2025 年。工具无法猜测。以"歧义"失败很烦人；在三个上全部执行则是灾难性的。

Roots 修复第一个问题：客户端在 `initialize` 时声明服务器可以访问的 URI 集合。Elicitation 修复第二个问题：服务器暂停工具调用并发送 `elicitation/create` 请求用户选择哪一个。

## The Concept

### Roots

客户端在 `initialize` 时声明 root 列表：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

然后服务器可以调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

服务器必须将 roots 视为边界：任何超出 root 集合的文件读写都将被拒绝。这不是由客户端强制执行的（服务器仍然是用户信任的代码），但符合规范的服务器会遵守这一约定。

当用户添加或删除 root 时，客户端发送 `notifications/roots/list_changed`。服务器重新调用 `roots/list` 并更新其边界。

### 为什么 roots 是客户端原语

Roots 由客户端声明，因为它们代表用户的同意模型。用户告诉 Claude Desktop"给这个笔记服务器访问这两个目录的权限"。服务器不能扩大该范围。

### Elicitation：默认的表单模式

`elicitation/create` 接受一个表单 schema 加上自然语言提示：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "删除 'TPS report'？有多个笔记匹配；请选择一个。",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

客户端渲染一个表单，收集用户的答案，返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

三种可能的操作：`accept`（用户填写了表单）、`decline`（用户关闭了表单）、`cancel`（用户中止了整个工具调用）。

表单 schema 是扁平的 — v1 不支持嵌套对象。SDK 通常会拒绝比单层更复杂的任何内容。

### Elicitation：URL 模式（SEP-1036，实验性）

2025-11-25 新增。服务器不发送 schema，而是发送一个 URL：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "登录 GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

客户端在浏览器中打开 URL，等待完成，用户返回后继续。适用于 OAuth 流程、支付授权和文档签署等表单无法胜任的场景。

漂移风险说明：SEP-1036 的响应形态仍在稳定中；一些 SDK 返回回调 URL，其他返回完成令牌。在生产环境中使用 URL 模式前，请阅读 SDK 的发布说明。

### 何时 elicitation 是正确的工具

- 破坏性操作前的用户确认（destructive hint + elicitation）。
- 消歧（从 N 个匹配中选择一个）。
- 首次运行设置（API 密钥、目录、偏好设置）。
- OAuth 风格的流程（URL 模式）。

### 何时 elicitation 是错误的选择

- 填充工具的必要参数，而模型本可以用文字询问。使用正常的重新提示，而非 elicitation 对话框。
- 高频调用。Elicitation 会打断对话；不要在循环中触发它。
- 服务器可以在事后验证的任何内容。进行验证，返回错误，让模型以文本方式询问用户。

### Human-in-the-loop 桥梁

Elicitation 与 sampling 结合实现了 MCP 的"human-in-the-loop"模型。服务器的 agent 循环可以因用户输入（elicitation）或模型推理（sampling）而暂停。Phase 13 · 11 涵盖了 sampling；本课涵盖 elicitation。将两者结合可实现完整的循环中途控制。

## Use It

`code/main.py` 扩展了笔记服务器，增加了：

- `roots/list` 响应，服务器在 root 列表变更通知后重新查询。
- 一个 `notes_delete` 工具，当多个笔记匹配时使用 `elicitation/create` 进行消歧。
- 一个 `notes_setup` 工具，使用 URL 模式 elicitation 打开首次运行配置页面（模拟）。
- 一个边界检查，拒绝对声明 roots 之外 URI 的操作。

演示运行三个场景：正常路径（一个匹配）、消歧（三个匹配，触发 elicitation）、超出 root 写入（被拒绝）。

## Ship It

本课产出 `outputs/skill-elicitation-form-designer.md`。给定一个可能需要用户确认或消歧的工具，该技能设计 elicitation 表单 schema 和消息模板。

## Exercises

1. 运行 `code/main.py`。触发消歧路径；确认模拟的用户答案被路由回工具。

2. 添加一个新工具 `notes_archive`，每次都需要 elicitation 确认（destructive hint）。检查用户体验：这与模型以文本方式重新询问相比如何？

3. 为首次运行 OAuth 流程实现 URL 模式 elicitation。注意漂移风险并添加 SDK 版本保护。

4. 扩展 `roots/list` 处理：当通知到达时，服务器应以原子方式重新读取并重新扫描可能已超出范围的打开文件句柄。

5. 阅读 GitHub 上 SEP-1036 的 issue 讨论线程。找出一个影响服务器如何处理 URL 模式回调的未解决问题。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Root | "同意边界" | 客户端允许服务器访问的 URI |
| `roots/list` | "服务器请求作用域" | 客户端返回当前 root 集合 |
| `notifications/roots/list_changed` | "用户更改了作用域" | 客户端发出 root 集合已变更的信号 |
| Elicitation | "调用中途询问用户" | 服务器发起的结构化用户输入请求 |
| `elicitation/create` | "该方法" | 用于 elicitation 请求的 JSON-RPC 方法 |
| Form mode | "schema 驱动的表单" | 扁平 JSON Schema 在客户端 UI 中渲染为表单 |
| URL mode | "浏览器重定向" | SEP-1036 实验性；打开 URL 并等待 |
| `accept` / `decline` / `cancel` | "用户响应结果" | 服务器处理的三个分支 |
| Disambiguation | "选择一个" | 当工具有 N 个候选时的常见 elicitation 用例 |
| Flat form | "仅限顶层属性" | Elicitation schema 不能嵌套 |

## Further Reading

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — 规范的 roots 参考
- [MCP — Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation) — 规范的 elicitation 参考
- [Cisco — What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 新增内容演练
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL 模式 elicitation 提案（实验性，存在漂移风险）
- [The New Stack — How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — UX 演练
