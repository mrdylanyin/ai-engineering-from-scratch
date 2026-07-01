# MCP 安全 II — OAuth 2.1、资源指示器、增量作用域

> 远程 MCP 服务器需要的是授权（authorization），而不仅仅是认证（authentication）。2025-11-25 规范与 OAuth 2.1 + PKCE + 资源指示器（RFC 8707）+ 受保护资源元数据（RFC 9728）对齐。SEP-835 增加了增量作用域同意，并在 403 WWW-Authenticate 时进行逐步升级授权。本课将逐步升级流程实现为一个状态机，以便你追踪每一个跳转。

**Type:** Build
**Languages:** Python（stdlib，OAuth 状态机模拟器）
**Prerequisites:** Phase 13 · 09（传输层），Phase 13 · 15（安全 I）
**Time:** ~75 分钟

## Learning Objectives

- 区分资源服务器与授权服务器的职责。
- 走通受 PKCE 保护的 OAuth 2.1 授权码流程。
- 使用 `resource`（RFC 8707）和受保护资源元数据（RFC 9728）防止混淆代理攻击。
- 实现逐步升级授权：服务器以 403 响应并在 WWW-Authenticate 中要求更高作用域；客户端重新提示用户同意并重试。

## 问题所在

早期 MCP（2025 年之前）的远程服务器使用临时 API 密钥甚至完全没有认证。2025-11-25 规范通过完整的 OAuth 2.1 配置方案填补了这一空白。

三个现实需求：

- **普通远程服务器。** 用户安装了一个访问其 Notion / GitHub / Gmail 的远程 MCP 服务器。OAuth 2.1 + PKCE 是正确的方案。
- **作用域升级。** 一个被授予 `notes:read` 的笔记服务器，后续可能需要对某个特定操作使用 `notes:write`。逐步升级（SEP-835）无需重走整个流程，只需请求额外作用域。
- **防止混淆代理。** 客户端持有一个受众限定为服务器 A 的令牌。服务器 A 是恶意的，试图将该令牌出示给服务器 B。资源指示器（RFC 8707）将令牌锁定到其预期受众。

OAuth 2.1 本身并不新鲜。新的是 MCP 的配置方案：特定的必需流程（仅授权码 + PKCE；默认不使用隐式流程，不使用客户端凭证），每个令牌请求必须携带资源指示器，以及发布受保护资源元数据以便客户端知道该去哪里。

## 核心概念

### 角色

- **客户端（Client）。** MCP 客户端（Claude Desktop、Cursor 等）。
- **资源服务器（Resource server）。** MCP 服务器（笔记、GitHub、Postgres 等）。
- **授权服务器（Authorization server）。** 签发令牌。可以与资源服务器是同一服务，也可以是独立的身份提供者（IdP）（Auth0、Keycloak、Cognito）。

在 MCP 的配置方案中，资源服务器和授权服务器可以是同一主机，但 SHOULD 通过 URL 加以区分。

### 授权码 + PKCE

流程如下：

1. 客户端生成 `code_verifier`（随机值）和 `code_challenge`（SHA256）。
2. 客户端将用户重定向到 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意。授权服务器重定向到 `redirect_uri?code=...`。
4. 客户端 POST 到 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`。
5. 授权服务器验证 verifier 的哈希与存储的 challenge 是否匹配，然后签发访问令牌。
6. 客户端使用令牌：对资源服务器的每个请求携带 `Authorization: Bearer ...`。

PKCE 防止授权码拦截攻击。资源指示器防止令牌在其他地方生效。

### 受保护资源元数据（RFC 9728）

资源服务器发布一个 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

客户端从资源服务器发现授权服务器。减少了配置——客户端只需要资源 URL。

### 资源指示器（RFC 8707）

令牌请求中的 `resource` 参数锁定了令牌的预期受众。签发的令牌包含 `aud: "https://notes.example.com"`。另一个收到此令牌的 MCP 服务器检查 `aud` 并拒绝它。

### 作用域模型

作用域是以空格分隔的字符串。常见的 MCP 约定：

- `notes:read`、`notes:write`、`notes:delete`
- `admin:*` 用于管理员功能（谨慎使用）
- `profile:read` 用于身份标识

作用域选择应遵循最小权限原则：先请求当前所需，需要更多时再逐步升级。

### 逐步升级授权（SEP-835）

用户授予了 `notes:read`。之后他们要求代理删除一条笔记。服务器响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

客户端看到 `insufficient_scope` 错误，向用户弹出额外作用域的同意对话框，为其执行一个迷你 OAuth 流程，然后用新令牌重试请求。

### 令牌受众验证

每次请求：服务器检查 `token.aud == self.resource_url`。不匹配 = 401。这阻止了跨服务器令牌复用。

### 短期令牌与轮换

访问令牌 SHOULD 是短期的（默认 1 小时）。刷新令牌在每次刷新时轮换。客户端在后台静默处理刷新。

### 禁止令牌透传

采样服务器（Phase 13 · 11）MUST NOT 将客户端的令牌透传给其他服务。采样请求就是边界。

### 混淆代理防护

令牌绑定到 `aud`。客户端绑定到 `client_id`。每个请求都针对两者进行验证。规范明确禁止了 MCP 之前远程工具生态中常见的"传递令牌"模式。

### 客户端 ID 发现

每个 MCP 客户端在一个固定 URL 发布其元数据。授权服务器可以获取客户端的元数据文档以发现重定向 URI 和联系信息。这消除了手动客户端注册。

### 网关与 OAuth

Phase 13 · 17 展示了企业网关如何处理 OAuth：网关持有上游服务器的凭证，发给客户端的令牌由网关签发，上游令牌永远不会离开网关。这翻转了信任模型——用户只需向网关认证一次；网关处理 N 个服务器的授权。

## Use It

`code/main.py` 将完整的 OAuth 2.1 逐步升级流程模拟为一个状态机。它实现了：

- PKCE code-verifier / challenge 生成。
- 带资源指示器的授权码流程。
- 受保护资源元数据端点。
- 带受众检查的令牌验证。
- 在 `insufficient_scope` 时逐步升级。

本课没有 HTTP 服务器；状态机在内存中运行，以便你追踪每一个跳转。Phase 13 · 17 的网关课程将其连接到实际传输层。

## Ship It

本课产出 `outputs/skill-oauth-scope-planner.md`。给定一个带有工具的远程 MCP 服务器，该技能设计作用域集合、锁定规则和逐步升级策略。

## Exercises

1. 运行 `code/main.py`。追踪两作用域的逐步升级流程。注意哪些跳转在逐步升级时会重复。

2. 添加刷新令牌轮换：每次刷新签发新的刷新令牌并使旧令牌失效。模拟一个被盗的刷新令牌在轮换后被使用，确认它会失败。

3. 使用 stdlib `http.server` 将受保护资源元数据端点实现为真实的 HTTP 响应。镜像第 09 课的 `/mcp` 端点。

4. 为一个 GitHub MCP 服务器设计作用域层次：读取仓库、提交 PR、审批 PR、合并 PR、管理员。在每层之间使用逐步升级。

5. 阅读 RFC 8707 和 RFC 9728。找出 9728 中 MCP 使用方式与 RFC 示例不同的一个字段。（提示：与 `scopes_supported` 有关。）

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| OAuth 2.1 | "现代 OAuth" | 整合后的 RFC，强制要求 PKCE 并禁止隐式流程 |
| PKCE | "持有证明" | Code verifier + challenge，用于防御授权码拦截 |
| Resource indicator | "令牌受众" | RFC 8707 的 `resource` 参数，将令牌锁定到一个服务器 |
| Protected-resource metadata | "发现文档" | RFC 9728 的 `.well-known/oauth-protected-resource` |
| Step-up authorization | "增量同意" | SEP-835 按需添加作用域的流程 |
| `insufficient_scope` | "带 WWW-Authenticate 的 403" | 服务器信号，要求对更大作用域重新同意 |
| Confused deputy | "跨服务令牌复用" | 受信任持有者不当转发令牌的攻击 |
| Short-lived token | "访问令牌 TTL" | 快速过期的 Bearer 令牌；通过刷新令牌续期 |
| Scope hierarchy | "最小权限栈" | 分级的作用域集合，层级间逐步升级 |
| Client ID metadata | "客户端发现文档" | 客户端发布其自身 OAuth 元数据的 URL |

## Further Reading

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 规范的 MCP OAuth 配置方案
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 变更的详细介绍
- [RFC 8707 — Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — 受众锁定 RFC
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 发现文档 RFC
- [Aembit — MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 逐步升级流程的实践演练
