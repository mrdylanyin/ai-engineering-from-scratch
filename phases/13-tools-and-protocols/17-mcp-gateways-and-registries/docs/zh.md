# MCP 网关与注册表 — 企业控制平面

> 企业不能让每个开发者随意安装 MCP 服务器。网关集中处理认证、RBAC、审计、速率限制、缓存和工具投毒检测，然后将合并后的工具表面暴露为单个 MCP 端点。官方 MCP 注册表（Anthropic + GitHub + PulseMCP + Microsoft，命名空间验证）是规范的上游来源。本课说明网关的定位，走通一个最小实现，并调研 2026 年的厂商格局。

**Type:** Learn
**Languages:** Python（stdlib，最小网关）
**Prerequisites:** Phase 13 · 15（工具投毒），Phase 13 · 16（OAuth 2.1）
**Time:** ~45 分钟

## Learning Objectives

- 解释 MCP 网关的位置（位于 MCP 客户端和多个后端 MCP 服务器之间）。
- 实现网关的五大职责：认证、RBAC、审计、速率限制、策略。
- 在网关层强制执行固定工具哈希清单。
- 区分官方 MCP 注册表与元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题所在

一家财富 500 强企业有 30 个已批准的 MCP 服务器、5000 名开发者、合规和审计要求，以及一个希望集中管理策略的安全团队。让每个开发者在 IDE 中随意安装任意服务器是不可接受的。

网关模式：

1. 网关作为单个 Streamable HTTP 端点运行，开发者连接到此端点。
2. 网关持有每个后端 MCP 服务器的凭证。
3. 每个开发者请求都通过网关自身的 OAuth 进行认证和作用域限定。
4. 网关将调用路由到后端服务器，同时应用策略。
5. 所有调用都记录到审计日志。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway — 都在 2025-2026 年推出了网关或网关功能。

与此同时，官方 MCP 注册表作为规范的上游来源上线：经过策展、命名空间验证、使用反向 DNS 命名的服务器，网关可以从中拉取。元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）则聚合来自多个来源的服务器。

## 核心概念

### 网关的五大职责

1. **认证（Auth）。** 通过 OAuth 2.1 识别开发者身份；映射到用户角色。
2. **RBAC。** 每用户策略：哪些服务器、哪些工具、哪些作用域。
3. **审计（Audit）。** 每次调用都记录谁、做了什么、何时、结果如何。
4. **速率限制（Rate limit）。** 每用户 / 每工具 / 每服务器的上限，防止滥用。
5. **策略（Policy）。** 拒绝投毒描述、执行二要素规则（Rule of Two）、脱敏 PII。

### 网关作为单一端点

对开发者而言，网关看起来像一个 MCP 服务器。内部它将请求路由到 N 个后端。会话 ID（Phase 13 · 09）在边界处被重写。

### 凭证保管

开发者永远看不到后端令牌。网关持有它们（或代理到一个持有凭证的身份提供者）。一个在网关上拥有 `notes:read` 权限的开发者，可能通过网关自身的后端凭证间接访问笔记 MCP 服务器——但仅在绑定该间接访问的策略下。

### 网关层的工具哈希固定

网关持有一份已批准工具描述的清单（SHA256 哈希）。在发现阶段，它获取每个后端的 `tools/list`，将哈希与清单比对，并移除任何描述已变更的工具。这就是 Phase 13 · 15 中地毯拉扯（rug-pull）防御的集中化应用。

### 策略即代码

高级网关使用 OPA/Rego、Kyverno 或 Styra 表达策略。像"用户 `alice` 只能在 `acme` 组织的仓库上调用 `github.open_pr`"这样的规则以声明方式编码。简单网关使用手写 Python。两种形态都是有效的。

### 会话感知路由

当用户的会话包含多个服务器的混合时，网关进行多路复用：开发者的单个 MCP 会话持有 N 个后端会话，每个服务器一个。来自任何后端的通知都通过网关路由到开发者的会话。

### 命名空间合并

网关合并来自所有后端的工具命名空间，通常在冲突时添加前缀。`github.open_pr`、`notes.search`。这使得路由明确无歧义。

### 注册表

- **官方 MCP 注册表（`registry.modelcontextprotocol.io`）。** 在 Anthropic、GitHub、PulseMCP、Microsoft 的管理下推出。命名空间验证（反向 DNS：`io.github.user/server`）。经过基本质量预筛选。
- **Glama。** 以搜索为中心的元注册表，聚合多个来源。
- **MCPMarket。** 偏向商业的目录，包含厂商列表。
- **MCP.so。** 社区目录；开放提交。
- **Smithery。** 包管理器风格的安装流程。
- **LobeHub。** 在其 LobeChat 应用中集成的 UI 注册表。

企业网关默认从官方注册表拉取，允许管理员从元注册表策展添加，并拒绝任何未固定的内容。

### 反向 DNS 命名

官方注册表要求公共服务器使用反向 DNS 名称：`io.github.alice/notes`。命名空间防止抢注，并使信任委托更加清晰。

### 厂商调研，2026 年 4 月

| Vendor | Strength |
|--------|----------|
| Cloudflare MCP Portals | 边缘托管；集成 OAuth；免费层 |
| Kong AI Gateway | K8s 原生；细粒度策略；日志输出到 OpenTelemetry |
| IBM ContextForge | 企业 IAM；合规；审计导出 |
| TrueFoundry | 偏 DevOps；指标优先 |
| MintMCP | 面向开发者平台 |
| Envoy AI Gateway | 开源；可定制过滤器 |

Phase 17（生产基础设施）将深入探讨网关运维。

## Use It

`code/main.py` 用约 150 行代码实现了一个最小网关：通过伪造的 Bearer 令牌认证用户，持有每用户 RBAC 策略，将请求路由到两个后端 MCP 服务器，将每次调用写入审计日志，执行速率限制，并拒绝任何描述哈希与固定清单不匹配的后端工具。

需要关注的部分：

- `RBAC` 字典以 `user_id` 为键，包含允许的 `server_tool` 条目。
- `AUDIT_LOG` 是一个仅追加的事件列表。
- 速率限制使用每用户的令牌桶。
- 固定清单是一个 `server::tool -> hash` 的字典。

## Ship It

本课产出 `outputs/skill-gateway-bootstrap.md`。给定一个企业 MCP 规划（用户、后端、合规），该技能生成一份网关配置规范。

## Exercises

1. 运行 `code/main.py`。以允许的用户发起调用；然后以不允许的用户；然后以超出速率限制的突发请求。验证所有三种流程。

2. 添加一个策略，在返回结果给客户端之前对 PII 进行脱敏。使用简单的正则表达式匹配 SSN 格式的字符串；注意其不足（电子邮件、电话号码）。

3. 扩展审计日志以发出 OpenTelemetry GenAI span。Phase 13 · 20 涵盖了确切的属性。

4. 为一个 50 人开发团队设计 RBAC 策略，包含五个后端（notes、github、postgres、jira、slack）。谁对每个后端只有只读权限？谁有写权限？

5. 从头到尾阅读 Cloudflare 的企业 MCP 文章。找出 Cloudflare 提供的、本 stdlib 网关没有的一个功能。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Gateway | "MCP 代理" | 位于客户端和后端之间的集中化服务器 |
| Credential vaulting | "后端令牌留在服务端" | 开发者永远看不到上游令牌 |
| Session-aware routing | "多后端会话" | 网关为每个开发者会话多路复用 N 个后端会话 |
| Tool-hash pinning | "已批准清单" | 每个已批准工具描述的 SHA256；集中阻止地毯拉扯 |
| RBAC | "每用户策略" | 针对工具和服务器的基于角色的访问控制 |
| Policy-as-code | "声明式规则" | 在网关执行的 OPA/Rego、Kyverno、Styra 策略 |
| Audit log | "谁、做了什么、何时" | 用于合规的仅追加事件日志 |
| Rate limit | "每用户令牌桶" | 每分钟上限以防止滥用 |
| Official MCP Registry | "规范上游" | `registry.modelcontextprotocol.io`，命名空间验证 |
| Reverse-DNS naming | "注册表命名空间" | `io.github.user/server` 约定 |

## Further Reading

- [Official MCP Registry](https://registry.modelcontextprotocol.io/) — 规范上游，命名空间验证
- [Cloudflare — Enterprise MCP](https://blog.cloudflare.com/enterprise-mcp/) — 带 OAuth 和策略的网关模式
- [agentic-community — MCP gateway registry](https://github.com/agentic-community/mcp-gateway-registry) — 开源参考网关
- [TrueFoundry — What is an MCP gateway?](https://www.truefoundry.com/blog/what-is-mcp-gateway) — 功能对比文章
- [IBM — MCP context forge](https://github.com/IBM/mcp-context-forge) — IBM 的企业网关
