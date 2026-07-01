# LLM 路由层 — LiteLLM、OpenRouter、Portkey

> 供应商锁定代价高昂。不同的工具调用场景适合不同的模型。路由网关提供统一的 API 接口、重试、故障转移、成本追踪和安全护栏。2026 年主流的三种形态：LiteLLM（开源自托管）、OpenRouter（托管 SaaS）、Portkey（生产级，2026 年 3 月开源）。本课梳理决策标准，并实现一个基于标准库的路由网关。

**Type:** Learn
**Languages:** Python（stdlib，路由 + 故障转移 + 成本追踪）
**Prerequisites:** Phase 13 · 02（函数调用）、Phase 13 · 17（网关）
**Time:** ~45 分钟

## 学习目标

- 区分自托管、托管和生产级路由方案。
- 实现一个按定义优先级顺序在供应商故障时重试的回退链（fallback chain）。
- 跨供应商追踪每次请求的成本和 token 用量。
- 根据给定的生产约束在 LiteLLM、OpenRouter 和 Portkey 之间做出选择。

## 问题所在

以下场景中供应商路由至关重要：

1. **成本。** Claude Sonnet 的价格是 Haiku 的 3 倍。对于分诊任务，Haiku 就够了；对于综合任务，Sonnet 物有所值。按请求路由。

2. **故障转移。** OpenAI 宕机一小时。所有请求都失败。你希望自动回退到 Anthropic，而无需重新部署。

3. **延迟。** 实时聊天 UI 需要快速的首 token 响应时间。批量摘要器则不需要。按延迟 SLA 路由。

4. **合规。** 欧盟用户必须留在欧盟区域。按区域路由。

5. **实验。** 在同一工作负载上 A/B 测试两个模型。按测试分组路由。

为每个集成手动编写所有这些逻辑非常重复。路由网关提供一个 OpenAI 兼容的 API，其余一切由它处理。

## 核心概念

### OpenAI 兼容代理形态

大家都说 OpenAI 方言。路由网关暴露 `/v1/chat/completions`，接受 OpenAI 的 schema，内部代理到 Anthropic / Gemini / Cohere / Ollama / 任意后端。客户端不关心细节。

### 模型别名

你的代码不再写 `claude-3-5-sonnet-20251022`，而是写 `our_smart_model`。网关将别名映射到真实模型。当 Anthropic 发布 Claude 4 时，你只需在服务端修改别名映射；代码一行都不用动。

### 回退链

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

网关在配置中定义回退链。重试有预算上限，避免回退级联导致成本失控。

### 语义缓存

相同或近似相同的 prompt 命中缓存而非供应商。在重复的 agent 循环中，节省可达 30% 到 60%。缓存键基于 embedding；近似 prompt 共享同一缓存槽位。

### 安全护栏

网关级别：

- **PII 脱敏。** 在发送 prompt 前通过正则或 ML 模型进行脱敏处理。
- **策略违规。** 拒绝包含禁止内容的 prompt。
- **输出过滤。** 对补全内容进行泄露检查。

Portkey 和 Kong 都提供开箱即用的安全护栏。LiteLLM 将其作为可选项。

### 按密钥的速率限制

一个 API 密钥 = 一个团队。按密钥的预算防止某个团队消耗共享配额。大多数网关都支持此功能。

### 自托管 vs 托管的权衡

| 因素 | LiteLLM（自托管） | OpenRouter（托管） | Portkey（生产级） |
|------|-------------------|-------------------|-------------------|
| 代码 | 开源，Python | 托管 SaaS | 开源（2026 年 3 月）+ 托管 |
| 部署 | 部署一个代理 | 注册即用 | 两者皆可 |
| 供应商 | 100+ | 300+ | 100+ |
| 计费 | 使用自己的密钥 | OpenRouter 积分 | 使用自己的密钥 |
| 可观测性 | OpenTelemetry | 仪表盘 | 完整 OTel + PII 脱敏 |
| 最适合 | 需要完全控制的团队 | 快速原型开发 | 有合规要求的生产环境 |

LiteLLM 适合有 SRE 团队且需要数据主权的场景。OpenRouter 适合想要单一订阅、无需管理基础设施的场景。Portkey 适合需要开箱即用的安全护栏和合规能力的场景。

### 成本追踪

每个请求携带 `provider`、`model`、`input_tokens`、`output_tokens`。乘以每个模型每个 token 的价格（从网关维护的价格表中获取）。支持按用户 / 按团队 / 按项目聚合。

### MCP 与路由

网关可以同时路由 LLM 调用和 MCP 采样请求。当采样请求的 modelPreferences 偏好特定模型时，网关会将其翻译到正确的后端。这正是 Phase 13 · 17（MCP 网关）和本课的路由网关有时会合并为一个服务的原因。

### 路由策略

- **静态优先级。** 按列表顺序；出错时回退。
- **负载均衡。** 轮询或加权。
- **成本感知。** 选择满足延迟 / 质量要求的最便宜模型。
- **延迟感知。** 选择过去 N 分钟内最快的模型。
- **任务感知。** prompt 分类器将编码任务路由到一个模型，摘要任务路由到另一个模型。

## 动手实践

`code/main.py` 用约 150 行代码实现了一个路由网关：接受 OpenAI 格式的请求，翻译为各供应商的 stub，运行优先级回退链，追踪每次请求的成本，并对输入进行 PII 脱敏。运行三个场景：正常请求、主供应商宕机触发回退、PII 泄露被脱敏拦截。

重点查看：

- `ROUTES` 字典：别名 -> 按优先级排序的具体供应商列表。
- 回退循环在 5xx 时重试。
- 成本追踪器将 token 用量乘以各模型费率。
- PII 脱敏器在转发前清除 SSN 格式的模式。

## 交付成果

本课产出 `outputs/skill-routing-config-designer.md`。给定工作负载画像（延迟、成本、合规），该 skill 选择 LiteLLM / OpenRouter / Portkey 并生成路由配置。

## 练习

1. 运行 `code/main.py`。触发宕机场景；确认回退落到第二个供应商，且成本正确归属。

2. 添加语义缓存：对 prompt 做 SHA256 作为查找键；缓存命中时直接返回。测量重复调用的成本节省。

3. 添加一个 prompt 分类器，将 "code ..." 开头的 prompt 路由到偏向智能的别名，将 "summarize ..." 开头的 prompt 路由到偏向速度的别名。

4. 设计按团队预算：每个团队有月度消费上限；网关在达到上限后拒绝请求。选择执行粒度（按请求或按时间窗口）。

5. 并排阅读 LiteLLM、OpenRouter 和 Portkey 的文档。指出每个产品各自独有的一个功能。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Routing gateway | "LLM 代理" | 位于多个供应商前面的统一 API 层 |
| OpenAI-compatible | "说 OpenAI 方言" | 接受 `/v1/chat/completions` 格式，翻译到任意后端 |
| Model alias | "our_smart_model" | 代码中的名称，由网关映射到具体模型 |
| Fallback chain | "重试列表" | 故障时按顺序尝试的供应商列表 |
| Semantic caching | "Prompt embedding 缓存" | 缓存键为 prompt 的 embedding；近似重复共享缓存命中 |
| Guardrails | "输入/输出过滤器" | 脱敏 PII，拒绝策略违规 |
| Per-key rate limit | "团队预算" | 限定在 API 密钥范围内的配额 |
| Cost tracking | "按请求花费" | 聚合 token 用量 × 每模型价格 |
| LiteLLM | "开源代理" | 可自托管的开源路由网关 |
| OpenRouter | "托管 SaaS" | 基于积分计费的托管网关 |
| Portkey | "生产级选项" | 开源 + 托管，内置安全护栏 |

## 延伸阅读

- [LiteLLM — 文档](https://docs.litellm.ai/) — 自托管路由网关
- [OpenRouter — 快速入门](https://openrouter.ai/docs/quickstart) — 托管路由 SaaS
- [Portkey — 文档](https://portkey.ai/docs) — 带安全护栏的生产级路由
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — 决策指南
- [Relayplane — LLM 网关对比 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — 供应商调研
