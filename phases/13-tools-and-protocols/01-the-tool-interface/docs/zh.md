# 工具接口（Tool Interface）—— 为什么 Agent 需要结构化 I/O

> 语言模型生成 token，程序执行动作。两者之间的鸿沟就是工具接口：一份让模型请求动作、由宿主执行的契约。2026 年的每一种技术栈——OpenAI、Anthropic、Gemini 的 function calling；MCP 的 `tools/call`；A2A 的 task parts——都是同一个四步循环的不同编码方式。本课命名这个循环，并展示运行它所需的最小机制。

**Type:** Learn
**Languages:** Python (stdlib, no LLM)
**Prerequisites:** Phase 11 (LLM completion APIs)
**Time:** ~45 分钟

## 学习目标

- 解释为什么一个只能生成文本的 LLM 无法独立地对真实世界采取行动。
- 画出四步工具调用循环（描述 → 决策 → 执行 → 观察），并说明每一步由谁负责。
- 将一个工具描述编写为三个部分：名称、JSON Schema 输入、以及确定性执行器函数。
- 区分纯工具（pure）和产生副作用的工具（consequential），并说明这种区分对安全性的重要性。

## 问题

LLM 输出的是下一个 token 的概率分布，这就是它全部的输出面。如果你问一个聊天模型"现在班加罗尔的天气如何"，它可以写出一句看似合理的话，但它无法调用天气 API。这句话可能碰巧是对的，也可能是三天前的过时信息。

弥合这一鸿沟正是工具接口的使命。宿主程序——你的 agent 运行时、Claude Desktop、ChatGPT、Cursor 或自定义脚本——向模型公布一组可调用的工具列表。当模型判断需要执行某个动作时，它会输出一个结构化载荷（payload），指定工具名称及其参数。宿主解析该载荷，真正执行工具，然后将结果反馈给模型。循环持续进行，直到模型判断不再需要更多调用。

这一契约的第一个版本于 2023 年 6 月随 OpenAI 的 "functions" 参数发布。Anthropic 随后在 Claude 2.1 中引入了 `tool_use` 块。Gemini 在几个月后加入了 `functionDeclarations`。如今每个提供商都暴露了相同的形态：JSON Schema 类型的工具列表输入，JSON 载荷的工具调用输出。模型上下文协议（Model Context Protocol，2024 年 11 月）将这一契约泛化，使一个工具注册表可以服务于所有模型。A2A（2026 年 4 月，v1.0）在此基础上为 agent 间委托增加了同样的原语。

四步循环是这一切之下的不变量。Phase 13 的其余内容都是它的展开。

## 概念

### 第一步：描述

宿主用三个字段声明每个工具。

- **名称。** 稳定的、机器可读的标识符。`get_weather`，而不是 "weather thing"。
- **描述。** 一段自然语言简介。"当用户询问特定城市的当前天气时使用。不要用于历史数据。"
- **输入 Schema。** 一个 JSON Schema 对象（draft 2020-12），描述工具的参数。

模型接收该列表。现代提供商使用各自特定的模板将这些声明序列化到系统提示中，因此你作为调用者只需处理结构化形式。

### 第二步：决策

给定用户消息和可用工具，模型选择以下三种行为之一。

1. **直接回答**，以文本形式。不发起工具调用。
2. **调用一个或多个工具。** 输出结构化的调用对象。在 `parallel_tool_calls: true`（OpenAI 和 Gemini 默认开启，Anthropic 需手动开启）下，模型可以在一个回合中发出多个调用。
3. **拒绝。** 严格模式下的结构化输出可以产生类型化的 `refusal` 块，而不是调用。

工具调用载荷有三个稳定字段：调用 `id`、工具 `name`、以及 JSON `arguments` 对象。id 的存在使宿主能够将后续结果与特定调用关联起来，这在并行调用以乱序返回时尤为重要。

### 第三步：执行

宿主接收调用，根据声明的 Schema 验证参数，然后运行执行器。参数无效意味着模型幻觉了一个字段或使用了错误的类型——这在弱模型上是非常常见的失败模式。生产环境中的宿主对无效参数通常做以下三种处理之一：快速失败并将错误反馈给模型、用约束解析器修复 JSON、或在提示中包含验证错误后重试模型。

执行器本身是普通代码。Python、TypeScript、shell 命令、数据库查询。它产生一个结果，通常是字符串，但也可以是任何 JSON 值或结构化内容块（MCP 中的文本、图像或资源引用）。结果必须是可序列化的。

### 第四步：观察

宿主将工具结果追加到对话中（作为带有匹配 `id` 的 `tool` 角色消息），并重新调用模型。此时模型在上下文中拥有了工具输出，可以生成最终答案或请求更多调用。循环持续进行，直到模型停止发出调用或宿主达到迭代次数的安全上限。

### 信任划分

工具在安全性方面分为两类。

- **纯工具（Pure）。** 只读、确定性、无副作用。`get_weather`、`search_docs`、`get_current_time`。可以安全地推测性调用。
- **有后果的工具（Consequential）。** 变更状态、花费资金、触及用户数据。`send_email`、`delete_file`、`execute_trade`。必须进行门控。

Meta 2026 年的 agent 安全"二规则"（Rule of Two）指出，一个回合最多可以组合以下三项中的两项：不可信输入、敏感数据、有后果的动作。工具接口正是你执行这一规则的地方——通过拒绝调用、要求用户确认或提升权限范围。完整的安全章节见 Phase 13 · 15，agent 级权限策略见 Phase 14 · 09。

### 循环在哪里运行

| 场景 | 谁描述 | 谁决策 | 谁执行 |
|------|--------|--------|--------|
| 单轮 function calling（OpenAI/Anthropic/Gemini） | 应用开发者 | LLM | 应用开发者 |
| MCP | MCP 服务器 | LLM（通过 MCP 客户端） | MCP 服务器 |
| A2A | Agent Card 发布者 | 调用方 agent | 被调用方 agent |
| Web 浏览器（function-calling agent） | 浏览器扩展 / WebMCP | LLM | 浏览器运行时 |

无论在哪里，都是同样的四步。列名在变，结构不变。

### 为什么不直接让模型输出 JSON？

"让模型以 JSON 格式回复"是 function calling 之前的做法。在前沿模型上大约有 5% 到 15% 的失败率，在更小的模型上则高得多。失败模式包括缺少花括号、尾随逗号、幻觉字段和类型错误。然后你需要一个 JSON 修复步骤、重试或约束解码器。

原生 function calling 更好，原因有三。首先，提供商针对确切的调用形态对模型进行端到端训练，因此在严格模式下有效 JSON 率可升至 98% 到 99%。其次，调用载荷位于独立的协议槽位中，而非自由文本内——因此工具调用永远不会泄漏到用户可见的回复中。第三，提供商通过约束解码（constrained decoding）强制执行 Schema 合规性（OpenAI 的 strict mode、Anthropic 的 `tool_use`、Gemini 的 `responseSchema`）。输出保证通过验证。

Phase 13 · 02 并排对比三个提供商的 API。Phase 13 · 04 深入讲解结构化输出。

### 熔断器

当模型停止发出调用或宿主达到最大回合数时，循环终止。生产环境的宿主通常将此值设为 5 到 20 个回合。超过这个限度，你几乎肯定陷入了模型无法退出的循环。Claude Code 默认为 20；OpenAI Assistants 为 10；Cursor 的 agent 模式为 25。

另一种选择——无界循环——每隔六个月就会以"agent 一夜之间花了 400 美元 API 调用费"的事后分析形式出现。不要在没有上限的情况下发布。

Phase 14 · 12 深入介绍错误恢复和自愈；Phase 17 介绍生产环境的速率限制。

### Phase 13 后续走向

- 第 02 至 05 课打磨提供商层面的工具调用接口。
- 第 06 至 14 课将循环泛化到 MCP。
- 第 15 至 18 课保护循环免受恶意服务器、对抗性用户和未认证远程认证面的攻击。
- 第 19 至 22 课将模式扩展到 agent 间协作、可观测性、路由和打包。
- 第 23 课发布一个使用所有原语的完整生态系统。

每一节剩余课程都是这个四步循环的展开。请将它作为不变量牢记在心。

## 动手实践

`code/main.py` 在没有 LLM 的情况下运行四步循环。一个假的"决策者"函数通过模式匹配用户消息来模拟模型；执行器、Schema 验证器和观察步骤框架是真实的。运行它可以看到完整的请求/响应编排过程，并可打印中间状态，然后在后续课程中用任何真实提供商替换假决策者。

关注点：

- 工具注册表为每个工具保存三个字段：名称、描述、Schema 和执行器引用。
- 验证器是一个最小的 JSON Schema 子集（types、required、enum、min/max），仅使用标准库。Phase 13 · 04 提供了更完整的版本。
- 循环将迭代次数上限设为 5。生产 agent 正需要这种熔断器。

## 发布成果

本课产出 `outputs/skill-tool-interface-reviewer.md`。给定一个工具定义草稿（名称 + 描述 + Schema + 执行器概要），该技能审计其循环适配性：名称是否机器稳定、描述是否是完整的使用简介、Schema 是否正确使用了 JSON Schema 2020-12、以及纯工具与有后果工具的分类是否明确。

## 练习

1. 在 `code/main.py` 中添加第四个工具 `get_stock_price(ticker)`。将其描述写为"当用户通过股票代码查询当前股价时使用。不要用于历史价格或市场摘要。"运行框架，确认假决策器会将提到股票代码的查询路由到新工具。

2. 破坏 Schema 验证器。传入一个 `arguments` 对象缺少必填字段的调用，确认宿主在执行前拒绝了它。然后传入一个包含额外未知字段的调用。决定：宿主应该拒绝还是忽略？用安全性论据证明你的选择。

3. 将框架中的每个工具分类为纯工具或有后果工具。为需要它的注册表条目添加 `consequential: true` 标志，并修改循环，使其在选择有后果工具时打印"将向用户确认"一行。这就是每个生产宿主都需要的确认门控的形状。

4. 在纸上画出四步循环，并填入上方提供商列对应表中最喜欢的客户端（Claude Desktop、Cursor、ChatGPT 或自定义技术栈）的信息。与 Phase 13 · 06 中 MCP 特定的变体进行交叉参考。

5. 从头到尾阅读 OpenAI 的 function calling 指南。找出存在于请求中但不在本课所述四步循环中的一个字段。解释它增加了什么，以及为什么它是便利性的而非本质性的。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Tool | "模型可以调用的东西" | 名称 + JSON Schema 类型输入 + 执行器函数的三元组 |
| Function calling | "原生工具使用" | 提供商级别的 API 支持，用于发出结构化工具调用而非自然语言 |
| Tool call | "模型的行动请求" | 模型发出的包含 `id`、`name`、`arguments` 的 JSON 载荷 |
| Tool result | "工具返回的内容" | 执行器的输出，包装在带有匹配 id 的 `tool` 角色消息中 |
| Parallel tool calls | "一次多个调用" | 一个模型回合中的多个调用对象，彼此独立，可通过 id 排序 |
| Strict mode | "保证 JSON" | 约束解码，强制模型输出通过声明的 Schema 验证 |
| Pure tool | "只读工具" | 无副作用；可安全重复运行 |
| Consequential tool | "动作工具" | 变更外部状态；需要门控、审计或用户确认 |
| Four-step loop | "工具调用周期" | 描述 → 决策 → 执行 → 观察 |
| Host | "Agent 运行时" | 持有工具注册表、调用模型并运行执行器的程序 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — OpenAI 风格工具声明和调用形态的权威参考
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — Claude 的 `tool_use` / `tool_result` 块格式
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 中的 `functionDeclarations` 和并行调用语义
- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 工具接口的提供商无关泛化
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 每个现代工具 API 使用的 Schema 方言
