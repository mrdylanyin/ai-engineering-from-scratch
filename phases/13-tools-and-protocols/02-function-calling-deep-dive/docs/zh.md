# Function Calling 深入——OpenAI、Anthropic、Gemini

> 三大前沿提供商在 2024 年收敛到了相同的工具调用循环，然后在其他所有方面产生了分歧。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 和唯一 id 关联。本课并排对比三者，让你在一个提供商上编写的代码在移植时不会出问题。

**Type:** Build
**Languages:** Python (stdlib, schema translators)
**Prerequisites:** Phase 13 · 01（工具接口）
**Time:** ~75 分钟

## 学习目标

- 说出 OpenAI、Anthropic 和 Gemini function calling 载荷之间的三个形态差异（声明、调用、结果）。
- 在三种提供商格式之间翻译一个工具声明，并预测严格模式约束的差异点。
- 在每个提供商中使用 `tool_choice` 来强制、禁止或自动选择工具调用。
- 了解每个提供商的硬限制（工具数量、Schema 深度、参数长度）以及违反限制时各自发出的错误特征。

## 问题

Function calling 请求的形态因提供商而异。以下是 2026 年生产栈中的三个具体示例：

**OpenAI Chat Completions / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型的响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是一个你需要解析的 JSON 字符串。严格模式（`strict: true`）通过约束解码强制执行 Schema 合规性。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 的形式返回。`input` 已经是解析好的（对象，不是字符串）。你需要用一个包含 `{type: "tool_result", tool_use_id, content}` 块的新 `user` 消息来回复。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 的形式到达，其中 `id` 在 Gemini 3 及以上版本中是唯一的，用于并行调用关联。你需要用 `{functionResponse: {name, id, response}}` 来回复。

同样的循环。不同的字段名、不同的嵌套方式、不同的字符串与对象约定、不同的关联机制。一个在 OpenAI 上编写天气 agent 的团队，仅为了适配管道就需要两天移植到 Anthropic，再加一天移植到 Gemini。

本课构建一个翻译器，将三种格式统一为一个规范的工具声明，并在边缘进行路由。Phase 13 · 17 将同样的模式泛化为一个 LLM 网关。

## 概念

### 共同结构

每个提供商都需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入 Schema。
2. **工具选择（Tool choice）。** 强制特定工具、禁止工具，或让模型自行决定。
3. **调用发出。** 命名工具和参数的结构化输出。
4. **调用 id。** 将响应关联到正确的调用（对并行调用很重要）。
5. **结果注入。** 将结果绑定回调用的消息或块。

### 逐字段形态对比

| 方面 | OpenAI | Anthropic | Gemini |
|------|--------|-----------|--------|
| 声明信封 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema 字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | assistant 消息上的 `tool_calls[]` | `content[]` 中类型为 `tool_use` 的块 | `parts[]` 中类型为 `functionCall` 的条目 |
| 参数类型 | 字符串化的 JSON | 解析好的对象 | 解析好的对象 |
| Id 格式 | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | role `tool`，`tool_call_id` | `user` 消息中的 `tool_result`，`tool_use_id` | `functionResponse` 中匹配 `id` |
| 强制工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格 Schema | `strict: true` | schema 即契约（始终强制执行） | 请求级别的 `responseSchema` |

### 你会实际遇到的限制

- **OpenAI。** 每个请求最多 128 个工具。Schema 深度 5。参数字符串 <= 8192 字节。严格模式要求无 `$ref`、无重叠的 `oneOf`/`anyOf`/`allOf`、每个属性都列在 `required` 中。
- **Anthropic。** 每个请求最多 64 个工具。Schema 深度实际上无上限，但实际限制约为 10。没有严格模式标志；Schema 是契约，模型倾向于遵守。
- **Gemini。** 每个请求最多 64 个函数。Schema 类型是 OpenAPI 3.0 子集（与 JSON Schema 2020-12 有细微差异）。自 Gemini 3 起并行调用使用唯一 id。

### `tool_choice` 行为

三种模式大家都支持，只是命名不同。

- **Auto。** 模型选择工具或文本。默认。
- **Required / Any。** 模型必须调用至少一个工具。
- **None。** 模型不得调用工具。

再加上每个提供商各自独有的一种模式：

- **OpenAI。** 按名称强制特定工具。
- **Anthropic。** 按名称强制特定工具；`disable_parallel_tool_use` 标志区分单调用和多调用。
- **Gemini。** `mode: "VALIDATED"` 将每个响应通过 Schema 验证器路由，无论模型意图如何。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）在一个 assistant 消息中发出多个调用。你全部运行它们，然后用一个批量 tool-role 消息回复，每个 `tool_call_id` 一个条目。Anthropic 历史上只做单调用；`disable_parallel_tool_use: false`（自 Claude 3.5 起默认）启用多调用。Gemini 2 允许并行调用但不提供稳定 id；Gemini 3 增加了 UUID，使乱序响应可以干净地关联。

### 流式传输

三者都支持流式工具调用。线路格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的 delta 块增量到达。你持续累积直到 `finish_reason: "tool_calls"`。
- **Anthropic。** block-start / block-delta / block-stop 事件。`input_json_delta` 块携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）发出带有 `functionCallId` 的块，使多个并行调用可以交错传输。

Phase 13 · 03 深入讲解并行 + 流式重组。本课聚焦于声明和单调用形态。

### 错误与修复

无效参数错误看起来也不同。

- **OpenAI（非严格模式）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，你注入错误消息并重新调用。
- **OpenAI（严格模式）。** 验证在解码过程中发生；无效 JSON 不可能出现，但 `refusal` 可能出现。
- **Anthropic。** `input` 可能包含意外字段；Schema 是建议性的。在服务端验证。
- **Gemini。** OpenAPI 3.0 怪癖：对象字段上的 `enum` 被静默忽略；需要自行验证。

### 翻译器模式

你的代码中的一个规范工具声明看起来像这样（你选择形态）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数将其翻译为三种提供商形态。`code/main.py` 中的框架正是这样做的，然后通过每个提供商的响应形态对一个假的工具调用进行往返测试。不需要网络——本课教的是形态，而非 HTTP。

生产团队将此翻译器包装在 `AbstractToolSet`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。Phase 13 · 17 发布了一个网关，在三个提供商中的任何一个前面暴露 OpenAI 形态的 API。

## 动手实践

`code/main.py` 定义了一个规范的 `Tool` 数据类和三个翻译器，分别输出 OpenAI、Anthropic 和 Gemini 的声明 JSON。然后它将每种形态的手工构造的提供商响应解析为同一个规范调用对象，证明底层语义是相同的。运行它并并排对比三种声明。

关注点：

- 三个声明块仅在信封和字段名上有差异。
- 三个响应块在调用的存放位置上不同（顶层 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 一个 `canonical_call()` 函数从所有三种响应形态中提取 `{id, name, args}`。

## 发布成果

本课产出 `outputs/skill-provider-portability-audit.md`。给定一个针对单个提供商的 function calling 集成，该技能生成一份可移植性审计报告：它依赖哪些提供商限制、哪些字段需要重命名、以及移植到每个其他提供商时会出什么问题。

## 练习

1. 运行 `code/main.py` 并验证三个提供商的声明 JSON 都序列化了同一个底层 `Tool` 对象。修改规范工具添加一个 enum 参数，确认只有 Gemini 翻译器需要处理 OpenAPI 怪癖。

2. 为每个提供商添加一个 `ListToolsResponse` 解析器，提取模型在 `list_tools` 或发现调用后返回的工具列表。OpenAI 原生没有这个功能；注意这个不对称性。

3. 实现 `tool_choice` 转换：将规范的 `ToolChoice(mode="force", tool_name="x")` 映射到三种提供商形态。然后映射 `mode="any"` 和 `mode="none"`。对照本课的对比表检查。

4. 选择三个提供商之一，从头到尾阅读其 function calling 指南。在其 Schema 规范中找到一个其他两个不支持的字段。候选项：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 编写一个测试向量：一个参数违反声明 Schema 的工具调用。通过每个提供商的验证器运行它（第 01 课的标准库验证器可以作为代理），记录哪些错误被触发。记录你会在生产中使用哪个提供商以获得最严格的一致性。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Function calling | "工具使用" | 提供商级别的结构化工具调用发出 API |
| Tool declaration | "工具规格" | 名称 + 描述 + JSON Schema 输入载荷 |
| `tool_choice` | "强制/禁止" | Auto / required / none / 特定名称模式 |
| Strict mode | "Schema 强制执行" | OpenAI 标志，通过约束解码强制执行 Schema |
| `tool_use` block | "Anthropic 的调用形态" | 包含 id、name、input 的内联内容块 |
| `functionCall` part | "Gemini 的调用形态" | `parts[]` 中包含 name、args 和 id 的条目 |
| Arguments-as-string | "字符串化的 JSON" | OpenAI 将参数作为 JSON 字符串而非对象返回 |
| Parallel tool calls | "一个回合中的扇出" | 一个 assistant 消息中的多个工具调用 |
| Refusal | "模型拒绝" | 严格模式下拒绝块替代调用 |
| OpenAPI 3.0 subset | "Gemini Schema 怪癖" | Gemini 使用一种与 JSON Schema 略有差异的方言 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — 权威参考，包括严格模式和并行调用
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 和 `tool_result` 块语义
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一 id 和 OpenAPI 子集
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企业级接口
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式 Schema 强制执行细节
