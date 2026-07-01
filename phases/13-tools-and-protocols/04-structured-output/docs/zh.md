# 结构化输出——JSON Schema、Pydantic、Zod、约束解码

> "好好请求模型返回 JSON"即使在前言模型上也有 5% 到 15% 的失败率。结构化输出通过约束解码弥合了这一差距：模型被字面上阻止发出违反 Schema 的 token。OpenAI 的 strict mode、Anthropic 的 Schema 类型 tool use、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type` 和 Zod 的 `.parse` 是同一思想的五种表面形式。本课构建 Schema 验证器和严格模式契约，学习者将在每个生产提取管道中使用它们。

**Type:** Build
**Languages:** Python (stdlib, JSON Schema 2020-12 subset)
**Prerequisites:** Phase 13 · 02（function calling 深入）
**Time:** ~75 分钟

## 学习目标

- 为一个提取目标编写 JSON Schema 2020-12，使用正确的约束（enum、min/max、required、pattern）。
- 解释为什么严格模式和约束解码提供的是不同于"生成后验证"的保证。
- 区分三种失败模式：解析错误、Schema 违规、模型拒绝。
- 发布一个带有类型化修复和类型化拒绝处理的提取管道。

## 问题

一个 agent 阅读采购订单邮件需要将自由文本转换为 `{customer, line_items, total_usd}`。三种方法。

**方法一：提示要求 JSON。** "以 JSON 格式回复，字段为 customer、line_items、total_usd。"在前沿模型上 85% 到 95% 的时间有效。有六种失败方式：缺少花括号、尾随逗号、类型错误、幻觉字段、在 token 限制处被截断、泄漏的自然语言如"这是你的 JSON："。

**方法二：生成后验证。** 自由生成、解析、根据 Schema 验证、失败时重试。可靠但昂贵——你要为每次重试付费，截断 bug 每次出现都要多花一个回合。

**方法三：约束解码。** 提供商在解码时强制执行 Schema。无效 token 从采样分布中被屏蔽。输出保证可以解析且保证通过验证。失败塌缩为一种模式：拒绝（模型判断输入不适合该 Schema）。

每个 2026 年的前沿提供商都提供了某种形式的方法三。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}` 加上响应中的 `refusal`（如果模型拒绝）。
- **Anthropic。** 对 `tool_use` 输入的 Schema 强制执行；没有 `stop_reason: "refusal"` 这种东西，但 `end_turn` 且没有工具调用就是信号。
- **Gemini。** 请求级别的 `responseSchema`；2026 年 Gemini 为选定类型提供 token 级别的语法约束。
- **Pydantic AI。** `output_type=InvoiceModel` 发出一个类型化为 `InvoiceModel` 的结构化 `RunResult`。
- **Zod（TypeScript）。** 运行时解析器，根据 Zod Schema 验证提供商输出；与 OpenAI 的 `beta.chat.completions.parse` 配合使用。

共同线索：声明一次 Schema，端到端强制执行。

## 概念

### JSON Schema 2020-12——通用语言

每个提供商都接受 JSON Schema 2020-12。你最常用的构造：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：字段名到子 Schema 的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许的值的封闭集合。
- `minimum` / `maximum`（数字）、`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用于每个数组元素的子 Schema。
- `additionalProperties`：`false` 禁止额外字段（默认值因模式而异）。

OpenAI 严格模式增加了三个要求：每个属性必须列在 `required` 中、所有地方 `additionalProperties: false`、以及无未解析的 `$ref`。如果你违反了这些，API 在请求时返回 400。

### Pydantic，Python 绑定

Pydantic v2 通过 `model_json_schema()` 从数据类形态的模型生成 JSON Schema。Pydantic AI 对此进行了封装，你可以这样写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

agent 框架在边缘将 Schema 翻译为 OpenAI strict mode、Anthropic `input_schema` 或 Gemini `responseSchema`。模型的输出以类型化的 `Invoice` 实例返回。验证错误抛出带有类型化错误路径的 `ValidationError`。

### Zod，TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TypeScript 的等价物。OpenAI 的 Node SDK 暴露了 `zodResponseFormat(Invoice)`，将其翻译为 API 的 JSON Schema 载荷。

### 拒绝

严格模式不能强制模型回答。如果输入无法适配 Schema（"邮件是一首诗，不是发票"），模型会发出一个包含原因的 `refusal` 字段。你的代码必须将其作为一等结果而非失败来处理。拒绝也可以作为安全信号使用：当被要求从受保护内容的邮件中提取信用卡号时，模型返回附带安全原因的拒绝。

### 开放环境中的约束解码

开放权重的实现使用三种技术。

1. **基于语法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：从 Schema 构建确定性有限自动机；在每一步，屏蔽会违反 FSM 的 token 的 logits。
2. **使用 JSON 解析器的 logit 屏蔽**：让流式 JSON 解析器与模型同步运行；在每一步，计算有效的下一个 token 集合。
3. **带验证器的推测解码**：廉价的草稿模型提议 token，验证器强制执行 Schema。

商业提供商在幕后选择其中一种。2026 年的最先进技术在短结构化输出上比纯生成更快，在长输出上速度大致相同。

### 三种失败模式

1. **解析错误。** 输出不是有效的 JSON。在严格模式下不可能发生。在非严格提供商上仍可能发生。
2. **Schema 违规。** 输出已解析但违反了 Schema。在严格模式下不可能发生。在严格模式之外很常见。
3. **拒绝。** 模型拒绝。必须作为类型化结果处理。

### 重试策略

当你在严格模式之外时（Anthropic tool use、非严格 OpenAI、较旧的 Gemini），恢复模式是：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

一次重试通常就够了。三次重试能捕获弱模型的偶发问题。超过三次则是 Schema 不佳的信号：模型对某些输入无法满足它，需要修复提示或 Schema。

### 小模型支持

约束解码在小模型上也能工作。一个 30 亿参数的开放模型配合语法强制执行，在结构化任务上优于一个 700 亿参数模型配合原始提示。这是结构化输出对生产环境如此重要的主要原因：它将可靠性与模型大小解耦。

## 动手实践

`code/main.py` 发布了一个标准库中的最小 JSON Schema 2020-12 验证器（types、required、enum、min/max、pattern、items、additionalProperties）。它包装了一个 `Invoice` Schema，并将一个假的 LLM 输出通过验证器运行，演示解析错误、Schema 违规和拒绝路径。在生产中将假输出替换为任何提供商的真实响应。

关注点：

- 验证器返回一个带有路径和消息的类型化 `[ValidationError]` 列表。这就是你希望传递给重试提示的形态。
- 拒绝分支不会重试。它记录日志并返回类型化的拒绝。Phase 14 · 09 将拒绝用作安全信号。
- `additionalProperties: false` 检查在对抗性测试输入上触发，展示了为什么严格模式要关上幻觉字段的大门。

## 发布成果

本课产出 `outputs/skill-structured-output-designer.md`。给定一个自由文本提取目标（发票、支持工单、简历等），该技能生成一个与严格模式兼容的 JSON Schema 2020-12 以及一个与之镜像的 Pydantic 模型，并附带类型化拒绝和重试处理的存根。

## 练习

1. 运行 `code/main.py`。添加第四个测试用例，其 `total_usd` 为负数。确认验证器以 `minimum` 约束路径拒绝它。

2. 扩展验证器以支持带鉴别器的 `oneOf`。常见情况：`line_item` 是产品或服务，通过 `kind` 标记。严格模式在这里有微妙的规则；查看 OpenAI 的结构化输出指南。

3. 将相同的 Invoice Schema 编写为 Pydantic BaseModel，并将 `model_json_schema()` 输出与你手写的 Schema 进行比较。找出 Pydantic 默认设置但手写版本遗漏的一个字段。

4. 测量拒绝率。构造十个不应可提取的输入（一段歌词、一个数学证明、一封空白邮件），并通过一个启用了严格模式的真实提供商运行它们。统计拒绝数与幻觉输出数。这是你进行拒绝感知重试的真实基准。

5. 从头到尾阅读 OpenAI 的结构化输出指南。找出它明确禁止在严格模式中使用但普通 JSON Schema 允许的一个构造。然后设计一个非本质地使用被禁止构造的 Schema，并将其重构为严格模式兼容的。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| JSON Schema 2020-12 | "Schema 规范" | 每个现代提供商使用的 IETF 草案 Schema 方言 |
| Strict mode | "保证 Schema" | OpenAI 标志，通过约束解码强制执行 Schema |
| Constrained decoding | "Logit 屏蔽" | 解码时强制执行，屏蔽无效的下一个 token |
| Refusal | "模型拒绝" | 输入无法适配 Schema 时的类型化结果 |
| Parse error | "无效 JSON" | 输出未解析为 JSON；严格模式下不可能 |
| Schema violation | "形态错误" | 已解析但违反了 types / required / enum / range |
| `additionalProperties: false` | "不允许额外字段" | 禁止未知字段；OpenAI 严格模式必需 |
| Pydantic BaseModel | "类型化输出" | 发出并验证 JSON Schema 的 Python 类 |
| Zod schema | "TypeScript 输出类型" | 用于提供商输出验证的 TS 运行时 Schema |
| Grammar enforcement | "开放权重约束解码" | 基于 FSM 的 logit 屏蔽，如 outlines / guidance |

## 延伸阅读

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式、拒绝和 Schema 要求
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月发布博文，解释解码保证
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — 序列化为每个提供商的类型化 output_type 绑定
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 权威规范
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企业部署注意事项和严格模式注意事项
