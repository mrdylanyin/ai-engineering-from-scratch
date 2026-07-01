# 并行工具调用与工具流式传输

> 三次独立的天气查询串行执行就是三次往返。并行运行它们，总时间就塌缩到最慢的那一次调用。每个前沿提供商现在都能在单个回合中发出多个工具调用。收益是实实在在的；管道是微妙的。本课讲解两个部分：并行扇出（fan-out）和流式参数重组，重点关注 id 关联陷阱。

**Type:** Build
**Languages:** Python (stdlib, thread pool + streaming harness)
**Prerequisites:** Phase 13 · 02（function calling 深入）
**Time:** ~75 分钟

## 学习目标

- 解释 `parallel_tool_calls: true` 为什么存在，以及何时应该禁用它。
- 在并行扇出期间将流式参数块关联到正确的工具调用 id。
- 将部分 `arguments` 字符串重组为完整的 JSON，而不提前解析。
- 运行一个三城市天气基准测试，演示串行与并行的延迟差异。

## 问题

没有并行调用时，一个回答"班加罗尔、东京和苏黎世的天气如何"的 agent 会这样做：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

三次 LLM 往返，每次还要额外承担执行器延迟。大约是理想实际耗时的 4 倍。

有了并行调用：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

一次 LLM 往返。执行器时间是三者中的最大值，而非总和。在 OpenAI、Anthropic 和 Gemini 上的生产基准测试显示，扇出工作负载的实际耗时减少了 60% 到 70%。

代价是关联复杂度。当三个调用以乱序完成时，你的结果必须携带匹配的 `tool_call_id`，以便模型能够对齐它们。当结果以流式传输时，你必须在执行前将部分参数片段组装成完整的 JSON。Gemini 3 增加唯一 id，部分原因是为了解决一个现实问题：两个对同一工具的并行调用无法区分。

## 概念

### 启用并行

- **OpenAI。** `parallel_tool_calls: true` 默认开启。设为 `false` 强制串行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false`（Claude 3.5 及以上版本默认）启用并行。设为 `true` 则为串行。
- **Gemini。** 始终支持并行；`tool_config.function_calling_config.mode = "AUTO"` 让模型自行决定。

在以下情况禁用并行：工具之间存在顺序依赖（`create_file` 然后 `write_file`）、一个调用的输出影响另一个调用的输入、或速率限制器无法处理扇出。

### Id 关联

模型发出的每个调用都有一个 `id`。宿主返回的每个结果必须包含相同的 id。否则结果就会产生歧义。

- **OpenAI。** 每个 tool-role 消息上的 `tool_call_id`。
- **Anthropic。** 每个 `tool_result` 块上的 `tool_use_id`。
- **Gemini。** 每个 `functionResponse` 上的 `id`（Gemini 3 及以上；Gemini 2 按名称匹配，同名并行调用会出错）。

### 并发运行调用

宿主在每个调用的执行器上运行自己的线程、协程或远程 worker。最简单的框架使用线程池；生产环境使用 asyncio 配合 `asyncio.gather` 或结构化并发。完成顺序是不可预测的——id 是唯一的标识符。

一个常见 bug：按调用列表顺序而非完成顺序回复结果。这通常能工作，因为模型只关心 `tool_call_id`，但如果结果被丢弃或重复，乱序提交会使调试更加困难。优先按完成顺序回复并附带显式 id。

### 流式工具调用

当模型以流式传输时，`arguments` 分片到达。三个并行调用的三个独立块流在线路上交错。你需要为每个 id 维护一个累加器。

各提供商的形态：

- **OpenAI。** 每个块是 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。块携带 `index`（调用列表中的位置）。你按 index 累积，在 id 首次出现时读取它，并在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic。** 流事件为 `message_start`，然后每个块有一个 `content_block_start`，类型为 `tool_use`（包含 id、name、空的 input）。`content_block_delta` 事件携带 `input_json_delta` 块。`content_block_stop` 关闭每个块。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及以上）发出带有 `functionCallId` 的块，使调用可以干净地交错。在 Gemini 3 之前，流式传输一次返回一个完整的调用。

### 部分 JSON 与提前解析陷阱

在 `arguments` 完整之前你不能解析它。像 `{"city": "Beng` 这样的部分 JSON 不是有效的，会抛出异常。正确的门控是提供商的调用结束信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop`、或 Gemini 的流结束事件。只有在那之后才尝试 `json.loads`。更健壮的方法是使用增量 JSON 解析器，在结构完成时产生事件；OpenAI 的流式传输指南推荐这种方式用于显示实时"思考中"指示器的用户体验。花括号计数作为完整性测试是不可靠的（引号字符串或转义内容中的花括号会导致误报），只应用作非正式的调试启发式方法。

### 乱序完成

```
call_A: 快速 API，最先返回
call_B: 慢速 API，第二个返回
call_C: 中速 API，第三个返回
```

宿主回复仍然必须引用 id：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

回复中的顺序对 OpenAI 和 Anthropic 的正确性没有影响。Gemini 接受任何顺序，只要 id 匹配即可。

### 基准测试：串行 vs 并行

`code/main.py` 中的框架模拟了三个延迟分别为 400、600 和 800 毫秒的执行器。串行运行总耗时 1800 毫秒。并行运行耗时 max(400, 600, 800) = 800 毫秒。差异是恒定的，而非成比例的，因此节省的时间随工具数量增长。

现实注意事项：并行调用会给下游 API 带来压力。对速率受限服务的 10 路扇出会失败。Phase 13 · 17 介绍网关级别的背压控制；重试语义计划在未来阶段覆盖。

### 流式扇出的实际耗时

如果模型本身以流式传输，你可以在一个调用的参数完成后立即开始执行，而不必等待所有调用完成。这是 OpenAI 文档中提到但并非所有 SDK 都暴露的优化。本课的框架这样做了：一旦模拟流产生一个完整的参数对象，宿主就启动该调用。

## 动手实践

`code/main.py` 有两个部分。第一部分使用 `concurrent.futures.ThreadPoolExecutor` 分别以串行和并行方式运行三个模拟天气调用，并打印实际耗时。第二部分重放一个假的流式响应——三个并行调用的 `arguments` 块在一个流上交错——并使用 `StreamAccumulator` 按 id 重组它们。没有 LLM，没有网络，只有重组逻辑。

关注点：

- 串行计时器显示 1.8 秒。并行计时器在相同的假延迟下显示 0.8 秒。
- 累加器通过按 id 缓冲并仅在每个调用的 JSON 完整时才解析来处理乱序到达的块。
- 执行器在某个 id 的参数完成后立即启动，而不是等所有流结束。

## 发布成果

本课产出 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表，该技能审计哪些工具可以安全地并行化、哪些有顺序依赖、哪些会压垮下游速率限制——返回一个带有每个工具 `parallel_safe` 标志的修订注册表。

## 练习

1. 运行 `code/main.py` 并改变模拟延迟。确认并行与串行的比率大约为 `max/sum`（实际运行由于线程调度、序列化和框架开销会略微偏离理想值）。在什么延迟分布下并行不再重要？

2. 扩展累加器以处理"调用在流式传输中途被取消"的情况，丢弃其缓冲区并发出 `cancelled` 事件。哪个提供商明确文档化了这种情况？查看 Anthropic 的 `content_block_stop` 语义和 OpenAI 的 `finish_reason: "length"` 行为。

3. 用 `asyncio.gather` 替换线程池。对两者进行基准测试。你应该看到 async 由于更低的上下文切换成本而有小幅提升，但前提是执行器进行真正的 I/O。

4. 选择两个不应并行化的工具（例如 `create_file` 然后 `write_file`）。在注册表中添加一个 `ordering_dependency` 图，并在并行扇出时根据该图进行门控。这是依赖感知调度的最小机制，未来的 agent 工程阶段会将其形式化。

5. 阅读 OpenAI 的并行 function calling 章节和 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 建议禁用并行的唯一真实工具类型。（提示：对同一资源的有后果变更操作。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Parallel tool calls | "一个回合中的扇出" | 模型在单个 assistant 消息中发出多个工具调用 |
| `parallel_tool_calls` | "OpenAI 的标志" | 启用或禁用多调用发出 |
| `disable_parallel_tool_use` | "Anthropic 的反向标志" | 退出标志；默认启用并行 |
| Tool call id | "关联句柄" | 每个调用的标识符，结果消息必须回显 |
| Accumulator | "流缓冲区" | 每个 id 的字符串缓冲区，用于部分 `arguments` 块 |
| Out-of-order completion | "最快的先完成" | 并行调用以不可预测的顺序完成；id 是粘合剂 |
| Dependency graph | "顺序约束" | 输出作为其他工具输入的工具；不能并行化 |
| Parse-early trap | "JSON.parse 炸了" | 尝试解析不完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | "Gemini 3 特性" | 每个调用带唯一 id 的流式参数块 |
| Completion-order reply | "不要等所有完成" | 按到达顺序回复结果，以 id 为键 |

## 延伸阅读

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为和退出标志
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 和结果批量处理
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — 从 Gemini 3 起的 id 关联并行调用
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI 流的块化参数重组
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — 带有 `input_json_delta` 的 `content_block_delta`
