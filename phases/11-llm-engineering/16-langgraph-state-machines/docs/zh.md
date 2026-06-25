# LangGraph — 面向代理的状态机

> 手写的 ReAct 循环是一个 `while True`。用 LangGraph 编写的 ReAct 循环是一个你可以设置检查点（Checkpoint）、中断（Interrupt）、分支和时光旅行（Time-travel）的图。代理没有变，变的是它周围的框架。

**类型：** 构建
**语言：** Python
**前置知识：** 第 11 阶段 · 09（函数调用），第 11 阶段 · 14（模型上下文协议）
**时间：** 约 75 分钟

## 问题

你发布了一个函数调用代理。它运行了三轮后出问题了：模型尝试调用一个返回 500 的工具，用户在中途改变主意，或者代理决定在没有人工批准的情况下退款。`while True:` 循环没有任何钩子。你不能暂停它，不能回退它，也不能分支到"如果模型选择了另一个工具会怎样"。一旦你把这个代理发布出去超越演示阶段，它就变成了一个要么成功要么失败的黑箱。

下一步在你看到它之后就变得显而易见。代理本身就是一个状态机——系统提示加上消息历史加上待处理的工具调用加上下一个动作。让状态机显式化：节点表示"模型思考"、"工具运行"、"人类批准"，边表示它们之间的条件转换。一旦图是显式的，框架就能免费获得四样东西：检查点保存（在步骤之间保存状态）、中断（暂停等待人类）、流式传输（流式输出 token 和中间事件）以及时光旅行（回退到先前状态并尝试不同的分支）。

LangGraph 就是提供这种抽象的库。它不是 LangChain 意义上的代理框架（"这是一个 AgentExecutor，祝你好运"）。它是一个图运行时，具有一等状态、一等持久化和一等中断。代理循环是你画的，而不是你手写的。

## 概念

![LangGraph StateGraph：节点、边和检查点保存器](../assets/langgraph-stategraph.svg)

一个 `StateGraph` 包含三样东西。

1. **状态（State）。** 一个在图中流转的类型化字典（TypedDict 或 Pydantic 模型）。每个节点接收完整状态并返回部分更新，LangGraph 使用每个字段的*归约器（Reducer）* 来合并——`operator.add` 用于应累积的列表，默认是覆盖。
2. **节点（Node）。** Python 函数 `state -> partial_state`。每个节点是一个离散步骤："调用模型"、"运行工具"、"摘要"。
3. **边（Edge）。** 节点之间的转换。静态边只去往一个地方。条件边使用路由器函数 `state -> next_node_name`，使图能够根据模型输出进行分支。

你编译这个图。编译会绑定拓扑，附加一个检查点保存器（Checkpointer，可选的但对生产至关重要），并返回一个可运行对象。你用初始状态和一个 `thread_id` 调用它。执行的每一步都会持久化一个以 `(thread_id, checkpoint_id)` 为键的检查点。

### 四个超能力

**检查点保存（Checkpointing）。** 每次节点转换都将新状态写入存储（测试用内存，生产用 Postgres/Redis/SQLite）。使用相同的 `thread_id` 再次调用图即可恢复。图会从暂停的位置继续执行。

**中断（Interrupts）。** 用 `interrupt_before=["human_review"]` 标记一个节点，执行会在该节点运行前停止。状态被持久化。你的 API 向用户回复"等待批准"。后续对同一 `thread_id` 的请求使用 `Command(resume=...)` 恢复执行。

**流式传输（Streaming）。** `graph.stream(state, mode="updates")` 在发生时产生状态增量。`mode="messages"` 流式传输模型节点内的 LLM token。`mode="values"` 产生完整快照。你选择要在 UI 中展示的内容。

**时光旅行（Time-travel）。** `graph.get_state_history(thread_id)` 返回完整的检查点日志。将任意先前的 `checkpoint_id` 传递给 `graph.invoke`，即可从该点分叉。非常适合调试（"如果模型选择了工具 B 而不是 A 会怎样？"）和重放生产跟踪的回归测试。

### 归约器是重点

每个状态字段都有一个归约器。大多数默认值都很好——新值覆盖旧值。但消息列表需要 `operator.add`，这样新消息会追加而不是替换。并行边通过归约器合并它们的更新。如果两个节点都更新了 `messages` 而你忘记了 `Annotated[list, add_messages]`，第二个会静默胜出，你就丢失了一半的轮次。归约器是这个库中唯一微妙的点；把它搞对了，其他的就组合起来了。

### 四个节点的 ReAct 图

生产级 ReAct 代理是四个节点和两条边：

1. `agent`——用当前消息历史调用 LLM。返回助手消息（可能包含 tool_calls）。
2. `tools`——执行最后一条助手消息中的任何 tool_calls，将工具结果作为工具消息追加。
3. 从 `agent` 出发的条件边：如果最后一条消息有 tool_calls 则路由到 `tools`，否则路由到 `END`。
4. 从 `tools` 回到 `agent` 的静态边。

就这些。你在大约 40 行代码中获得完整的 ReAct 循环（思考 → 行动 → 观察 → 思考 → ...），并带有检查点保存、中断和流式传输。

### StateGraph vs Send（扇出）

`Send(node_name, state)` 允许一个节点派发并行子图。示例：代理决定同时查询三个检索器。每个 `Send` 生成目标节点的一个并行执行；它们的输出通过状态归约器合并。这就是 LangGraph 表达编排者-工作者（Orchestrator-Workers）模式的方式，无需线程原语。

### 子图

一个编译后的图可以作为另一个图中的一个节点。外部图看到一个单一的节点；内部图有它自己的状态和检查点。这就是团队构建监督者-工作者（Supervisor-Worker）代理的方式：监督者图将用户意图路由到每个领域的 worker 子图。

## 构建它

### 步骤 1：状态和节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是使消息列表追加而不是覆盖的归约器。忘记它是 LangGraph 最常见的 bug。

### 步骤 2：使用线程运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("查找 Anthropic 总部地址")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每次更新都是一个字典 `{node_name: state_delta}`。你的前端可以将其流式传输到 UI，用户可以看到"代理正在思考...正在调用 search_web...获取了结果...正在回答"。

### 步骤 3：添加人机协同中断

标记一个节点，以便在执行前暂停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # 在每个工具调用前暂停
)

state = app.invoke({"messages": [HumanMessage("删除生产数据库")]}, config)
# state["__interrupt__"] 已设置。检查提议的工具调用。
# 如果批准：
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# 如果拒绝：写一条拒绝消息并恢复
app.update_state(config, {"messages": [AIMessage("被人审核者阻止。")]})
```

状态、检查点和线程在中止期间全部持久化。除了执行期间，没有任何东西在内存中。

### 步骤 4：用时光旅行调试

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# 从先前的检查点分叉
target = history[3].config  # 回退三步
for event in app.stream(None, target, stream_mode="values"):
    pass  # 从该点向前重放
```

以 `None` 作为输入表示从给定检查点重放；传递一个值表示将其作为更新追加到该检查点的状态后再恢复。这就是你在不重新运行整个对话的情况下重现一次坏的代理运行的方式。

### 步骤 5：为生产环境更换检查点保存器

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis 和 Postgres 都已内置。`MemorySaver` 仅用于测试。任何需要跨重启持久化的内容都需要真正的存储。

## 技能

> 你将代理构建为图，而不是 `while True` 循环。

在诉诸 LangGraph 之前，做一个 60 秒的设计：

1. **命名节点。** 每个离散的决策或副效应操作都是一个节点。"代理思考"、"工具运行"、"审核者批准"、"响应流式输出"。如果你不能列出它们，说明任务还不是代理形态。
2. **声明状态。** 最简 TypedDict，每个列表字段都有一个归约器。不要把所有东西塞进 `messages`；将任务特定的字段（一个工作中的 `plan`、一个 `budget` 计数器、一个 `retrieved_docs` 列表）提升到顶层。
3. **画边。** 静态边，除非下一步取决于模型输出。每一条条件边都需要一个带命名分支的路由器函数。
4. **预先选择检查点保存器。** 测试用 `MemorySaver`，其他情况用 Postgres/Redis/SQLite。没有检查点保存器就不要交付——没有它就没有恢复、没有中断、没有时光旅行。
5. **在工具运行前而非运行后决定中断。** 审批放在进入副效应节点的边上，这样你可以在造成损害之前取消；验证放在离开模型的边上，这样你可以廉价地拒绝错误的调用。
6. **默认开启流式传输。** UI 用 `mode="updates"`，模型节点内 token 级流式输出用 `mode="messages"`，评估时的完整快照用 `mode="values"`。

拒绝交付没有检查点保存器的 LangGraph 代理。拒绝交付在*副效应之后*中断的代理。拒绝交付没有 `add_messages` 作为归约器的 `messages` 字段。

## 练习

1. **简单。** 用计算器工具和网页搜索工具实现上述四节点 ReAct 图。验证 `list(app.get_state_history(config))` 在一次两轮对话中返回至少四个检查点。
2. **中等。** 添加一个在 `agent` 之前运行的 `planner` 节点，将结构化的 `plan: list[str]` 写入状态。让 `agent` 将计划步骤标记为已完成。如果检查点恢复后 `plan` 丢失则测试失败（归约器错误）。
3. **困难。** 构建一个监督者图，使用 `Send` 在三个子图之间路由（`researcher`、`writer`、`reviewer`）。每个子图有自己的状态和检查点保存器。在外部图上添加 `interrupt_before=["writer"]`，以便人类可以批准研究简报。确认从先前检查点的时光旅行只重新运行分叉的分支。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| StateGraph | "LangGraph 的图" | 构建器对象，在编译之前向其中添加节点和边。 |
| 归约器（Reducer） | "字段如何合并" | 当节点返回该字段的更新时应用的函数 `(old, new) -> merged`；默认是覆盖，`add_messages` 是追加。 |
| 线程（Thread） | "对话 ID" | 一个 `thread_id` 字符串，限定一个会话的所有检查点范围。 |
| 检查点（Checkpoint） | "暂停的状态" | 节点转换后完整图状态的持久化快照，以 `(thread_id, checkpoint_id)` 为键。 |
| 中断（Interrupt） | "等待人类" | `interrupt_before` / `interrupt_after` 在节点边界停止执行；用 `Command(resume=...)` 恢复。 |
| 时光旅行（Time-travel） | "从先前步骤分叉" | `graph.invoke(None, config_with_old_checkpoint_id)` 从该检查点向前重放。 |
| Send | "并行子图派发" | 一个节点可以返回的构造函数，用于生成目标节点的 N 个并行执行。 |
| 子图（Subgraph） | "作为节点的编译图" | 在另一个图中作为节点使用的编译后 StateGraph；保留自己的状态范围。 |

## 扩展阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)——StateGraph、归约器、检查点保存器和中断的权威参考。
- [LangGraph 概念：状态、归约器、检查点保存器](https://langchain-ai.github.io/langgraph/concepts/low_level/)——本课程使用的思维模型，直接来自源头。
- [LangGraph 持久化与检查点](https://langchain-ai.github.io/langgraph/concepts/persistence/)——关于 Postgres/SQLite/Redis 存储、检查点命名空间和线程 ID 的细节。
- [LangGraph 人机协同](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)——`interrupt_before`、`interrupt_after`、`Command(resume=...)` 和编辑状态模式。
- [Yao 等，"ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629)——每个 LangGraph 代理都实现的模式；阅读它以了解推理轨迹的基本原理。
- [Anthropic — Building effective agents (2024 年 12 月)](https://www.anthropic.com/research/building-effective-agents)——应优先选择哪些图形状（链、路由器、编排者-工作者、评估者-优化器）以及何时选用。
- 第 11 阶段 · 09（函数调用）——每个 LangGraph 代理节点重用的工具调用原语。
- 第 11 阶段 · 14（模型上下文协议）——通过 MCP 适配器插入 LangGraph `ToolNode` 的外部工具发现。
- 第 11 阶段 · 17（代理框架权衡）——何时选择 LangGraph 而非 CrewAI、AutoGen 或 Agno。
