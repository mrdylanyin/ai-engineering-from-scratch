# 代理框架权衡——LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架都卖相同的演示（研究代理撰写报告）并隐藏相同的 bug（状态 schema 与编排层冲突）。选择其抽象与问题形态匹配的框架；其余的都是你需要写两次的胶水代码。

**类型：** 学习
**语言：** Python
**前置知识：** 第 11 阶段 · 09（函数调用），第 11 阶段 · 16（LangGraph）
**时间：** 约 45 分钟

## 问题

你有一个需要不止一次 LLM 调用的任务。也许它是一个研究工作流（规划、搜索、摘要、引用）。也许它是一个代码审查流水线（解析 diff、评论、修补、验证）。也许它是一个多轮助手，负责订机票、写邮件和报销费用。你选择了一个框架。

三天后，你发现框架的抽象存在泄漏。CrewAI 给了你角色，但当"研究员"需要将结构化计划交给"作家"时它就会和你打架。AutoGen 给了你代理之间的聊天，但没有一等状态，所以你的检查点只是对话日志的 pickle。LangGraph 给了你状态图，但强迫你在不知道代理会做什么之前就命名每个转换。Agno 给了你一个单代理抽象，但当你试图扇出到三个并发工作器时它就会尖叫。

解决办法不是"选最好的框架"。而是将框架的核心抽象与问题的形态匹配起来。本课程绘制了这张地图。

## 概念

![代理框架矩阵：核心抽象 vs 问题形态](../assets/framework-matrix.svg)

2026 年有四个主流框架。它们的核心抽象并不相同。

| 框架 | 核心抽象 | 最适合 | 最不适合 |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph`——类型化状态、节点、条件边、检查点保存器。 | 具有显式状态和人机协同中断的工作流；需要时光旅行调试的生产级代理。 | 松散、角色驱动的头脑风暴，其中拓扑结构未知。 |
| **CrewAI** | `Crew`——角色（目标、背景故事）、任务、流程（顺序或层级）。 | 角色扮演或角色驱动的工作流，具有短的线性/层级计划。 | 任何超出 crew 轮次历史的有状态内容；复杂的分支。 |
| **AutoGen** | `ConversableAgent` 对——两个或多个代理轮流对话，直到满足退出条件。 | 多代理*对话*（教师-学生、提议者-批评者、演员-审查者），在这种场景中思考从聊天中涌现。 | 具有已知 DAG 的确定性工作流；任何需要在重启后保持持久状态的内容。 |
| **Agno** | `Agent`——单个 LLM + 工具 + 记忆，可组合成团队。 | 快速构建的单个代理和轻量级团队；强大的多模态和内置存储驱动。 | 需要深度显式分支图和自定义归约器的场景。 |

### "抽象"实际上意味着什么

框架的核心抽象是你在介绍架构时在白板上画的东西。

- **LangGraph** → 你画一个图。节点是步骤，边是转换，每一点的状态对象都是类型化的。心智模型是状态机。
- **CrewAI** → 你画一个组织结构图。每个角色有职位描述和一个管理者来路由任务。心智模型是一个专员小团队。
- **AutoGen** → 你画一个 Slack 私聊。两个代理互相发消息；如果你需要一个协调者，第三个加入。心智模型是聊天。
- **Agno** → 你画一个带有挂载工具的单个盒子。把盒子并排放置就构成一个团队。心智模型是"开箱即用的代理"。

### 状态问题

状态是大多数框架选型在生产中崩塌的地方。

- **LangGraph。** 类型化状态（`TypedDict` 或 Pydantic 模型），每个字段有归约器，一等检查点保存器（SQLite/Postgres/Redis）。恢复、中断和时光旅行是自带的。*(参见第 11 阶段 · 16。)*
- **CrewAI。** 状态通过 `context` 字段在任务间以字符串形式流动，或通过 `output_pydantic` 以结构化形式流动。没有开箱即用的持久团队存储；如果 crew 必须经受重启，你需要自己焊接上去。
- **AutoGen。** 状态是聊天历史和任何用户定义的 `context`。对话抄本持久化；任意工作流状态不会，除非你编写适配器。
- **Agno。** 内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB），通过 `storage=` 附加到 `Agent`——对话会话和用户记忆自动持久化。不是完整的图检查点保存器；而是一个会话存储。

### 分支问题

每个非平凡代理都会分支。谁决定分支是关键。

- **LangGraph**——你决定，通过条件边。路由是一个带有命名分支的 Python 函数。分支在编译后的图中是一等公民；检查点保存器记录走了哪个分支。
- **CrewAI**——在层级模式下由管理者决定；在顺序模式下由你在构建时决定。路由隐含在任务列表中；在管理者的提示之外没有一等"if"。
- **AutoGen**——代理通过聊天决定。分支从谁接下来说话中涌现。`GroupChatManager` 选择下一个发言者；你可以手写 `speaker_selection_method`，但默认是 LLM 驱动的。
- **Agno**——代理通过调用哪个工具来决定。团队有协调者/路由器/协作者模式；超出此范围的分支是开发者的责任。

### 可观测性问题

- **LangGraph**——通过 LangSmith 或任何 OTel 导出器提供 OpenTelemetry。每次节点转换都是一个 trace span；检查点兼作可重放的 trace。LangSmith 是第一方选项；Langfuse/Phoenix 也有适配器。
- **CrewAI**——自 2025 年底起提供一等 OpenTelemetry；集成 Langfuse、Phoenix、Opik、AgentOps。
- **AutoGen**——通过 `autogen-core` 集成 OpenTelemetry；AgentOps 和 Opik 有连接器。跟踪粒度为每个代理消息，而非每个节点。
- **Agno**——内置 `monitoring=True` 标志加 OpenTelemetry 导出器；与 Langfuse 紧密集成用于会话跟踪。

### 成本和延迟

所有四个框架都会增加每次调用的开销（框架逻辑、验证、序列化）。开销递增的大致顺序：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要取决于框架做了多少额外的 LLM 路由。CrewAI 的层级管理者花费 token 来决定谁下一步；AutoGen 的 `GroupChatManager` 同理。LangGraph 只在你的 `llm.invoke` 处花费 token。Agno 的单代理路径很薄。

当每次运行的成本很重要时，优先选择显式路由（LangGraph 边、AutoGen 的 `speaker_selection_method`）而不是 LLM 选择的路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 工具、检索器、LLM。一等 MCP 适配器（工具以 MCP 服务器形式导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具和 MCP 工具都可以适配接入。Crew 间委派通过 `allow_delegation=True`。
- **AutoGen** → `FunctionTool` 包装任何 Python 可调用对象；MCP 适配器可用。代理间模式与 AG2 生态紧密耦合。
- **Agno** → `@tool` 装饰器或 BaseTool 子类；MCP 适配器；工具可以在代理和团队之间共享。

## 技能

> 你能用一句话解释为什么某个框架适合给定的代理问题。

构建前检查清单：

1. **画出形态。** 这是一个图（类型化状态，命名转换）？一个角色扮演（专员移交工作）？一个聊天（代理聊到完成）？一个带工具的单个代理？
2. **决定谁分支。** 开发者决定的分支 → LangGraph。管理者代理决定 → CrewAI 层级。聊天涌现 → AutoGen。工具调用决定 → Agno。
3. **检查状态预算。** 你需要从检查点恢复吗？时光旅行？中途人工中断？如果需要，LangGraph 是默认选择；Agno 会话覆盖对话范围的状态。
4. **检查成本预算。** LLM 选择的路由每轮花费额外 token。如果代理每天运行数千次，优先选择显式路由。
5. **评估框架开销。** 每个框架都是额外的依赖。如果任务是两次 LLM 调用和一个工具，写 30 行纯 Python；没有框架比没有框架更便宜。

拒绝在你能画出图、组织结构图、聊天或代理框之前就诉诸框架。拒绝选择一个强迫你为其状态模型而与你真正需要的东西抗争的框架。

## 决策矩阵

| 问题形态 | 推荐框架 | 理由 |
|---------------|---------------------|-----|
| 具有类型化状态、人工审批、长时间运行的工作流 DAG | LangGraph | 一等状态、检查点保存器、中断、时光旅行。 |
| 具有不同角色的研究/写作流水线 | CrewAI（顺序）或 LangGraph 子图 | 角色-任务映射在 CrewAI 中表达成本低廉；当分支变复杂时用 LangGraph 扩展。 |
| 提议者-批评者或教师-学生对话 | AutoGen | 双代理聊天是其原生形态。 |
| 带工具、会话、记忆的单个代理 | Agno | 设置最简单，内置存储和记忆。 |
| 具有归约器的数千个并行扇出 | LangGraph + `Send` | 唯一具有一等并行派发 API 的框架。 |
| 快速原型，无框架承诺 | 纯 Python + 供应商 SDK | 没有框架是最快的框架。 |

## 练习

1. **简单。** 用同一个任务——"研究 Anthropic 的总部，写一份 200 字的简报，引用来源"——分别在 LangGraph（四个节点：plan、search、write、cite）和 CrewAI（三个角色：researcher、writer、editor）中实现。报告每次运行的 token 成本和代码行数。
2. **中等。** 在 AutoGen（researcher ↔ writer 聊天，editor 通过 `GroupChat` 加入）和 Agno（一个带 `search_tools` 和 `write_tools` 的单个代理，加上会话存储）中构建同一个任务。从以下方面对四种实现排名：(a) 每次运行成本，(b) 崩溃后的恢复能力，(c) 在写入步骤前注入人工审批的能力。
3. **困难。** 构建一个决策树脚本 `pick_framework.py`，接收一个简短的问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`）并返回一个带一句话理由的推荐。用你自行设计的六个案例验证它。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 编排（Orchestration） | "代理如何协调" | 决定哪个节点/角色/代理下一步运行的层。 |
| 持久状态 | "重启后恢复" | 在进程死亡后幸存的状态，附加到检查点或会话存储。 |
| LLM 选择的路由 | "让模型决定" | 一个规划 LLM 每轮选择下一步；灵活但每个决策都消耗 token。 |
| 显式路由 | "开发者决定" | 一个 Python 函数或静态边选择下一步；廉价且可审计。 |
| Crew | "一个 CrewAI 团队" | 角色 + 任务 + 流程（顺序或层级），绑定为单次运行。 |
| GroupChat | "AutoGen 的多代理聊天" | 一个 N 个代理之间的托管对话，带有发言者选择器。 |
| Team（Agno） | "多代理 Agno" | 对一组代理的路由/协调/协作模式。 |
| StateGraph | "LangGraph 的图" | 类型化状态、节点、条件边、检查点保存器的抽象。 |

## 扩展阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)——StateGraph、检查点保存器、中断、时光旅行。
- [CrewAI 文档](https://docs.crewai.com/)——Crews、Flows、Agents、Tasks、Processes。
- [AutoGen 文档](https://microsoft.github.io/autogen/)——ConversableAgent、GroupChat、teams、tools。
- [Agno 文档](https://docs.agno.com/)——Agent、Team、Workflow、storage、memory。
- [Anthropic — Building effective agents (2024 年 12 月)](https://www.anthropic.com/research/building-effective-agents)——模式库（提示链式调用、路由、并行化、编排者-工作者、评估者-优化器），与框架无关。
- [Yao 等，"ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629)——每个框架都在装饰的循环。
- [Wu 等，"AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155)——AutoGen 的设计论文。
- [Park 等，"Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442)——CrewAI 风格角色栈所基于的角色扮演基础。
- 第 11 阶段 · 16（LangGraph）——本课程进行基准对比的框架。
- 第 11 阶段 · 19（Reflexion）——一种能清晰映射到 LangGraph 但对 CrewAI 来说很别扭的模式。
- 第 11 阶段 · 22（生产可观测性）——无论选择哪个框架，如何为其添加仪表。
