# Skills 与 Agent SDK — Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 定义"有哪些工具"。Skills 定义"如何完成任务"。2026 年的技术栈将两者分层叠加。Anthropic 的 Agent Skills（2025 年 12 月发布的开放标准）以 SKILL.md 形式交付，支持渐进式披露（progressive disclosure）。OpenAI 的 Apps SDK 是 MCP 加上 widget 元数据。AGENTS.md（已被 60,000+ 仓库采用）位于仓库根目录，作为项目级的 agent 上下文。本课梳理它们各自的职责，并构建一个可跨 agent 移植的最小 SKILL.md + AGENTS.md 组合。

**Type:** Learn
**Languages:** Python（stdlib，SKILL.md 解析器和加载器）
**Prerequisites:** Phase 13 · 07（MCP 服务器）
**Time:** ~45 分钟

## 学习目标

- 区分三个层次：AGENTS.md（项目上下文）、SKILL.md（可复用的操作知识）、MCP（工具）。
- 编写一个带 YAML frontmatter 和渐进式披露的 SKILL.md。
- 以文件系统方式将 skills 加载到 agent 运行时中。
- 将一个 skill 与 MCP 服务器和 AGENTS.md 组合，使一个包在 Claude Code、Cursor 和 Codex 中都能工作。

## 问题所在

一位工程师将编写发布说明的工作流提炼为一个多步 prompt："读取最新合并的 PR。按领域分组。逐个总结。按团队风格撰写 changelog 条目。发布到 Slack 草稿。" 他把这个流程放在 Notion 文档中供团队使用。

现在他想在 Claude Code、Cursor 和 Codex CLI 中使用这个工作流。每个 agent 加载指令的方式不同：Claude Code 用斜杠命令，Cursor 用规则，Codex 用 `.codex.md`。工程师把流程复制了三份，维护三份副本。

AGENTS.md 和 SKILL.md 一起解决了这个问题：

- **AGENTS.md** 位于仓库根目录。每个兼容的 agent 在会话启动时读取它。"这个项目怎么运作？有哪些约定？哪些命令运行测试？"
- **SKILL.md** 是一个可移植的包：YAML frontmatter（名称、描述）+ markdown 正文 + 可选资源。支持 skills 的 agent 按需按名称加载它们。
- **MCP**（Phase 13 · 06-14）处理 skill 需要调用的工具。

三层结构，一个可移植的产物。

## 核心概念

### AGENTS.md（agents.md）

2025 年末发布，到 2026 年 4 月已被 60,000+ 仓库采用。一个文件位于仓库根目录。格式如下：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

Agent 在会话启动时读取此文件，并据此校准其在该项目中的行为。2026 年的每个编码 agent 都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的 Agent Skills（2025 年 12 月作为开放标准发布）：

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

Frontmatter 声明 skill 的身份标识。正文是 skill 加载时展示给模型的 prompt。

### 渐进式披露

Skills 可以引用子资源，agent 仅在需要时才获取这些资源。示例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 中写道"样式规则请参阅 style-guide.md"。Agent 仅在 skill 实际运行时才拉取 style-guide.md。这避免了将模型可能不需要的细节塞满 prompt。

### 文件系统发现

Agent 运行时扫描已知目录中的 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- 项目 `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

加载方式基于文件夹名称和 frontmatter 中的 `name`。Claude Code、Anthropic Claude Agent SDK 和 SkillKit（跨 agent）都遵循这一模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）在会话启动时加载 skills，将它们作为运行时内可调用的"agents"暴露。当用户调用某个 skill 时，agent 循环将任务分派给它。

### OpenAI Apps SDK

2025 年 10 月发布；直接构建在 MCP 之上。将 OpenAI 之前的 Connectors 和 Custom GPT Actions 统一到单一开发者接口下。一个 Apps SDK 应用包含：

- 一个 MCP 服务器（工具、资源、prompts）。
- 加上 ChatGPT UI 的 widget 元数据。
- 加上可选的 MCP Apps `ui://` 资源，用于交互式界面。

同一协议，更丰富的用户体验。

### 通过 SkillKit 实现跨 agent 可移植性

SkillKit 等跨 agent 分发工具将单一的 SKILL.md 翻译为 32+ 个 AI agent（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）各自的原生格式。一个真实来源；多个消费者。

### 三层技术栈

| 层次 | 文件 | 加载时机 | 用途 |
|------|------|---------|------|
| AGENTS.md | 仓库根目录 | 会话启动 | 项目级约定 |
| SKILL.md | skills 目录 | skill 被调用 | 可复用工作流 |
| MCP 服务器 | 外部进程 | 需要工具时 | 可调用操作 |

三者组合使用：agent 在会话启动时读取 AGENTS.md，用户调用一个 skill，skill 的指令包含 MCP 工具调用，agent 通过 MCP 客户端分派执行。

## 动手实践

`code/main.py` 提供了一个基于标准库的 SKILL.md 解析器和加载器。它发现 `./skills/` 下的 skills，解析 YAML frontmatter 和 markdown 正文，生成以 skill 名称为键的字典。然后模拟一个 agent 循环，按名称调用 `release-notes-writer`。

重点查看：

- YAML frontmatter 使用最小化的 stdlib 解析器解析（无 `pyyaml` 依赖）。
- Skill 正文原样存储；agent 在调用时将其前置到系统 prompt 中。
- 通过 `read_subresource` 函数演示渐进式披露，按需拉取引用的文件。

## 交付成果

本课产出 `outputs/skill-agent-bundle.md`。给定一个工作流，该 skill 生成组合的 SKILL.md + AGENTS.md + MCP 服务器蓝图包，可跨 agent 移植。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个 skill，确认加载器能发现它。

2. 为本课程仓库编写一个 AGENTS.md。包含测试命令、代码风格约定和 Phase 13 的心智模型。

3. 将团队内部文档中的一个多步工作流移植为 SKILL.md。验证它能在 Claude Code 中加载。

4. 手动将 skill 翻译为 Cursor 和 Codex 的原生规则格式。计算两种格式之间的差异——这就是 SkillKit 自动化的翻译面。

5. 阅读 Anthropic Agent Skills 博客文章。找出 Claude Agent SDK 中本课加载器未涵盖的一个功能。（提示：agent 子调用。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| SKILL.md | "skill 文件" | YAML frontmatter 加 markdown 正文，由 agent 运行时加载 |
| AGENTS.md | "仓库根目录的 agent 上下文" | 会话启动时读取的项目级约定文件 |
| Progressive disclosure | "延迟加载子资源" | Skill 正文引用仅在需要时才拉取的文件 |
| Frontmatter | "顶部的 YAML 块" | `---` 分隔符中的元数据（名称、描述） |
| Claude Agent SDK | "Anthropic 的 skill 运行时" | `@anthropic-ai/claude-agent-sdk`，加载 skills 并路由 |
| OpenAI Apps SDK | "MCP + widget 元数据" | OpenAI 基于 MCP 构建的开发者接口，加上 ChatGPT UI 钩子 |
| Skill discovery | "文件系统扫描" | 遍历已知目录查找 SKILL.md，按名称索引 |
| Cross-agent portability | "一个 skill 多个 agent" | 通过 SkillKit 类工具将一个 SKILL.md 翻译为 32+ 个 agent |
| Agent Skill | "可移植的操作知识" | 超越 MCP 工具概念的可复用任务模板 |
| Apps SDK | "MCP 加 ChatGPT UI" | Connectors 和 Custom GPTs 在 MCP 上的统一 |

## 延伸阅读

- [Anthropic — Agent Skills 发布公告](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布
- [Anthropic — Agent Skills 文档](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md 格式参考
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — 基于 MCP 的 ChatGPT 开发者平台
- [agents.md](https://agents.md/) — AGENTS.md 格式和采用列表
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 官方 skill 示例
