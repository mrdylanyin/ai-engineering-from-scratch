# 函数调用与工具使用

> LLM 不能做任何事情。它们只会生成文本。这是它们的全部能力。它们不能查询天气、查询数据库、发送邮件、运行代码或读取文件。你见过的每一个"AI 代理（AI agent）"都是 LLM 生成 JSON，告诉你该调用哪个函数 —— 然后由你的代码实际调用它。模型是大脑，工具是双手，函数调用是连接二者的神经系统。

**Type:** 构建（Build）
**Languages:** Python
**Prerequisites:** 第十一阶段第 03 课（结构化输出）
**Time:** ~75 分钟
**Related:** 第十一阶段 · 14（模型上下文协议 MCP）—— 当工具需要跨主机共享时，从内联函数调用升级为 MCP 服务器。本课涵盖内联情形；MCP 涵盖协议情形。

## 学习目标

- 实现函数调用循环：定义工具 Schema，解析模型的工具调用 JSON，执行函数，返回结果
- 设计包含清晰描述和类型化参数的工具 Schema，使模型能够可靠地调用
- 构建多轮代理循环，将多个函数调用串联起来回答复杂查询
- 处理函数调用的边界情况：并行工具调用、错误传播、防止无限工具循环

## 问题

你做了一个聊天机器人。用户问："东京现在天气怎么样？"

模型回答："我没有实时天气数据的访问权限，但根据季节，东京目前大概在 15 摄氏度左右……"

这是一个披着免责声明外衣的幻觉（hallucination）。模型不知道天气。它永远不会知道。天气每小时都在变化。模型的训练数据是几个月前的。

正确的回答需要调用 OpenWeatherMap API，获取当前温度，返回真实的数字。模型无法调用 API。你的代码可以。缺失的一环：一个结构化的协议，让模型能够说"我需要用这些参数调用天气 API"，并让你的代码执行它，将结果反馈回去。

这就是函数调用（Function Calling）。模型输出结构化 JSON，描述要调用哪个函数以及参数是什么。你的应用程序执行该函数。结果返回给对话上下文。模型利用结果生成最终答案。

没有函数调用，LLM 只是百科全书。有了它，它们就成了代理。

## 概念

### 函数调用循环

每一次工具使用交互都遵循相同的 5 步循环。

```mermaid
sequenceDiagram
    participant U as User
    participant A as Application
    participant M as Model
    participant T as Tool

    U->>A: "What's the weather in Tokyo?"
    A->>M: messages + tool definitions
    M->>A: tool_call: get_weather(city="Tokyo")
    A->>T: Execute get_weather("Tokyo")
    T->>A: {"temp": 18, "condition": "cloudy"}
    A->>M: tool_result + conversation
    M->>A: "It's 18C and cloudy in Tokyo."
    A->>U: Final response
```

第 1 步：用户发送消息。第 2 步：模型接收到消息以及工具定义（描述可用函数的 JSON Schema）。第 3 步：模型不直接回复文本，而是输出一个工具调用 —— 一个包含函数名和参数的结构化 JSON 对象。第 4 步：你的代码执行该函数并捕获结果。第 5 步：结果返回给模型，模型现在有了真实数据来生成最终答案。

模型从不执行任何操作。它只决定调用什么以及用什么参数。你的代码是执行者。

### 工具定义：JSON Schema 契约

每个工具由 JSON Schema 定义，告诉模型该函数做什么、接受什么参数以及参数的类型。

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a city. Returns temperature in Celsius and conditions.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "City name, e.g. 'Tokyo' or 'San Francisco'"
        },
        "units": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "Temperature units"
        }
      },
      "required": ["city"]
    }
  }
}
```

`description`（描述）字段至关重要。模型通过读取它们来决定何时以及如何使用工具。模糊的描述如"查询天气"会产生较差的工具选择效果，而"获取城市的当前天气。返回摄氏度温度和天气状况"则更好。描述就是工具选择的提示词。

### 各厂商对比

所有主要厂商都支持函数调用，但 API 接口有差异。

| 厂商 | API 参数 | 工具调用格式 | 并行调用 | 强制调用 |
|------|---------|------------|---------|---------|
| OpenAI (GPT-5, o4) | `tools` | `tool_calls[].function` | 支持（每轮可多个） | `tool_choice="required"` |
| Anthropic (Claude 4.6/4.7) | `tools` | `content[].type="tool_use"` | 支持（多个块） | `tool_choice={"type":"any"}` |
| Google (Gemini 3) | `function_declarations` | `functionCall` | 支持 | `function_calling_config` |
| 开源模型 (Llama 4, Qwen3, DeepSeek-V3) | Llama 4 原生 `tools`；其他为 Hermes 或 ChatML | 混合 | 因模型而异 | 基于提示词，若支持则通过 `tool_choice` |

到 2026 年，三家闭源厂商已趋同于几乎相同的基于 JSON Schema 的格式。Llama 4 提供了与 OpenAI 形状匹配的 `tools` 字段。开源微调模型仍然各异 —— Hermes 格式（NousResearch）是第三方微调中最常见的格式。对于跨主机共享的工具，优先选择 MCP（第十一阶段 · 14）而非内联函数调用 —— 服务器对所有主机保持一致。

### 工具选择：自动、强制、指定

你可以控制模型何时使用工具。

**Auto（自动）**（默认）：模型自行决定是调用工具还是直接回复。"2+2 等于几？"—— 直接回复。"天気怎么样？"—— 调用工具。

**Required（强制）**：模型必须调用至少一个工具。当你确定用户意图需要工具时使用此模式。防止模型在应该查询真实数据时猜测。

**Specific function（指定函数）**：强制模型调用某个特定函数。`tool_choice={"type":"function", "function": {"name": "get_weather"}}` 确保无论查询内容如何，都会调用天气工具。用于路由场景 —— 上游逻辑已经确定需要哪个工具。

### 并行函数调用

GPT-4o 和 Claude 可以在单轮中调用多个函数。用户问："东京和纽约的天气怎么样？" 模型同时输出两个工具调用：

```json
[
  {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  {"name": "get_weather", "arguments": {"city": "New York"}}
]
```

你的代码执行两者（理想情况下并发执行），返回两个结果，模型合成单个响应。这将往返次数从 2 次减少到 1 次。对于每次查询需要 5-10 次工具调用的代理，并行调用可将延迟降低 60-80%。

### 结构化输出 vs 函数调用

第 03 课涵盖了结构化输出。函数调用使用相同的 JSON Schema 机制，但目的不同。

**结构化输出**：强制模型以特定形状生成数据。输出就是最终产物。例如：从文本中提取产品信息 `{name, price, in_stock}`。

**函数调用**：模型声明执行某个操作的意图。输出是中间步骤。例如：`get_weather(city="Tokyo")` —— 模型请求执行一个操作，而非生成最终答案。

当你需要数据提取时使用结构化输出。当你需要模型与外部系统交互时使用函数调用。

### 安全性：不可妥协的规则

函数调用是你能赋予 LLM 的最危险的能力。模型选择执行什么。如果你的工具集包含数据库查询，模型会构造查询。如果包含 shell 命令，模型会编写它们。

**规则 1：永远不要将模型生成的 SQL 直接传给数据库。** 模型可以且将会生成 DROP TABLE、UNION 注入或返回每一行的查询。始终使用参数化查询。始终验证。始终使用操作白名单。

**规则 2：函数白名单。** 模型只能调用你显式定义的函数。永远不要构建一个通用的"按名称执行任意函数"工具。如果你有 50 个内部函数，只暴露用户需要的 5 个。

**规则 3：验证参数。** 模型可能传入城市名为 `"; DROP TABLE users; --"`。在执行之前，验证每个参数是否符合预期类型、范围和格式。

**规则 4：清洗工具结果。** 如果工具返回敏感数据（API 密钥、个人身份信息 PII、内部错误），在将其发回模型之前进行过滤。模型会逐字地将工具结果包含在响应中。

**规则 5：限制工具调用频率。** 处于循环中的模型可能调用工具数百次。设置最大限制（每次会话 10-20 次是合理的）。打破无限循环。

### 错误处理

工具会失败。API 会超时。数据库会宕机。文件不存在。模型需要知道工具何时失败以及原因。

将错误作为结构化工具结果返回，而不是异常：

```json
{
  "error": true,
  "message": "City 'Toky' not found. Did you mean 'Tokyo'?",
  "code": "CITY_NOT_FOUND"
}
```

模型会读取这个信息，调整参数，然后重试。模型擅长从结构化错误消息中自我纠正，但不擅长从空响应或通用的"出错了"这种错误中恢复。

### MCP：模型上下文协议

MCP 是 Anthropic 提出的工具互操作性开放标准。每个应用程序不再需要定义自己的工具，MCP 提供了一个通用协议：工具由 MCP 服务器提供，由 MCP 客户端（如 Claude Code、Cursor 或你的应用程序）消费。

一个 MCP 服务器可以向任何兼容的客户端暴露工具。一个 Postgres MCP 服务器赋予任何 MCP 兼容代理数据库访问能力。一个 GitHub MCP 服务器赋予任何代理仓库访问能力。工具定义一次，随处使用。

MCP 之于函数调用，就像 HTTP 之于网络。它将传输层标准化，使工具变得可移植。

## 构建它

### 第 1 步：定义工具注册表

构建一个注册表，存储工具定义及其实现。每个工具有一个 JSON Schema 定义（模型看到的内容）和一个 Python 函数（你代码执行的内容）。

```python
import json
import math
import time
import hashlib


TOOL_REGISTRY = {}


def register_tool(name, description, parameters, function):
    TOOL_REGISTRY[name] = {
        "definition": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        },
        "function": function,
    }
```

### 第 2 步：实现 5 个工具

构建计算器、天气查询、网页搜索模拟器、文件读取器和代码运行器。

```python
def calculator(expression, precision=2):
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": True, "message": f"Invalid characters in expression: {expression}"}
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return {"result": round(float(result), precision), "expression": expression}
    except Exception as e:
        return {"error": True, "message": str(e)}


WEATHER_DB = {
    "tokyo": {"temp_c": 18, "condition": "cloudy", "humidity": 72, "wind_kph": 14},
    "new york": {"temp_c": 22, "condition": "sunny", "humidity": 45, "wind_kph": 8},
    "london": {"temp_c": 12, "condition": "rainy", "humidity": 88, "wind_kph": 22},
    "san francisco": {"temp_c": 16, "condition": "foggy", "humidity": 80, "wind_kph": 18},
    "sydney": {"temp_c": 25, "condition": "sunny", "humidity": 55, "wind_kph": 10},
}


def get_weather(city, units="celsius"):
    key = city.lower().strip()
    if key not in WEATHER_DB:
        suggestions = [c for c in WEATHER_DB if c.startswith(key[:3])]
        return {
            "error": True,
            "message": f"City '{city}' not found.",
            "suggestions": suggestions,
            "code": "CITY_NOT_FOUND",
        }
    data = WEATHER_DB[key].copy()
    if units == "fahrenheit":
        data["temp_f"] = round(data["temp_c"] * 9 / 5 + 32, 1)
        del data["temp_c"]
    data["city"] = city
    return data


SEARCH_DB = {
    "python function calling": [
        {"title": "OpenAI Function Calling Guide", "url": "https://platform.openai.com/docs/guides/function-calling", "snippet": "Learn how to connect LLMs to external tools."},
        {"title": "Anthropic Tool Use", "url": "https://docs.anthropic.com/en/docs/tool-use", "snippet": "Claude can interact with external tools and APIs."},
    ],
    "MCP protocol": [
        {"title": "Model Context Protocol", "url": "https://modelcontextprotocol.io", "snippet": "An open standard for connecting AI models to data sources."},
    ],
    "weather API": [
        {"title": "OpenWeatherMap API", "url": "https://openweathermap.org/api", "snippet": "Free weather API with current, forecast, and historical data."},
    ],
}


def web_search(query, max_results=3):
    key = query.lower().strip()
    for db_key, results in SEARCH_DB.items():
        if db_key in key or key in db_key:
            return {"query": query, "results": results[:max_results], "total": len(results)}
    return {"query": query, "results": [], "total": 0}


FILE_SYSTEM = {
    "data/config.json": '{"model": "gpt-4o", "temperature": 0.7, "max_tokens": 4096}',
    "data/users.csv": "name,email,role\nAlice,alice@example.com,admin\nBob,bob@example.com,user",
    "README.md": "# My Project\nA tool-use agent built from scratch.",
}


def read_file(path):
    if ".." in path or path.startswith("/"):
        return {"error": True, "message": "Path traversal not allowed.", "code": "FORBIDDEN"}
    if path not in FILE_SYSTEM:
        available = list(FILE_SYSTEM.keys())
        return {"error": True, "message": f"File '{path}' not found.", "available_files": available, "code": "NOT_FOUND"}
    content = FILE_SYSTEM[path]
    return {"path": path, "content": content, "size_bytes": len(content), "lines": content.count("\n") + 1}


def run_code(code, language="python"):
    if language != "python":
        return {"error": True, "message": f"Language '{language}' not supported. Only 'python' is available."}
    forbidden = ["import os", "import sys", "import subprocess", "exec(", "eval(", "__import__", "open("]
    for pattern in forbidden:
        if pattern in code:
            return {"error": True, "message": f"Forbidden operation: {pattern}", "code": "SECURITY_VIOLATION"}
    try:
        local_vars = {}
        exec(code, {"__builtins__": {"print": print, "range": range, "len": len, "str": str, "int": int, "float": float, "list": list, "dict": dict, "sum": sum, "min": min, "max": max, "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "math": math}}, local_vars)
        result = local_vars.get("result", None)
        return {"success": True, "result": result, "variables": {k: str(v) for k, v in local_vars.items() if not k.startswith("_")}}
    except Exception as e:
        return {"error": True, "message": f"{type(e).__name__}: {e}"}
```

### 第 3 步：注册所有工具

```python
def register_all_tools():
    register_tool(
        "calculator", "Evaluate a mathematical expression. Supports +, -, *, /, parentheses, and decimals. Returns the numeric result.",
        {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression, e.g. '(10 + 5) * 3'"}, "precision": {"type": "integer", "description": "Decimal places in result", "default": 2}}, "required": ["expression"]},
        calculator,
    )
    register_tool(
        "get_weather", "Get current weather for a city. Returns temperature, condition, humidity, and wind speed.",
        {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. 'Tokyo' or 'San Francisco'"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units, defaults to celsius"}}, "required": ["city"]},
        get_weather,
    )
    register_tool(
        "web_search", "Search the web for information. Returns a list of results with title, URL, and snippet.",
        {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}, "max_results": {"type": "integer", "description": "Maximum results to return", "default": 3}}, "required": ["query"]},
        web_search,
    )
    register_tool(
        "read_file", "Read the contents of a file. Returns the file content, size, and line count.",
        {"type": "object", "properties": {"path": {"type": "string", "description": "Relative file path, e.g. 'data/config.json'"}}, "required": ["path"]},
        read_file,
    )
    register_tool(
        "run_code", "Execute Python code in a sandboxed environment. Set a 'result' variable to return output.",
        {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute"}, "language": {"type": "string", "enum": ["python"], "description": "Programming language"}}, "required": ["code"]},
        run_code,
    )
```

### 第 4 步：构建函数调用循环

这是核心引擎。它模拟模型决定调用哪个工具，执行工具，并反馈结果。

```python
def simulate_model_decision(user_message, tools, conversation_history):
    msg = user_message.lower()

    if any(word in msg for word in ["weather", "temperature", "forecast"]):
        cities = []
        for city in WEATHER_DB:
            if city in msg:
                cities.append(city)
        if not cities:
            for word in msg.split():
                if word.capitalize() in [c.title() for c in WEATHER_DB]:
                    cities.append(word)
        if not cities:
            cities = ["tokyo"]
        calls = []
        for city in cities:
            calls.append({"name": "get_weather", "arguments": {"city": city.title()}})
        return calls

    if any(word in msg for word in ["calculate", "compute", "math", "what is", "how much"]):
        for token in msg.split():
            if any(c in token for c in "+-*/"):
                return [{"name": "calculator", "arguments": {"expression": token}}]
        if "+" in msg or "-" in msg or "*" in msg or "/" in msg:
            expr = "".join(c for c in msg if c in "0123456789+-*/.() ")
            if expr.strip():
                return [{"name": "calculator", "arguments": {"expression": expr.strip()}}]
        return [{"name": "calculator", "arguments": {"expression": "0"}}]

    if any(word in msg for word in ["search", "find", "look up", "google"]):
        query = msg.replace("search for", "").replace("look up", "").replace("find", "").strip()
        return [{"name": "web_search", "arguments": {"query": query}}]

    if any(word in msg for word in ["read", "file", "open", "cat", "show"]):
        for path in FILE_SYSTEM:
            if path.split("/")[-1].split(".")[0] in msg:
                return [{"name": "read_file", "arguments": {"path": path}}]
        return [{"name": "read_file", "arguments": {"path": "README.md"}}]

    if any(word in msg for word in ["run", "execute", "code", "python"]):
        return [{"name": "run_code", "arguments": {"code": "result = 'Hello from the sandbox!'", "language": "python"}}]

    return []


def execute_tool_call(tool_call):
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name not in TOOL_REGISTRY:
        return {"error": True, "message": f"Unknown tool: {name}", "code": "UNKNOWN_TOOL"}

    tool = TOOL_REGISTRY[name]
    func = tool["function"]
    start = time.time()

    try:
        result = func(**args)
    except TypeError as e:
        result = {"error": True, "message": f"Invalid arguments: {e}"}

    elapsed_ms = round((time.time() - start) * 1000, 2)
    return {"tool": name, "result": result, "execution_time_ms": elapsed_ms}


def run_function_calling_loop(user_message, max_iterations=5):
    conversation = [{"role": "user", "content": user_message}]
    tool_definitions = [t["definition"] for t in TOOL_REGISTRY.values()]
    all_tool_results = []

    for iteration in range(max_iterations):
        tool_calls = simulate_model_decision(user_message, tool_definitions, conversation)

        if not tool_calls:
            break

        results = []
        for call in tool_calls:
            result = execute_tool_call(call)
            results.append(result)

        conversation.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

        for result in results:
            conversation.append({"role": "tool", "content": json.dumps(result["result"]), "tool_name": result["tool"]})

        all_tool_results.extend(results)
        break

    return {"conversation": conversation, "tool_results": all_tool_results, "iterations": iteration + 1 if tool_calls else 0}
```

### 第 5 步：参数验证

构建一个验证器，在执行之前根据 JSON Schema 检查工具调用的参数。

```python
def validate_tool_arguments(tool_name, arguments):
    if tool_name not in TOOL_REGISTRY:
        return [f"Unknown tool: {tool_name}"]

    schema = TOOL_REGISTRY[tool_name]["definition"]["function"]["parameters"]
    errors = []

    if not isinstance(arguments, dict):
        return [f"Arguments must be an object, got {type(arguments).__name__}"]

    for required_field in schema.get("required", []):
        if required_field not in arguments:
            errors.append(f"Missing required argument: {required_field}")

    properties = schema.get("properties", {})
    for arg_name, arg_value in arguments.items():
        if arg_name not in properties:
            errors.append(f"Unknown argument: {arg_name}")
            continue

        prop_schema = properties[arg_name]
        expected_type = prop_schema.get("type")

        type_checks = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
        if expected_type in type_checks:
            if not isinstance(arg_value, type_checks[expected_type]):
                errors.append(f"Argument '{arg_name}': expected {expected_type}, got {type(arg_value).__name__}")

        if "enum" in prop_schema and arg_value not in prop_schema["enum"]:
            errors.append(f"Argument '{arg_name}': '{arg_value}' not in {prop_schema['enum']}")

    return errors
```

### 第 6 步：运行演示

```python
def run_demo():
    register_all_tools()

    print("=" * 60)
    print("  Function Calling & Tool Use Demo")
    print("=" * 60)

    print("\n--- Registered Tools ---")
    for name, tool in TOOL_REGISTRY.items():
        desc = tool["definition"]["function"]["description"][:60]
        params = list(tool["definition"]["function"]["parameters"].get("properties", {}).keys())
        print(f"  {name}: {desc}...")
        print(f"    params: {params}")

    print(f"\n--- Argument Validation ---")
    validation_tests = [
        ("get_weather", {"city": "Tokyo"}, "Valid call"),
        ("get_weather", {}, "Missing required arg"),
        ("get_weather", {"city": "Tokyo", "units": "kelvin"}, "Invalid enum value"),
        ("calculator", {"expression": 123}, "Wrong type (int for string)"),
        ("unknown_tool", {"x": 1}, "Unknown tool"),
    ]
    for tool_name, args, label in validation_tests:
        errors = validate_tool_arguments(tool_name, args)
        status = "VALID" if not errors else f"ERRORS: {errors}"
        print(f"  {label}: {status}")

    print(f"\n--- Tool Execution ---")
    direct_tests = [
        {"name": "calculator", "arguments": {"expression": "(10 + 5) * 3 / 2"}},
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"name": "get_weather", "arguments": {"city": "Mars"}},
        {"name": "web_search", "arguments": {"query": "python function calling"}},
        {"name": "read_file", "arguments": {"path": "data/config.json"}},
        {"name": "read_file", "arguments": {"path": "../etc/passwd"}},
        {"name": "run_code", "arguments": {"code": "result = sum(range(1, 101))"}},
        {"name": "run_code", "arguments": {"code": "import os; os.system('rm -rf /')"}},
    ]
    for call in direct_tests:
        result = execute_tool_call(call)
        print(f"\n  {call['name']}({json.dumps(call['arguments'])})")
        print(f"    -> {json.dumps(result['result'], indent=None)[:100]}")
        print(f"    time: {result['execution_time_ms']}ms")

    print(f"\n--- Full Function Calling Loop ---")
    test_queries = [
        "What's the weather in Tokyo?",
        "Calculate (100 + 250) * 0.15",
        "Search for MCP protocol",
        "Read the config file",
        "Run some Python code",
        "Tell me a joke",
    ]
    for query in test_queries:
        print(f"\n  User: {query}")
        result = run_function_calling_loop(query)
        if result["tool_results"]:
            for tr in result["tool_results"]:
                print(f"    Tool: {tr['tool']} ({tr['execution_time_ms']}ms)")
                print(f"    Result: {json.dumps(tr['result'], indent=None)[:90]}")
        else:
            print(f"    [No tool called -- direct response]")
        print(f"    Iterations: {result['iterations']}")

    print(f"\n--- Parallel Tool Calls ---")
    multi_city_query = "What's the weather in tokyo and london?"
    print(f"  User: {multi_city_query}")
    result = run_function_calling_loop(multi_city_query)
    print(f"  Tool calls made: {len(result['tool_results'])}")
    for tr in result["tool_results"]:
        city = tr["result"].get("city", "unknown")
        temp = tr["result"].get("temp_c", "N/A")
        print(f"    {city}: {temp}C, {tr['result'].get('condition', 'N/A')}")

    print(f"\n--- Security Checks ---")
    security_tests = [
        ("read_file", {"path": "../../etc/passwd"}),
        ("run_code", {"code": "import subprocess; subprocess.run(['ls'])"}),
        ("calculator", {"expression": "__import__('os').system('ls')"}),
    ]
    for tool_name, args in security_tests:
        result = execute_tool_call({"name": tool_name, "arguments": args})
        blocked = result["result"].get("error", False)
        print(f"  {tool_name}({list(args.values())[0][:40]}): {'BLOCKED' if blocked else 'ALLOWED'}")
```

## 使用它

### OpenAI 函数调用

```python
# from openai import OpenAI
#
# client = OpenAI()
#
# tools = [{
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }
# }]
#
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
#     tools=tools,
#     tool_choice="auto",
# )
#
# tool_call = response.choices[0].message.tool_calls[0]
# args = json.loads(tool_call.function.arguments)
# result = get_weather(**args)
#
# final = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         response.choices[0].message,
#         {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)},
#     ],
# )
# print(final.choices[0].message.content)
```

OpenAI 将工具调用作为 `response.choices[0].message.tool_calls` 返回。每个调用都有一个 `id`，你在返回结果时必须包含它。模型使用此 ID 来匹配结果与调用。GPT-4o 可以在单个响应中返回多个工具调用 —— 迭代并执行所有调用。

### Anthropic 工具使用

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     tools=[{
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }],
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
# )
#
# tool_block = next(b for b in response.content if b.type == "tool_use")
# result = get_weather(**tool_block.input)
#
# final = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     tools=[...],
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         {"role": "assistant", "content": response.content},
#         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": json.dumps(result)}]},
#     ],
# )
```

Anthropic 将工具调用作为类型为 `"tool_use"` 的内容块返回。工具结果放在用户消息中，类型为 `"tool_result"`。注意关键区别：Anthropic 使用 `input_schema` 来定义工具参数，而 OpenAI 使用 `parameters`。

### MCP 集成

```python
# MCP servers expose tools over a standardized protocol.
# Any MCP-compatible client can discover and call these tools.
#
# Example: connecting to a Postgres MCP server
#
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
#
# server_params = StdioServerParameters(
#     command="npx",
#     args=["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
# )
#
# async with stdio_client(server_params) as (read, write):
#     async with ClientSession(read, write) as session:
#         await session.initialize()
#         tools = await session.list_tools()
#         result = await session.call_tool("query", {"sql": "SELECT count(*) FROM users"})
```

MCP 将工具实现与工具消费解耦。Postgres 服务器知道 SQL。GitHub 服务器知道 API。你的代理只需发现并调用工具 —— 不需要为每个集成编写特定于厂商的代码。

## 交付物

本课产出 `outputs/prompt-tool-designer.md` —— 一个用于设计工具定义的可复用提示词模板。向它描述你希望工具做什么，它会生成完整的 JSON Schema 定义，包括描述、类型和约束。

还产出 `outputs/skill-function-calling-patterns.md` —— 一个在生产环境中实现函数调用的决策框架，涵盖工具设计、错误处理、安全性以及各厂商特定的模式。

## 练习

1. **添加第 6 个工具：数据库查询。** 使用内存表实现一个模拟的 SQL 工具。该工具接受表名和过滤条件（而非原始 SQL）。验证表名在白名单中，过滤操作符限制为 `=`、`>`、`<`、`>=`、`<=`。将匹配的行以 JSON 形式返回。

2. **实现带错误反馈的重试。** 当工具调用失败时（例如，城市未找到），将错误消息反馈给模型决策函数，让它修正参数。记录每次调用所需的尝试次数。设置每个工具调用最多 3 次重试。

3. **构建多步骤代理。** 某些查询需要串联工具调用："读取配置文件，告诉我配置了什么模型，然后在网络上搜索该模型的定价。"实现一个循环，直到模型决定不再需要更多工具为止，将累积结果传入每个决策步骤。限制为 10 次迭代以防止无限循环。

4. **测量工具选择准确率。** 创建 30 个测试查询及其期望的工具名称。对所有 30 个查询运行你的决策函数，测量选择正确工具的百分比。识别哪些查询在工具之间造成最多的混淆。

5. **实现工具调用缓存。** 如果在 60 秒内以相同参数调用同一工具，返回缓存结果而非重新执行。使用以 `(tool_name, frozenset(args.items()))` 为键的字典。在一次包含 20 个查询的对话中测量缓存命中率。

## 关键术语

| 术语（英文） | 人们常说的 | 实际含义 |
|-------------|-----------|---------|
| Function calling（函数调用） | "工具使用" | 模型输出结构化 JSON，描述要以特定参数调用的函数 —— 你的代码执行它，而非模型 |
| Tool definition（工具定义） | "函数 Schema" | 一个 JSON Schema 对象，描述工具的名称、用途、参数和类型 —— 模型通过读取它来决定何时以及如何使用工具 |
| Tool choice（工具选择） | "调用模式" | 控制模型是必须调用工具（强制）、可以调用工具（自动），还是必须调用特定工具（指定名称） |
| Parallel calling（并行调用） | "多工具" | 模型在单轮中输出多个工具调用，减少往返次数 —— GPT-4o 和 Claude 都支持此功能 |
| Tool result（工具结果） | "函数输出" | 执行工具的返回值，作为消息发送回模型，以便模型在响应中使用真实数据 |
| Argument validation（参数验证） | "输入检查" | 在执行工具之前，验证模型生成的参数是否符合预期的类型、范围和约束 |
| MCP（模型上下文协议） | "工具协议" | Model Context Protocol —— Anthropic 提出的开放标准，通过服务器暴露工具，任何兼容的客户端都可以发现和调用 |
| Agent loop（代理循环） | "ReAct 循环" | 模型决定工具、代码执行工具、结果反馈回去的迭代循环，直到模型有足够信息来响应 |
| Tool poisoning（工具投毒） | "通过工具的提示注入" | 一种攻击方式，工具结果中包含操控模型行为的指令 —— 清洗所有工具输出 |
| Rate limiting（频率限制） | "调用预算" | 设置每次会话的最大工具调用次数，以防止无限循环和失控的 API 成本 |

## 拓展阅读

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) —— GPT-4o 工具使用的权威参考，包括并行调用、强制调用和结构化参数
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use) —— Claude 的工具使用实现，包含 input_schema、多工具响应和 tool_choice 配置
- [Model Context Protocol Specification](https://modelcontextprotocol.io) —— 跨 AI 应用程序工具互操作性的开放标准，包含服务器/客户端架构
- [Schick et al., 2023 -- "Toolformer: Language Models Can Teach Themselves to Use Tools"](https://arxiv.org/abs/2302.04761) —— 关于训练 LLM 自行决定何时以及如何调用外部工具的基础论文
- [Patil et al., 2023 -- "Gorilla: Large Language Model Connected with Massive APIs"](https://arxiv.org/abs/2305.15334) —— 对 LLM 进行微调以在 1,645 个 API 上实现准确的 API 调用，减少幻觉
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) —— 实时基准测试，比较 GPT-4o、Claude、Gemini 和开源模型的函数调用准确率
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) —— 思考-行动-观察循环，是每个工具调用外围的代理循环；本课结束的地方，第十四阶段接续
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) —— 五种可组合模式（提示链式、路由、并行、编排器-工作器、评估器-优化器），均基于单个工具使用原语构建
