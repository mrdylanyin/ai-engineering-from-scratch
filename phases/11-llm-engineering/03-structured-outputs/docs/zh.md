# 结构化输出：JSON、Schema 验证与约束解码

> 你的 LLM 返回一个字符串。你的应用需要 JSON。这个差距导致的生产系统崩溃比任何模型幻觉都要多。结构化输出是自然语言与类型化数据之间的桥梁。做对了，你的 LLM 就变成了一个可靠的 API。做错了，你将在凌晨三点用正则表达式解析自由文本。

**类型：** 构建
**语言：** Python
**前置要求：** 第 10 阶段，第 01-05 课（从零构建 LLM）
**时间：** 约 90 分钟
**相关：** 第 5 阶段 · 第 20 课（结构化输出与约束解码）涵盖解码器级理论（FSM/CFG logit 处理器、Outlines、XGrammar）。本课聚焦于生产 SDK 层面（OpenAI `response_format`、Anthropic 工具使用、Instructor）——如果想了解 API 之下的原理，先阅读第 5 阶段 · 第 20 课。

## 学习目标

- 使用 OpenAI 和 Anthropic API 参数实现 JSON 模式和 schema 约束输出
- 构建一个 Pydantic 验证层，拒绝格式错误的 LLM 输出并通过错误反馈进行重试
- 解释约束解码如何在 token 级别强制有效 JSON，无需后处理
- 设计鲁棒的提取提示，将非结构化文本可靠地转换为类型化数据结构

## 问题

你让 LLM："从这段文本中提取产品名称、价格和库存状态。"它回复：

```
The product is the Sony WH-1000XM5 headphones, which cost $348.00 and are currently in stock.
```

这是一个完全正确的答案。但对你的应用来说完全没用。你的库存系统需要的是 `{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}`。你需要一个 JSON 对象，包含特定的键、特定的类型和特定的值约束。你不需要一个句子。

天真的解决方案：在提示中添加"以 JSON 格式回复"。这在 90% 的情况下有效。剩下 10% 的情况，模型在 markdown 代码块中包装 JSON，或添加"这里是 JSON："之类的前言，或因为过早闭合括号而产生语法无效的 JSON。你的 JSON 解析器崩溃了。你的管道中断了。你添加了 try/except 和重试循环。重试有时会产生不同的数据。现在你在解析问题之上又有了一个一致性问题。

这不是提示工程问题，而是解码问题。模型从左到右生成 token。在每个位置，它从 10 万+ 选项的词汇表中选择最可能的下一个 token。大多数选项在当前位置会产生无效的 JSON。如果模型刚刚输出了 `{"price":`，下一个 token 必须是数字、引号（用于字符串）、`null`、`true`、`false` 或负号。任何其他内容都会产生无效 JSON。没有约束时，模型可能选择一个完全合理的英语单词，但在语法上是灾难性的错误。

## 概念

### 结构化输出光谱

结构化输出控制有四个层次，每一层比上一层更可靠。

```mermaid
graph LR
    subgraph Spectrum["Structured Output Spectrum"]
        direction LR
        A["Prompt-based\n'Return JSON'\n~90% valid"] --> B["JSON Mode\nGuaranteed valid JSON\nNo schema guarantee"]
        B --> C["Schema Mode\nJSON + matches schema\nGuaranteed compliance"]
        C --> D["Constrained Decoding\nToken-level enforcement\n100% compliance"]
    end

    style A fill:#1a1a2e,stroke:#ff6b6b,color:#fff
    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D fill:#1a1a2e,stroke:#0f3460,color:#fff
```

**基于提示**（"以有效 JSON 回复"）：无强制执行。模型通常会遵守，但有时不会。可靠性：约 90%。失败模式：markdown 代码块、前言文本、截断输出、错误结构。

**JSON 模式**：API 保证输出是语法上有效的 JSON。OpenAI 的 `response_format: { type: "json_object" }` 启用此功能。输出解析不会出错。但可能不匹配你期望的 schema——多余的键、错误的类型、缺失的字段。

**Schema 模式**：API 接收一个 JSON Schema 并保证输出与之匹配。到 2026 年，每个主要提供商都原生支持此功能：OpenAI 的 `response_format: { type: "json_schema", json_schema: {...} }`（也以 `tool_choice="required"` 形式提供）、Anthropic 的带 `input_schema` 的工具使用、Gemini 的 `response_schema` + `response_mime_type: "application/json"`。输出具有你指定的确切键、类型和约束。

**约束解码（Constrained decoding）**：在生成过程中的每个 token 位置，解码器屏蔽所有会生成无效输出的 token。如果 schema 要求一个数字而模型即将输出一个字母，该 token 的概率被设为零。模型只能生成通向有效输出的 token。这是 OpenAI 的结构化输出模式以及 Outlines 和 Guidance 等库在底层实现的方式。

### JSON Schema：契约语言

JSON Schema 是你告诉模型（或验证层）输出必须具有什么形状的方式。每个主要的结构化输出系统都使用它。

```json
{
  "type": "object",
  "properties": {
    "product": { "type": "string" },
    "price": { "type": "number", "minimum": 0 },
    "in_stock": { "type": "boolean" },
    "categories": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["product", "price", "in_stock"]
}
```

这个 schema 表示：输出必须是一个对象，有一个字符串 `product`、一个非负数 `price`、一个布尔值 `in_stock`，以及一个可选的字符串数组 `categories`。任何不匹配的输出都会被拒绝。

Schema 处理了困难的情况：嵌套对象、类型化项的数组、枚举（将字符串约束为特定值）、模式匹配（字符串上的正则表达式）和组合器（oneOf、anyOf、allOf 用于多态输出）。

### Pydantic 模式

在 Python 中，你不必手工编写 JSON Schema。你定义一个 Pydantic 模型，它会自动生成 schema。

```python
from pydantic import BaseModel

class Product(BaseModel):
    product: str
    price: float
    in_stock: bool
    categories: list[str] = []
```

这会生成与上面相同的 JSON Schema。Instructor 库（以及 OpenAI 的 SDK）直接接受 Pydantic 模型：传入模型类，返回一个验证后的实例。如果 LLM 输出不匹配，Instructor 会自动重试。

### 函数调用/工具使用

这是解决相同问题的另一种接口。你定义带有类型参数的"工具"（函数），而非要求模型直接生成 JSON。模型输出带有结构化参数的函数调用。OpenAI 称之为"函数调用（function calling）"，Anthropic 称之为"工具使用（tool use）"。结果是一样的：结构化数据。

```mermaid
graph TD
    subgraph ToolUse["Tool Use Flow"]
        U["User: Extract product info\nfrom this review text"] --> M["Model processes input"]
        M --> TC["Tool Call:\nextract_product(\n  product='Sony WH-1000XM5',\n  price=348.00,\n  in_stock=true\n)"]
        TC --> V["Validate against\nfunction schema"]
        V --> R["Structured Result:\n{product, price, in_stock}"]
    end

    style U fill:#1a1a2e,stroke:#0f3460,color:#fff
    style TC fill:#1a1a2e,stroke:#e94560,color:#fff
    style V fill:#1a1a2e,stroke:#ffa500,color:#fff
    style R fill:#1a1a2e,stroke:#51cf66,color:#fff
```

当模型需要选择调用哪个函数（而非仅仅填充参数）时，工具使用更为可取。如果你有 10 种不同的提取 schema，模型必须根据输入选择正确的一种，工具使用给你 schema 选择和结构化输出双重保障。

### 常见失败模式

即使有 schema 强制执行，结构化输出也可能以微妙的方式失败。

**虚构的值（Hallucinated values）**：输出匹配 schema 但包含编造的数据。当文本写着 $348 时，模型生成 `{"price": 299.99}`。Schema 验证无法捕获此问题——类型正确，值错误。

**枚举混淆（Enum confusion）**：你将字段约束为 `["in_stock", "out_of_stock", "preorder"]`。模型输出 `"available"`——语义上正确，但不在允许的集合内。好的约束解码能防止此问题，基于提示的方法则不能。

**嵌套对象深度（Nested object depth）**：深度嵌套的 schema（4 层以上）产生更多错误。每一层嵌套都是模型可能丢失结构跟踪的另一处。

**数组长度（Array length）**：模型可能在数组中生成太多或太少的项。Schema 支持 `minItems` 和 `maxItems`，但并非所有提供商在解码层面强制执行。

**可选字段遗漏（Optional field omission）**：模型省略了技术上可选但对你的用例语义上重要的字段。即使数据有时缺失，也在 schema 中将其设为必需——强制模型显式生成 `null`。

## 构建它

### 步骤 1：JSON Schema 验证器

从零构建一个验证器，检查 Python 对象是否匹配 JSON Schema。这是输出端运行的合规性验证。

```python
import json

def validate_schema(data, schema):
    errors = []
    _validate(data, schema, "", errors)
    return errors

def _validate(data, schema, path, errors):
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object, got {type(data).__name__}")
            return
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: required field missing")
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}", errors)

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"{path}: expected array, got {type(data).__name__}")
            return
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", float("inf"))
        if len(data) < min_items:
            errors.append(f"{path}: array has {len(data)} items, minimum is {min_items}")
        if len(data) > max_items:
            errors.append(f"{path}: array has {len(data)} items, maximum is {max_items}")
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            _validate(item, items_schema, f"{path}[{i}]", errors)

    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: expected string, got {type(data).__name__}")
            return
        enum_values = schema.get("enum")
        if enum_values and data not in enum_values:
            errors.append(f"{path}: '{data}' not in allowed values {enum_values}")

    elif schema_type == "number":
        if not isinstance(data, (int, float)):
            errors.append(f"{path}: expected number, got {type(data).__name__}")
            return
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: {data} is less than minimum {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: {data} is greater than maximum {maximum}")

    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path}: expected boolean, got {type(data).__name__}")

    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{path}: expected integer, got {type(data).__name__}")
```

### 步骤 2：Pydantic 风格模型到 Schema 转换

构建一个最简的类到 schema 的转换器。定义一个 Python 类并自动生成其 JSON Schema。

```python
class SchemaField:
    def __init__(self, field_type, required=True, default=None, enum=None, minimum=None, maximum=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.enum = enum
        self.minimum = minimum
        self.maximum = maximum

def python_type_to_schema(field):
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    schema = {}

    if field.field_type in type_map:
        schema["type"] = type_map[field.field_type]
    elif field.field_type == list:
        schema["type"] = "array"
        schema["items"] = {"type": "string"}
    elif isinstance(field.field_type, dict):
        schema = field.field_type

    if field.enum:
        schema["enum"] = field.enum
    if field.minimum is not None:
        schema["minimum"] = field.minimum
    if field.maximum is not None:
        schema["maximum"] = field.maximum

    return schema

def model_to_schema(name, fields):
    properties = {}
    required = []

    for field_name, field in fields.items():
        properties[field_name] = python_type_to_schema(field)
        if field.required:
            required.append(field_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
```

### 步骤 3：约束 Token 过滤器

模拟约束解码。给定一个部分 JSON 字符串和一个 schema，确定当前位置哪些 token 类别是有效的。

```python
def next_valid_tokens(partial_json, schema):
    stripped = partial_json.strip()

    if not stripped:
        return ["{"]

    try:
        json.loads(stripped)
        return ["<EOS>"]
    except json.JSONDecodeError:
        pass

    last_char = stripped[-1] if stripped else ""

    if last_char == "{":
        return ['"', "}"]
    elif last_char == '"':
        if stripped.endswith('":'):
            return ['"', "0-9", "true", "false", "null", "[", "{"]
        return ["a-z", '"']
    elif last_char == ":":
        return [" ", '"', "0-9", "true", "false", "null", "[", "{"]
    elif last_char == ",":
        return [" ", '"', "{", "["]
    elif last_char in "0123456789":
        return ["0-9", ".", ",", "}", "]"]
    elif last_char == "}":
        return [",", "}", "]", "<EOS>"]
    elif last_char == "]":
        return [",", "}", "<EOS>"]
    elif last_char == "[":
        return ['"', "0-9", "true", "false", "null", "{", "[", "]"]
    else:
        return ["any"]

def demonstrate_constrained_decoding():
    partial_states = [
        '',
        '{',
        '{"product"',
        '{"product":',
        '{"product": "Sony"',
        '{"product": "Sony",',
        '{"product": "Sony", "price":',
        '{"product": "Sony", "price": 348',
        '{"product": "Sony", "price": 348}',
    ]

    print(f"{'Partial JSON':<45} {'Valid Next Tokens'}")
    print("-" * 80)
    for state in partial_states:
        valid = next_valid_tokens(state, {})
        display = state if state else "(empty)"
        print(f"{display:<45} {valid}")
```

### 步骤 4：提取流水线

将所有内容组合成一个提取流水线：定义 schema、模拟 LLM 产生结构化输出、验证输出并处理重试。

```python
def simulate_llm_extraction(text, schema, attempt=0):
    if "headphones" in text.lower() or "sony" in text.lower():
        if attempt == 0:
            return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true, "categories": ["audio", "headphones"]}'
        return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}'

    if "laptop" in text.lower():
        return '{"product": "MacBook Pro 16", "price": 2499.00, "in_stock": false, "categories": ["computers"]}'

    return '{"product": "Unknown", "price": 0, "in_stock": false}'

def extract_with_retry(text, schema, max_retries=3):
    for attempt in range(max_retries):
        raw = simulate_llm_extraction(text, schema, attempt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}: JSON parse error -- {e}")
            continue

        errors = validate_schema(data, schema)
        if not errors:
            return data

        print(f"  Attempt {attempt + 1}: Schema validation errors -- {errors}")

    return None

product_schema = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product", "price", "in_stock"],
}
```

### 步骤 5：运行完整流水线

```python
def run_demo():
    print("=" * 60)
    print("  Structured Output Pipeline Demo")
    print("=" * 60)

    print("\n--- Schema Definition ---")
    product_fields = {
        "product": SchemaField(str),
        "price": SchemaField(float, minimum=0),
        "in_stock": SchemaField(bool),
        "categories": SchemaField(list, required=False),
    }
    generated_schema = model_to_schema("Product", product_fields)
    print(json.dumps(generated_schema, indent=2))

    print("\n--- Schema Validation ---")
    test_cases = [
        ({"product": "Test", "price": 10.0, "in_stock": True}, "Valid object"),
        ({"product": "Test", "price": -5.0, "in_stock": True}, "Negative price"),
        ({"product": "Test", "in_stock": True}, "Missing price"),
        ({"product": "Test", "price": "ten", "in_stock": True}, "String as price"),
        ("not an object", "String instead of object"),
    ]

    for data, label in test_cases:
        errors = validate_schema(data, product_schema)
        status = "PASS" if not errors else f"FAIL: {errors}"
        print(f"  {label}: {status}")

    print("\n--- Constrained Decoding Simulation ---")
    demonstrate_constrained_decoding()

    print("\n--- Extraction Pipeline ---")
    texts = [
        "The Sony WH-1000XM5 headphones are priced at $348 and currently available.",
        "The new MacBook Pro 16-inch laptop costs $2499 but is sold out.",
        "This is a random sentence with no product info.",
    ]

    for text in texts:
        print(f"\n  Input: {text[:60]}...")
        result = extract_with_retry(text, product_schema)
        if result:
            print(f"  Output: {json.dumps(result)}")
        else:
            print(f"  Output: FAILED after retries")
```

## 使用它

### OpenAI 结构化输出

```python
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = OpenAI()
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# response = client.beta.chat.completions.parse(
#     model="gpt-5-mini",
#     messages=[
#         {"role": "system", "content": "Extract product information."},
#         {"role": "user", "content": "Sony WH-1000XM5, $348, in stock"},
#     ],
#     response_format=Product,
# )
#
# product = response.choices[0].message.parsed
# print(product.product, product.price, product.in_stock)
```

OpenAI 的结构化输出模式在内部使用约束解码。模型生成的每个 token 都保证产生匹配 Pydantic schema 的输出。无需重试，无需验证。约束被内置到解码过程中。

### Anthropic 工具使用

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-opus-4-7",
#     max_tokens=1024,
#     tools=[{
#         "name": "extract_product",
#         "description": "Extract product information from text",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "product": {"type": "string"},
#                 "price": {"type": "number"},
#                 "in_stock": {"type": "boolean"},
#             },
#             "required": ["product", "price", "in_stock"],
#         },
#     }],
#     messages=[{"role": "user", "content": "Extract: Sony WH-1000XM5, $348, in stock"}],
# )
```

Anthropic 通过工具使用实现结构化输出。模型发出一个带有匹配 input_schema 的结构化参数的工具调用。相同的结果，不同的 API 接口。

### Instructor 库

```python
# pip install instructor
# import instructor
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = instructor.from_openai(OpenAI())
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# product = client.chat.completions.create(
#     model="gpt-5-mini",
#     response_model=Product,
#     messages=[{"role": "user", "content": "Sony WH-1000XM5, $348, in stock"}],
# )
```

Instructor 封装任何 LLM 客户端，并添加了带验证的自动重试。如果第一次尝试未通过验证，它将错误作为上下文发送回模型并要求其修正输出。这适用于任何提供商，不仅仅是 OpenAI。

## 提交成果

本课产出 `outputs/prompt-structured-extractor.md`——一个可复用的提示模板，给定一个 schema 定义，可从任意文本中提取结构化数据。提供 JSON Schema 和非结构化文本，返回验证后的 JSON。

还产出 `outputs/skill-structured-outputs.md`——一个决策框架，根据提供商、可靠性要求和 schema 复杂度选择正确的结构化输出策略。

## 练习

1. 扩展 schema 验证器以支持 `oneOf`（数据必须匹配多个 schema 中的恰好一个）。这处理多态输出——例如，一个字段可以是 `Product` 或 `Service` 对象，它们具有不同的形状。

2. 构建一个"schema diff"工具，比较两个 schema 并识别破坏性变更（移除必需字段、更改类型）和非破坏性变更（添加可选字段、放宽约束）。这对在生产环境中对你的提取 schema 进行版本管理至关重要。

3. 实现一个更真实的约束解码模拟器。给定一个 JSON Schema 和一个 100 个 token 的词汇表（字母、数字、标点、关键字），逐步走过生成过程，在每个位置屏蔽无效 token。测量每个有效步骤词汇表中有效 token 的百分比。

4. 构建一个提取评估套件。创建 50 个带有手工标注 JSON 输出的产品描述。在所有 50 个上运行提取流水线，测量精确匹配、字段级准确率和类型合规性。识别哪些字段最难正确提取。

5. 向提取流水线添加"置信度分数"。对于每个提取的字段，估计模型的置信度（基于 token 概率，或通过运行 3 次提取并测量一致性）。将低置信度字段标记为人工审核。

## 关键术语

| 术语 | 人们怎么说 | 它实际上意味着什么 |
|------|----------------|----------------------|
| JSON 模式 | "返回 JSON" | 保证语法上有效 JSON 输出的 API 标志，但不强制执行任何特定 schema |
| 结构化输出（Structured output） | "类型化 JSON" | 匹配特定 JSON Schema 的输出，具有正确的键、类型和约束 |
| 约束解码（Constrained decoding） | "引导式生成" | 在每个 token 位置屏蔽会生成无效输出的 token——保证 100% schema 合规性 |
| JSON Schema | "一个 JSON 模板" | 一种声明式语言，用于描述 JSON 数据的结构、类型和约束（OpenAPI、JSON Forms 等使用） |
| Pydantic | "Python 数据类+" | 定义带类型验证的数据模型的 Python 库，FastAPI 和 Instructor 使用它生成 JSON Schema |
| 函数调用（Function calling） | "工具使用" | LLM 输出结构化函数调用（名称 + 类型化参数）而非自由文本——OpenAI 和 Anthropic 都支持 |
| Instructor | "LLM 的 Pydantic" | 封装 LLM 客户端以返回验证后的 Pydantic 实例，验证失败时自动重试的 Python 库 |
| Token 屏蔽（Token masking） | "过滤词汇表" | 在生成期间将特定 token 概率设为零，使模型无法生成它们 |
| Schema 合规性（Schema compliance） | "匹配形状" | 输出具有每个必需字段、正确类型、值在约束范围内，且没有额外的禁止字段 |
| 重试循环（Retry loop） | "重试直到成功为止" | 将验证错误发送回模型并要求其修正输出——Instructor 自动执行此操作，最多可配置的最大次数 |

## 进一步阅读

- [OpenAI 结构化输出指南](https://platform.openai.com/docs/guides/structured-outputs) —— OpenAI API 中基于 JSON Schema 的约束解码官方文档
- [Willard & Louf, 2023 —— "Efficient Guided Generation for Large Language Models"](https://arxiv.org/abs/2307.09702) —— Outlines 论文，描述了如何将 JSON Schema 编译为有限状态机以实现 token 级约束
- [Instructor 文档](https://python.useinstructor.com/) —— 从任何 LLM 获取结构化输出的标准库，带有 Pydantic 验证和重试
- [Anthropic 工具使用指南](https://docs.anthropic.com/en/docs/tool-use) —— Claude 如何通过带 JSON Schema input_schema 的工具使用实现结构化输出
- [JSON Schema 规范](https://json-schema.org/) —— 每个主要结构化输出系统使用的 schema 语言完整规范
- [Outlines 库](https://github.com/outlines-dev/outlines) —— 开源约束生成，使用正则表达式和编译为有限状态机的 JSON Schema
- [Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models" (MLSys 2025)](https://arxiv.org/abs/2411.15100) —— 当前最先进的语法引擎；下推自动机编译，以约 100 ns/token 的速度屏蔽 token
- [Beurer-Kellner et al., "Prompting Is Programming: A Query Language for Large Language Models" (LMQL)](https://arxiv.org/abs/2212.06094) —— LMQL 论文将约束解码框架化为带类型和值约束的查询语言
- [Microsoft Guidance（框架文档）](https://github.com/guidance-ai/guidance) —— 模板驱动的约束生成；Outlines 和 XGrammar 的跨厂商补充
