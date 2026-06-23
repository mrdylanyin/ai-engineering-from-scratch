# 从零构建一个 Tokenizer

> 第 01 课给了你一个玩具。本课给你一件武器。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 01 (Tokenizers: BPE, WordPiece, SentencePiece)
**Time:** ~90 minutes

## Learning Objectives

- 构建一个生产级的 BPE（字节对编码）tokenizer，处理 Unicode（统一码）、空白字符归一化（whitespace normalization）以及特殊 token
- 实现字节级回退（byte-level fallback），使 tokenizer 可以编码任何输入（包括 emoji、CJK 字符和代码），不会产生未知 token
- 添加预分词（pre-tokenization）正则模式，在应用 BPE 合并之前按词边界拆分文本
- 在一个语料库上训练自定义 tokenizer，并在多语言文本上评估其相对于 tiktoken 的压缩比

## The Problem

你在第 01 课中构建的 BPE tokenizer 可以处理英文文本。现在试试输入日语。或 emoji。或带有混合制表符和空格的 Python 代码。

它会崩溃。

不是因为 BPE 有问题——而是因为实现不完整。一个生产级 tokenizer 需要处理任意编码的原始字节、在拆分前对 Unicode 进行归一化、管理永远不被合并的特殊 token、将预分词与子词拆分串联起来，并且所有这些都要足够快，不能成为处理 15 万亿 token 的训练流水线的瓶颈。

GPT-2 的 tokenizer 有 50,257 个 token。Llama 3 有 128,256 个。GPT-4 大约有 100,000 个。这些不是玩具数字。这些词表背后的合并表（merge table）是在数百 GB 的文本上训练出来的，而周围的一系列机制——归一化、预分词、特殊 token 注入、聊天模板格式化——正是区分"只能处理 hello world 的 tokenizer"和"能处理整个互联网的 tokenizer"的关键。

你将亲手构建这些机制。

## The Concept

### 完整流水线

一个生产级 tokenizer 不是一个算法，而是一个由五个阶段组成的流水线，每个阶段解决不同的问题。

```mermaid
graph LR
    A[原始文本] --> B[归一化]
    B --> C[预分词]
    C --> D[BPE 合并]
    D --> E[特殊 Token]
    E --> F[Token ID]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段有特定的职责：

| 阶段 | 做什么 | 为什么重要 |
|-------|-------------|----------------|
| 归一化 | NFKC Unicode 归一化，可选小写化、去重音符 | "fi" 连字符（U+FB01）变成 "fi"（两个字符）。没有这个，同一个词会得到不同的 token。 |
| 预分词 | 在 BPE 之前将文本拆分为块 | 防止 BPE 跨词边界合并。"the cat" 永远不应该产生 "e c" 这种 token。 |
| BPE 合并 | 将学到的合并规则应用于字节序列 | 核心压缩。将原始字节转换为子词 token。 |
| 特殊 Token | 注入 [BOS]、[EOS]、[PAD]、聊天模板标记 | 这些 token 有固定的 ID，永远不参与 BPE 合并。模型需要它们来理解结构。 |
| ID 映射 | 将 token 字符串转换为整数 ID | 模型看到的是整数，而非字符串。 |

### 字节级 BPE

第 01 课的 tokenizer 在 UTF-8 字节上操作。这是正确的选择。但我们跳过了一件重要的事：当这些字节不是合法的 UTF-8 时会怎样？

字节级 BPE 通过将每一个可能的字节值（0-255）视为一个合法 token 来解决这个问题。你的基础词表恰好是 256 个条目。任何文件——文本、二进制、损坏的——都可以被 tokenize，而不会产生未知 token。

GPT-2 添加了一个技巧：将每个字节映射到一个可打印的 Unicode 字符，使词表保持人类可读。字节 0x20（空格）在其映射中变为字符 "G"。这纯粹是表面的，算法并不关心。

真正的威力在于：字节级 BPE 能处理地球上的每一种语言。中文字符每个是 3 个 UTF-8 字节。日语可以是 3-4 个字节。阿拉伯语、梵文天成体、emoji——都只是字节序列。BPE 算法在这些字节序列中寻找模式，与在英文 ASCII 字节中寻找模式的方式完全相同。

### 预分词

在 BPE 接触你的文本之前，你需要将其拆分为块。这可以防止合并算法创建跨词边界的 token。

GPT-2 使用正则表达式模式来拆分文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

这个模式按缩写拆分（"don't" 变成 "don" + "'t"）、带可选前导空格的单词、数字、标点符号和空白字符。前导空格保持附加在单词上——因此 "the cat" 变成 [" the", " cat"] 而不是 ["the", " ", "cat"]。

Llama 使用 SentencePiece，它完全跳过了正则表达式。它将原始字节流视为一个长序列，让 BPE 算法自己找出边界。这更简单，但给了 BPE 更多的自由来创建跨词的 token。

这个选择很重要。GPT-2 的正则表达式阻止 tokenizer 学到"一个词的末尾的 'the' 和下一个词的起始的 'the' 应该合并"。SentencePiece 允许这样做，这有时会产生更高效的压缩但更不易解释的 token。

### 特殊 Token

每个生产级 tokenizer 都为结构标记预留了 token ID：

| Token | 用途 | 使用者 |
|-------|---------|---------|
| `[BOS]` / `<s>` | 序列开始 | Llama 3, GPT |
| `[EOS]` / `</s>` | 序列结束 | 所有模型 |
| `[PAD]` | 批次对齐的填充 | BERT, T5 |
| `[UNK]` | 未知 token（字节级 BPE 已消除这个需求） | BERT, WordPiece |
| `<\|im_start\|>` | 聊天消息边界开始 | ChatGPT, Qwen |
| `<\|im_end\|>` | 聊天消息边界结束 | ChatGPT, Qwen |
| `<\|user\|>` | 用户发言标记 | Llama 3 |
| `<\|assistant\|>` | 助手发言标记 | Llama 3 |

特殊 token 永远不会被 BPE 拆分。它们在合并算法运行之前被精确匹配，替换为固定 ID，周围的文本则正常 tokenize。

### 聊天模板

这是大多数人感到困惑、也是大多数实现出问题的地方。

当你向聊天模型发送消息时，API 接受一个消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看到的不是 JSON，而是一个扁平的 token 序列。聊天模板使用特殊 token 将消息转换为该扁平序列。每个模型都做的不一样：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

模板搞错了，模型就会输出垃圾。它是在一种精确格式上训练的。任何偏差——缺少的换行、调换的 token、多余的空格——都会将输入放到训练分布之外。

### 速度

Python 对生产级 tokenization 来说太慢了。

tiktoken（OpenAI）是用 Rust 写的，带有 Python 绑定。HuggingFace tokenizers 也是 Rust。SentencePiece 是 C++。这些比纯 Python 快 10-100 倍。

感性认识一下：以每秒 100 万 token 的速度（快速 Python）处理 Llama 3 预训练的 15 万亿 token 需要 174 天。以每秒 1 亿 token 的速度（Rust）只需要 1.7 天。

你将在 Python 中构建以理解算法。在生产环境中，你会使用编译的实现，只接触 Python 包装器。

```figure
weight-tying
```

## Build It

### Step 1: Byte-Level Encoding

基础部分。将任意字符串转换为字节序列，将每个字节映射到可打印字符以便显示，并逆转该过程。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

在多语言文本上测试以查看字节数：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" 是 5 个字节。"你好" 是 6 个字节（每个字符 3 个）。火焰 emoji 是 4 个字节。字节级 tokenizer 不关心是什么语言。字节就是字节。

### Step 2: Pre-Tokenizer with Regex

使用 GPT-2 的正则模式将文本拆分为块。每个块独立由 BPE tokenize。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 模块支持 Unicode 属性转义（`\p{L}` 表示字母，`\p{N}` 表示数字）。标准库 `re` 模块不支持，所以我们回退到 ASCII 字符类。对于生产级多语言 tokenizer，请安装 `regex`。

试试看：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格保持附加在单词上。缩写按撇号拆分。标点符号成为独立的块。BPE 永远不会跨这些边界合并 token。

### Step 3: BPE on Byte Sequences

第 01 课的核心算法，但现在在预分词后的块上独立操作。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### Step 4: Special Token Handling

特殊 token 需要精确匹配和固定 ID。它们完全绕过 BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### Step 5: Full Tokenizer Class

将所有部分串联起来：归一化、按特殊 token 拆分、预分词、BPE 合并、映射到 ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### Step 6: Multilingual Test

真正的测试。把英文、中文、emoji 和代码一起扔进去。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

中文字符每个产生 3 个字节。emoji 产生 4 个字节。没有任何一个会让 tokenizer 崩溃，也没有任何一个会产生未知 token。这就是字节级 BPE 的威力。

## Use It

### Comparing Real Tokenizers

加载 Llama 3、GPT-4 和 Mistral 的实际 tokenizer。看每个如何处理同一段多语言文本。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

你会看到同一段文本的不同 token 数量。Llama 3 拥有 128K 词表，在合并常见模式上更激进。GPT-4 拥有 100K 词表，处于中间位置。Mistral 拥有 32K 词表，产生更多 token，但嵌入层（embedding layer）更小。

权衡总是相同的：更大的词表意味着更短的序列，但更多的参数。

## Ship It

本课产出一个用于构建和调试生产级 tokenizer 的提示词。参见 `outputs/prompt-tokenizer-builder.md`。

## Exercises

1. **Easy:** 添加一个 `get_token_bytes(id)` 方法，显示任意 token ID 的原始字节。用它来检查你最常用的合并 token 实际代表什么。
2. **Medium:** 实现 Llama 风格的预分词器，按空白字符和数字拆分但保留前导空格。将其词表与 GPT-2 正则方法在相同语料库上进行比较。
3. **Hard:** 添加一个聊天模板方法，接受 `{"role": ..., "content": ...}` 消息列表，并产生 Llama 3 聊天格式的正确 token 序列。将其与 HuggingFace 实现进行对比测试。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Byte-level BPE | "工作在字节上的 tokenizer" | 基础词表为 256 个字节值的 BPE——可以处理任何输入，不会产生未知 token |
| Pre-tokenization | "BPE 之前的拆分" | 基于正则或规则的拆分，防止 BPE 跨词边界合并 |
| NFKC normalization | "Unicode 清理" | 规范分解后接兼容性合成——"fi" 连字符变成 "fi"，全角 "A" 变成 "A" |
| Chat template | "消息如何变成 token" | 将角色/内容消息列表转换为扁平 token 序列的确切格式——因模型而异，必须匹配训练格式 |
| Special tokens | "控制 token" | 绕过 BPE 的保留 token ID——[BOS]、[EOS]、[PAD]、聊天标记——在合并前精确匹配 |
| Fertility | "每个词对应多少 token" | 输出 token 与输入词数的比率——GPT-4 在英文上约为 1.3，韩文为 2-3，比率越高意味着上下文窗口越浪费 |
| tiktoken | "OpenAI 的 tokenizer" | Rust 的 BPE 实现，带有 Python 绑定——比纯 Python 快 10-100 倍 |
| Merge table | "词表" | 训练中学习到的有序字节对合并列表——这本身就是 tokenizer 学到的知识 |

## Further Reading

- [OpenAI tiktoken 源码](https://github.com/openai/tiktoken) -- GPT-3.5/4 使用的 Rust BPE 实现
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- 支持 BPE、WordPiece、Unigram 的 Rust tokenizer 库
- [Llama 3 论文 (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- 关于 128K 词表和 tokenizer 训练的细节
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- 语言无关的 tokenization
- [GPT-2 tokenizer 源码](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- 原始的字节到 Unicode 映射
