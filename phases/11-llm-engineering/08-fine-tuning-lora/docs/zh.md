# 使用 LoRA 和 QLoRA 进行微调

> 全量微调一个 7B 模型需要 56GB 显存。你没有这个条件。大多数公司也没有。LoRA 允许你在 6GB 显存中微调同一个模型，只需训练不到 1% 的参数。这不是一种妥协——它在大多数任务上匹配全量微调的质量。整个开源微调生态系统都建立在这一个技术上。

**类型：** 构建
**语言：** Python
**前置要求：** 第 10 阶段，第 6 课（指令微调/SFT）
**时间：** ~75 分钟
**相关课程：** 第 10 阶段从零覆盖了 SFT/DPO 循环。本课程将这些连接到 2026 年的 PEFT 工具链（PEFT、TRL、Unsloth、Axolotl、LLaMA-Factory）。

## 学习目标

- 通过将低秩适配器矩阵（A 和 B）注入预训练模型的注意力层来实现 LoRA
- 计算 LoRA 与全量微调相比的参数节省：秩 r 在 d_model 维度下训练 2*r*d 个参数，而非 d²
- 使用 QLoRA（4 比特量化基座 + LoRA 适配器）微调模型，使其放入消费级 GPU 内存
- 将 LoRA 权重合并回基座模型以进行部署，并比较有适配器和无适配器的推理速度

## 问题

你有一个基座模型：Llama 3 8B。你希望它用你公司的口吻回复客户支持工单。SFT 是答案。但 SFT 有成本问题。

全量微调会更新模型中的每个参数。Llama 3 8B 有 80 亿个参数。在 fp16 下，每个参数占用 2 个字节。只加载权重就需要 16GB。在训练过程中，你还需要梯度（16GB）、Adam 的优化器状态（32GB，用于动量 + 方差）和激活值。总计：一个 8B 模型的训练大约需要 56GB 显存。

一张 A100 80GB 勉强能装下。两张 A100 在云服务商那里的费用是 $3-4/小时。在 50,000 个样本上训练 3 个 Epoch 需要 6-10 小时。每次实验花费 $30-40。做 10 次实验来调试超参数，还没部署就花了 $400。

将规模扩大到 Llama 3 70B，数字变得荒谬。仅权重就需要 140GB。你需要一个集群。每次实验 $100+。

还有一个更深层的问题。全量微调会修改模型中的每个权重。如果你用客户支持数据微调，你可能会退化模型的一般能力。这叫做灾难性遗忘（Catastrophic Forgetting）。模型在你的任务上变得更好，但在其他所有任务上变得更差。

你需要一种方法，能够训练更少的参数、使用更少的内存，并且不会破坏模型已有的知识。

## 概念

### LoRA：低秩适配（Low-Rank Adaptation）

微软的 Edward Hu 及其同事于 2021 年 6 月发表了 LoRA。该论文的洞察：微调过程中的权重更新具有低的内在秩。你不需要更新一个 4096x4096 权重矩阵中全部的 1670 万个参数。更新中有用的信息可以被一个秩为 16 或 32 的矩阵捕捉到。

数学如下。一个标准的线性层计算：

```
y = Wx
```

其中 W 是一个 d_out x d_in 的矩阵。对于一个 4096x4096 的注意力投影，那是 16,777,216 个参数。

LoRA 冻结 W，并添加一个低秩分解：

```
y = Wx + BAx
```

其中 B 是 (d_out x r)，A 是 (r x d_in)。秩 r 远小于 d——通常为 8、16 或 32。

对于 4096x4096 层，r=16：
- 原始参数：4096 x 4096 = 16,777,216
- LoRA 参数：(4096 x 16) + (16 x 4096) = 65,536 + 65,536 = 131,072
- 减少比例：131,072 / 16,777,216 = 0.78%

你只训练了 0.78% 的参数，却获得了 95-100% 的质量。

```mermaid
graph LR
    X["输入 x"] --> W["冻结的 W (d x d)"]
    X --> A["A (r x d)"]
    A --> B["B (d x r)"]
    W --> Plus["+ (合并)"]
    B --> Plus
    Plus --> Y["输出 y"]

    style W fill:#1a1a2e,stroke:#e94560,color:#fff
    style A fill:#0f3460,stroke:#16213e,color:#fff
    style B fill:#0f3460,stroke:#16213e,color:#fff
```

A 用随机高斯分布初始化。B 初始化为零。这意味着 LoRA 的贡献从零开始——模型从原始行为开始训练，逐步学习适配。

### 缩放因子：Alpha

LoRA 引入了一个缩放因子 alpha，控制低秩更新对输出影响的程度：

```
y = Wx + (alpha / r) * BAx
```

当 alpha = r 时，缩放是 1 倍。当 alpha = 2r（常用默认值）时，缩放是 2 倍。这个超参数独立于基础学习率来控制 LoRA 路径的学习率。

实践指南：
- alpha = 2 * rank 是常见的社区惯例（原始论文在大多数实验中使用 alpha = rank）
- alpha = rank 给出 1 倍缩放，保守但稳定
- 更高的 alpha 意味着每步更大的更新，可以加速收敛或导致不稳定

### 在哪里应用 LoRA

一个 Transformer 有很多线性层。你不需要对所有层都加 LoRA。原始论文测试了不同的组合：

| 目标层 | 可训练参数（7B） | 质量 |
|--------|----------------|------|
| 仅 q_proj | 4.7M | 好 |
| q_proj + v_proj | 9.4M | 更好 |
| q_proj + k_proj + v_proj + o_proj | 18.9M | 注意力最佳 |
| 所有线性层（注意力 + MLP） | 37.7M | 边际增益，参数量翻倍 |

大多数任务的最佳点：q_proj + v_proj。这定位到自注意力中的查询和值投影，它们控制模型关注什么以及提取什么信息。增加 MLP 层对复杂任务（如代码生成）有帮助，但在更简单的任务上会使参数量翻倍但收益递减。

### 秩的选择

秩 r 控制适配的表达能力：

| 秩 | 每层可训练参数 | 最适合 |
|----|------------|-------|
| 4 | 32,768 | 简单分类、情感分析 |
| 8 | 65,536 | 单领域问答、摘要 |
| 16 | 131,072 | 多领域任务、指令遵循 |
| 32 | 262,144 | 复杂推理、代码生成 |
| 64 | 524,288 | 大多数任务收益递减 |
| 128 | 1,048,576 | 很少有必要 |

Hu 等人表明，r=4 已经能捕获简单任务的大部分适配。r=8 和 r=16 是实践中最常见的选择。超过 r=64 很少能提升质量，并开始失去 LoRA 的内存优势。

### QLoRA：4 比特量化 + LoRA

华盛顿大学的 Tim Dettmers 及其同事于 2023 年 5 月发表了 QLoRA。其思想：将冻结的基座模型量化为 4 比特精度，然后在其上附加 fp16 的 LoRA 适配器。

这极大地改变了内存方程：

| 方法 | 权重内存（7B） | 训练内存（7B） | 所需 GPU |
|------|--------------|-------------|---------|
| 全量微调（fp16） | 14GB | ~56GB | 1x A100 80GB |
| LoRA（fp16 基座） | 14GB | ~18GB | 1x A100 40GB |
| QLoRA（4 比特基座） | 3.5GB | ~6GB | 1x RTX 3090 24GB |

QLoRA 做出了三项技术贡献：

**NF4（Normal Float 4-bit）**：一种专门为神经网络权重设计的新数据类型。神经网络权重大致遵循正态分布。NF4 将其 16 个量化水平放置在标准正态分布的分位数上。这对于正态分布数据是在信息论上最优的。它比均匀 4 比特量化（INT4）或标准 Float4 损失更少的信息。

**双重量化**：量化常量本身占用内存。每个 64 个权重的块需要一个 fp32 的缩放因子（4 字节）。对于一个 7B 模型，这是额外的 0.4GB。双重量化将这些常量量化为 fp8，将开销减少到 0.1GB。虽小但积少成多。

**分页优化器（Paged Optimizers）**：在训练过程中，优化器状态（Adam 的动量和方差）在长序列上可能超出 GPU 内存。分页优化器使用 NVIDIA 的统一内存（Unified Memory），在 GPU 内存耗尽时自动将优化器状态换出到 CPU RAM，需要时再换回。这防止了 OOM 崩溃，代价是一些吞吐量。

### 质量问题

减少参数或量化基座会损害质量吗？多篇论文的结果：

| 方法 | MMLU (5-shot) | MT-Bench | HumanEval |
|------|-------------|----------|-----------|
| 全量微调（Llama 2 7B） | 48.3 | 6.72 | 14.6 |
| LoRA r=16 | 47.9 | 6.68 | 14.0 |
| QLoRA r=16 (NF4) | 47.5 | 6.61 | 13.4 |
| QLoRA r=64 (NF4) | 48.1 | 6.70 | 14.2 |

r=16 的 LoRA 在大多数基准测试上比全量微调差不到 1%。r=16 的 QLoRA 再损失零点几个百分点。r=64 的 QLoRA 基本匹配全量微调，同时使用少 90% 的内存。

### 真实成本

在 50,000 个样本上微调 Llama 3 8B（3 个 Epoch）：

| 方法 | GPU | 时间 | 成本 |
|------|----|------|------|
| 全量微调 | 2x A100 80GB | 8 hours | ~$32 |
| LoRA r=16 | 1x A100 40GB | 4 hours | ~$8 |
| QLoRA r=16 | 1x RTX 4090 24GB | 6 hours | ~$5 |
| QLoRA r=16 (Unsloth) | 1x RTX 4090 24GB | 2.5 hours | ~$2 |
| QLoRA r=16 | 1x T4 16GB | 12 hours | ~$4 |

在单个消费级 GPU 上用 QLoRA 微调的成本比一顿饭还便宜。这就是为什么开源权重微调社区在 2023 年爆发，以及为什么 2026 年下面所列的每个训练框架默认都支持 QLoRA。

### 2026 年 PEFT 工具栈

| 框架 | 它是什么 | 何时选择 |
|------|---------|--------|
| **Hugging Face PEFT** | 规范的 LoRA/QLoRA/DoRA/IA3 库 | 你需要原始控制，且训练循环已基于 `transformers.Trainer` |
| **TRL** | Hugging Face 的强化反馈训练器（SFT、DPO、GRPO、PPO、ORPO） | 你在 SFT 之后需要 DPO/GRPO；构建在 PEFT 之上 |
| **Unsloth** | 前向/后向传播的 Triton 内核重写 | 你需要 2-5 倍加速 + 一半显存且无精度损失；Llama/Mistral/Qwen 系列 |
| **Axolotl** | PEFT + TRL + DeepSpeed + Unsloth 上的 YAML 配置封装 | 你需要可复现、版本控制的训练运行 |
| **LLaMA-Factory** | PEFT + TRL 上的 GUI/CLI/API | 你需要零代码微调；支持 100+ 模型系列 |
| **torchtune** | 原生 PyTorch 方案，无 `transformers` 依赖 | 你需要最少依赖，且组织已标准化使用 PyTorch |

经验法则：研究用途或一次性实验 → PEFT。可复现的生产流水线 → 启用 Unsloth 内核的 Axolotl。随手原型开发 → LLaMA-Factory。

### 合并适配器

训练后，你有两样东西：冻结的基座模型和一个小的 LoRA 适配器（通常 10-100MB）。你可以：

1. **保持分离**：加载基座模型，在其上加载适配器。为不同任务交换适配器。这就是你从一个基座模型提供多个微调变体的方式。

2. **永久合并**：计算 W' = W + (alpha/r) * BA，将结果保存为一个新的全量模型。合并后的模型与原始模型大小相同。没有推理开销。无需管理适配器。

对于提供多个任务（客户支持适配器、代码适配器、翻译适配器）的场景，保持分离。对于部署单个专用模型，合并。

用于合并多个适配器的高级合并技术：

- **TIES-Merging**（Yadav et al. 2023）：修剪小幅值参数，解决符号冲突，然后合并。减少适配器之间的干扰。
- **DARE**（Yu et al. 2023）：在合并前随机丢弃适配器参数，并重新缩放其余部分。在组合能力方面出奇地有效。
- **任务算术（Task Arithmetic）**：简单地相加或相减适配器权重。添加一个"代码"适配器和一个"数学"适配器通常产生一个两者都擅长的模型。

### 何时不微调

微调是第三种选择，不是第一种。

**第一：提示工程。**编写更好的系统提示。添加少样本示例。使用思维链。这不用花钱，几分钟就能完成。如果提示工程能让你达到 80% 的效果，你可能就不需要微调。

**第二：RAG。**如果模型需要了解你的特定数据（文档、知识库、产品目录），检索比将其烘焙到权重中更便宜、更易维护。参见第 6 课。

**第三：微调。**当你需要模型采用特定的风格、格式或推理模式，而这些无法通过提示实现时，使用微调。当你需要一致的结构化输出时。当你需要将更大的模型蒸馏到一个更小的模型中时。当延迟很重要，你无法承受少样本提示带来的额外 Token 消耗时。

```mermaid
graph TD
    Start["需要更好的模型行为？"] --> PE["尝试提示工程"]
    PE -->|"有效"| Done["交付"]
    PE -->|"不够"| RAG["需要外部知识？"]
    RAG -->|"需要"| RAGBuild["构建 RAG 流水线"]
    RAG -->|"不需要，需要风格/格式变更"| FT["用 LoRA/QLoRA 微调"]
    RAGBuild -->|"有效"| Done
    RAGBuild -->|"还需要风格变更"| FT
    FT --> Done

    style Start fill:#1a1a2e,stroke:#e94560,color:#fff
    style Done fill:#0f3460,stroke:#16213e,color:#fff
```

```figure
lora-params
```

## 构建

我们在纯 PyTorch 中从零实现 LoRA。不用库。不用魔法。你将构建 LoRA 层，将其注入模型，训练它，并合并权重回去。

### 步骤 1：LoRA 层

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Parameter(torch.randn(in_features, rank) * (1 / math.sqrt(rank)))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling
```

A 用缩放后的随机值初始化。B 初始化为零。乘积 BA 从零开始，所以模型从其原始行为开始。

### 步骤 2：LoRA 包装的线性层

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

原始的线性层被冻结。只有 LoRA 参数（A 和 B）是可训练的。

### 步骤 3：将 LoRA 注入模型

```python
def inject_lora(model, target_modules, rank=8, alpha=16):
    for param in model.parameters():
        param.requires_grad = False

    lora_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = dict(model.named_modules())[parent_name]
                lora_linear = LinearWithLoRA(module, rank, alpha)
                setattr(parent, child_name, lora_linear)
                lora_layers[name] = lora_linear
    return lora_layers
```

首先，冻结模型中的每个参数。然后遍历模型树，找到匹配目标名称的线性层，用 LoRA 包装版本替换它们。LoRA 的 A 和 B 矩阵是整个模型中唯一可训练的参数。

### 步骤 4：统计参数

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0
    }
```

### 步骤 5：合并权重回去

```python
def merge_lora_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            with torch.no_grad():
                merged = (
                    module.lora.A @ module.lora.B
                ) * module.lora.scaling
                module.linear.weight.data += merged.T
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            setattr(parent, child_name, module.linear)
```

合并后，LoRA 层消失了。模型与原始大小相同，适配已烘焙到权重中。没有推理开销。

### 步骤 6：模拟的 QLoRA 量化

```python
def quantize_to_nf4(tensor, block_size=64):
    blocks = tensor.reshape(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True).values / 7.0
    scales = torch.clamp(scales, min=1e-8)
    quantized = torch.round(blocks / scales).clamp(-8, 7).to(torch.int8)
    return quantized, scales

def dequantize_from_nf4(quantized, scales, original_shape):
    dequantized = quantized.float() * scales
    return dequantized.reshape(original_shape)
```

这通过将权重映射到 64 个块的 16 个离散级别来模拟 4 比特量化。生产级 QLoRA 使用 bitsandbytes 库在 GPU 上实现真正的 NF4。

### 步骤 7：训练循环

```python
def train_lora(model, data, epochs=5, lr=1e-3, batch_size=4):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        indices = torch.randperm(len(data["inputs"]))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x = data["inputs"][batch_idx]
            y = data["targets"][batch_idx]

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

    return losses
```

### 步骤 8：完整演示

```python
def demo():
    torch.manual_seed(42)
    d_model = 256
    n_classes = 10

    model = nn.Sequential(
        nn.Linear(d_model, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, n_classes),
    )

    n_samples = 500
    x = torch.randn(n_samples, d_model)
    y = torch.randint(0, n_classes, (n_samples,))
    y_onehot = torch.zeros(n_samples, n_classes).scatter_(1, y.unsqueeze(1), 1.0)

    data = {"inputs": x, "targets": y_onehot}

    params_before = count_parameters(model)

    lora_layers = inject_lora(
        model, target_modules=["0", "2"], rank=8, alpha=16
    )

    params_after = count_parameters(model)

    losses = train_lora(model, data, epochs=20, lr=1e-3)

    merge_lora_weights(model)
    params_merged = count_parameters(model)

    return {
        "params_before": params_before,
        "params_after": params_after,
        "params_merged": params_merged,
        "losses": losses,
    }
```

演示创建一个小模型，将 LoRA 注入两层，训练它，然后将权重合并回去。参数数量从全部可训练下降到 LoRA 训练期间约 1% 的可训练参数，合并后恢复到原始架构。

## 使用

使用 Hugging Face 生态系统，在真实模型上使用 LoRA 大约需要 20 行代码：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

对于 QLoRA，添加 bitsandbytes 量化：

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

model = get_peft_model(model, lora_config)
```

就这样。相同的训练循环。相同的数据流水线。基座模型现在以 4 比特存储，LoRA 适配器在 fp16 下训练，整个东西放入 6GB 显存。

使用 Hugging Face Trainer 进行训练：

```python
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

training_args = TrainingArguments(
    output_dir="./lora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./lora-adapter")
```

保存的适配器只有 10-100MB。基座模型保持不变。你可以在 Hugging Face Hub 上分享适配器而无需重新分发完整模型。

## 交付物

本课程产出：
- `outputs/prompt-lora-advisor.md`——一个帮助你就特定任务决定 LoRA 秩、目标模块和超参数的提示词
- `outputs/skill-fine-tuning-guide.md`——一个教授 Agent 何时以及如何进行微调决策树的技能

## 练习

1. **秩消融研究。**以秩 2、4、8、16、32 和 64 运行演示。绘制最终损失与秩的关系。找到收益递减的点，即加倍秩不再能使损失减半的位置。对于 256 维特征的简单分类任务，这个点应该在大约 r=8-16 附近。

2. **目标模块比较。**修改 `inject_lora` 仅针对第 "0" 层、仅第 "2" 层、仅第 "4" 层以及所有三层。训练每个变体 20 个 Epoch。比较收敛速度和最终损失。这反映了在实际中选择 q_proj vs v_proj vs 所有线性层的决策。

3. **量化误差分析。**在 `quantize_to_nf4` / `dequantize_from_nf4` 前后获取训练模型的权重矩阵。计算均方误差、最大绝对误差以及原始权重与重建权重之间的相关性。尝试 block_size 值为 32、64、128 和 256。

4. **多适配器服务。**在不同的数据子集（偶数索引 vs 奇数索引）上训练两个 LoRA 适配器。保存两个适配器。加载基座模型一次，然后交换适配器并验证每个适配器对相同输入产生不同的输出。这就是生产系统从一个基座模型提供多个微调模型的方式。

5. **合并 vs 未合并推理。**比较 LoRA 模型在 `merge_lora_weights` 前后对相同 100 个输入的输出。验证输出完全相同（在 1e-5 的浮点容差内）。然后对两者的推理速度进行基准测试——合并后应该稍快，因为它是单个矩阵乘法，而不是两个。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------|--------|
| LoRA | "高效微调" | 低秩适配：冻结基座权重，训练两个小矩阵 A 和 B，其乘积近似完整的权重更新 |
| QLoRA | "在笔记本上微调" | 量化 LoRA：以 4 比特 NF4 加载基座模型，在上层以 fp16 训练 LoRA 适配器，使 7B 模型能在 6GB 显存中微调 |
| 秩（r） | "模型能学到多少" | A 和 B 矩阵的内部维度；控制表达能力 vs 参数数量 |
| Alpha | "LoRA 的学习率" | 应用于 LoRA 输出的缩放因子；alpha/r 决定了适配对最终输出的贡献程度 |
| NF4 | "4 比特量化" | Normal Float 4：一种 4 比特数据类型，量化水平位于正态分布分位数上，对神经网络权重最优 |
| 适配器（Adapter） | "小的已训练部分" | 保存为单独文件（10-100MB）的 LoRA A 和 B 矩阵，可加载到基座模型的任意副本上 |
| 目标模块（Target Modules） | "哪些层加 LoRA" | 注入 LoRA 适配器的特定线性层（如 q_proj、v_proj 等） |
| 合并（Merging） | "烘焙进去" | 计算 W + (alpha/r) * BA 并替换原始权重，消除推理时的适配器开销 |
| 分页优化器（Paged Optimizers） | "训练时不要 OOM" | 当 GPU 内存耗尽时，将优化器状态（Adam 动量、方差）卸载到 CPU |
| 灾难性遗忘（Catastrophic Forgetting） | "微调破坏了其他一切" | 当更新所有权重导致模型丧失之前所学的能力 |

## 进一步阅读

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"（2021）——引入低秩分解方法的原始论文，在 GPT-3 175B 上以低至 4 的秩进行测试
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized Language Models"（2023）——引入 NF4、双重量化和分页优化器，使 65B 模型能在单张 48GB GPU 上微调
- PEFT 库文档（huggingface.co/docs/peft）——Hugging Face 生态系统中 LoRA、QLoRA 及其他参数高效方法的标准库
- Yadav et al., "TIES-Merging: Resolving Interference When Merging Models"（2023）——在不降低质量的情况下组合多个 LoRA 适配器的技术
- [Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (NeurIPS 2023)](https://arxiv.org/abs/2305.18290)——DPO 推导；SFT 之后的偏好调优阶段，无需奖励模型
- [TRL 文档](https://huggingface.co/docs/trl/)——`SFTTrainer`、`DPOTrainer`、`KTOTrainer` 及与 PEFT/bitsandbytes/Unsloth 集成接口的官方参考
- [Unsloth 文档](https://docs.unsloth.ai/)——使微调吞吐量翻倍、内存减半的融合内核；TRL 下的性能层
- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/)——基于 YAML 配置的多 GPU SFT/DPO/QLoRA 训练器；手写脚本的配置即代码替代方案
