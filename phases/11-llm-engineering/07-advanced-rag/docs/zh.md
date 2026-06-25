# 进阶 RAG（切块、重排序、混合搜索）

> 基本 RAG 检索 Top-K 个最相似的块，对于简单问题这可以工作。但在多跳推理、模糊查询和大规模语料库上就会崩溃。进阶 RAG 是在 10 份文档上工作的演示与在 1000 万份文档上工作的系统之间的区别。

**类型：** 构建
**语言：** Python
**前置要求：** 第 11 阶段，第 6 课（RAG）
**时间：** ~90 分钟
**相关课程：** 第 5 阶段·第 23 课（RAG 的切块策略）涵盖了所有六种切块算法——递归、语义、句子、父文档（Parent-Document）、延迟切块（Late Chunking）、上下文检索（Contextual Retrieval）——并包含 Vectara/Anthropic 的基准测试数据。本课程在此基础上构建：混合搜索、重排序、查询变换。

## 学习目标

- 实现进阶的切块策略（语义、递归、父子块），以保留文档结构和上下文
- 构建混合搜索流水线，结合 BM25 关键词匹配、语义向量搜索和交叉编码器（Cross-Encoder）重排序器
- 应用查询变换技术（HyDE、多查询、后退式查询）以提高对模糊或复杂问题的检索效果
- 诊断并修复常见的 RAG 失败：检索到错误的块、答案不在上下文中、多跳推理断裂

## 问题

你在第 6 课中构建了一个基本 RAG 流水线。它能在小型语料库上处理直接的问题。现在试试这些：

**模糊查询**："上季度的收入是多少？"语义搜索返回关于收入策略、收入预测以及 CFO 对收入增长看法的块。所有这些在语义上都与"收入"一词相似，但都不包含实际数字。正确的块说"2025 年 Q3 为 $47.2M"，但使用了"盈利"而非"收入"这个词。嵌入模型认为"收入策略"比"Q3 盈利为 $47.2M"更接近查询。

**多跳问题**："哪个团队的客户满意度评分提升最大？"这需要找到每个团队的满意度分数，比较它们，并识别最大值。没有单个块包含答案。信息分散在各团队的报告中。

**大规模语料库问题**：你有 200 万个块。正确答案在第 1,847,293 号块中。你的 Top-5 检索拉出了第 14、89,201、1,200,000、44 和 901,333 号块。在嵌入空间中很接近，但都不包含答案。在这种规模下，近似最近邻搜索引入了足够的误差，导致相关结果被挤出 Top-K。

基本 RAG 失败是因为向量相似度并不等同于相关性。一个块可以在语义上与查询相似，却对回答它毫无用处。进阶 RAG 用四种技术解决这个问题：混合搜索（添加关键词匹配）、重排序（更细致地为候选结果评分）、查询变换（在搜索前修正查询）以及更好的切块（以正确的粒度检索）。

## 概念

### 混合搜索：语义 + 关键词

语义搜索（向量相似度）擅长理解含义。"如何取消订阅？"能匹配"终止计划的步骤"，尽管它们不共享任何单词。但它会遗漏精确匹配。"错误代码 E-4021"可能无法匹配包含"E-4021"的块，如果嵌入模型将其视为噪音的话。

关键词搜索（BM25）则相反。它擅长精确匹配。"E-4021"完美匹配。但如果文档说"终止你的计划"，"取消订阅"这种查询会返回零结果。

混合搜索同时运行两者，然后合并结果。

**BM25**（最佳匹配 25，Best Matching 25）是标准的关键词搜索算法。自 1990 年代以来一直是搜索引擎的支柱。其公式为：

```
BM25(q, d) = 对查询 q 中的每个词项 t 求和：
    IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

其中 tf(t,d) 是词项 t 在文档 d 中的词频，IDF(t) 是逆文档频率，|d| 是文档长度，avgdl 是平均文档长度，k1 控制词频饱和（默认 1.2），b 控制长度归一化（默认 0.75）。

通俗解释：BM25 在文档包含查询词项（尤其是稀有词项）时给出更高评分，但对反复出现的词项有递减的回报。一篇出现 50 次"收入"的文档不会比出现 1 次的文档相关性高 50 倍。

### 倒数排名融合（RRF）

你有两个排序列表：一个来自向量搜索，一个来自 BM25。如何合并它们？倒数排名融合（Reciprocal Rank Fusion）是标准方法。

```
RRF_score(d) = 对所有排序列表 R 求和：
    1 / (k + rank_R(d))
```

其中 k 是一个常量（通常为 60），防止排名第一的结果压倒其他结果。

一篇在向量搜索中排第 1、BM25 中排第 5 的文档得到：1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318。

一篇在向量搜索中排第 3、BM25 中排第 2 的文档得到：1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320。

RRF 自然地平衡两种信号。在两个列表中都排名高的文档获得最佳分数。在一个列表中排第 1 但在另一个列表中不存在的文档获得中等分数。这种方法具有鲁棒性，因为它使用排名而非原始分数，所以两个系统的分数分布差异不会造成问题。

### 重排序

检索（无论是向量、关键词还是混合搜索）快速但不精确。它使用双编码器（Bi-Encoder）：查询和每个文档分别独立嵌入，然后进行比较。嵌入是预先计算并缓存的。这可以扩展到数百万个文档。

重排序使用交叉编码器（Cross-Encoder）：查询和候选文档一起输入模型，模型输出一个相关性分数。模型同时看到两段文本，能够捕获它们之间细粒度的交互。交叉编码器能够理解"What were Q3 earnings?"与包含"$47.2M in Q3"的块高度相关，即使双编码器错过了这种联系。

权衡：交叉编码器比双编码器慢 100-1000 倍，因为它们联合处理查询-文档对。你无法为一百万份文档预先计算交叉编码器分数。解决方案：检索一个更大的候选集（混合搜索得到 Top-50），然后用交叉编码器重排序得到最终的 Top-5。

```mermaid
graph LR
    Q["查询"] --> H["混合搜索"]
    H --> C50["Top 50 候选"]
    C50 --> RR["交叉编码器重排序"]
    RR --> C5["Top 5 最终结果"]
    C5 --> P["构建提示词"]
    P --> LLM["生成答案"]
```

常用的重排序模型（2026 年阵容）：
- Cohere Rerank 3.5：托管 API，多语言，在混合语料库上召回增益最佳
- Voyage rerank-2.5：托管 API，托管选项中延迟最低
- Jina-Reranker-v2 Multilingual：开放权重，100+ 语言
- bge-reranker-v2-m3：开放权重，强基线
- cross-encoder/ms-marco-MiniLM-L-6-v2：开放权重，可在 CPU 上运行用于原型开发
- ColBERTv2 / Jina-ColBERT-v2：延迟交互（Late Interaction）多向量重排序器——评分时复杂度与 Token 数量而非文档数量相关

### 查询变换

有时问题不在于检索，而在于查询本身。"关于新政策变更的那个事情是什么？"是一个糟糕的搜索查询。它不包含任何具体的词项。嵌入是模糊的。任何检索系统都无法从中找到正确的文档。

**查询重写**：将用户的查询重新表述为更好的搜索查询。LLM 可以做到这一点：

```
用户："关于新政策变更的那个事情是什么？"
重写后："近期的政策变更和更新"
```

**HyDE（假设文档嵌入，Hypothetical Document Embeddings）**：不直接用查询搜索，而是生成一个假设的答案，用该假设答案进行嵌入，然后搜索与之相似的真实的文档。

```
查询："企业版的退款政策是什么？"
假设答案："企业客户有资格在购买后 60 天内获得全额退款。
退款按剩余订阅周期按比例计算，并在 5-7 个工作日内处理。"
```

嵌入假设答案，搜索与之相似的真实的文档。直觉：假设答案在嵌入空间中比原始问题更接近真正的答案。问题和答案有不同的语言结构。通过生成假设答案，你弥合了嵌入空间中"问题空间"和"答案空间"之间的差距。

HyDE 在检索前增加了一次 LLM 调用。这会增加 500-2000ms 的延迟。当原始查询的检索质量较差时，这是值得的。

### 父子切块

标准切块强制做一个权衡：小块用于精确检索，大块用于充足上下文。父子切块消除了这种权衡。

对小块（128 Token）进行索引以用于检索。当一个小块被检索到时，返回其父块（512 Token）用于提示词。小块精确匹配查询。父块为 LLM 生成良好答案提供足够的上下文。

```mermaid
graph TD
    P["父块（512 Token）\n关于退款政策的完整章节"]
    C1["子块（128 Token）\n标准计划：30 天退款"]
    C2["子块（128 Token）\n企业版：60 天按比例退款"]
    C3["子块（128 Token）\n处理时间：5-7 天"]
    C4["子块（128 Token）\n如何提交请求"]

    P --> C1
    P --> C2
    P --> C3
    P --> C4

    Q["查询：企业退款？"] -.->|"匹配子块"| C2
    C2 -.->|"返回父块"| P
```

查询"企业退款？"精确匹配子块 C2。但提示词接收到的是完整的父块 P，其中包含处理时间和提交流程的周边上下文。

### 元数据过滤

在运行向量搜索之前，先按元数据过滤语料库：日期、来源、类别、作者、语言。这缩小了搜索空间，避免了不相关的结果。

"上个月的安全策略有什么变化？"应该只搜索过去 30 天内安全类别的文档。没有元数据过滤，你会搜索整个语料库，可能检索到一份碰巧在语义上相似的两年旧安全文档。

生产级 RAG 系统在每个块旁边存储元数据：源文档、创建日期、类别、作者、版本。向量数据库支持在相似度搜索之前按元数据进行预过滤，这对大规模性能至关重要。

### 评估

你构建了一个 RAG 系统，如何知道它是否有效？三个指标：

**检索相关性（Recall@k）**：对于一组带有已知相关文档的测试问题，相关文档在 Top-K 结果中出现的百分比是多少？如果问题的答案在块 #47 中，块 #47 是否出现在 Top-5 里？

**忠实度**：生成的答案是否以检索到的文档为依据？如果检索到的块说"60 天退款窗口"，而模型说"90 天退款窗口"，这是忠实度失败。模型尽管有正确的上下文，但仍然产生了幻觉。

**答案正确性**：生成的答案是否匹配预期答案？这是端到端的指标。它结合了检索质量和生成质量。

一个简单的忠实度检查：取生成答案中的每个声明，验证它是否（本质上）出现在检索到的块中。如果答案包含任何检索块中都没有的事实，那很可能是幻觉。

```mermaid
graph TD
    subgraph "评估框架"
        Q["测试问题\n+ 预期答案\n+ 相关文档 ID"]
        Q --> Ret["检索评估\nRecall@k：正确的\n文档被检索到了吗？"]
        Q --> Faith["忠实度评估\n答案是否以检索\n到的文档为依据？"]
        Q --> Correct["正确性评估\n答案是否匹配\n预期答案？"]
    end
```

## 构建

### 步骤 1：BM25 实现

```python
import math
from collections import Counter

class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_lengths = []
        self.avg_dl = 0
        self.doc_freqs = {}
        self.n_docs = 0

    def index(self, documents):
        self.docs = documents
        self.n_docs = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            words = doc.lower().split()
            self.doc_lengths.append(len(words))
            unique_words = set(words)
            for word in unique_words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        self.avg_dl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 1

    def score(self, query, doc_idx):
        query_words = query.lower().split()
        doc_words = self.docs[doc_idx].lower().split()
        doc_len = self.doc_lengths[doc_idx]
        word_counts = Counter(doc_words)
        score = 0.0

        for term in query_words:
            if term not in word_counts:
                continue
            tf = word_counts[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator

        return score

    def search(self, query, top_k=10):
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

### 步骤 2：倒数排名融合

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused
```

### 步骤 3：混合搜索流水线

```python
def hybrid_search(query, chunks, vector_embeddings, vocab, idf, bm25_index, top_k=5, fusion_k=60):
    query_emb = tfidf_embed(query, vocab, idf)
    vector_results = search(query_emb, vector_embeddings, top_k=top_k * 3)
    bm25_results = bm25_index.search(query, top_k=top_k * 3)
    fused = reciprocal_rank_fusion([vector_results, bm25_results], k=fusion_k)
    return fused[:top_k]
```

### 步骤 4：简单重排序器

在生产环境中，你会使用交叉编码器模型。这里我们构建一个重排序器，使用单词重叠、词项重要性和短语匹配来评分查询-文档相关性。

```python
def rerank(query, candidates, chunks):
    query_words = set(query.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "do", "does", "for", "of", "in", "to",
                  "and", "or", "on", "at", "by", "it", "its", "this", "that",
                  "with", "from", "be", "has", "have", "had", "not", "but"}
    query_terms = query_words - stop_words

    scored = []
    for doc_id, initial_score in candidates:
        chunk = chunks[doc_id].lower()
        chunk_words = set(chunk.split())

        term_overlap = len(query_terms & chunk_words)

        query_bigrams = set()
        q_list = [w for w in query.lower().split() if w not in stop_words]
        for i in range(len(q_list) - 1):
            query_bigrams.add(q_list[i] + " " + q_list[i + 1])
        bigram_matches = sum(1 for bg in query_bigrams if bg in chunk)

        position_boost = 0
        for term in query_terms:
            pos = chunk.find(term)
            if pos != -1 and pos < len(chunk) // 3:
                position_boost += 0.5

        rerank_score = (
            term_overlap * 1.0
            + bigram_matches * 2.0
            + position_boost
            + initial_score * 5.0
        )
        scored.append((doc_id, rerank_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
```

### 步骤 5：HyDE（假设文档嵌入）

```python
def hyde_generate_hypothesis(query):
    templates = {
        "what": "The answer to '{query}' is as follows: Based on our documentation, {topic} involves specific policies and procedures that define how the process works.",
        "how": "To address '{query}': The process involves several steps. First, you need to initiate the request. Then, the system processes it according to the defined rules.",
        "default": "Regarding '{query}': Our records indicate specific details and policies related to this topic that provide a comprehensive answer."
    }
    query_lower = query.lower()
    if query_lower.startswith("what"):
        template = templates["what"]
    elif query_lower.startswith("how"):
        template = templates["how"]
    else:
        template = templates["default"]

    topic_words = [w for w in query.lower().split()
                   if w not in {"what", "is", "the", "how", "do", "does", "a", "an",
                                "for", "of", "to", "in", "on", "at", "by", "and", "or"}]
    topic = " ".join(topic_words) if topic_words else "this topic"

    return template.format(query=query, topic=topic)


def hyde_search(query, chunks, vector_embeddings, vocab, idf, top_k=5):
    hypothesis = hyde_generate_hypothesis(query)
    hypothesis_emb = tfidf_embed(hypothesis, vocab, idf)
    results = search(hypothesis_emb, vector_embeddings, top_k)
    return results, hypothesis
```

### 步骤 6：父子切块

```python
def create_parent_child_chunks(text, parent_size=200, child_size=50):
    words = text.split()
    parents = []
    children = []
    child_to_parent = {}

    parent_idx = 0
    start = 0
    while start < len(words):
        parent_end = min(start + parent_size, len(words))
        parent_text = " ".join(words[start:parent_end])
        parents.append(parent_text)

        child_start = start
        while child_start < parent_end:
            child_end = min(child_start + child_size, parent_end)
            child_text = " ".join(words[child_start:child_end])
            child_idx = len(children)
            children.append(child_text)
            child_to_parent[child_idx] = parent_idx
            child_start += child_size

        parent_idx += 1
        start += parent_size

    return parents, children, child_to_parent
```

### 步骤 7：忠实度评估

```python
def evaluate_faithfulness(answer, retrieved_chunks):
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 1.0, []

    grounded = 0
    ungrounded = []
    context = " ".join(retrieved_chunks).lower()

    for sentence in answer_sentences:
        words = set(sentence.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                      "to", "of", "in", "for", "on", "at", "by", "it", "this", "that"}
        content_words = words - stop_words
        if not content_words:
            grounded += 1
            continue

        matched = sum(1 for w in content_words if w in context)
        ratio = matched / len(content_words) if content_words else 0

        if ratio >= 0.5:
            grounded += 1
        else:
            ungrounded.append(sentence)

    score = grounded / len(answer_sentences) if answer_sentences else 1.0
    return score, ungrounded


def evaluate_retrieval_recall(queries_with_relevant, retrieval_fn, k=5):
    total_recall = 0.0
    results = []

    for query, relevant_indices in queries_with_relevant:
        retrieved = retrieval_fn(query, k)
        retrieved_indices = set(idx for idx, _ in retrieved)
        relevant_set = set(relevant_indices)
        hits = len(retrieved_indices & relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 1.0
        total_recall += recall
        results.append({
            "query": query,
            "recall": recall,
            "hits": hits,
            "total_relevant": len(relevant_set)
        })

    avg_recall = total_recall / len(queries_with_relevant) if queries_with_relevant else 0
    return avg_recall, results
```

## 使用

使用真正的交叉编码器进行重排序：

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_cross_encoder(query, candidates, chunks, top_k=5):
    pairs = [(query, chunks[doc_id]) for doc_id, _ in candidates]
    scores = reranker.predict(pairs)
    scored = list(zip([doc_id for doc_id, _ in candidates], scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

使用 Cohere 的托管重排序器：

```python
import cohere

co = cohere.Client()

def rerank_with_cohere(query, candidates, chunks, top_k=5):
    docs = [chunks[doc_id] for doc_id, _ in candidates]
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=top_k
    )
    return [(candidates[r.index][0], r.relevance_score) for r in response.results]
```

使用真实 LLM 进行 HyDE：

```python
import anthropic

client = anthropic.Anthropic()

def hyde_with_llm(query):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"Write a short paragraph that would be a good answer to this question. Do not say you don't know. Just write what the answer would look like.\n\nQuestion: {query}"
        }]
    )
    return response.content[0].text
```

使用 Weaviate 进行生产级混合搜索：

```python
import weaviate

client = weaviate.connect_to_local()

collection = client.collections.get("Documents")
response = collection.query.hybrid(
    query="enterprise refund policy",
    alpha=0.5,
    limit=10
)
```

alpha 参数控制平衡：0.0 = 纯关键词（BM25），1.0 = 纯向量，0.5 = 等权重。大多数生产系统使用 0.3 到 0.7 之间的 alpha 值。

## 交付物

本课程产出：
- `outputs/prompt-advanced-rag-debugger.md`——一个用于诊断和修复 RAG 质量问题的提示词
- `outputs/skill-advanced-rag.md`——一个用于构建带有混合搜索和重排序的生产级 RAG 的技能

## 练习

1. 在示例文档上比较 BM25 vs 向量搜索 vs 混合搜索。对于 5 个测试查询中的每一个，记录哪种方法在第 1 位返回最相关的块。混合搜索应在至少 5 个中的 3 个上胜出。

2. 实现元数据过滤器。为每个文档添加一个"类别"字段（security、billing、api、product）。在运行向量搜索之前，将块过滤为仅相关类别。用"使用了什么加密方式？"进行测试，验证它只搜索安全类别的块。

3. 使用第 6 课中的简单生成函数构建完整的 HyDE 流水线。在所有 5 个测试查询上比较直接查询搜索和 HyDE 的检索质量（Top-3 相关性）。HyDE 应该对模糊查询有更好的结果。

4. 在示例文档上实现父子切块策略。使用 child_size=30 和 parent_size=100。用子块搜索但返回父块到提示词中。将生成的答案与 chunk_size=50 的标准切块方案进行比较。

5. 创建一个评估数据集：10 个带有已知答案块的问题。测量以下方法的 Recall@3、Recall@5 和 Recall@10：(a) 仅向量搜索，(b) 仅 BM25，(c) 混合搜索，(d) 混合搜索 + 重排序。绘制结果图表，识别重排序在哪些场景帮助最大。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------|--------|
| BM25 | "关键词搜索" | 一种概率排名算法，根据词频、逆文档频率和文档长度归一化为文档评分 |
| 混合搜索（Hybrid Search） | "两个世界的优点" | 同时运行语义（向量）和关键词（BM25）搜索，然后用排名融合合并结果 |
| 倒数排名融合（Reciprocal Rank Fusion） | "合并排序列表" | 对每个文档在所有列表中求和 1/(k + rank) 来合并多个排序列表 |
| 重排序（Reranking） | "第二轮评分" | 使用更昂贵的交叉编码器模型重新给初始检索的候选集打分 |
| 交叉编码器（Cross-Encoder） | "联合查询-文档模型" | 将查询和文档作为单一输入处理的模型，产生相关性评分；比双编码器更准确但对全语料库搜索来说太慢 |
| 双编码器（Bi-Encoder） | "独立嵌入模型" | 分别嵌入查询和文档的模型；因嵌入可预先计算而快速，但不如交叉编码器准确 |
| HyDE | "用假答案搜索" | 生成一个假设的查询答案，嵌入该假设答案，然后搜索与之相似的真实的文档 |
| 父子切块（Parent-Child Chunking） | "小搜索，大上下文" | 对小块索引以实现精确检索，但返回更大的父块以提供足够的上下文 |
| 元数据过滤（Metadata Filtering） | "搜索前缩小范围" | 在运行向量搜索之前按属性（日期、来源、类别）过滤文档以缩小搜索空间 |
| 忠实度（Faithfulness） | "是否保持 grounded" | 生成的答案是否被检索到的文档所支撑，而不是从模型的训练数据中产生的幻觉 |

## 进一步阅读

- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond"（2009）——BM25 的权威参考文献，解释了公式背后的概率基础
- Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods"（2009）——原始 RRF 论文，表明它优于更复杂的融合方法
- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels"（2022）——HyDE 论文，证明假设文档嵌入在没有任何训练数据的情况下改进了检索
- Nogueira & Cho, "Passage Re-ranking with BERT"（2019）——表明在 BM25 基础上使用交叉编码器重排序能显著提高检索质量
- [Khattab et al., "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" (2023)](https://arxiv.org/abs/2310.03714)——将提示词构建和权重选择视为检索流水线上的优化问题；理解"编程 LLM"而非"提示 LLM"
- [Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Microsoft Research 2024)](https://arxiv.org/abs/2404.16130)——GraphRAG 论文：实体-关系提取 + Leiden 社区检测用于以查询为中心的摘要；全局检索与局部检索的区别
- [Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (ICLR 2024)](https://arxiv.org/abs/2310.11511)——带有反思 Token 的自评估 RAG；超越静态检索再生成的 Agent 前沿
- [LangChain Query Construction blog](https://blog.langchain.dev/query-construction/)——如何将自然语言查询转化为结构化数据库查询（Text-to-SQL、Cypher），作为检索前步骤
