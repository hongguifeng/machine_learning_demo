"""
第五章 5.1：词嵌入与 Word2Vec
==============================

在 NLP 中，第一个问题是：如何让计算机"理解"文字？

本节内容：
1. One-Hot 编码的问题
2. 词嵌入的直觉
3. Word2Vec 原理
4. 从零实现 Skip-gram
5. 词向量的有趣性质
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 60)
print("第五章 5.1：词嵌入与 Word2Vec")
print("=" * 60)

# ============================================================
# 1. One-Hot 的问题
# ============================================================
print("\n" + "=" * 60)
print("1. One-Hot 编码的问题")
print("=" * 60)

print("""
【One-Hot 编码回顾】
假设词表有 5 个词: [猫, 狗, 鱼, 苹果, 香蕉]

  猫   = [1, 0, 0, 0, 0]
  狗   = [0, 1, 0, 0, 0]
  鱼   = [0, 0, 1, 0, 0]
  苹果 = [0, 0, 0, 1, 0]
  香蕉 = [0, 0, 0, 0, 1]

【问题】
1. 维度灾难：实际词表有 10 万+ 个词
   每个词是 100000 维的向量！太大了

2. 稀疏：向量中只有 1 个位置是 1，其余全是 0
   浪费存储和计算

3. 无语义关系：
   cos(猫, 狗) = 0
   cos(猫, 苹果) = 0
   → 所有词的距离都一样！猫和狗不比猫和苹果更"近"

【我们需要什么？】
  一种低维、稠密的表示，能捕捉词的语义关系：
  
  猫 ≈ [0.8, -0.2, 0.5, ...]  (比如 128 维)
  狗 ≈ [0.7, -0.3, 0.4, ...]  (和猫接近！因为都是动物)
  苹果 ≈ [-0.1, 0.6, -0.3, ...] (和猫很远，因为不是动物)
  
  这就是词嵌入 (Word Embedding)！
""")

# 演示 One-Hot 的问题
vocab = ['猫', '狗', '鱼', '苹果', '香蕉']
one_hot = np.eye(len(vocab))

print("One-Hot 编码:")
for word, vec in zip(vocab, one_hot):
    print(f"  {word}: {vec}")

# 计算余弦相似度
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

print(f"\n余弦相似度:")
print(f"  cos(猫, 狗) = {cosine_similarity(one_hot[0], one_hot[1]):.4f}")
print(f"  cos(猫, 苹果) = {cosine_similarity(one_hot[0], one_hot[3]):.4f}")
print(f"  → 所有词对之间的相似度都是 0！无法区分语义关系")


# ============================================================
# 2. 词嵌入的直觉
# ============================================================
print("\n" + "=" * 60)
print("2. 词嵌入的直觉")
print("=" * 60)

print("""
【词嵌入 (Word Embedding)】
将每个词映射到一个低维稠密向量（如 100-300 维），
使得语义相近的词，向量也相近。

【如何学习词嵌入？】
核心思想（分布假说）：
  "You shall know a word by the company it keeps."
  一个词的含义由它周围的词决定。

例如：
  "我养了一只__，它会喵喵叫" → 猫
  "我养了一只__，它会汪汪叫" → 狗
  
  因为"猫"和"狗"经常出现在相似的上下文中，
  所以它们的向量应该很接近。

而 "苹果" 出现在 "我吃了一个__" 这样的上下文中，
和动物的上下文很不同，所以向量应该很远。

【Word2Vec 的两种模式】
1. Skip-gram: 给定中心词，预测周围的词
   输入: "猫" → 预测: "养", "一只", "它", "喵喵叫"
   
2. CBOW: 给定周围的词，预测中心词
   输入: "养", "一只", "它", "喵喵叫" → 预测: "猫"
""")


# ============================================================
# 3. 从零实现 Skip-gram
# ============================================================
print("\n" + "=" * 60)
print("3. 从零实现 Skip-gram Word2Vec")
print("=" * 60)

# 简单语料
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat chased the dog",
    "the dog chased the cat",
    "the cat and the dog are friends",
]

# 构建词表
def build_vocab(corpus):
    words = set()
    for sentence in corpus:
        for word in sentence.split():
            words.add(word)
    word2idx = {w: i for i, w in enumerate(sorted(words))}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word

word2idx, idx2word = build_vocab(corpus)
vocab_size = len(word2idx)
print(f"词表: {word2idx}")
print(f"词表大小: {vocab_size}")

# 生成 Skip-gram 训练数据
def generate_skipgram_data(corpus, word2idx, window_size=2):
    """生成 (中心词, 上下文词) 对"""
    pairs = []
    for sentence in corpus:
        words = sentence.split()
        for i, word in enumerate(words):
            center_idx = word2idx[word]
            # 窗口内的上下文词
            for j in range(max(0, i-window_size), min(len(words), i+window_size+1)):
                if j != i:
                    context_idx = word2idx[words[j]]
                    pairs.append((center_idx, context_idx))
    return pairs

pairs = generate_skipgram_data(corpus, word2idx, window_size=2)
print(f"\n训练样本数: {len(pairs)}")
print(f"前5个样本:")
for center, context in pairs[:5]:
    print(f"  ({idx2word[center]}, {idx2word[context]})")

# 实现 Skip-gram 模型
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # 两个嵌入矩阵：中心词和上下文词
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, center_words, context_words):
        # 获取嵌入向量
        center_embeds = self.center_embeddings(center_words)   # (batch, embed_dim)
        context_embeds = self.context_embeddings(context_words) # (batch, embed_dim)
        
        # 计算得分（点积）
        scores = (center_embeds * context_embeds).sum(dim=1)   # (batch,)
        return scores

# 训练
embed_dim = 10  # 小词表用小维度就够
model = SkipGram(vocab_size, embed_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 准备数据
center_words = torch.LongTensor([p[0] for p in pairs])
context_words = torch.LongTensor([p[1] for p in pairs])

print(f"\n--- 训练 Skip-gram ---")
print(f"嵌入维度: {embed_dim}")

n_epochs = 200
for epoch in range(n_epochs):
    # 正样本得分
    pos_scores = model(center_words, context_words)
    pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-10).mean()
    
    # 负采样：随机选择不是上下文的词作为负样本
    neg_words = torch.randint(0, vocab_size, (len(pairs),))
    neg_scores = model(center_words, neg_words)
    neg_loss = -torch.log(torch.sigmoid(-neg_scores) + 1e-10).mean()
    
    loss = pos_loss + neg_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"  Epoch {epoch}: loss = {loss.item():.4f}")

print("\n✓ 训练完成")


# ============================================================
# 4. 词向量的性质
# ============================================================
print("\n" + "=" * 60)
print("4. 词向量的有趣性质")
print("=" * 60)

# 获取学到的词向量
embeddings = model.center_embeddings.weight.detach().numpy()

print("学到的词向量:")
for word, idx in sorted(word2idx.items()):
    print(f"  {word:8s}: {embeddings[idx][:5].round(3)}...")

# 计算相似度
print(f"\n--- 词语相似度 ---")
def word_similarity(word1, word2, embeddings, word2idx):
    v1 = embeddings[word2idx[word1]]
    v2 = embeddings[word2idx[word2]]
    return cosine_similarity(v1, v2)

word_pairs = [('cat', 'dog'), ('cat', 'mat'), ('dog', 'log'), 
              ('sat', 'chased'), ('the', 'on')]
for w1, w2 in word_pairs:
    sim = word_similarity(w1, w2, embeddings, word2idx)
    print(f"  cos({w1}, {w2}) = {sim:.4f}")

print("""
\n注意: 由于语料很小，结果可能不太理想。
在真实的大规模语料(如 Wikipedia)上训练，词向量会展现出令人惊奇的性质：

经典例子 (使用预训练的 Word2Vec):
  king - man + woman ≈ queen
  paris - france + japan ≈ tokyo
  
这说明词向量捕捉到了 语义关系 和 类比关系！
""")

# ============================================================
# 5. 嵌入层在实际中的使用
# ============================================================
print("\n" + "=" * 60)
print("5. 嵌入层在神经网络中的使用")
print("=" * 60)

print("""
【nn.Embedding 的本质】
nn.Embedding 其实就是一个查找表（lookup table）：
  - 内部存储一个 (vocab_size × embed_dim) 的矩阵
  - 给定词的索引，返回对应行的向量

等价于：对 one-hot 向量做矩阵乘法
  one_hot × W = embedding
  [0,0,1,0,0] × W(5×3) = W 的第 3 行

但直接查表比矩阵乘法快！

【在完整模型中】
文本 → 分词 → 词索引 → Embedding 层 → 向量序列 → RNN/Transformer → 输出
""")

# 演示 Embedding 层
embedding_layer = nn.Embedding(num_embeddings=10, embedding_dim=4)
print(f"Embedding 权重矩阵形状: {embedding_layer.weight.shape}")

# 查询
input_indices = torch.LongTensor([2, 5, 7])
output_vectors = embedding_layer(input_indices)
print(f"\n输入索引: {input_indices.tolist()}")
print(f"输出向量:\n{output_vectors.detach().numpy().round(4)}")

# 验证：直接索引等价
manual = embedding_layer.weight[2]
assert torch.allclose(output_vectors[0], manual)
print("\n✓ Embedding(2) == weight[2]，验证通过")

# 批量处理句子
sentences = torch.LongTensor([
    [1, 3, 5, 2],   # 句子1 (4个词)
    [4, 6, 8, 0],   # 句子2 (4个词)
])
embedded = embedding_layer(sentences)
print(f"\n批量嵌入:")
print(f"  输入: {list(sentences.shape)} (2个句子, 每句4个词)")
print(f"  输出: {list(embedded.shape)} (2个句子, 4个词, 4维向量)")

print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点：
1. One-Hot 高维稀疏，无法表达语义关系
2. 词嵌入：低维稠密向量，语义相近的词距离也近
3. Word2Vec 通过预测上下文来学习词向量
4. Skip-gram: 中心词→预测上下文词
5. nn.Embedding 就是一个可学习的查找表
6. 词嵌入是所有 NLP 模型的第一层

现代演进：
  Word2Vec (2013) → GloVe (2014) → ELMo (2018) → BERT (2018)
  
  区别：
  - Word2Vec/GloVe: 每个词一个固定向量
  - ELMo/BERT: 同一个词在不同语境中有不同向量（上下文化表示）
    例: "苹果很好吃" vs "苹果发布了新手机"
        → "苹果"在两句中的向量不同！

下一节：注意力机制 → Transformer 的核心
""")
