"""
第五章 5.2：注意力机制
====================

注意力机制是 Transformer 和所有现代大语言模型的核心。
理解注意力 = 理解 GPT/BERT/LLaMA 等模型的基础。

本节内容：
1. 注意力的直觉
2. 缩放点积注意力 (Scaled Dot-Product Attention)
3. Q/K/V 的含义
4. 多头注意力 (Multi-Head Attention)
5. 自注意力 (Self-Attention)
6. 从零实现注意力机制
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("第五章 5.2：注意力机制")
print("=" * 60)

# ============================================================
# 1. 注意力的直觉
# ============================================================
print("\n" + "=" * 60)
print("1. 注意力的直觉")
print("=" * 60)

print("""
【为什么需要注意力？】

RNN 的问题：信息瓶颈
  "I love machine learning and deep learning is amazing"
  
  RNN 把整个句子压缩成一个固定大小的向量 h
  → 句子越长，信息损失越多
  → 翻译长句子时，开头的信息容易丢失

【注意力的核心思想】
不再只看最后一个隐藏状态，而是可以"注意"序列中任何位置。

比喻：读一篇文章回答问题
  问题: "作者住在哪里？"
  你不会从头到尾重新读一遍，而是"注意力"集中在提到地点的段落。

【注意力的计算方式】
对于要预测的每个位置，计算它和序列中所有位置的"相关度"：
  1. 计算相关度分数 (score)
  2. Softmax 归一化为权重 (attention weights)
  3. 用权重对所有位置的值加权求和

结果：每个位置的表示是"它最关注的那些位置"的信息的加权组合
""")


# ============================================================
# 2. Q/K/V 的含义
# ============================================================
print("\n" + "=" * 60)
print("2. Query / Key / Value 的含义")
print("=" * 60)

print("""
【Q/K/V 类比：信息检索系统】

想象你在图书馆找书：
  Query (查询): 你想找什么？ ("关于深度学习的书")
  Key (键):     每本书的标签/标题 ("机器学习入门", "深度学习实战", "烹饪指南")
  Value (值):   每本书的内容

查找过程：
  1. 用 Query 和每个 Key 比较 → 相关度分数
     "深度学习" vs "机器学习入门" → 中等相关
     "深度学习" vs "深度学习实战" → 高度相关
     "深度学习" vs "烹饪指南" → 不相关
  
  2. Softmax → 注意力权重
     [0.3, 0.65, 0.05]
  
  3. 用权重加权所有 Value → 最终结果
     = 0.3×书1内容 + 0.65×书2内容 + 0.05×书3内容
     → 主要包含"深度学习实战"的内容

【在 Self-Attention 中】
Q, K, V 都来自同一个输入序列（通过不同的线性变换得到）：
  Q = X @ W_Q    (这个位置"想找什么")
  K = X @ W_K    (这个位置"提供什么标签")
  V = X @ W_V    (这个位置"包含什么信息")
""")


# ============================================================
# 3. 缩放点积注意力
# ============================================================
print("\n" + "=" * 60)
print("3. 缩放点积注意力 (Scaled Dot-Product Attention)")
print("=" * 60)

print("""
【公式】
  Attention(Q, K, V) = softmax(Q × Kᵀ / √d_k) × V

  Q: (seq_len, d_k)  — 查询矩阵
  K: (seq_len, d_k)  — 键矩阵  
  V: (seq_len, d_v)  — 值矩阵
  d_k: Key 的维度

步骤分解：
  1. Q × Kᵀ → (seq_len, seq_len)  — 每对位置的相关度
  2. / √d_k → 缩放（防止点积值太大导致 softmax 饱和）
  3. softmax → 注意力权重（每行和为 1）
  4. × V → 加权组合的输出

为什么除以 √d_k？
  如果 d_k 很大，点积的值会很大，softmax 会接近 one-hot，
  梯度会很小（softmax 的梯度在极端值处接近 0）。
  除以 √d_k 让方差稳定在 1 附近。
""")

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力
    Q: (batch, seq_len, d_k)
    K: (batch, seq_len, d_k)
    V: (batch, seq_len, d_v)
    """
    d_k = Q.shape[-1]
    
    # 1. 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    # scores: (batch, seq_len, seq_len)
    
    # 2. 可选的 mask（用于因果注意力，防止看到未来的信息）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 3. Softmax 归一化
    attention_weights = F.softmax(scores, dim=-1)
    
    # 4. 加权求和
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

# 示例
print("\n--- 示例：3个词的自注意力 ---")
seq_len = 3
d_k = 4

# 假设输入是3个词的嵌入
X = torch.randn(1, seq_len, d_k)  # (batch=1, seq=3, dim=4)
print(f"输入 X: shape={list(X.shape)}")
print(f"  (3个词，每个用4维向量表示)")

# 线性变换得到 Q, K, V（简化版，实际中权重是学习的）
W_Q = torch.randn(d_k, d_k)
W_K = torch.randn(d_k, d_k)
W_V = torch.randn(d_k, d_k)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

output, attn_weights = scaled_dot_product_attention(Q, K, V)

print(f"\nQ, K, V shape: {list(Q.shape)}")
print(f"注意力权重 shape: {list(attn_weights.shape)}")
print(f"注意力权重:\n{attn_weights[0].detach().numpy().round(4)}")
print(f"  每行表示: 该位置对各位置的注意力分配")
print(f"  每行之和: {attn_weights[0].sum(dim=-1).detach().numpy().round(4)} (都是1)")
print(f"\n输出 shape: {list(output.shape)}")


# ============================================================
# 4. 因果注意力 (Causal Attention)
# ============================================================
print("\n" + "=" * 60)
print("4. 因果注意力（GPT 使用的）")
print("=" * 60)

print("""
【因果注意力 / 掩码注意力】
在语言生成中，预测下一个词时不能"偷看"后面的词！

例如: "The cat sat on the ___"
  预测第6个词时，只能看到前5个词。

实现方式：用一个上三角 mask，把未来位置的分数设为 -∞
  softmax(-∞) = 0 → 完全忽略未来位置

  mask:
  [[1, 0, 0, 0],     位置1只能看到自己
   [1, 1, 0, 0],     位置2能看到位置1和自己
   [1, 1, 1, 0],     位置3能看到位置1-3
   [1, 1, 1, 1]]     位置4能看到所有

这就是 GPT 使用的 "decoder-only" 注意力！
""")

# 创建因果 mask
seq_len = 5
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)  # 下三角
print(f"因果 mask (下三角):\n{causal_mask[0].int().numpy()}")

# 用因果 mask 做注意力
Q_causal = torch.randn(1, seq_len, d_k)
K_causal = torch.randn(1, seq_len, d_k)
V_causal = torch.randn(1, seq_len, d_k)

output_causal, weights_causal = scaled_dot_product_attention(
    Q_causal, K_causal, V_causal, mask=causal_mask)

print(f"\n因果注意力权重:\n{weights_causal[0].detach().numpy().round(3)}")
print(f"\n验证: 上三角部分全为 0（不看未来）")
upper = weights_causal[0].detach().numpy()
for i in range(seq_len):
    for j in range(i+1, seq_len):
        assert abs(upper[i][j]) < 1e-6, f"位置({i},{j})不为0!"
print("✓ 验证通过：每个位置只注意到自己和之前的位置")


# ============================================================
# 5. 多头注意力
# ============================================================
print("\n" + "=" * 60)
print("5. 多头注意力 (Multi-Head Attention)")
print("=" * 60)

print("""
【为什么要多头？】

单个注意力头只能捕捉一种"关系模式"。
但语言中有多种关系需要同时捕捉：

  "The cat sat on the mat because it was tired"
  
  - 语法关系: "it" → "cat" (代词指代)
  - 位置关系: "sat" → "on" (动作+方位)
  - 语义关系: "tired" → "cat" (状态的主体)

多头注意力 = 多个注意力头并行工作，各关注不同模式
  
【实现方式】
  1. 将 d_model 维的 Q/K/V 分成 h 个头
     每个头的维度: d_k = d_model / h
  2. 每个头独立计算注意力
  3. 拼接所有头的输出
  4. 通过一个线性层投影回 d_model 维

例: d_model=512, h=8 → 每个头 d_k=64
""")

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性投影层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)  # 输出投影
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        seq_len = Q.shape[1]
        
        # 线性投影
        Q = self.W_Q(Q)  # (batch, seq, d_model)
        K = self.W_K(K)
        V = self.W_V(V)
        
        # 分头: (batch, seq, d_model) → (batch, n_heads, seq, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # 广播到所有头
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # 合并头: (batch, n_heads, seq, d_k) → (batch, seq, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.W_O(context)
        
        return output, attention_weights

# 测试多头注意力
d_model = 32
n_heads = 4
mha = MultiHeadAttention(d_model, n_heads)

X = torch.randn(2, 5, d_model)  # 2个句子，每个5个词，32维
output, attn = mha(X, X, X)  # Self-attention: Q=K=V=X

print(f"输入: {list(X.shape)} (2句, 5词, 32维)")
print(f"输出: {list(output.shape)}")
print(f"注意力权重: {list(attn.shape)} (2句, 4头, 5×5)")
print(f"\n第1句第1头的注意力权重:")
print(attn[0, 0].detach().numpy().round(3))
print(f"\n第1句第2头的注意力权重 (不同的模式!):")
print(attn[0, 1].detach().numpy().round(3))
print("\n→ 不同头学到不同的注意力模式！")


# ============================================================
# 6. 位置编码
# ============================================================
print("\n" + "=" * 60)
print("6. 位置编码 (Positional Encoding)")
print("=" * 60)

print("""
【问题】
注意力机制是"无序"的！它不知道词的位置。
  "dog bites man" 和 "man bites dog" 的注意力结果一样！

但位置信息很重要！

【解决方案：位置编码】
给每个位置加上一个固定的"位置向量"：
  input = word_embedding + position_encoding

Transformer 使用正弦/余弦位置编码：
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

为什么用正弦/余弦？
  1. 值域有界 [-1, 1]
  2. 不同位置的编码不同
  3. 相对位置可以通过线性变换得到
     (sin(a+b) = sin(a)cos(b) + cos(a)sin(b))
""")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

# 测试
pe = PositionalEncoding(d_model=32)
x = torch.randn(1, 10, 32)  # 10个词
x_with_pos = pe(x)

print(f"输入: {list(x.shape)}")
print(f"加位置编码后: {list(x_with_pos.shape)}")
print(f"\n位置编码的前4个位置的前8维:")
pe_values = pe.pe[0, :4, :8].numpy().round(3)
for pos in range(4):
    print(f"  位置{pos}: {pe_values[pos]}")
print(f"\n→ 每个位置都有独特的编码")

print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点：
1. 注意力 = 动态地关注序列中最相关的位置
2. Q·Kᵀ/√d_k → softmax → ×V (缩放点积注意力公式)
3. Q=查询, K=键(标签), V=值(内容)
4. 因果 mask: 生成模型不能偷看未来 (GPT)
5. 多头注意力: 多种关系模式并行学习
6. 位置编码: 补充注意力缺失的位置信息

这是 Transformer 的核心！

下一节：完整的 Transformer 架构 → 把所有组件组合在一起
""")
