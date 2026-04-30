"""
第五章 5.3：Transformer 架构
============================

Transformer 是现代 AI 的基础架构。
GPT、BERT、LLaMA、ChatGPT 等全部基于 Transformer。

"Attention Is All You Need" (2017) - 最重要的 AI 论文之一

本节内容：
1. Transformer 整体架构
2. 从零实现 Transformer Block
3. GPT 风格的解码器
4. 简单的文本生成
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print("=" * 60)
print("第五章 5.3：Transformer 架构")
print("=" * 60)

# ============================================================
# 1. Transformer 整体架构
# ============================================================
print("\n" + "=" * 60)
print("1. Transformer 整体架构")
print("=" * 60)

print("""
【Transformer 的两种变体】

1. Encoder-only (如 BERT):
   - 双向注意力（可以看到完整上下文）
   - 适合理解任务（分类、抽取等）

2. Decoder-only (如 GPT):
   - 因果注意力（只能看到左边的内容）
   - 适合生成任务（写文章、对话等）
   - 现在最流行的 LLM 架构！

3. Encoder-Decoder (如原始 Transformer):
   - 编码器处理输入，解码器生成输出
   - 适合翻译等序列到序列任务

【单个 Transformer Block 的结构】(Decoder 版本)

  输入
   ↓
  [多头自注意力 (Masked)]  ← 核心！
   ↓  + 残差连接
  [Layer Normalization]
   ↓
  [前馈网络 (FFN)]        ← 两层全连接
   ↓  + 残差连接
  [Layer Normalization]
   ↓
  输出

然后堆叠 N 个这样的 Block（GPT-3 有 96 层！）

【关键组件】
- 多头注意力: 学习词之间的关系
- FFN: 对每个位置独立地非线性变换
- 残差连接: 缓解深层网络的梯度问题
- Layer Norm: 稳定训练
""")


# ============================================================
# 2. 从零实现各组件
# ============================================================
print("\n" + "=" * 60)
print("2. 从零实现 Transformer 组件")
print("=" * 60)

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape
        
        Q = self.W_Q(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_O(context)


class FeedForward(nn.Module):
    """前馈网络 (Position-wise FFN)"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()  # GPT 使用 GELU 而不是 ReLU
    
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))


class TransformerBlock(nn.Module):
    """单个 Transformer Block (Decoder 风格)"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + LayerNorm
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # FFN + 残差连接 + LayerNorm
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        
        return x

# 测试
print("--- 测试 TransformerBlock ---")
d_model = 64
n_heads = 4
d_ff = 256

block = TransformerBlock(d_model, n_heads, d_ff)
x = torch.randn(2, 10, d_model)  # 2句话, 10个词, 64维

# 因果 mask
mask = torch.tril(torch.ones(10, 10)).unsqueeze(0).unsqueeze(0)
output = block(x, mask)

print(f"输入: {list(x.shape)}")
print(f"输出: {list(output.shape)}")
print(f"✓ TransformerBlock 正常工作")

# 验证残差连接（输出和输入维度相同）
assert x.shape == output.shape
print("✓ 输入输出形状一致（残差连接）")


# ============================================================
# 3. 完整的 GPT 模型
# ============================================================
print("\n" + "=" * 60)
print("3. 完整的 GPT 风格模型")
print("=" * 60)

class MiniGPT(nn.Module):
    """
    迷你 GPT 模型
    
    完整结构:
    Token Embedding + Position Embedding
    → N × TransformerBlock
    → LayerNorm
    → Linear (投影回词表大小)
    """
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer 层（堆叠 N 个）
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        
        # 嵌入
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)
        
        # 因果 mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # 通过所有 Transformer Block
        for block in self.blocks:
            x = block(x, mask)
        
        # 输出 logits
        x = self.norm(x)
        logits = self.output_projection(x)  # (batch, seq_len, vocab_size)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        """自回归生成"""
        for _ in range(max_new_tokens):
            # 截断到最大序列长度
            x = input_ids[:, -self.max_seq_len:]
            
            # 前向传播
            logits = self.forward(x)
            
            # 只看最后一个位置的预测
            next_token_logits = logits[:, -1, :] / temperature
            
            # 采样（或 argmax）
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 拼接
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids

# 创建迷你 GPT
print("""
创建一个迷你 GPT 模型:
  词表大小: 50
  嵌入维度: 64
  注意力头: 4
  层数: 3
  FFN 维度: 256
  最大序列长度: 32
""")

mini_gpt = MiniGPT(
    vocab_size=50,
    d_model=64,
    n_heads=4,
    n_layers=3,
    d_ff=256,
    max_seq_len=32
)

total_params = sum(p.numel() for p in mini_gpt.parameters())
print(f"模型参数量: {total_params:,}")
print(f"\n对比真实模型参数量:")
print(f"  GPT-2:    117M (1.17亿)")
print(f"  GPT-3:    175B (1750亿)")
print(f"  LLaMA-7B: 7B  (70亿)")
print(f"  我们的:   {total_params:,} (约{total_params//1000}K)")

# 测试前向传播
input_ids = torch.randint(0, 50, (2, 10))  # 2句话, 每句10个token
logits = mini_gpt(input_ids)
print(f"\n前向传播测试:")
print(f"  输入: {list(input_ids.shape)}")
print(f"  输出 logits: {list(logits.shape)}")
assert logits.shape == (2, 10, 50)
print("✓ 模型结构正确")


# ============================================================
# 4. 训练迷你 GPT
# ============================================================
print("\n" + "=" * 60)
print("4. 训练迷你 GPT (字符级语言模型)")
print("=" * 60)

print("""
【任务】
训练模型学会一个简单的模式：重复序列
输入: [1, 2, 3, 4, 5] → 预测下一个: [2, 3, 4, 5, 1]

这就是"下一个 token 预测"(Next Token Prediction)
= GPT 的核心训练目标！
""")

# 生成简单的训练数据（重复模式）
def generate_repeat_data(n_samples=500, seq_len=8, vocab_size=10):
    """生成重复模式的数据"""
    data = []
    for _ in range(n_samples):
        # 随机选一个短序列，然后重复
        pattern_len = np.random.randint(2, 5)
        pattern = np.random.randint(1, vocab_size, pattern_len)
        # 重复 pattern 填满 seq_len
        full_seq = np.tile(pattern, seq_len // pattern_len + 1)[:seq_len + 1]
        data.append(full_seq)
    return np.array(data)

np.random.seed(42)
data = generate_repeat_data(n_samples=1000, seq_len=16, vocab_size=10)
print(f"训练数据示例:")
for i in range(3):
    print(f"  输入: {data[i][:-1].tolist()}")
    print(f"  目标: {data[i][1:].tolist()}")
    print()

# 准备数据
train_data = torch.LongTensor(data[:800])
test_data = torch.LongTensor(data[800:])

# 创建较小的模型
small_gpt = MiniGPT(
    vocab_size=10,
    d_model=32,
    n_heads=4,
    n_layers=2,
    d_ff=128,
    max_seq_len=16
)

optimizer = optim.Adam(small_gpt.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练
print("--- 开始训练 ---")
n_epochs = 30
batch_size = 64

for epoch in range(n_epochs):
    small_gpt.train()
    total_loss = 0
    
    indices = np.random.permutation(len(train_data))
    for i in range(0, len(train_data), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch = train_data[batch_idx]
        
        input_ids = batch[:, :-1]   # 输入: 前15个token
        targets = batch[:, 1:]       # 目标: 后15个token（右移一位）
        
        # 前向传播
        logits = small_gpt(input_ids)
        
        # 计算损失
        loss = criterion(logits.view(-1, 10), targets.reshape(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if epoch % 5 == 0 or epoch == n_epochs - 1:
        # 评估
        small_gpt.eval()
        with torch.no_grad():
            test_input = test_data[:, :-1]
            test_target = test_data[:, 1:]
            test_logits = small_gpt(test_input)
            test_loss = criterion(test_logits.view(-1, 10), test_target.reshape(-1))
            
            # 准确率
            pred = test_logits.argmax(dim=-1)
            acc = (pred == test_target).float().mean()
        
        print(f"  Epoch {epoch:2d}: train_loss={total_loss/(len(train_data)//batch_size):.4f}, "
              f"test_loss={test_loss:.4f}, accuracy={acc:.4f}")

# 生成测试
print("\n--- 文本生成测试 ---")
small_gpt.eval()
# 给定前几个 token，让模型续写
prompts = [[1, 2, 3], [5, 6, 7], [3, 4]]

for prompt in prompts:
    input_ids = torch.LongTensor([prompt])
    generated = small_gpt.generate(input_ids, max_new_tokens=8, temperature=0.5)
    print(f"  输入: {prompt} → 生成: {generated[0].tolist()}")

print("\n(模型应该学会了重复模式)")


# ============================================================
# 5. 理解 GPT 的训练
# ============================================================
print("\n" + "=" * 60)
print("5. 理解大语言模型的训练")
print("=" * 60)

print("""
【GPT 的训练本质上就是我们刚才做的事情，只是规模大得多】

我们的模型:
  数据: 1000个简单重复序列
  参数: ~10K
  训练: 30 epochs, 几秒钟

GPT-3:
  数据: 整个互联网文本 (45TB)
  参数: 175B (1750亿)
  训练: 数千个 GPU 训练数月
  成本: 约 460 万美元

但核心思想完全一样：
  输入: "The cat sat on the"
  目标: "cat sat on the mat"
  损失: CrossEntropy(预测的下一个词, 真实的下一个词)

【为什么这么简单的目标能产生"智能"？】

预测下一个词迫使模型理解：
  - 语法: "He ___(goes/go) to school" → 需要理解主语单复数
  - 事实: "The capital of France is ___" → 需要"知道"地理知识
  - 推理: "If A>B and B>C, then A___(>/<)C" → 需要逻辑推理
  - 情感: "I'm so happy because ___" → 需要理解因果和情感

通过在海量文本上预测下一个词，模型被迫学到了语言的各种规律。

【关键技术突破】
1. 规模: 更大的模型 + 更多的数据 = 更好的效果 (Scaling Law)
2. Transformer: 可以高效并行训练（不像 RNN）
3. 对齐: RLHF (人类反馈强化学习) → 让模型输出更有帮助
""")

print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点：
1. Transformer Block = Multi-Head Attention + FFN + 残差 + LayerNorm
2. GPT = Token Embedding + Position Embedding + N × TransformerBlock + 输出投影
3. 训练目标: 给定前文，预测下一个 token (Next Token Prediction)
4. 生成方式: 自回归 — 一次生成一个 token，把生成的接到输入后面继续
5. 大语言模型的"智能"来自于在海量数据上做下一个词预测

我们的迷你 GPT 展示了完整的架构，
真实的 GPT 只是规模更大（更多层、更多头、更大维度、更多数据）

下一节：使用预训练模型 → Hugging Face 实践
""")
