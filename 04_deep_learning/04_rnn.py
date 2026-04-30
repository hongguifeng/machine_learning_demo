"""
第四章 4.4：循环神经网络 (RNN)
==============================

RNN 用于处理序列数据（文本、时间序列、语音等）。
核心思想：有"记忆"，前面的信息会影响后面的判断。

本节内容：
1. RNN 的直觉与结构
2. 从零实现 RNN
3. 梯度消失问题
4. LSTM（长短期记忆）
5. 用 PyTorch 实现序列预测
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 60)
print("第四章 4.4：循环神经网络 (RNN)")
print("=" * 60)

# ============================================================
# 1. RNN 的直觉
# ============================================================
print("\n" + "=" * 60)
print("1. RNN 的直觉与结构")
print("=" * 60)

print("""
【为什么需要 RNN？】

全连接网络和 CNN 处理的是固定大小的输入。
但很多数据是序列/变长的：
  - 文本: "我 爱 学习 AI" (4个词)
  - 时间序列: 股票价格 [100, 102, 98, 105, ...]
  - 语音: 每秒上万个采样点

而且序列中的顺序很重要！
  "狗 咬 人" vs "人 咬 狗" — 词一样，意思完全不同

【RNN 的核心思想】

普通网络: input → output (独立处理每个输入)
RNN:      input₁ → hidden₁ → input₂ → hidden₂ → ... → output

每一步都有一个"隐藏状态" h，像是网络的"记忆"：
  h_t = f(W_h × h_{t-1} + W_x × x_t + b)
  
  当前记忆 = 激活(上一步记忆的变换 + 当前输入的变换)

展开来看:
  时间步1: h₁ = tanh(W_h × h₀ + W_x × x₁ + b)
  时间步2: h₂ = tanh(W_h × h₁ + W_x × x₂ + b)
  时间步3: h₃ = tanh(W_h × h₂ + W_x × x₃ + b)
  ...
  
  所有时间步共享同一组权重 W_h, W_x, b！

关键：h₃ 包含了 x₁, x₂, x₃ 的信息（通过 h₁, h₂ 传递）
""")


# ============================================================
# 2. 从零实现 RNN
# ============================================================
print("\n" + "=" * 60)
print("2. 从零实现 RNN")
print("=" * 60)

class SimpleRNN:
    """从零实现的 RNN（前向传播）"""
    
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        
        # 权重初始化
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01   # 输入→隐藏
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # 隐藏→隐藏
        self.b_h = np.zeros((1, hidden_size))                          # 隐藏偏置
        
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01  # 隐藏→输出
        self.b_y = np.zeros((1, output_size))                          # 输出偏置
    
    def forward(self, X_sequence):
        """
        X_sequence: (seq_len, input_size) — 一个序列
        返回: 所有时间步的输出和隐藏状态
        """
        seq_len = X_sequence.shape[0]
        
        # 初始隐藏状态
        h = np.zeros((1, self.hidden_size))
        
        all_h = []  # 记录所有隐藏状态
        all_y = []  # 记录所有输出
        
        for t in range(seq_len):
            x_t = X_sequence[t:t+1]  # (1, input_size)
            
            # RNN 核心公式
            h = np.tanh(x_t @ self.W_xh + h @ self.W_hh + self.b_h)
            
            # 输出
            y_t = h @ self.W_hy + self.b_y
            
            all_h.append(h.copy())
            all_y.append(y_t.copy())
        
        return np.vstack(all_y), np.vstack(all_h)

# 演示
rnn = SimpleRNN(input_size=3, hidden_size=5, output_size=2)

# 模拟一个长度为4的序列，每步有3个特征
sequence = np.random.randn(4, 3)
outputs, hidden_states = rnn.forward(sequence)

print(f"输入序列形状: {sequence.shape} (4步, 3特征)")
print(f"输出形状: {outputs.shape} (4步, 2输出)")
print(f"隐藏状态形状: {hidden_states.shape} (4步, 5维隐藏)")
print(f"\n最后一步的隐藏状态包含了整个序列的信息!")
print(f"h_final = {hidden_states[-1].round(4)}")


# ============================================================
# 3. 梯度消失问题
# ============================================================
print("\n" + "=" * 60)
print("3. 梯度消失/爆炸问题")
print("=" * 60)

print("""
【问题】
RNN 展开后相当于一个很深的网络（每个时间步是一层）。
反向传播时梯度要通过很多层：

  ∂L/∂h₁ = ∂L/∂h_T × ∂h_T/∂h_{T-1} × ... × ∂h₂/∂h₁

每次乘以 ∂h_t/∂h_{t-1}，包含 tanh 的导数和 W_hh:
  - 如果 W_hh 的特征值 < 1 → 连乘后趋向 0 → 梯度消失
  - 如果 W_hh 的特征值 > 1 → 连乘后趋向 ∞ → 梯度爆炸

后果：
  梯度消失 → 模型无法学到长距离依赖
    ("The cat, which sat on the mat, was ___" 
     → 隔了很多词，记不住 "cat" 是主语)
  
  梯度爆炸 → 训练不稳定
    (解决方法: 梯度裁剪 gradient clipping)

【解决方案】
  1. LSTM (Long Short-Term Memory) ← 最经典
  2. GRU (Gated Recurrent Unit) ← 简化版 LSTM
  3. Transformer + Attention ← 现在的主流（下一章讲）
""")

# 演示梯度消失
print("--- 梯度消失演示 ---")
W = np.array([[0.5, 0], [0, 0.5]])  # 特征值 < 1
gradient = np.ones(2)
print(f"W 的特征值 < 1 (0.5):")
for t in range(10):
    gradient = W @ gradient
    if t in [0, 2, 5, 9]:
        print(f"  经过 {t+1} 步: 梯度范数 = {np.linalg.norm(gradient):.8f}")
print(f"  → 梯度指数级衰减到 0！")

W2 = np.array([[1.5, 0], [0, 1.5]])  # 特征值 > 1
gradient2 = np.ones(2)
print(f"\nW 的特征值 > 1 (1.5):")
for t in range(10):
    gradient2 = W2 @ gradient2
    if t in [0, 2, 5, 9]:
        print(f"  经过 {t+1} 步: 梯度范数 = {np.linalg.norm(gradient2):.4f}")
print(f"  → 梯度指数级爆炸！")


# ============================================================
# 4. LSTM
# ============================================================
print("\n" + "=" * 60)
print("4. LSTM（长短期记忆网络）")
print("=" * 60)

print("""
【LSTM 的核心思想】
引入"细胞状态" C (cell state)，像一条传送带：
  信息可以不经修改地直接传递 → 解决了梯度消失

LSTM 有三个"门"来控制信息流：
  
  1. 遗忘门 (Forget Gate): 决定丢弃哪些旧信息
     f_t = sigmoid(W_f × [h_{t-1}, x_t] + b_f)
     f_t ∈ (0,1): 0=完全遗忘, 1=完全保留
  
  2. 输入门 (Input Gate): 决定存入哪些新信息
     i_t = sigmoid(W_i × [h_{t-1}, x_t] + b_i)
     候选信息: C̃_t = tanh(W_C × [h_{t-1}, x_t] + b_C)
  
  3. 输出门 (Output Gate): 决定输出什么
     o_t = sigmoid(W_o × [h_{t-1}, x_t] + b_o)

状态更新：
  C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t   (旧信息×遗忘 + 新信息×输入)
  h_t = o_t ⊙ tanh(C_t)               (输出)

直觉比喻（学习笔记）：
  C = 你的笔记本
  遗忘门 = 划掉过时的笔记
  输入门 = 写入新的重要内容
  输出门 = 决定现在需要看哪部分笔记
""")


# ============================================================
# 5. PyTorch LSTM 实现序列预测
# ============================================================
print("\n" + "=" * 60)
print("5. PyTorch 实现：正弦波预测")
print("=" * 60)

print("""
任务：给定正弦波的前 N 个时间步，预测下一个值。
这是时间序列预测的经典入门问题。
""")

# 生成正弦波数据
def create_sine_data(seq_length=20, n_samples=1000):
    """生成正弦波序列数据"""
    X, y = [], []
    # 生成一段长的正弦波
    t = np.linspace(0, 100, n_samples + seq_length)
    data = np.sin(t)
    
    for i in range(n_samples):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    
    X = np.array(X).reshape(-1, seq_length, 1)  # (samples, seq_len, features)
    y = np.array(y).reshape(-1, 1)
    return X, y

# 创建数据
X, y = create_sine_data(seq_length=20, n_samples=800)
X_train = torch.FloatTensor(X[:600])
y_train = torch.FloatTensor(y[:600])
X_test = torch.FloatTensor(X[600:])
y_test = torch.FloatTensor(y[600:])

print(f"训练集: {X_train.shape} → {y_train.shape}")
print(f"测试集: {X_test.shape} → {y_test.shape}")
print(f"任务: 用前20个时间步预测第21个值")

# 定义 LSTM 模型
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 只用最后一步的输出
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        prediction = self.fc(last_output)
        return prediction

model = LSTMPredictor(input_size=1, hidden_size=32)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(f"\nLSTM 模型: {model}")

# 训练
print("\n--- 训练 ---")
n_epochs = 50
batch_size = 32

for epoch in range(n_epochs):
    model.train()
    indices = np.random.permutation(len(X_train))
    total_loss = 0
    n_batches = 0
    
    for i in range(0, len(X_train), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch_X = X_train[batch_idx]
        batch_y = y_train[batch_idx]
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test)
        print(f"  Epoch {epoch:3d}: train_loss={total_loss/n_batches:.6f}, test_loss={test_loss:.6f}")

# 最终评估
model.eval()
with torch.no_grad():
    predictions = model(X_test).numpy()
    
mse = np.mean((predictions - y_test.numpy()) ** 2)
print(f"\n最终 MSE: {mse:.6f}")
print(f"预测 vs 真实 (前10个):")
for i in range(10):
    print(f"  预测: {predictions[i][0]:7.4f}, 真实: {y_test[i].item():7.4f}, "
          f"误差: {abs(predictions[i][0] - y_test[i].item()):7.4f}")

assert mse < 0.01, f"MSE 太高: {mse}"
print("\n✓ LSTM 成功学会了预测正弦波！")


# ============================================================
# 6. 文本分类示例（简化版）
# ============================================================
print("\n" + "=" * 60)
print("6. RNN 用于文本分类（简化示例）")
print("=" * 60)

print("""
【文本处理流程】
1. 分词: "I love AI" → ["I", "love", "AI"]
2. 构建词表: {"I": 0, "love": 1, "AI": 2, ...}
3. 转为数字: [0, 1, 2]
4. 词嵌入: 每个数字 → 一个向量 (下一章详讲)
5. RNN 处理序列 → 得到最终隐藏状态
6. 全连接层分类
""")

# 简化的情感分类
# 模拟数据：用随机序列代替真实文本
vocab_size = 100    # 词表大小
embed_dim = 16      # 嵌入维度
hidden_dim = 32     # LSTM 隐藏维度
max_len = 10        # 序列最大长度
n_classes = 2       # 正面/负面

class TextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # 词嵌入层
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, x):
        # x: (batch, seq_len) — 词的索引
        embedded = self.embedding(x)         # (batch, seq_len, embed_dim)
        lstm_out, (h_n, _) = self.lstm(embedded)
        last_hidden = h_n.squeeze(0)         # (batch, hidden_dim)
        output = self.fc(last_hidden)        # (batch, n_classes)
        return output

# 生成模拟数据
np.random.seed(42)
n_train = 500
X_text_train = torch.randint(0, vocab_size, (n_train, max_len))
y_text_train = torch.randint(0, n_classes, (n_train,))

X_text_test = torch.randint(0, vocab_size, (100, max_len))
y_text_test = torch.randint(0, n_classes, (100,))

# 训练
text_model = TextClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(text_model.parameters(), lr=0.01)

print(f"模型结构: Embedding → LSTM → Linear")
print(f"  Embedding: {vocab_size} 词 → {embed_dim} 维向量")
print(f"  LSTM: {embed_dim} 输入 → {hidden_dim} 隐藏")
print(f"  Linear: {hidden_dim} → {n_classes} 类")

text_model.train()
for epoch in range(20):
    outputs = text_model(X_text_train)
    loss = criterion(outputs, y_text_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

text_model.eval()
with torch.no_grad():
    test_outputs = text_model(X_text_test)
    _, predicted = test_outputs.max(1)
    acc = (predicted == y_text_test).float().mean()

print(f"\n文本分类准确率: {acc:.4f}")
print(f"(注：随机数据所以准确率约50%，用真实数据效果会好很多)")
print(f"✓ 模型结构正确，训练流程正确")

print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点：
1. RNN 处理序列数据，通过隐藏状态传递"记忆"
2. 普通 RNN 有梯度消失问题，无法捕捉长距离依赖
3. LSTM 通过"门"机制解决了梯度消失
4. 文本处理: 分词→词表→嵌入→RNN→分类

RNN 的局限性：
  - 串行计算（不能并行，训练慢）
  - 长序列仍然有信息损失
  
→ 这就是为什么 Transformer 取代了 RNN（下一章）

下一章：大语言模型 → 词嵌入、注意力机制、Transformer
""")
