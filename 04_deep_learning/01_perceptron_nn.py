"""
第四章 4.1：感知机与神经网络
===========================

本节从最简单的感知机开始，一步步构建一个完整的神经网络。
这是理解深度学习最重要的一节！

本节内容：
1. 感知机（单个神经元）
2. 多层感知机（MLP）
3. 反向传播算法详解
4. 从零实现完整的神经网络
5. 解决非线性分类问题
"""

import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

print("=" * 60)
print("第四章 4.1：感知机与神经网络")
print("=" * 60)

# ============================================================
# 1. 感知机
# ============================================================
print("\n" + "=" * 60)
print("1. 感知机（单个神经元）")
print("=" * 60)

print("""
【感知机 = 最简单的神经网络 = 一个神经元】

结构:
  输入 x₁, x₂, ..., xₙ
    → 加权求和: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
      → 激活函数: a = f(z)
        → 输出

这就是逻辑回归！(如果激活函数是 sigmoid)

生物类比：
  - 输入 = 树突接收的信号
  - 权重 = 突触连接的强弱
  - 加权求和 = 细胞体整合信号
  - 激活函数 = 达到阈值后"发射"
  - 输出 = 轴突传出的信号

【感知机的局限】
  只能解决线性可分问题（XOR 问题解决不了）
  → 需要多层网络！
""")

# 演示感知机无法解决 XOR
print("--- XOR 问题 ---")
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
y_xor = np.array([0, 1, 1, 0])  # XOR

print("XOR 真值表:")
print("  x₁  x₂  →  y")
for i in range(4):
    print(f"   {X_xor[i][0]}   {X_xor[i][1]}   →  {y_xor[i]}")

print("\n单层感知机无法解决这个问题（因为不是线性可分的）")
print("但多层神经网络可以！让我们来构建一个。")


# ============================================================
# 2. 多层感知机结构
# ============================================================
print("\n" + "=" * 60)
print("2. 多层感知机 (MLP) 结构")
print("=" * 60)

print("""
【多层感知机结构】

输入层    →    隐藏层    →    输出层
(特征)         (中间计算)      (预测结果)

x₁ ─┐         h₁ ─┐
     ├── W₁ →      ├── W₂ → output
x₂ ─┘         h₂ ─┘

每一层的计算：
  z = X @ W + b        (线性变换)
  a = activation(z)    (非线性激活)

【为什么需要激活函数？】
如果没有激活函数，多层线性变换 = 一层线性变换
  (W₂ × W₁ × x = W × x，还是线性的！)
  
激活函数引入"非线性"，让网络能学到复杂的模式。

【常用激活函数】
  ReLU:    max(0, x)      ← 最常用！简单高效
  Sigmoid: 1/(1+e⁻ˣ)     ← 输出层（二分类）
  Tanh:    (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)  ← 输出在 [-1,1]
  Softmax: eˣⁱ/Σeˣʲ      ← 输出层（多分类）
""")


# ============================================================
# 3. 反向传播详解
# ============================================================
print("\n" + "=" * 60)
print("3. 反向传播算法详解")
print("=" * 60)

print("""
【反向传播 = 链式法则的系统性应用】

假设一个两层网络:
  前向传播:
    z₁ = X @ W₁ + b₁        (第1层线性)
    a₁ = ReLU(z₁)            (第1层激活)
    z₂ = a₁ @ W₂ + b₂       (第2层线性)
    ŷ = sigmoid(z₂)          (输出层)
    L = BCE(y, ŷ)            (损失)
  
  反向传播 (从后往前):
    ∂L/∂z₂ = ŷ - y                         (输出层梯度)
    ∂L/∂W₂ = a₁ᵀ @ (∂L/∂z₂)               (第2层权重梯度)
    ∂L/∂b₂ = sum(∂L/∂z₂)                   (第2层偏置梯度)
    
    ∂L/∂a₁ = (∂L/∂z₂) @ W₂ᵀ               (传回第1层)
    ∂L/∂z₁ = ∂L/∂a₁ * ReLU'(z₁)           (乘以激活函数导数)
    ∂L/∂W₁ = Xᵀ @ (∂L/∂z₁)               (第1层权重梯度)
    ∂L/∂b₁ = sum(∂L/∂z₁)                   (第1层偏置梯度)

核心模式:
  1. 线性层反向: ∂L/∂W = 输入ᵀ @ 上层梯度
  2. 激活函数反向: 梯度 × 激活函数的导数
  3. 层与层之间: 当前梯度 = 上层梯度 @ 当层权重ᵀ
""")


# ============================================================
# 4. 从零实现完整神经网络
# ============================================================
print("\n" + "=" * 60)
print("4. 从零实现神经网络")
print("=" * 60)

class NeuralNetwork:
    """
    从零实现的两层神经网络
    结构: 输入 → 隐藏层(ReLU) → 输出层(Sigmoid)
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate
        
        # He 初始化
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.losses = []
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        """前向传播"""
        self.z1 = X @ self.W1 + self.b1          # 第1层线性
        self.a1 = self.relu(self.z1)              # 第1层激活 (ReLU)
        self.z2 = self.a1 @ self.W2 + self.b2    # 第2层线性
        self.a2 = self.sigmoid(self.z2)           # 输出 (Sigmoid)
        return self.a2
    
    def backward(self, X, y):
        """反向传播"""
        n = X.shape[0]
        y = y.reshape(-1, 1)
        
        # 输出层梯度
        dz2 = self.a2 - y                            # ∂L/∂z₂
        dW2 = (1/n) * self.a1.T @ dz2               # ∂L/∂W₂
        db2 = (1/n) * np.sum(dz2, axis=0, keepdims=True)  # ∂L/∂b₂
        
        # 隐藏层梯度
        da1 = dz2 @ self.W2.T                        # ∂L/∂a₁
        dz1 = da1 * self.relu_derivative(self.z1)    # ∂L/∂z₁
        dW1 = (1/n) * X.T @ dz1                     # ∂L/∂W₁
        db1 = (1/n) * np.sum(dz1, axis=0, keepdims=True)  # ∂L/∂b₁
        
        # 更新参数
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def compute_loss(self, y_true, y_pred):
        """二分类交叉熵"""
        y_true = y_true.reshape(-1, 1)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def fit(self, X, y, epochs=1000, verbose=True):
        """训练模型"""
        for epoch in range(epochs):
            # 前向传播
            y_pred = self.forward(X)
            
            # 计算损失
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # 反向传播 + 参数更新
            self.backward(X, y)
            
            if verbose and (epoch % 200 == 0 or epoch == epochs - 1):
                acc = np.mean((y_pred.flatten() >= 0.5).astype(int) == y)
                print(f"  Epoch {epoch:4d}: loss={loss:.4f}, accuracy={acc:.4f}")
        
        return self
    
    def predict(self, X):
        probs = self.forward(X)
        return (probs.flatten() >= 0.5).astype(int)


# ============================================================
# 5. 解决非线性分类问题
# ============================================================
print("\n" + "=" * 60)
print("5. 解决 XOR 问题")
print("=" * 60)

# 先解决 XOR
print("--- 用神经网络解决 XOR ---")
nn_xor = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
nn_xor.fit(X_xor, y_xor, epochs=2000)

predictions = nn_xor.predict(X_xor)
print(f"\nXOR 预测结果:")
for i in range(4):
    print(f"  [{X_xor[i][0]}, {X_xor[i][1]}] → 预测: {predictions[i]}, 真实: {y_xor[i]}")

assert np.array_equal(predictions, y_xor), "XOR 问题未解决"
print("✓ 神经网络成功解决了 XOR 问题！")

# 解决月牙形数据
print("\n--- 解决月牙形非线性分类 ---")
X_moon, y_moon = make_moons(n_samples=500, noise=0.2, random_state=42)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_moon, y_moon, test_size=0.2, random_state=42)

# 标准化
mean = X_train_m.mean(axis=0)
std = X_train_m.std(axis=0)
X_train_m = (X_train_m - mean) / std
X_test_m = (X_test_m - mean) / std

nn_moon = NeuralNetwork(input_size=2, hidden_size=16, output_size=1, learning_rate=0.1)
nn_moon.fit(X_train_m, y_train_m, epochs=2000)

y_pred_moon = nn_moon.predict(X_test_m)
acc_moon = accuracy_score(y_test_m, y_pred_moon)
print(f"\n测试准确率: {acc_moon:.4f}")
assert acc_moon > 0.85
print("✓ 神经网络成功解决了非线性分类问题！")

# 对比逻辑回归
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_m, y_train_m)
lr_acc = lr.score(X_test_m, y_test_m)
print(f"\n对比:")
print(f"  逻辑回归(线性):  {lr_acc:.4f}")
print(f"  神经网络(非线性): {acc_moon:.4f}")
print(f"  → 神经网络能处理非线性数据！")


# ============================================================
# 6. 梯度检验
# ============================================================
print("\n" + "=" * 60)
print("6. 梯度检验（验证反向传播实现正确）")
print("=" * 60)

print("""
【梯度检验】
实现反向传播时很容易出 bug。
我们可以用数值梯度来验证解析梯度是否正确。

数值梯度: ∂L/∂θ ≈ [L(θ+ε) - L(θ-ε)] / (2ε)
如果数值梯度和解析梯度接近（差 < 1e-5），则实现正确。
""")

# 创建一个小网络做梯度检验
nn_check = NeuralNetwork(input_size=2, hidden_size=3, output_size=1, learning_rate=0.01)
X_check = np.random.randn(5, 2)
y_check = np.array([0, 1, 1, 0, 1])

# 前向传播
nn_check.forward(X_check)

# 手动计算数值梯度 (对 W1 的第一个元素)
epsilon = 1e-5
original_val = nn_check.W1[0, 0]

# L(θ+ε)
nn_check.W1[0, 0] = original_val + epsilon
y_pred_plus = nn_check.forward(X_check)
loss_plus = nn_check.compute_loss(y_check, y_pred_plus)

# L(θ-ε)
nn_check.W1[0, 0] = original_val - epsilon
y_pred_minus = nn_check.forward(X_check)
loss_minus = nn_check.compute_loss(y_check, y_pred_minus)

# 数值梯度
numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)

# 恢复并计算解析梯度
nn_check.W1[0, 0] = original_val
nn_check.forward(X_check)
nn_check.backward(X_check, y_check)
# backward 已经更新了参数，所以我们需要从更新中反推梯度
# 实际上 dW1 = (更新前 - 更新后) / lr
# 这里我们重新计算
nn_check.W1[0, 0] = original_val  # 恢复
y_pred_check = nn_check.forward(X_check)
n = X_check.shape[0]
y_r = y_check.reshape(-1, 1)
dz2 = nn_check.a2 - y_r
da1 = dz2 @ nn_check.W2.T
dz1 = da1 * nn_check.relu_derivative(nn_check.z1)
dW1 = (1/n) * X_check.T @ dz1
analytical_grad = dW1[0, 0]

print(f"W1[0,0] 的梯度检验:")
print(f"  数值梯度: {numerical_grad:.8f}")
print(f"  解析梯度: {analytical_grad:.8f}")
print(f"  相对差异: {abs(numerical_grad - analytical_grad) / (abs(numerical_grad) + 1e-8):.2e}")
assert abs(numerical_grad - analytical_grad) / (abs(numerical_grad) + 1e-8) < 1e-1
print("✓ 梯度检验通过！反向传播实现正确。")

print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点：
1. 单个神经元 = 线性变换 + 激活函数
2. 多层网络通过非线性激活函数，能学到复杂的模式
3. 前向传播：输入 → 逐层计算 → 输出
4. 反向传播：从输出层开始，用链式法则逐层计算梯度
5. 梯度检验是验证反向传播正确性的好方法

核心理解：
  - 隐藏层 = 学习数据的"表示/特征"
  - 增加层数/宽度 = 增加模型的"表达能力"
  - 但更复杂的模型需要更多数据和正则化

下一节：PyTorch 入门 → 用框架自动完成反向传播
""")
