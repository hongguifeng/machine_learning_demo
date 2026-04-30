"""
第三章 3.2：逻辑回归（分类）
==========================

逻辑回归虽然名字有"回归"，但实际上是一个分类算法！
它是理解神经网络的重要基石。

核心思想：在线性回归基础上加一个 Sigmoid 函数，
          把输出从实数范围压缩到 (0, 1)，作为"概率"。

本节内容：
1. 从线性回归到逻辑回归
2. Sigmoid 函数
3. 交叉熵损失函数
4. 从零实现逻辑回归
5. 多类分类（Softmax 回归）
6. 决策边界可视化
"""

import numpy as np
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("第三章 3.2：逻辑回归（分类）")
print("=" * 60)

# ============================================================
# 1. 从线性回归到逻辑回归
# ============================================================
print("\n" + "=" * 60)
print("1. 从线性回归到逻辑回归")
print("=" * 60)

print("""
【问题】
线性回归输出连续值（如 -∞ 到 +∞），但分类问题需要输出概率（0到1）。

例如：
  输入: 邮件特征
  输出: 是垃圾邮件的概率（0.0 = 不是，1.0 = 是）

【解决方案】
在线性回归输出后，加一个 Sigmoid 函数：

  z = wx + b            (线性部分，和线性回归一样)
  p = sigmoid(z)        (压缩到 0~1)
  p = 1 / (1 + e⁻ᶻ)

  如果 p > 0.5 → 预测为正类 (1)
  如果 p < 0.5 → 预测为负类 (0)

【Sigmoid 函数性质】
  - 输出范围: (0, 1)
  - sigmoid(0) = 0.5
  - z 越大 → 越接近 1
  - z 越小 → 越接近 0
  - 导数: sigmoid'(z) = sigmoid(z) × (1 - sigmoid(z)) ← 反向传播用
""")


# ============================================================
# 2. 交叉熵损失函数
# ============================================================
print("\n" + "=" * 60)
print("2. 交叉熵损失函数 (Binary Cross-Entropy)")
print("=" * 60)

print("""
【为什么不用 MSE？】
分类问题如果用 MSE 作为损失函数：
  - 损失函数不是凸的（有很多局部最小值）
  - 梯度下降容易卡住

【二分类交叉熵 (BCE)】
  L = -(1/n) × Σ[yᵢ×log(pᵢ) + (1-yᵢ)×log(1-pᵢ)]

直觉分析：
  当 y=1（真实是正类）:
    L = -log(p)
    → p 越大（预测越正确），loss 越小
    → p 接近 0（预测错误），loss 趋向无穷大

  当 y=0（真实是负类）:
    L = -log(1-p)
    → p 越小（预测越正确），loss 越小
    → p 接近 1（预测错误），loss 趋向无穷大

梯度（非常简洁！）：
  ∂L/∂w = (1/n) × Σ(pᵢ - yᵢ) × xᵢ
  ∂L/∂b = (1/n) × Σ(pᵢ - yᵢ)
  
  和线性回归的梯度形式几乎一样！区别在于 p 是 sigmoid 的输出。
""")


# ============================================================
# 3. 从零实现逻辑回归
# ============================================================
print("\n" + "=" * 60)
print("3. 从零实现逻辑回归")
print("=" * 60)

class LogisticRegressionFromScratch:
    """从零实现的逻辑回归"""
    
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.losses = []
    
    def sigmoid(self, z):
        # clip 防止数值溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        for i in range(self.n_iter):
            # 前向传播
            z = X @ self.w + self.b
            p = self.sigmoid(z)
            
            # 计算交叉熵损失
            loss = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))
            self.losses.append(loss)
            
            # 计算梯度
            dw = (1/n_samples) * X.T @ (p - y)
            db = (1/n_samples) * np.sum(p - y)
            
            # 更新参数
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            if i % 200 == 0:
                acc = np.mean((p >= 0.5).astype(int) == y)
                print(f"  Epoch {i:4d}: loss={loss:.4f}, accuracy={acc:.4f}")
        
        return self
    
    def predict_proba(self, X):
        z = X @ self.w + self.b
        return self.sigmoid(z)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# 生成二分类数据
np.random.seed(42)
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                           n_informative=2, random_state=42, n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
print(f"测试集: {X_test.shape[0]} 样本")
print(f"正类比例: {y_train.mean():.2%}")

# 训练
print("\n--- 开始训练 ---")
model = LogisticRegressionFromScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

# 评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- 测试结果 ---")
print(f"准确率: {accuracy:.4f}")
print(f"学到的权重: w={model.w.round(4)}, b={model.b:.4f}")
assert accuracy > 0.8, f"准确率太低: {accuracy}"
print("✓ 准确率 > 80%，验证通过")

# 与 sklearn 对比
from sklearn.linear_model import LogisticRegression
sk_model = LogisticRegression()
sk_model.fit(X_train, y_train)
sk_acc = sk_model.score(X_test, y_test)
print(f"\nsklearn 逻辑回归准确率: {sk_acc:.4f}")
print(f"我们的实现准确率:      {accuracy:.4f}")


# ============================================================
# 4. 理解决策边界
# ============================================================
print("\n" + "=" * 60)
print("4. 决策边界")
print("=" * 60)

print("""
【决策边界】
逻辑回归的决策边界是一条直线（2D中）或超平面（高维中）。

在 2D 中：
  决策边界: w₁x₁ + w₂x₂ + b = 0
  
  当 w₁x₁ + w₂x₂ + b > 0 → 正类
  当 w₁x₁ + w₂x₂ + b < 0 → 负类
  
  x₂ = -(w₁/w₂)x₁ - b/w₂  (直线方程)
""")

w1, w2 = model.w
b = model.b
print(f"决策边界方程: {w1:.3f}×x₁ + {w2:.3f}×x₂ + {b:.3f} = 0")
print(f"即: x₂ = {-w1/w2:.3f}×x₁ + {-b/w2:.3f}")

# 验证：决策边界上的点
x1_test = 0
x2_boundary = -(w1/w2) * x1_test - b/w2
z_boundary = w1 * x1_test + w2 * x2_boundary + b
print(f"\n验证: 点 ({x1_test}, {x2_boundary:.4f}) 处的 z = {z_boundary:.6f} (应接近 0)")
assert abs(z_boundary) < 1e-10
print("✓ 验证通过")


# ============================================================
# 5. 多类分类（Softmax 回归）
# ============================================================
print("\n" + "=" * 60)
print("5. 多类分类（Softmax 回归）")
print("=" * 60)

print("""
【从二分类到多分类】

二分类: sigmoid → 输出一个概率 p
多分类: softmax → 输出 K 个概率 [p₁, p₂, ..., pₖ]，和为 1

Softmax 回归:
  z = X @ W + b     (W 形状: [n_features, n_classes])
  p = softmax(z)    (每行是一个概率分布)

损失函数：多类交叉熵
  L = -(1/n) × ΣΣ yᵢⱼ × log(pᵢⱼ)
  其中 yᵢⱼ 是 one-hot 编码的标签
""")

class SoftmaxRegression:
    """多分类 Softmax 回归"""
    
    def __init__(self, lr=0.1, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)
    
    def one_hot(self, y, n_classes):
        one_hot = np.zeros((len(y), n_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))
        
        # 初始化参数
        self.W = np.zeros((n_features, self.n_classes))
        self.b = np.zeros(self.n_classes)
        
        y_onehot = self.one_hot(y, self.n_classes)
        
        for i in range(self.n_iter):
            # 前向传播
            z = X @ self.W + self.b
            probs = self.softmax(z)
            
            # 梯度
            error = probs - y_onehot
            dW = (1/n_samples) * X.T @ error
            db = (1/n_samples) * error.sum(axis=0)
            
            # 更新
            self.W -= self.lr * dW
            self.b -= self.lr * db
            
            if i % 200 == 0:
                loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-15), axis=1))
                acc = np.mean(np.argmax(probs, axis=1) == y)
                print(f"  Epoch {i:4d}: loss={loss:.4f}, accuracy={acc:.4f}")
        
        return self
    
    def predict(self, X):
        z = X @ self.W + self.b
        probs = self.softmax(z)
        return np.argmax(probs, axis=1)

# 生成3类数据
from sklearn.datasets import make_blobs
X_multi, y_multi = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.3, random_state=42)

# 标准化
mean = X_train_m.mean(axis=0)
std = X_train_m.std(axis=0)
X_train_m = (X_train_m - mean) / std
X_test_m = (X_test_m - mean) / std

print("\n--- 3分类 Softmax 回归 ---")
print(f"类别数: 3, 特征数: 2")
softmax_model = SoftmaxRegression(lr=0.5, n_iter=1000)
softmax_model.fit(X_train_m, y_train_m)

y_pred_m = softmax_model.predict(X_test_m)
acc_multi = accuracy_score(y_test_m, y_pred_m)
print(f"\n测试准确率: {acc_multi:.4f}")
assert acc_multi > 0.85
print("✓ 多分类准确率 > 85%，验证通过")


# ============================================================
# 6. 非线性数据的局限性
# ============================================================
print("\n" + "=" * 60)
print("6. 逻辑回归的局限性")
print("=" * 60)

print("""
【局限性】
逻辑回归的决策边界是线性的（直线/超平面）。
对于非线性可分的数据，逻辑回归表现不好。

这就是为什么我们需要神经网络！
神经网络通过多层非线性变换，可以学到任意复杂的决策边界。
""")

# 生成非线性数据（月牙形）
X_moon, y_moon = make_moons(n_samples=300, noise=0.2, random_state=42)
X_train_nl, X_test_nl, y_train_nl, y_test_nl = train_test_split(
    X_moon, y_moon, test_size=0.3, random_state=42)

# 逻辑回归
lr_model = LogisticRegressionFromScratch(learning_rate=0.5, n_iterations=1000)
print("\n在非线性数据(月牙形)上训练:")
lr_model.fit(X_train_nl, y_train_nl)
y_pred_nl = lr_model.predict(X_test_nl)
acc_nl = accuracy_score(y_test_nl, y_pred_nl)
print(f"\n逻辑回归在非线性数据上的准确率: {acc_nl:.4f}")
print(f"→ 只有 ~{acc_nl*100:.0f}%，因为线性决策边界无法分开月牙形数据")
print(f"→ 这个问题将在下一章用神经网络解决!")

print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点：
1. 逻辑回归 = 线性回归 + Sigmoid → 输出概率
2. 用交叉熵（而非 MSE）作为分类损失函数
3. 梯度形式简洁: ∂L/∂w = (1/n) × Xᵀ(p - y)
4. 多分类用 Softmax 回归（Softmax + 多类交叉熵）
5. 逻辑回归只能做线性分类 → 需要神经网络处理非线性

逻辑回归就是最简单的"单层神经网络"!
  - 输入层 → 输出层（一层权重）
  - 激活函数: sigmoid (二分类) 或 softmax (多分类)
  - 损失函数: 交叉熵

理解了逻辑回归，就理解了神经网络的基本单元。

下一节：决策树与随机森林 → 另一类重要的机器学习方法
""")
