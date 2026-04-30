"""
第三章 3.1：线性回归
==================

线性回归是最简单的机器学习算法，但其中的核心思想贯穿所有 AI 算法：
  1. 定义模型（假设函数）
  2. 定义损失函数（衡量预测与真实值的差距）
  3. 优化参数（梯度下降最小化损失）

本节内容：
1. 线性回归的直觉与数学
2. 从零实现线性回归（梯度下降）
3. 解析解（正规方程）
4. 使用 scikit-learn 实现
5. 多元线性回归
6. 过拟合与正则化
"""

import numpy as np

print("=" * 60)
print("第三章 3.1：线性回归")
print("=" * 60)

# ============================================================
# 1. 线性回归的直觉与数学
# ============================================================
print("\n" + "=" * 60)
print("1. 线性回归的直觉与数学")
print("=" * 60)

print("""
【什么是线性回归？】

目标：找到一条直线，最好地"拟合"数据点。

模型：y = wx + b
  w = 斜率（权重）
  b = 截距（偏置）

例: 用房子面积(x)预测价格(y)
  y = 5000 × x + 10000
  意思是：每平方米 5000 元，基础价格 10000 元

【损失函数：均方误差 (MSE)】
  L = (1/n) × Σ(yᵢ - ŷᵢ)²
  
  其中 ŷᵢ = w×xᵢ + b 是模型预测值
  
  直觉：所有预测误差的平方的平均值
  为什么用平方？
  1. 正负误差不会相互抵消
  2. 更惩罚大的误差（平方放大了大误差）
  3. 数学上可以求导

【梯度下降更新】
  ∂L/∂w = (2/n) × Σ(ŷᵢ - yᵢ) × xᵢ
  ∂L/∂b = (2/n) × Σ(ŷᵢ - yᵢ)
  
  w = w - lr × ∂L/∂w
  b = b - lr × ∂L/∂b
""")

# ============================================================
# 2. 从零实现线性回归
# ============================================================
print("\n" + "=" * 60)
print("2. 从零实现线性回归（梯度下降）")
print("=" * 60)

# 生成模拟数据: y = 3x + 2 + 噪声
np.random.seed(42)
n_samples = 100
X = np.random.uniform(0, 10, n_samples)
y_true_w, y_true_b = 3.0, 2.0
y = y_true_w * X + y_true_b + np.random.randn(n_samples) * 1.5

print(f"数据: y = {y_true_w}x + {y_true_b} + 噪声")
print(f"样本数: {n_samples}")
print(f"X 范围: [{X.min():.2f}, {X.max():.2f}]")
print(f"y 范围: [{y.min():.2f}, {y.max():.2f}]")

class LinearRegressionFromScratch:
    """从零实现的线性回归"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.w = 0.0
        self.b = 0.0
        self.losses = []
    
    def predict(self, X):
        return self.w * X + self.b
    
    def compute_loss(self, X, y):
        """均方误差"""
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)
    
    def fit(self, X, y):
        """训练模型"""
        n = len(X)
        
        for i in range(self.n_iter):
            # 前向传播：计算预测值
            y_pred = self.predict(X)
            
            # 计算梯度
            dw = (2/n) * np.sum((y_pred - y) * X)
            db = (2/n) * np.sum(y_pred - y)
            
            # 更新参数
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            # 记录损失
            loss = self.compute_loss(X, y)
            self.losses.append(loss)
            
            if i % 200 == 0 or i == self.n_iter - 1:
                print(f"  Epoch {i:4d}: loss={loss:.4f}, w={self.w:.4f}, b={self.b:.4f}")
        
        return self

# 训练
print("\n--- 开始训练 ---")
model = LinearRegressionFromScratch(learning_rate=0.005, n_iterations=1000)
model.fit(X, y)

print(f"\n--- 训练结果 ---")
print(f"学到的参数: w={model.w:.4f}, b={model.b:.4f}")
print(f"真实参数:   w={y_true_w:.4f}, b={y_true_b:.4f}")
print(f"最终损失:   {model.losses[-1]:.4f}")
assert abs(model.w - y_true_w) < 0.3, f"w 偏差太大: {model.w}"
assert abs(model.b - y_true_b) < 1.0, f"b 偏差太大: {model.b}"
print("✓ 参数接近真实值，验证通过")


# ============================================================
# 3. 解析解（正规方程）
# ============================================================
print("\n" + "=" * 60)
print("3. 解析解（正规方程）")
print("=" * 60)

print("""
【正规方程】
对于线性回归，有直接的数学解（不需要迭代）:

  θ = (XᵀX)⁻¹ Xᵀy

其中 X 是增加了一列全 1 的特征矩阵（对应偏置）

优缺点：
  优点: 一步得到精确解
  缺点: 需要求逆矩阵，数据量大或特征多时计算慢 O(n³)
  
实际中: 数据量 < 10000 且特征 < 1000 时用正规方程更快
        数据量大时用梯度下降
""")

# 构造设计矩阵（加一列 1 表示偏置）
X_design = np.column_stack([X, np.ones(n_samples)])
print(f"设计矩阵形状: {X_design.shape}")
print(f"前3行:\n{X_design[:3]}")

# 正规方程求解
theta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
w_analytical, b_analytical = theta[0], theta[1]

print(f"\n正规方程结果: w={w_analytical:.4f}, b={b_analytical:.4f}")
print(f"梯度下降结果: w={model.w:.4f}, b={model.b:.4f}")
print(f"真实值:       w={y_true_w:.4f}, b={y_true_b:.4f}")
assert abs(w_analytical - y_true_w) < 0.3
print("✓ 正规方程验证通过")


# ============================================================
# 4. 使用 scikit-learn
# ============================================================
print("\n" + "=" * 60)
print("4. 使用 scikit-learn 实现")
print("=" * 60)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# sklearn 需要 2D 输入
X_sklearn = X.reshape(-1, 1)

# 训练
sk_model = LinearRegression()
sk_model.fit(X_sklearn, y)

print(f"sklearn 结果: w={sk_model.coef_[0]:.4f}, b={sk_model.intercept_:.4f}")
print(f"我们的实现:   w={model.w:.4f}, b={model.b:.4f}")

# 评估指标
y_pred = sk_model.predict(X_sklearn)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"\n评估指标:")
print(f"  MSE (均方误差): {mse:.4f}")
print(f"  R² (决定系数): {r2:.4f}")
print(f"    R²=1 表示完美拟合, R²=0 表示和均值预测一样差")
print(f"    R²={r2:.4f} 表示模型解释了 {r2*100:.1f}% 的数据变化")


# ============================================================
# 5. 多元线性回归
# ============================================================
print("\n" + "=" * 60)
print("5. 多元线性回归")
print("=" * 60)

print("""
【多元线性回归】
当有多个特征时：
  y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

向量形式：y = Xw + b

例: 预测房价 = w₁×面积 + w₂×房间数 + w₃×楼龄 + b
""")

# 生成多特征数据
np.random.seed(42)
n = 200
# 3个特征
X_multi = np.random.randn(n, 3)
true_weights = np.array([3.0, -1.5, 2.0])
true_bias = 5.0
y_multi = X_multi @ true_weights + true_bias + np.random.randn(n) * 0.5

print(f"数据: y = 3x₁ - 1.5x₂ + 2x₃ + 5 + 噪声")
print(f"特征矩阵形状: {X_multi.shape}")

class MultiLinearRegression:
    """多元线性回归 - 从零实现"""
    
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.losses = []
        
        for i in range(self.n_iter):
            y_pred = X @ self.w + self.b
            
            # 梯度
            dw = (2/n_samples) * X.T @ (y_pred - y)
            db = (2/n_samples) * np.sum(y_pred - y)
            
            # 更新
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            loss = np.mean((y_pred - y) ** 2)
            self.losses.append(loss)
        
        return self
    
    def predict(self, X):
        return X @ self.w + self.b

# 训练
multi_model = MultiLinearRegression(lr=0.05, n_iter=500)
multi_model.fit(X_multi, y_multi)

print(f"\n学到的权重: {multi_model.w.round(4)}")
print(f"真实权重:   {true_weights}")
print(f"学到的偏置: {multi_model.b:.4f}")
print(f"真实偏置:   {true_bias}")
assert np.allclose(multi_model.w, true_weights, atol=0.2)
assert abs(multi_model.b - true_bias) < 0.5
print("✓ 多元线性回归验证通过")


# ============================================================
# 6. 过拟合与正则化
# ============================================================
print("\n" + "=" * 60)
print("6. 过拟合与正则化")
print("=" * 60)

print("""
【过拟合 (Overfitting)】
模型在训练数据上表现很好，但在新数据上表现很差。
= 模型"记住"了训练数据的噪声，而没有学到真正的规律。

比喻：学生背答案而不是理解知识。考试换了题就不会做了。

【检测过拟合】
  训练误差很小，但验证/测试误差大 → 过拟合！

【正则化 (Regularization)】
在损失函数中加入惩罚项，限制权重不能太大：

L2 正则化 (Ridge)：
  Loss = MSE + λ × Σ(wᵢ²)
  效果：权重值趋向于小但不为零（所有特征都参与，但影响被削弱）

L1 正则化 (Lasso)：
  Loss = MSE + λ × Σ|wᵢ|
  效果：某些权重直接变为零（自动特征选择！）

λ (alpha) 越大，正则化越强，模型越简单
""")

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# 创建一个容易过拟合的场景
np.random.seed(42)
n = 30
X_overfit = np.sort(np.random.uniform(0, 1, n))
y_overfit = np.sin(2 * np.pi * X_overfit) + np.random.randn(n) * 0.3

# 用高次多项式特征（容易过拟合）
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
    X_overfit.reshape(-1, 1), y_overfit, test_size=0.3, random_state=42)

# 创建 15 次多项式特征
poly = PolynomialFeatures(degree=15)
X_train_poly = poly.fit_transform(X_train_o)
X_test_poly = poly.transform(X_test_o)

# 无正则化
model_no_reg = LinearRegression()
model_no_reg.fit(X_train_poly, y_train_o)
train_score = model_no_reg.score(X_train_poly, y_train_o)
test_score = model_no_reg.score(X_test_poly, y_test_o)
print(f"\n无正则化 (15次多项式):")
print(f"  训练 R²: {train_score:.4f}")
print(f"  测试 R²: {test_score:.4f}")
print(f"  → 训练好但测试差 = 过拟合!")

# L2 正则化 (Ridge)
model_ridge = Ridge(alpha=0.1)
model_ridge.fit(X_train_poly, y_train_o)
train_score_r = model_ridge.score(X_train_poly, y_train_o)
test_score_r = model_ridge.score(X_test_poly, y_test_o)
print(f"\nRidge 正则化 (α=0.1):")
print(f"  训练 R²: {train_score_r:.4f}")
print(f"  测试 R²: {test_score_r:.4f}")
print(f"  → 测试分数大大提高!")

# L1 正则化 (Lasso)
model_lasso = Lasso(alpha=0.01)
model_lasso.fit(X_train_poly, y_train_o)
train_score_l = model_lasso.score(X_train_poly, y_train_o)
test_score_l = model_lasso.score(X_test_poly, y_test_o)
n_zero = np.sum(model_lasso.coef_ == 0)
print(f"\nLasso 正则化 (α=0.01):")
print(f"  训练 R²: {train_score_l:.4f}")
print(f"  测试 R²: {test_score_l:.4f}")
print(f"  权重为零的数量: {n_zero}/{len(model_lasso.coef_)} (自动特征选择!)")

print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点：
1. 线性回归: y = wx + b，用梯度下降最小化 MSE
2. 这是所有机器学习算法的基本模式：模型→损失→优化
3. 正规方程是解析解，梯度下降是迭代解
4. R² 衡量模型好坏 (越接近 1 越好)
5. 过拟合 = 训练好测试差，用正则化来防止
6. 多元线性回归就是向量化的单变量线性回归

下一节：逻辑回归 → 从回归到分类
""")
