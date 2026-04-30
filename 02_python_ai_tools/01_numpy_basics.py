"""
第二章 2.1：NumPy 基础
====================

NumPy 是 Python 科学计算的基石，几乎所有 AI 框架都建立在它之上。

为什么用 NumPy 而不是 Python 列表？
- 速度快 100 倍以上（底层 C 实现 + 向量化运算）
- 内存效率高（连续存储，固定类型）
- 丰富的数学函数
- 广播机制让代码更简洁

本节内容：
1. 数组创建
2. 索引和切片
3. 数组运算（向量化）
4. 广播机制
5. 常用函数
6. 实战：数据标准化和矩阵运算
"""

import numpy as np
import time

print("=" * 60)
print("第二章 2.1：NumPy 基础")
print("=" * 60)

# ============================================================
# 1. 数组创建
# ============================================================
print("\n" + "=" * 60)
print("1. 数组创建")
print("=" * 60)

# 从列表创建
a = np.array([1, 2, 3, 4, 5])
print(f"从列表创建: {a}")
print(f"  类型: {a.dtype}")
print(f"  形状: {a.shape}")
print(f"  维度: {a.ndim}")

# 2D 数组
b = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"\n2D 数组:\n{b}")
print(f"  形状: {b.shape} (2行3列)")

# 常用创建函数
print("\n--- 常用创建函数 ---")
print(f"zeros(3,4):\n{np.zeros((3, 4))}")
print(f"\nones(2,3):\n{np.ones((2, 3))}")
print(f"\neye(3) (单位矩阵):\n{np.eye(3)}")
print(f"\narange(0, 10, 2): {np.arange(0, 10, 2)}")
print(f"\nlinspace(0, 1, 5): {np.linspace(0, 1, 5)}")

# 随机数（AI 中最常用！）
print("\n--- 随机数生成 ---")
np.random.seed(42)  # 设置种子确保可重现
print(f"均匀分布 rand(2,3):\n{np.random.rand(2, 3)}")
print(f"\n标准正态分布 randn(2,3):\n{np.random.randn(2, 3)}")
print(f"\n整数随机 randint(0,10, size=5): {np.random.randint(0, 10, size=5)}")

# AI 中常用的初始化
print("\n--- AI 中的权重初始化 ---")
n_in, n_out = 784, 256  # 如: 输入层784个神经元 → 隐藏层256个神经元

# Xavier 初始化（适合 sigmoid/tanh）
xavier_weights = np.random.randn(n_in, n_out) * np.sqrt(2.0 / (n_in + n_out))
print(f"Xavier 初始化: 形状 {xavier_weights.shape}")
print(f"  均值: {xavier_weights.mean():.6f} (应接近 0)")
print(f"  标准差: {xavier_weights.std():.6f} (应接近 {np.sqrt(2.0/(n_in+n_out)):.6f})")

# He 初始化（适合 ReLU）
he_weights = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
print(f"\nHe 初始化: 形状 {he_weights.shape}")
print(f"  均值: {he_weights.mean():.6f}")
print(f"  标准差: {he_weights.std():.6f} (应接近 {np.sqrt(2.0/n_in):.6f})")


# ============================================================
# 2. 索引和切片
# ============================================================
print("\n" + "=" * 60)
print("2. 索引和切片")
print("=" * 60)

data = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]])
print(f"数据:\n{data}")

print(f"\n基本索引:")
print(f"  data[0, 0] = {data[0, 0]}  (第0行第0列)")
print(f"  data[1, 2] = {data[1, 2]}  (第1行第2列)")
print(f"  data[-1, -1] = {data[-1, -1]}  (最后一行最后一列)")

print(f"\n切片 (start:end:step):")
print(f"  data[0, :] = {data[0, :]}  (第0行所有列)")
print(f"  data[:, 1] = {data[:, 1]}  (所有行第1列)")
print(f"  data[0:2, 1:3] = \n{data[0:2, 1:3]}  (前2行, 第1-2列)")

print(f"\n布尔索引 (条件过滤):")
mask = data > 5
print(f"  data > 5 = \n{mask}")
print(f"  data[data > 5] = {data[data > 5]}")

# Fancy indexing
print(f"\n花式索引:")
rows = [0, 2]
cols = [1, 3]
print(f"  data[[0,2], [1,3]] = {data[rows, cols]}  (取(0,1)和(2,3)位置)")


# ============================================================
# 3. 向量化运算
# ============================================================
print("\n" + "=" * 60)
print("3. 向量化运算（为什么 NumPy 快）")
print("=" * 60)

print("""
向量化 = 对整个数组同时操作，避免 Python 循环

原理：NumPy 在底层用 C 语言一次性处理整个数组，
      而 Python 循环每次迭代都有解释器开销。
""")

# 速度对比
n = 1000000
a = np.random.randn(n)
b = np.random.randn(n)

# Python 循环
start = time.time()
c_loop = [a[i] + b[i] for i in range(n)]
loop_time = time.time() - start

# NumPy 向量化
start = time.time()
c_numpy = a + b
numpy_time = time.time() - start

print(f"数组大小: {n:,}")
print(f"Python 循环: {loop_time:.4f} 秒")
print(f"NumPy 向量化: {numpy_time:.6f} 秒")
print(f"加速比: {loop_time/numpy_time:.0f}x !")

# 逐元素运算
print("\n--- 逐元素运算 ---")
x = np.array([1, 2, 3, 4])
y = np.array([10, 20, 30, 40])
print(f"x = {x}")
print(f"y = {y}")
print(f"x + y = {x + y}")
print(f"x * y = {x * y}  (逐元素相乘，不是矩阵乘法！)")
print(f"x ** 2 = {x ** 2}")
print(f"np.sqrt(x) = {np.sqrt(x)}")
print(f"np.exp(x) = {np.exp(x)}")
print(f"np.log(x) = {np.log(x)}")


# ============================================================
# 4. 广播机制 (Broadcasting)
# ============================================================
print("\n" + "=" * 60)
print("4. 广播机制 (Broadcasting)")
print("=" * 60)

print("""
【广播规则】
当两个数组形状不同时，NumPy 会自动"广播"较小的数组以匹配较大的。

规则：从后往前比较维度
  1. 如果维度相同 → OK
  2. 如果其中一个维度为 1 → 扩展为另一个的大小
  3. 如果维度不同且都不为 1 → 报错

例: (3, 4) + (1, 4) → (3, 4)  ✓  (第一维 1→3)
例: (3, 4) + (4,)   → (3, 4)  ✓  (自动补充维度: (4,)→(1,4)→(3,4))
例: (3, 4) + (3,)   → 报错    ✗  (最后维度 4≠3)

在 AI 中常见的广播：
  - 给每个样本加上同一个偏置: (batch, features) + (features,)
  - 标准化: (data - mean) / std，其中 mean 和 std 可能形状更小
""")

# 示例 1: 矩阵 + 向量
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
row_vec = np.array([10, 20, 30])
print(f"矩阵 (3,3):\n{matrix}")
print(f"向量 (3,): {row_vec}")
print(f"矩阵 + 向量 (广播: 向量加到每一行):\n{matrix + row_vec}")

# 示例 2: AI 中的标准化
print("\n--- AI 中的数据标准化（广播应用）---")
# 假设有 5 个样本，每个有 3 个特征
data = np.array([
    [170, 65, 25],  # 身高(cm), 体重(kg), 年龄
    [165, 55, 30],
    [180, 80, 22],
    [175, 70, 28],
    [160, 50, 35],
], dtype=float)
print(f"原始数据 (5样本, 3特征):\n{data}")
print(f"  形状: {data.shape}")

# 计算每个特征的均值和标准差
mean = data.mean(axis=0)  # 沿行方向(对每列)求均值
std = data.std(axis=0)
print(f"\n每个特征的均值: {mean}")
print(f"每个特征的标准差: {std}")
print(f"  形状: {mean.shape}")

# 标准化（广播: (5,3) - (3,) → 自动扩展）
normalized = (data - mean) / std
print(f"\n标准化后:\n{np.round(normalized, 3)}")
print(f"  新均值: {normalized.mean(axis=0).round(10)}")
print(f"  新标准差: {normalized.std(axis=0).round(10)}")
assert np.allclose(normalized.mean(axis=0), 0, atol=1e-10)
assert np.allclose(normalized.std(axis=0), 1, atol=1e-10)
print("✓ 标准化验证通过 (均值≈0, 标准差≈1)")


# ============================================================
# 5. 常用函数
# ============================================================
print("\n" + "=" * 60)
print("5. 常用函数")
print("=" * 60)

arr = np.array([[1, 5, 3],
                [4, 2, 6]])
print(f"数组:\n{arr}")

print(f"\n--- 聚合函数 ---")
print(f"sum(): {arr.sum()}")
print(f"sum(axis=0) (按列): {arr.sum(axis=0)}")
print(f"sum(axis=1) (按行): {arr.sum(axis=1)}")
print(f"mean(): {arr.mean():.2f}")
print(f"max(): {arr.max()}")
print(f"argmax(): {arr.argmax()} (展平后最大值的索引)")
print(f"argmax(axis=1): {arr.argmax(axis=1)} (每行最大值的列索引)")

print(f"\n--- 形状操作 ---")
flat = arr.reshape(-1)  # -1 表示自动计算
print(f"reshape(-1) (展平): {flat}")
print(f"reshape(3,2):\n{arr.reshape(3, 2)}")
print(f"transpose / .T:\n{arr.T}")

# 在 AI 中很常用的操作
print(f"\n--- AI 中常用操作 ---")
# 矩阵乘法
A = np.random.randn(3, 4)
B = np.random.randn(4, 2)
C = A @ B  # 或 np.dot(A, B)
print(f"矩阵乘法: (3,4) @ (4,2) → {C.shape}")

# 拼接
x1 = np.array([[1, 2], [3, 4]])
x2 = np.array([[5, 6], [7, 8]])
print(f"\n垂直拼接:\n{np.vstack([x1, x2])}")
print(f"水平拼接:\n{np.hstack([x1, x2])}")

# clip（限制范围）
vals = np.array([-2, -1, 0, 1, 2, 3])
print(f"\nclip(-2~3 → 0~1): {np.clip(vals, 0, 1)}")
print(f"  (常用于将概率限制在 [ε, 1-ε] 避免 log(0))")


# ============================================================
# 6. 实战：从零实现 Sigmoid 和 Softmax
# ============================================================
print("\n" + "=" * 60)
print("6. 实战：常用激活函数实现")
print("=" * 60)

def sigmoid(x):
    """Sigmoid 激活函数: 将任意值压缩到 (0, 1)"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU 激活函数: max(0, x)"""
    return np.maximum(0, x)

def softmax(x):
    """Softmax: 将向量转为概率分布"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

x = np.array([-3, -1, 0, 1, 3])
print(f"输入 x = {x}")
print(f"\nSigmoid(x) = {np.round(sigmoid(x), 4)}")
print(f"  特点: 输出在(0,1)，适合二分类概率输出")
print(f"  sigmoid(0) = 0.5, sigmoid(大正数)→1, sigmoid(大负数)→0")

print(f"\nReLU(x) = {relu(x)}")
print(f"  特点: 负数→0, 正数不变。简单高效，深度学习最常用")

logits = np.array([2.0, 1.0, 0.1])
print(f"\nSoftmax([2.0, 1.0, 0.1]) = {np.round(softmax(logits), 4)}")
print(f"  总和 = {softmax(logits).sum():.6f} (=1，是合法的概率分布)")

# batch softmax
batch_logits = np.array([[2.0, 1.0, 0.1],
                          [0.5, 2.5, 1.0]])
batch_probs = softmax(batch_logits)
print(f"\nBatch Softmax:")
print(f"  输入:\n{batch_logits}")
print(f"  输出:\n{np.round(batch_probs, 4)}")
print(f"  每行之和: {batch_probs.sum(axis=1)}")
assert np.allclose(batch_probs.sum(axis=1), 1.0)
print("✓ 验证通过")

print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点：
1. NumPy 数组比 Python 列表快 100 倍以上
2. 向量化运算是核心 —— 避免 Python 循环
3. 广播机制让不同形状的数组也能运算
4. axis 参数很重要（0=沿行/按列，1=沿列/按行）
5. 形状(shape)操作在 AI 中无处不在

记住：在 AI 中调试代码，最常见的 bug 就是形状不匹配！
养成习惯：操作前后 print(x.shape) 检查形状。

下一节：Pandas 数据处理 → 实际数据的加载和清洗
""")
