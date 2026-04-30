"""
第一章 1.2：微积分基础
==================

微积分在 AI 中的核心作用：让模型能"学习"。

机器学习的本质：
1. 定义一个"损失函数"来衡量模型预测有多差
2. 通过求导，找到让损失最小的方向
3. 沿着这个方向调整模型参数

所以，导数/梯度 = 指导模型如何改进的"指南针"

本节内容：
1. 导数的概念与直觉
2. 常用求导法则
3. 偏导数（多变量函数）
4. 链式法则（反向传播的数学基础！）
5. 梯度的概念
"""

import numpy as np

print("=" * 60)
print("第一章 1.2：微积分基础")
print("=" * 60)

# ============================================================
# 1. 导数的概念
# ============================================================
print("\n" + "=" * 60)
print("1. 导数的概念")
print("=" * 60)

print("""
【什么是导数？】

导数 = 函数在某一点的"变化率"或"斜率"

生活中的例子：
  - 你开车，速度表显示 60 km/h
  - 速度 = 位置对时间的导数 = 位置变化了多快
  - 加速度 = 速度对时间的导数 = 速度变化了多快

数学定义：
  f'(x) = lim(h→0) [f(x+h) - f(x)] / h

直觉：在函数曲线上某一点画切线，切线的斜率就是导数。

在 AI 中：
  - f(x) 是损失函数（预测值和真实值的差距）
  - x 是模型的参数（权重）
  - f'(x) 告诉我们：参数往哪个方向调整能减少损失
""")

def numerical_derivative(f, x, h=1e-7):
    """用数值方法近似求导数（导数的定义）"""
    return (f(x + h) - f(x)) / h

# 例1: f(x) = x²，导数应该是 f'(x) = 2x
def f1(x):
    return x ** 2

x = 3.0
deriv = numerical_derivative(f1, x)
print(f"f(x) = x²")
print(f"f'(x) = 2x")
print(f"在 x=3 处：")
print(f"  数值导数: {deriv:.6f}")
print(f"  解析导数: {2*x:.6f}")
print(f"  误差: {abs(deriv - 2*x):.2e}")
assert abs(deriv - 2*x) < 1e-5, "导数计算误差过大"
print("✓ 验证通过")

# 例2: f(x) = x³，导数应该是 f'(x) = 3x²
def f2(x):
    return x ** 3

deriv2 = numerical_derivative(f2, x)
print(f"\nf(x) = x³")
print(f"f'(x) = 3x²")
print(f"在 x=3 处：")
print(f"  数值导数: {deriv2:.6f}")
print(f"  解析导数: {3*x**2:.6f}")
assert abs(deriv2 - 3*x**2) < 1e-4
print("✓ 验证通过")


# ============================================================
# 2. 常用求导法则
# ============================================================
print("\n" + "=" * 60)
print("2. 常用求导法则")
print("=" * 60)

print("""
【基本求导公式】（AI 中最常用的）

1. 常数: d/dx(c) = 0
   (常数不变，所以变化率为 0)

2. 幂函数: d/dx(xⁿ) = n·xⁿ⁻¹
   例: d/dx(x³) = 3x²
   
3. 指数函数: d/dx(eˣ) = eˣ  ← 自然指数的神奇之处！
   (它的导数等于自身)

4. 对数函数: d/dx(ln x) = 1/x

5. 加法法则: d/dx[f(x) + g(x)] = f'(x) + g'(x)

6. 乘法法则: d/dx[f(x)·g(x)] = f'(x)·g(x) + f(x)·g'(x)

7. 除法法则: d/dx[f(x)/g(x)] = [f'(x)·g(x) - f(x)·g'(x)] / g(x)²
""")

# 验证 e^x 的导数是自身
def exp_func(x):
    return np.exp(x)

x = 2.0
deriv_exp = numerical_derivative(exp_func, x)
print(f"f(x) = eˣ")
print(f"f'(x) = eˣ (导数等于自身!)")
print(f"在 x=2 处：")
print(f"  f(2) = e² = {np.exp(2):.6f}")
print(f"  f'(2) = {deriv_exp:.6f}")
print(f"  差值: {abs(deriv_exp - np.exp(2)):.2e}")
assert abs(deriv_exp - np.exp(2)) < 1e-5
print("✓ 验证通过")


# ============================================================
# 3. 偏导数
# ============================================================
print("\n" + "=" * 60)
print("3. 偏导数（多变量函数的导数）")
print("=" * 60)

print("""
【什么是偏导数？】

当函数有多个变量时，偏导数 = 只对一个变量求导，其他变量视为常数。

例: f(x, y) = x² + 2xy + y²

  ∂f/∂x = 2x + 2y    (对 x 求导，y 视为常数)
  ∂f/∂y = 2x + 2y    (对 y 求导，x 视为常数)

在 AI 中的意义：
  - 损失函数通常有成千上万个参数（权重）
  - 对每个参数求偏导数 = 找到每个参数的最佳调整方向
  - 所有偏导数合在一起 = 梯度（gradient）
""")

def f_multi(x, y):
    """f(x, y) = x² + 2xy + y²"""
    return x**2 + 2*x*y + y**2

def partial_x(f, x, y, h=1e-7):
    """对 x 求偏导数"""
    return (f(x + h, y) - f(x, y)) / h

def partial_y(f, x, y, h=1e-7):
    """对 y 求偏导数"""
    return (f(x, y + h) - f(x, y)) / h

x, y = 1.0, 2.0
px = partial_x(f_multi, x, y)
py = partial_y(f_multi, x, y)

print(f"f(x, y) = x² + 2xy + y²")
print(f"在 (x=1, y=2) 处：")
print(f"  ∂f/∂x = 2x + 2y = 2(1) + 2(2) = 6")
print(f"  数值计算: {px:.6f}")
assert abs(px - 6.0) < 1e-4
print(f"  ✓ 验证通过")

print(f"\n  ∂f/∂y = 2x + 2y = 2(1) + 2(2) = 6")
print(f"  数值计算: {py:.6f}")
assert abs(py - 6.0) < 1e-4
print(f"  ✓ 验证通过")


# ============================================================
# 4. 链式法则 —— 反向传播的数学基础！
# ============================================================
print("\n" + "=" * 60)
print("4. 链式法则（Chain Rule）—— 反向传播的数学基础！")
print("=" * 60)

print("""
【链式法则】

如果 y = f(g(x))，即函数的嵌套/组合：
  dy/dx = f'(g(x)) · g'(x)

或者更直觉的写法：
  如果 y = f(u), u = g(x)
  那么 dy/dx = dy/du · du/dx

直觉：变化率是可以"传递"和"相乘"的。
  - x 变化一点 → u 变化了 du/dx 倍
  - u 变化一点 → y 变化了 dy/du 倍
  - 所以 x 变化一点 → y 变化了 (dy/du × du/dx) 倍

【为什么这对 AI 至关重要？】

神经网络就是一系列函数的嵌套：
  output = f3(f2(f1(input)))

要训练网络，需要知道：改变最开始的权重，最终输出会怎么变？
这正是链式法则告诉我们的！

反向传播(Backpropagation) = 从输出层开始，用链式法则逐层
往回计算每个权重的梯度。
""")

# 例: y = (2x + 1)³
# 令 u = 2x + 1, y = u³
# dy/dx = dy/du · du/dx = 3u² · 2 = 6(2x+1)²

def composite_func(x):
    return (2*x + 1) ** 3

x = 2.0
numerical = numerical_derivative(composite_func, x)
analytical = 6 * (2*x + 1) ** 2  # 链式法则结果

print(f"y = (2x + 1)³")
print(f"令 u = 2x + 1, y = u³")
print(f"dy/dx = dy/du · du/dx = 3u² · 2 = 6(2x+1)²")
print(f"\n在 x=2 处：")
print(f"  2x+1 = 5")
print(f"  dy/dx = 6 × 5² = 6 × 25 = 150")
print(f"  数值计算: {numerical:.4f}")
print(f"  解析计算: {analytical:.4f}")
assert abs(numerical - analytical) < 1e-3
print("✓ 验证通过")

# 模拟一个简单的反向传播
print(f"""
【模拟简单的反向传播】

假设一个两层计算：
  第1层: z = w·x + b     (线性变换)
  第2层: a = sigmoid(z)  (激活函数)
  损失:  L = (a - y)²   (均方误差)

目标：求 ∂L/∂w（损失对权重的梯度）

链式法则：
  ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w

分别计算：
  ∂L/∂a = 2(a - y)      (L 对 a 求导)
  ∂a/∂z = a(1 - a)      (sigmoid 的导数)
  ∂z/∂w = x             (z 对 w 求导)

最终：
  ∂L/∂w = 2(a - y) · a(1 - a) · x
""")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 设定初始值
w = 0.5   # 权重
b = 0.1   # 偏置
x = 2.0   # 输入
y_true = 1.0  # 真实标签

# 前向传播
z = w * x + b
a = sigmoid(z)
L = (a - y_true) ** 2

print(f"前向传播：")
print(f"  w={w}, x={x}, b={b}")
print(f"  z = w·x + b = {z}")
print(f"  a = sigmoid(z) = {a:.6f}")
print(f"  L = (a - y)² = ({a:.6f} - {y_true})² = {L:.6f}")

# 反向传播（链式法则）
dL_da = 2 * (a - y_true)          # ∂L/∂a
da_dz = a * (1 - a)               # sigmoid 的导数
dz_dw = x                         # ∂z/∂w

dL_dw = dL_da * da_dz * dz_dw    # 链式法则

print(f"\n反向传播（链式法则）：")
print(f"  ∂L/∂a = 2(a - y) = {dL_da:.6f}")
print(f"  ∂a/∂z = a(1-a) = {da_dz:.6f}")
print(f"  ∂z/∂w = x = {dz_dw}")
print(f"  ∂L/∂w = {dL_da:.6f} × {da_dz:.6f} × {dz_dw} = {dL_dw:.6f}")

# 数值验证
def loss_wrt_w(w_val):
    z_val = w_val * x + b
    a_val = sigmoid(z_val)
    return (a_val - y_true) ** 2

numerical_grad = numerical_derivative(loss_wrt_w, w)
print(f"\n数值梯度验证: {numerical_grad:.6f}")
print(f"解析梯度:     {dL_dw:.6f}")
print(f"差值: {abs(numerical_grad - dL_dw):.2e}")
assert abs(numerical_grad - dL_dw) < 1e-5
print("✓ 验证通过！链式法则计算正确！")


# ============================================================
# 5. 梯度
# ============================================================
print("\n" + "=" * 60)
print("5. 梯度（Gradient）")
print("=" * 60)

print("""
【梯度 = 所有偏导数组成的向量】

对于函数 f(x₁, x₂, ..., xₙ)：
  梯度 ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

梯度的方向 = 函数增长最快的方向
梯度的反方向 = 函数下降最快的方向 ← 这就是梯度下降！

在 AI 中：
  - 参数 θ = [w₁, w₂, ..., wₙ] (所有权重)
  - 损失函数 L(θ)
  - 梯度 ∇L = [∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ]
  - 更新规则: θ_new = θ_old - learning_rate × ∇L
  
  学习率(learning rate)控制每步走多远。
""")

# 示例：求 f(x,y) = x² + y² 的梯度（这是一个"碗"形函数，最低点在原点）
def bowl(params):
    x, y = params
    return x**2 + y**2

def gradient_bowl(params):
    """解析梯度"""
    x, y = params
    return np.array([2*x, 2*y])

def numerical_gradient(f, params, h=1e-7):
    """数值梯度（通用方法）"""
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += h
        params_minus = params.copy()
        params_minus[i] -= h
        grad[i] = (f(params_plus) - f(params_minus)) / (2 * h)
    return grad

params = np.array([3.0, 4.0])
analytical_grad = gradient_bowl(params)
numerical_grad = numerical_gradient(bowl, params)

print(f"f(x, y) = x² + y²")
print(f"∇f = [2x, 2y]")
print(f"\n在 (3, 4) 处：")
print(f"  解析梯度: {analytical_grad}")
print(f"  数值梯度: {numerical_grad}")
assert np.allclose(analytical_grad, numerical_grad, atol=1e-5)
print("✓ 验证通过")

# 演示梯度下降
print(f"\n--- 梯度下降演示 ---")
print(f"目标：找到 f(x,y) = x² + y² 的最小值点")
print(f"(我们知道答案是 (0, 0)，看梯度下降能否找到)")

params = np.array([3.0, 4.0])  # 起始点
learning_rate = 0.1

print(f"\n起始点: ({params[0]:.4f}, {params[1]:.4f}), f = {bowl(params):.4f}")

for step in range(30):
  grad = gradient_bowl(params)
  params = params - learning_rate * grad
  if step < 5 or step in [9, 19, 29]:
    print(f"第{step+1}步: ({params[0]:.4f}, {params[1]:.4f}), f = {bowl(params):.4f}, |grad| = {np.linalg.norm(grad):.4f}")

print(f"\n最终结果接近 (0, 0)？ 是的！梯度下降成功找到了最小值点。")
assert bowl(params) < 0.01, "梯度下降未收敛"
print("✓ 验证通过")

print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点：
1. 导数 = 函数的变化率，告诉我们"往哪走能让函数变小"
2. 偏导数 = 多变量函数中，对单个变量的导数
3. 链式法则 = 复合函数求导，是反向传播的数学基础
4. 梯度 = 所有偏导数的向量，指向函数增长最快的方向
5. 梯度下降 = 沿梯度反方向走，逐步找到最小值

记住：AI 的"学习" = 通过梯度下降不断调整参数，让损失函数变小。

下一节：概率与统计 → 理解数据的不确定性
""")
