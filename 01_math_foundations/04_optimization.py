"""
第一章 1.4：优化基础
==================

优化 = 找到使目标函数最小（或最大）的参数值

在 AI 中：
  目标函数 = 损失函数（模型预测与真实值的差距）
  参数 = 模型的权重
  优化 = 找到让模型表现最好的权重

本节内容：
1. 梯度下降详解
2. 学习率的影响
3. 随机梯度下降 (SGD)
4. 动量 (Momentum)
5. Adam 优化器
6. 学习率调度
"""

import numpy as np

print("=" * 60)
print("第一章 1.4：优化基础")
print("=" * 60)

# ============================================================
# 1. 梯度下降详解
# ============================================================
print("\n" + "=" * 60)
print("1. 梯度下降详解")
print("=" * 60)

print("""
【梯度下降的完整过程】

目标：最小化损失函数 L(θ)，其中 θ 是模型参数

算法：
  1. 初始化参数 θ（通常随机）
  2. 计算梯度: g = ∂L/∂θ  (损失函数关于参数的导数)
  3. 更新参数: θ = θ - lr × g  (沿梯度反方向走一步)
  4. 重复步骤 2-3 直到收敛

其中 lr 是学习率（步长）

比喻：
  你蒙着眼睛站在山上，想到达最低点（山谷）。
  - 你用脚感受地面的坡度（计算梯度）
  - 朝下坡方向走一步（更新参数）
  - 步子大小就是学习率
  - 重复这个过程直到走到最低点
""")

# 示例：最小化 f(x) = (x-3)² + 1
# 最小值在 x=3, f(3)=1
def f(x):
    return (x - 3)**2 + 1

def df(x):
    """f 的导数: f'(x) = 2(x-3)"""
    return 2 * (x - 3)

print("目标函数: f(x) = (x-3)² + 1")
print("最小值在 x=3, f(3)=1")
print("导数: f'(x) = 2(x-3)")

# 梯度下降
x = 10.0  # 起始点
lr = 0.1  # 学习率
history = [(x, f(x))]

print(f"\n起始: x={x:.4f}, f(x)={f(x):.4f}")
print(f"学习率: {lr}")
print(f"\n{'步骤':<4} {'x':<10} {'f(x)':<10} {'梯度':<10}")
print("-" * 40)

for step in range(35):
  grad = df(x)
  x = x - lr * grad
  history.append((x, f(x)))
  if step < 8 or step >= 33:
    print(f"{step+1:<4} {x:<10.4f} {f(x):<10.4f} {grad:<10.4f}")

print(f"\n最终结果: x={x:.6f}, f(x)={f(x):.6f}")
print(f"真实最小值: x=3, f(3)=1")
assert abs(x - 3.0) < 0.01
print("✓ 梯度下降成功找到最小值")


# ============================================================
# 2. 学习率的影响
# ============================================================
print("\n" + "=" * 60)
print("2. 学习率的影响")
print("=" * 60)

print("""
学习率(Learning Rate)是最重要的超参数之一！

- 太小：收敛太慢，训练要很久很久
- 太大：来回震荡甚至发散，永远找不到最小值
- 刚好：快速且稳定地收敛

经验值：通常在 0.001 ~ 0.01 之间开始尝试
""")

learning_rates = [0.01, 0.1, 0.5, 1.0, 1.01]
print(f"\n比较不同学习率 (起始 x=10, 目标 x=3):")
print(f"{'学习率':<8} {'20步后的x':<12} {'f(x)':<12} {'状态'}")
print("-" * 50)

for lr in learning_rates:
    x = 10.0
    diverged = False
    for _ in range(20):
        grad = df(x)
        x = x - lr * grad
        if abs(x) > 1e10:
            diverged = True
            break
    
    if diverged:
        status = "❌ 发散了！"
        print(f"{lr:<8} {'发散':<12} {'发散':<12} {status}")
    elif abs(x - 3) < 0.1:
        status = "✓ 收敛"
        print(f"{lr:<8} {x:<12.4f} {f(x):<12.4f} {status}")
    else:
        status = "~ 震荡/慢"
        print(f"{lr:<8} {x:<12.4f} {f(x):<12.4f} {status}")


# ============================================================
# 3. 随机梯度下降 (SGD)
# ============================================================
print("\n" + "=" * 60)
print("3. 随机梯度下降 (SGD)")
print("=" * 60)

print("""
【批量梯度下降 vs 随机梯度下降】

批量梯度下降 (Batch GD)：
  - 每次用全部数据计算梯度
  - 优点：梯度准确，方向稳定
  - 缺点：数据量大时，一步计算太慢

随机梯度下降 (SGD)：
  - 每次只用一个样本计算梯度
  - 优点：计算快，有助于跳出局部最小值
  - 缺点：梯度噪声大，路径不稳定

小批量梯度下降 (Mini-batch SGD)：← 实际最常用！
  - 每次用一小批数据（如32, 64, 128个样本）
  - 折中：既不太慢，梯度也不太noisy
  - batch_size 是一个超参数

在 AI 中的术语：
  - Epoch: 过完一遍全部训练数据
  - Batch: 一小批数据
  - Iteration: 用一个 batch 更新一次参数
  - 1 Epoch = 数据量/batch_size 次 iterations
""")

# 模拟线性回归的 SGD vs Batch GD
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 1) * 2
y_true_val = 3.0
b_true_val = 1.0
y = X * y_true_val + b_true_val + np.random.randn(n_samples, 1) * 0.5

def compute_loss(X, y, w, b):
    """均方误差"""
    pred = X * w + b
    return np.mean((pred - y) ** 2)

def compute_gradient_batch(X, y, w, b):
    """全部数据的梯度"""
    pred = X * w + b
    error = pred - y
    dw = 2 * np.mean(error * X)
    db = 2 * np.mean(error)
    return dw, db

def compute_gradient_sgd(X, y, w, b, batch_size=16):
    """小批量梯度"""
    indices = np.random.choice(len(X), batch_size, replace=False)
    X_batch = X[indices]
    y_batch = y[indices]
    pred = X_batch * w + b
    error = pred - y_batch
    dw = 2 * np.mean(error * X_batch)
    db = 2 * np.mean(error)
    return dw, db

# 比较 Batch GD 和 Mini-batch SGD
print("\n--- Batch GD vs Mini-batch SGD ---")

# Batch GD
w_batch, b_batch = 0.0, 0.0
lr = 0.01
batch_losses = []
for _ in range(100):
    dw, db = compute_gradient_batch(X, y, w_batch, b_batch)
    w_batch -= lr * dw
    b_batch -= lr * db
    batch_losses.append(compute_loss(X, y, w_batch, b_batch))

# Mini-batch SGD
w_sgd, b_sgd = 0.0, 0.0
sgd_losses = []
for _ in range(100):
    dw, db = compute_gradient_sgd(X, y, w_sgd, b_sgd, batch_size=16)
    w_sgd -= lr * dw
    b_sgd -= lr * db
    sgd_losses.append(compute_loss(X, y, w_sgd, b_sgd))

print(f"真实参数: w={y_true_val}, b={b_true_val}")
print(f"Batch GD 结果: w={w_batch:.4f}, b={b_batch:.4f}, 最终loss={batch_losses[-1]:.4f}")
print(f"SGD 结果:      w={w_sgd:.4f}, b={b_sgd:.4f}, 最终loss={sgd_losses[-1]:.4f}")
print(f"\n两者都接近真实值，但 SGD 的路径更noisy")
assert abs(w_batch - y_true_val) < 0.5
print("✓ 验证通过")


# ============================================================
# 4. 动量 (Momentum)
# ============================================================
print("\n" + "=" * 60)
print("4. 动量 (Momentum)")
print("=" * 60)

print("""
【问题】
普通 SGD 的问题：
  1. 在狭长"峡谷"中来回震荡
  2. 在平坦区域走得太慢

【动量的直觉】
想象一个球从山上滚下来：
  - 球会越滚越快（积累速度）
  - 球不会在每个小坑都停下来
  - 球有惯性，会冲过小的颠簸

数学：
  v_t = β × v_{t-1} + (1-β) × g_t     (速度 = 上次速度×衰减 + 当前梯度)
  θ_t = θ_{t-1} - lr × v_t              (更新参数)
  
  β 通常取 0.9（保留 90% 的历史速度）

效果：
  - 加速收敛（在一致方向上越来越快）
  - 减少震荡（在震荡方向上正负抵消）
""")

# 二维优化问题：椭圆函数（有狭长峡谷）
# f(x,y) = 10*x² + y²（x方向很陡，y方向很平缓）
def f_2d(params):
    return 10 * params[0]**2 + params[1]**2

def grad_2d(params):
    return np.array([20 * params[0], 2 * params[1]])

# 普通 SGD
params_sgd = np.array([5.0, 5.0])
lr = 0.05
sgd_path = [params_sgd.copy()]
for _ in range(50):
    g = grad_2d(params_sgd)
    params_sgd = params_sgd - lr * g
    sgd_path.append(params_sgd.copy())

# SGD with Momentum
params_mom = np.array([5.0, 5.0])
velocity = np.array([0.0, 0.0])
beta = 0.9
mom_path = [params_mom.copy()]
for _ in range(50):
    g = grad_2d(params_mom)
    velocity = beta * velocity + (1 - beta) * g
    params_mom = params_mom - lr * velocity
    mom_path.append(params_mom.copy())

print(f"优化 f(x,y) = 10x² + y² (最小值在原点)")
print(f"起始点: (5, 5)")
print(f"\n50步后:")
print(f"  普通 SGD: ({params_sgd[0]:.6f}, {params_sgd[1]:.6f}), f = {f_2d(params_sgd):.6f}")
print(f"  动量 SGD: ({params_mom[0]:.6f}, {params_mom[1]:.6f}), f = {f_2d(params_mom):.6f}")
print(f"\n动量版本收敛更快更稳定！")


# ============================================================
# 5. Adam 优化器
# ============================================================
print("\n" + "=" * 60)
print("5. Adam 优化器（最流行的优化器）")
print("=" * 60)

print("""
【Adam = Adaptive Moment Estimation】

Adam 结合了两个思想：
  1. Momentum: 积累梯度的一阶矩（均值）→ 加速 + 减少震荡
  2. RMSProp: 积累梯度的二阶矩（方差）→ 自适应学习率

直觉：
  - 对于经常更新的参数，减小学习率（已经学够了）
  - 对于很少更新的参数，增大学习率（需要多学一点）
  - 同时保留动量的加速效果

算法：
  m_t = β₁ × m_{t-1} + (1-β₁) × g_t       (一阶矩/均值估计)
  v_t = β₂ × v_{t-1} + (1-β₂) × g_t²      (二阶矩/方差估计)
  m̂_t = m_t / (1 - β₁ᵗ)                    (偏差修正)
  v̂_t = v_t / (1 - β₂ᵗ)                    (偏差修正)
  θ_t = θ_{t-1} - lr × m̂_t / (√v̂_t + ε)   (更新)

默认超参数（几乎不需要调）：
  lr = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 1e-8
""")

class Adam:
    """从零实现 Adam 优化器"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # 一阶矩
        self.v = None  # 二阶矩
        self.t = 0     # 时间步
    
    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        
        # 更新一阶矩和二阶矩
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2
        
        # 偏差修正
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # 更新参数
        params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params

# 用 Adam 优化同一个问题
params_adam = np.array([5.0, 5.0])
adam = Adam(lr=0.1)
adam_path = [params_adam.copy()]

for _ in range(50):
    g = grad_2d(params_adam)
    params_adam = adam.step(params_adam, g)
    adam_path.append(params_adam.copy())

print(f"\n50步后各优化器结果:")
print(f"  普通 SGD:  f = {f_2d(params_sgd):.8f}")
print(f"  动量 SGD:  f = {f_2d(params_mom):.8f}")
print(f"  Adam:      f = {f_2d(params_adam):.8f}")
print(f"\nAdam 通常收敛最快，且几乎不需要调参！")
print(f"这就是为什么 Adam 是深度学习中最常用的优化器。")


# ============================================================
# 6. 学习率调度
# ============================================================
print("\n" + "=" * 60)
print("6. 学习率调度 (Learning Rate Scheduling)")
print("=" * 60)

print("""
【为什么需要学习率调度？】

训练初期：需要大学习率，快速接近最优解
训练后期：需要小学习率，精细调整到最优点

比喻：开车到目的地
  - 远处时加速（大步前进）
  - 到附近时减速（精细调整）

常见策略：
  1. Step Decay: 每隔 N 步，学习率乘以 γ（如 0.1）
  2. Cosine Annealing: 学习率按余弦曲线从大到小
  3. Warmup: 训练开始时从小学习率逐渐增大，然后再衰减
     (Transformer 中广泛使用)
""")

# 模拟不同学习率调度策略
n_steps = 100
initial_lr = 0.1

# 1. 常数学习率
constant_lr = [initial_lr] * n_steps

# 2. Step Decay
step_decay_lr = []
lr = initial_lr
for i in range(n_steps):
    if i > 0 and i % 30 == 0:
        lr *= 0.1
    step_decay_lr.append(lr)

# 3. Cosine Annealing
cosine_lr = [initial_lr * 0.5 * (1 + np.cos(np.pi * i / n_steps)) for i in range(n_steps)]

# 4. Warmup + Cosine
warmup_steps = 10
warmup_cosine_lr = []
for i in range(n_steps):
    if i < warmup_steps:
        lr = initial_lr * (i + 1) / warmup_steps
    else:
        progress = (i - warmup_steps) / (n_steps - warmup_steps)
        lr = initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
    warmup_cosine_lr.append(lr)

print(f"各策略在关键步骤的学习率:")
print(f"{'步骤':<6} {'常数':<10} {'Step Decay':<12} {'Cosine':<10} {'Warmup+Cos':<10}")
print("-" * 50)
for i in [0, 5, 10, 30, 50, 70, 99]:
    print(f"{i:<6} {constant_lr[i]:<10.5f} {step_decay_lr[i]:<12.5f} {cosine_lr[i]:<10.5f} {warmup_cosine_lr[i]:<10.5f}")

print("""
【实践建议】
- 简单任务：Adam + 常数学习率 0.001 就够了
- 大模型训练：Adam/AdamW + Warmup + Cosine Decay
- 图像分类：SGD + Momentum + Step Decay 效果也不错
- 大语言模型：通常用 AdamW + Linear Warmup + Cosine Decay
""")

print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点：
1. 梯度下降 = 沿损失函数下降方向走，反复迭代
2. 学习率太大会发散，太小会太慢
3. Mini-batch SGD 是实际最常用的（折中计算量和梯度质量）
4. 动量帮助加速收敛、减少震荡
5. Adam 是最流行的优化器（自适应学习率 + 动量）
6. 学习率调度可以进一步提升训练效果

恭喜！数学基础部分完成。
下一章：Python AI 工具库 → NumPy、Pandas、Matplotlib
""")
