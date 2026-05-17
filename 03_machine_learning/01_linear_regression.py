"""
第三章 3.1：线性回归（深度版）
============================

线性回归是最简单但最核心的机器学习算法，是理解所有 AI 算法的基石。

本节内容（从基础到进阶）：
 1. 什么是机器学习 & 什么是回归问题
 2. 线性回归的直觉与数学基础（假设函数）
 3. 损失函数深度剖析 —— 从直观到公式
 4. 梯度下降深度剖析 —— 原理、推导、变体、学习率
 5. 从零实现线性回归（带详细注释）
 6. 正规方程（解析解） —— 推导与代码
 7. 模型评估指标 —— MSE, MAE, RMSE, R²
 8. 多元线性回归（矩阵形式）
 9. 特征缩放 —— 标准化 vs 归一化
10. 梯度下降的三种变体 —— 批量、随机、小批量
11. 过拟合、欠拟合与正则化
12. 偏差-方差权衡（Bias-Variance Tradeoff）
13. 完整项目实战
14. 总结与思考题
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 配置中文字体，避免中文 Glyph 缺失警告
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 保存图表的目录
PLOT_DIR = "/home/hong/testcode/machine_learning_demo/03_machine_learning"

def save_plot(fig, filename, close=True):
    """保存图表到文件"""
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=100, bbox_inches='tight')
    print(f"  [图表已保存] {path}")
    if close:
        plt.close(fig)

print("=" * 70)
print("第三章 3.1：线性回归（深度版）")
print("=" * 70)

# ============================================================
# 0. 预备知识：什么是机器学习？什么是回归？
# ============================================================
print("\n" + "=" * 70)
print("0. 预备知识：什么是机器学习？什么是回归问题？")
print("=" * 70)

print("""
【什么是机器学习？】

传统编程：你写规则（如果...那么...），机器执行。
机器学习：你给机器数据（输入+输出），机器自己"学会"规则。

打个比方：
  传统编程 = 你教计算机如何钓鱼（写具体步骤）
  机器学习 = 你给计算机看很多钓鱼的视频，让它自己学会

【机器学习三大类别】

1. 监督学习（Supervised Learning）
   数据带有"标签"（正确答案），目标是学会从输入到输出的映射。
   
   ┌─────────┐     ┌──────────────┐     ┌────────┐
   │ 输入 X   │ ───→│  机器学习模型 │ ───→│ 输出 y │
   │ (特征)   │     │   (算法)     │     │ (标签) │
   └─────────┘     └──────────────┘     └────────┘
   
   监督学习又分为两类：
   
   a) 回归（Regression）：预测连续的数值
      - 预测房价（298万元）、气温（25.3度）、销售额（123.4万）
      - 输出可以是任意实数
   
   b) 分类（Classification）：预测离散的类别
      - 判断邮件是否为垃圾邮件（是/否）
      - 识别手写数字（0-9，10个类别）
      - 诊断疾病（良性/恶性）

2. 无监督学习（Unsupervised Learning）
   数据没有标签，目标是发现数据的内在结构。
   - 聚类：把相似的用户分组（客户分群）
   - 降维：把高维数据压缩到2D/3D以便可视化

3. 强化学习（Reinforcement Learning）
   智能体通过试错来学习，环境给予奖励或惩罚。
   - AlphaGo 下围棋、机器人控制、游戏AI

【回归问题的本质】

回归问题可以这样理解：
  我们有一堆数据点 (x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)
  我们想要找到一个函数 f，使得 f(xᵢ) ≈ yᵢ
  
  关键问题：什么样的 f 是"好"的？
  → 这就是线性回归要回答的问题！

【为什么从线性回归开始学？】

1. 最简单：只有一个公式 y = wx + b，容易理解
2. 最通用：后续所有深度学习模型都包含"线性变换"这一步
3. 最基础：损失函数、梯度下降、正则化等概念在这里首次出现
4. 最实用：很多实际问题用线性回归就足够了

记住这句话：
  所有复杂的 AI 模型 = 线性变换 + 非线性激活 + 优化算法
  线性回归提供了前两项的基础理解。
""")

# ============================================================
# 1. 线性回归的直觉与数学基础
# ============================================================
print("\n" + "=" * 70)
print("1. 线性回归的直觉与数学基础")
print("=" * 70)

print("""
【1.1 直觉理解：用例子说话】

假设你是房产中介，你想根据房子面积预测售价。

你收集了历史数据：
  面积(m²)     价格(万元)
  ─────────────────────────
    50           150
    60           180
    80           240
   100           310
   120           360
   150           450

你画了一个散点图（横轴=面积，纵轴=价格），发现这些点大致排成一条线。
→ 这就是"线性"的含义：两个变量之间存在近似的直线关系。

现在来了一个新客户，他的房子是90m²，值多少钱？
→ 你会在这条线上找到面积=90对应的价格。
→ 这就是"回归"——用已知的数据推断新的值。

【1.2 数学模型：假设函数】

线性回归的核心假设：目标变量 y 和特征 x 之间存在线性关系。

单变量（一个特征）：
  ŷ = wx + b
  
其中：
  ŷ (读作 "y hat") = 模型的预测值
  w (weight)       = 权重/斜率，表示 x 每增加 1 单位，y 增加多少
  b (bias)         = 偏置/截距，表示 x=0 时 y 的基础值
  x                = 输入特征

多变量（n个特征）：
  ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
  
向量形式（更简洁）：
  ŷ = w·x + b
  
其中 w·x 表示向量 w 和向量 x 的点积（内积）。

【1.3 "线性"到底是什么意思？】

"线性"不是指图像是一条直线，而是指：
  模型关于参数 w 和 b 是线性的。

换句话说：
  ŷ = w₁x₁ + w₂x₂ + b  ← 这是线性模型（w₁, w₂, b 的幂都是 1）
  ŷ = w₁x₁² + w₂x₂ + b  ← 这仍然是线性模型！（w₁ 的幂是 1，x₁² 只是一个"新特征"）
  ŷ = sin(w₁x₁) + b      ← 这不是线性模型（w₁ 在 sin 里面）

这个区别很重要！线性回归也可以用 x², x³, sin(x) 等作为特征，
只要模型关于参数是线性的就行。

【为什么 x² 仍然是线性模型？】

关键在于：线性模型要求的是"参数"（w, b）的幂为 1，而不是输入 x 的幂。

以 ŷ = w₁x₁² + w₂x₂ + b 为例：
  - w₁ 的幂是 1 → 参数是线性的 ✓
  - x₁² 不是参数，而是输入特征本身

你可以做一个变量替换：令 x₁' = x₁²，那么：
  ŷ = w₁x₁' + w₂x₂ + b

这就是标准的线性模型！参数 w₁ 仍然是 1 次方，没有出现在任何非线性函数内部。

【什么是"新特征"？】

"新特征"是指对原始输入进行变换后得到的新的输入变量。

例如：
  原始特征：x = [1, 2, 3, 4]
  新特征：x' = x² = [1, 4, 9, 16]

在机器学习中，你可以自由地对原始特征做各种变换（平方、立方、对数、sin等），
这些变换后的结果就是"新特征"。

这样做的好处：
  - 原始数据可能是非线性关系（如面积和房价是平方关系）
  - 通过增加新特征（如 x²），线性模型也能拟合非线性关系
  - 本质上是把非线性问题转化到更高维的特征空间中

但要注意：只有"特征变换"才是线性的，如果"参数"被放进非线性函数里面（如 sin(w₁x)），
那就变成非线性模型了。

【1.4 为什么是"回归"？】

"回归"这个词来自英国科学家弗朗西斯·高尔顿（Francis Galton）。
他研究父母和子女身高的关系时发现：
  高个子父母的孩子往往比父母矮一点（向平均值"回归"）
  矮个子父母的孩子往往比父母高一点（向平均值"回归"）

这种现象叫"回归均值"（Regression to the Mean），因此得名。

【1.5 我们的目标】

找到最优的 w 和 b，使得模型预测值 ŷ 尽可能接近真实值 y。

但"尽可能接近"怎么定义？
→ 这就引出了"损失函数"的概念。
""")

# ============================================================
# 2. 损失函数深度剖析
# ============================================================
print("\n" + "=" * 70)
print("2. 损失函数深度剖析 —— 衡量模型有多差")
print("=" * 70)

print("""
【2.1 什么是损失函数？为什么要损失函数？】

在回答"什么是损失函数"之前，先回答"为什么需要损失函数"。

假设你训练了一个线性回归模型，得到了 w=3.0, b=2.0。
你怎么知道这个模型好不好？

你需要一个数字来回答：这个模型的预测误差有多大？
这个数字就是"损失"（Loss）或"代价"（Cost）。

损失函数的作用：
  输入：模型的参数 (w, b) + 数据 (X, y)
  输出：一个数字，表示模型"有多差"

损失函数越小 → 模型越好
损失函数越大 → 模型越差

我们的目标：找到让损失函数最小的 w 和 b。

【2.2 直觉理解：射箭的例子】

想象你在练习射箭：
  靶心 = 真实值 y
  你的箭 = 预测值 ŷ
  
每次射箭，你的箭和靶心都有一个距离（误差）。
  误差 = |ŷ - y|

你想评价自己射得好不好，怎么办？
  → 把所有箭到靶心的距离加起来，得到一个"总误差"。
  → 总误差越小 = 你射得越好。

这就是损失函数的核心思想。

【2.3 为什么不能直接用"误差之和"？】

你可能会想：直接用 Σ(yᵢ - ŷᵢ) 不行吗？

不行！因为：
  有些误差是正的（预测值偏高），有些是负的（预测值偏低）。
  它们会互相抵消！

例子：
  真实值: [10, 20, 30]
  预测值: [12, 18, 30]
  误差:   [-2, +2,  0]  → 误差之和 = 0

误差之和 = 0 不代表预测准确！它只是正负抵消了。

【2.4 解决方案 1：平均绝对误差（MAE）】

MAE = (1/n) × Σ|yᵢ - ŷᵢ|

优点：
  - 直观：就是平均每个样本的误差
  - 对异常值不敏感（因为没用平方）

缺点：
  - 绝对值函数 |x| 在 x=0 处不可导
  - 数学上不好处理（梯度下降需要求导）

【2.5 解决方案 2：均方误差（MSE） ★ 最常用】

MSE = (1/n) × Σ(yᵢ - ŷᵢ)²

这就是线性回归最常用的损失函数。

为什么选 MSE？四个原因：

  1. 解决符号问题
     (yᵢ - ŷᵢ)² 永远是正数，不会互相抵消。

  2. 放大误差
     大误差会被平方放大。
     误差=2 → 平方=4
     误差=10 → 平方=100（比误差=2时放大了25倍！）
     
     这意味着模型会更"在意"大的错误，促使它尽快修正大的偏差。

  3. 可微性
     f(x) = x² 在任何点都可导，梯度下降非常顺滑。
     
     对比 |x| 在 x=0 处不可导，梯度下降在 x=0 附近会不稳定。

  4. 统计学基础（高斯-马尔可夫定理）
     如果误差服从正态分布（高斯分布），最小化 MSE 等价于
     最大似然估计（Maximum Likelihood Estimation, MLE）。
     这意味着 MSE 在统计学意义上是最优的。

【2.6 MSE 的可视化理解】

MSE 就像一个碗：
  
        L(w,b)
          │
         ╱│╲
        ╱ │ ╲      ← 碗的形状
       ╱  │  ╲
  ────┴───●───┴───  ← 碗底 = 最小损失 = 最优参数
       w*  b*

这个"碗"在数学上叫"凸函数"（Convex Function）。
凸函数有一个极佳的性质：只有一个最低点（全局最小值），没有"假谷"（局部最小值）。

这意味着：只要我们沿着碗壁往下走，一定能走到碗底！

【2.7 代码演示：不同损失函数的对比】
""")

# 生成数据并对比不同损失函数
np.random.seed(42)
n_demo = 50
X_demo = np.linspace(0, 10, n_demo)
true_w, true_b = 2.0, 5.0
y_demo = true_w * X_demo + true_b + np.random.randn(n_demo) * 2.0

# 用不同的 w 值来观察损失函数变化
w_range = np.linspace(0, 4, 100)
mse_values = []
mae_values = []

for w in w_range:
    b_fixed = 5.0  # 暂时固定 b
    y_pred = w * X_demo + b_fixed
    mse = np.mean((y_pred - y_demo) ** 2)
    mae = np.mean(np.abs(y_pred - y_demo))
    mse_values.append(mse)
    mae_values.append(mae)

mse_values = np.array(mse_values)
mae_values = np.array(mae_values)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# MSE 曲线
axes[0].plot(w_range, mse_values, 'b-', linewidth=2)
axes[0].axvline(x=true_w, color='r', linestyle='--', alpha=0.7, label=f'真实 w={true_w}')
axes[0].set_xlabel('权重 w')
axes[0].set_ylabel('MSE')
axes[0].set_title('均方误差 (MSE) vs 权重 w')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE 曲线
axes[1].plot(w_range, mae_values, 'g-', linewidth=2)
axes[1].axvline(x=true_w, color='r', linestyle='--', alpha=0.7, label=f'真实 w={true_w}')
axes[1].set_xlabel('权重 w')
axes[1].set_ylabel('MAE')
axes[1].set_title('平均绝对误差 (MAE) vs 权重 w')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

save_plot(fig, 'loss_function_comparison.png')

print("\nMSE vs MAE 对比：")
min_mse_w = w_range[np.argmin(mse_values)]
min_mae_w = w_range[np.argmin(mae_values)]
print(f"  MSE 最小在 w≈{min_mse_w:.2f}（真实值 {true_w}）")
print(f"  MAE 最小在 w≈{min_mae_w:.2f}（真实值 {true_w}）")
print("  两者都找到了接近真实值的 w，但 MSE 曲线更平滑，更容易用梯度下降找到最小值")

print("""
【2.8 损失函数的"地形图"理解】

把损失函数想象成一个地形：
  - 横轴 = w（权重）
  - 纵轴 = b（偏置）  
  - 海拔高度 = 损失值

你的任务：从山上某个位置出发，找到最低的山谷。
→ 这就是梯度下降要做的事！

下面我们用代码来可视化损失函数的 2D 地形图。
""")

# 2D 损失函数地形图
w_grid = np.linspace(0, 4, 50)
b_grid = np.linspace(0, 10, 50)
W_grid, B_grid = np.meshgrid(w_grid, b_grid)
Z = np.zeros_like(W_grid)

for i in range(len(w_grid)):
    for j in range(len(b_grid)):
        y_pred = W_grid[j, i] * X_demo + B_grid[j, i]
        Z[j, i] = np.mean((y_pred - y_demo) ** 2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 等高线图
contour = axes[0].contourf(W_grid, B_grid, Z, levels=30, cmap='viridis')
axes[0].plot(true_w, true_b, 'r*', markersize=15, label='最优参数')
axes[0].set_xlabel('w (权重)')
axes[0].set_ylabel('b (偏置)')
axes[0].set_title('损失函数等高线图 (MSE)')
axes[0].legend()
plt.colorbar(contour, ax=axes[0], label='MSE')

# 3D 图
ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(W_grid, B_grid, Z, cmap='viridis', alpha=0.8, edgecolor='none')
ax.plot([true_w], [true_b], [np.min(Z)], 'r*', markersize=15)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
ax.set_title('损失函数 3D 地形图')

save_plot(fig, 'loss_landscape_2d.png')

print("""
【2.9 总结：损失函数的选择】

┌────────────┬──────────────┬──────────────┬─────────────────┐
│ 损失函数   │ 公式         │ 优点         │ 缺点            │
├────────────┼──────────────┼──────────────┼─────────────────┤
│ MSE        │ (1/n)Σ(y-ŷ)² │ 可微、凸函数 │ 对异常值敏感    │
│ MAE        │ (1/n)Σ|y-ŷ|  │ 对异常值稳健 │ x=0处不可导     │
│ Huber      │ 组合MSE+MAE  │ 兼顾两者     │ 需要调参δ       │
│ RMSLE      │ 对数形式     │ 适合大数值   │ 解释性差        │
└────────────┴──────────────┴──────────────┴─────────────────┘

线性回归默认使用 MSE，这也是本课程的重点。
""")

# ============================================================
# 3. 梯度下降深度剖析
# ============================================================
print("\n" + "=" * 70)
print("3. 梯度下降深度剖析 —— 找到碗底的算法")
print("=" * 70)

print("""
【3.1 什么是梯度下降？】

有了损失函数 L(w, b)，我们知道：
  - L 越小，模型越好
  - 我们的目标是找到让 L 最小的 w 和 b

但怎么找？不可能一个个试（参数空间是连续的，有无穷多个值）。

梯度下降给了我们一个系统的方法：
  "沿着最陡的方向下山"

【3.2 直觉理解：下山的故事】

想象你站在一座山上，被蒙住了眼睛，你的目标是走到山谷最低处。

你怎么做？
  1. 用脚感受周围的地面
  2. 找到最陡的下坡方向
  3. 往那个方向走一步
  4. 重复 1-3，直到再也走不下去

梯度下降就是这个过程：
  1. 计算当前点的梯度（最陡方向）
  2. 沿梯度的反方向走一步
  3. 重复，直到收敛

【3.3 什么是梯度？】

梯度（Gradient）是一个向量，指向函数值增加最快的方向。

对于一元函数 f(w)：
  梯度 = df/dw（导数/斜率）
  
  如果 df/dw > 0：函数在增加，往左走（减小 w）
  如果 df/dw < 0：函数在减小，往右走（增大 w）

对于二元函数 L(w, b)：
  梯度 = (∂L/∂w, ∂L/∂b)
  
  这是一个二维向量，指向 L 增加最快的方向。

【3.4 MSE 的梯度推导（重要！）】

损失函数：
  L = (1/n) × Σ(yᵢ - ŷᵢ)²
  其中 ŷᵢ = w×xᵢ + b

对 w 求偏导（链式法则）：
  ∂L/∂w = (1/n) × Σ 2(yᵢ - ŷᵢ) × ∂(yᵢ - ŷᵢ)/∂w
        = (1/n) × Σ 2(yᵢ - ŷᵢ) × (-xᵢ)
        = (2/n) × Σ(ŷᵢ - yᵢ) × xᵢ

对 b 求偏导：
  ∂L/∂b = (1/n) × Σ 2(yᵢ - ŷᵢ) × ∂(yᵢ - ŷᵢ)/∂b
        = (1/n) × Σ 2(yᵢ - ŷᵢ) × (-1)
        = (2/n) × Σ(ŷᵢ - yᵢ)

注意：
  ∂(ŷᵢ - yᵢ)/∂w = ∂(wxᵢ + b - yᵢ)/∂w = xᵢ
  ∂(ŷᵢ - yᵢ)/∂b = ∂(wxᵢ + b - yᵢ)/∂b = 1

我们把 (ŷᵢ - yᵢ) 换成 -(yᵢ - ŷᵢ)，消去负号：
  ∂L/∂w = (2/n) × Σ(ŷᵢ - yᵢ) × xᵢ   ← 这就是代码中的 dw
  ∂L/∂b = (2/n) × Σ(ŷᵢ - yᵢ)         ← 这就是代码中的 db

【3.5 更新规则】

得到梯度后，更新参数：
  w = w - lr × ∂L/∂w
  b = b - lr × ∂L/∂b

为什么要减去梯度？
  梯度指向损失增加最快的方向 → 我们要的是减少 → 所以要反向走

lr（Learning Rate，学习率）是什么？
  它控制每一步走多大。

  lr 太大 → 步子太大 → 可能会跳过最低点，甚至越跳越高（发散）
  lr 太小 → 步子太小 → 走得很慢，可能走不到最低点就停了
  lr 适中 → 稳步下山 → 顺利收敛

【3.6 学习率的可视化】
""")

# 不同学习率的梯度下降演示
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 使用简化的一维 MSE 曲线来演示
w_values = np.linspace(0, 4, 200)
mse_curve = np.array([np.mean((w * X_demo + 5.0 - y_demo) ** 2) for w in w_values])

learning_rates = [0.0001, 0.001, 0.005, 0.02]
titles = ['学习率太小 (收敛慢)', '学习率偏小', '学习率适中 ★', '学习率太大 (发散)']

for idx, (lr, title) in enumerate(zip(learning_rates, titles)):
    ax = axes[idx // 2, idx % 2]
    ax.plot(w_values, mse_curve, 'b-', linewidth=2, alpha=0.5)
    
    # 梯度下降轨迹
    w_trace = [0.5]  # 起始点
    b_trace = 5.0
    
    for _ in range(200):
        y_pred = w_trace[-1] * X_demo + b_trace
        dw = (2/n_demo) * np.sum((y_pred - y_demo) * X_demo)
        w_next = w_trace[-1] - lr * dw
        
        if len(w_trace) > 1 and abs(w_next - w_trace[-1]) > 10:
            break  # 发散了
        w_trace.append(w_next)
    
    w_trace = np.array(w_trace)
    
    if len(w_trace) < 50:
        ax.scatter(w_trace[:10], 
                   [np.mean((w * X_demo + b_trace - y_demo)**2) for w in w_trace[:10]],
                   c='red', s=10, alpha=0.7)
        ax.set_title(title + ' [发散]')
    else:
        mse_trace = [np.mean((w * X_demo + b_trace - y_demo)**2) for w in w_trace]
        ax.plot(w_trace, mse_trace, 'r-o', markersize=3, linewidth=1, alpha=0.7)
        ax.set_title(title)
    
    ax.axvline(x=true_w, color='g', linestyle='--', alpha=0.5)
    ax.set_xlabel('w')
    ax.set_ylabel('MSE')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 5)

save_plot(fig, 'learning_rate_comparison.png')

print("""
【3.7 梯度下降收敛的判断】

什么时候停止迭代？
  1. 达到最大迭代次数（最常用，简单粗暴）
  2. 两次迭代的损失变化很小：|L_new - L_old| < ε（如 1e-6）
  3. 梯度的范数很小：||∇L|| < ε（说明已经很平了）

实际中通常用 1 或 2 的组合。

【3.8 损失下降曲线】

训练过程中，损失应该持续下降：

  Loss
   │
   │╲
   │ ╲
   │  ╲
   │   ╲───────  ← 收敛（变化很小）
   └─────────── Epoch

如果损失不下降 → 学习率太小或太大
如果损失震荡 → 学习率太大
如果损失 NaN/Inf → 学习率过大导致溢出
""")

# 损失下降曲线
fig, ax = plt.subplots(figsize=(8, 4))
epochs = np.arange(1, 101)
loss_curve = 50 * np.exp(-0.05 * epochs) + 2 + 0.5 * np.random.randn(100) * 0.01
loss_curve = np.maximum(loss_curve, 0)  # 确保非负

ax.plot(epochs, loss_curve, 'b-', linewidth=2)
ax.set_xlabel('Epoch (迭代次数)')
ax.set_ylabel('Loss (MSE)')
ax.set_title('训练过程中的损失下降曲线')
ax.grid(True, alpha=0.3)

# 标记收敛区域
ax.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='收敛')
ax.fill_between([80, 100], 0, 60, alpha=0.1, color='green')
ax.text(85, 55, '← 收敛区域', fontsize=10, color='green')
ax.legend()

save_plot(fig, 'loss_curve_demo.png')

# ============================================================
# 4. 从零实现线性回归
# ============================================================
print("\n" + "=" * 70)
print("4. 从零实现线性回归（带详细解释）")
print("=" * 70)

print("""
现在我们用前面学到的所有知识，从零实现一个完整的线性回归模型。

实现步骤：
  1. 定义模型（假设函数）：ŷ = wx + b
  2. 定义损失函数（MSE）：L = (1/n) × Σ(yᵢ - ŷᵢ)²
  3. 计算梯度：∂L/∂w 和 ∂L/∂b
  4. 更新参数：w = w - lr × ∂L/∂w，b = b - lr × ∂L/∂b
  5. 重复步骤 3-4 直到收敛
""")

# 生成模拟数据：y = 3x + 2 + 噪声
np.random.seed(42)
n_samples = 100
X = np.random.uniform(0, 10, n_samples)
y_true_w, y_true_b = 3.0, 2.0
noise_std = 1.5
y = y_true_w * X + y_true_b + np.random.randn(n_samples) * noise_std

print(f"数据生成过程: y = {y_true_w}x + {y_true_b} + N(0, {noise_std}²)")
print(f"  样本数: {n_samples}")
print(f"  X 范围: [{X.min():.2f}, {X.max():.2f}]")
print(f"  y 范围: [{y.min():.2f}, {y.max():.2f}]")

# 可视化数据
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, y, alpha=0.6, s=30, label='数据点')
ax.plot([0, 10], [y_true_b, y_true_w * 10 + y_true_b], 'r--', linewidth=2, 
        label=f'真实直线: y = {y_true_w}x + {y_true_b}')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('原始数据散点图')
ax.legend()
ax.grid(True, alpha=0.3)
save_plot(fig, 'original_data.png')


class LinearRegressionFromScratch:
    """从零实现的线性回归 —— 每个步骤都有详细注释"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        初始化模型参数
        
        Args:
            learning_rate: 学习率 (lr)，控制每次更新步长
            n_iterations: 最大迭代次数 (epochs)
        """
        self.lr = learning_rate
        self.n_iter = n_iterations
        
        # 初始化参数
        # w = 0, b = 0 是常见的初始值（从零开始学）
        # 也可以用小的随机值初始化，效果通常更好
        self.w = 0.0
        self.b = 0.0
        
        # 记录训练过程中的损失，用于绘制学习曲线
        self.losses = []
        
        # 记录每一步的参数，用于可视化训练过程
        self.w_history = []
        self.b_history = []
    
    def predict(self, X):
        """
        前向传播：用当前参数预测
        
        公式: ŷ = wx + b
        
        Args:
            X: 输入特征 (numpy array)
        Returns:
            ŷ: 预测值 (numpy array)
        """
        return self.w * X + self.b
    
    def compute_loss(self, X, y):
        """
        计算均方误差 (MSE)
        
        公式: L = (1/n) × Σ(yᵢ - ŷᵢ)²
        
        Args:
            X: 输入特征
            y: 真实标签
        Returns:
            loss: 标量，损失值
        """
        predictions = self.predict(X)
        errors = predictions - y  # 每个样本的误差
        squared_errors = errors ** 2  # 平方误差
        loss = np.mean(squared_errors)  # 平均
        return loss
    
    def compute_gradients(self, X, y):
        """
        计算梯度（损失函数对 w 和 b 的偏导数）
        
        ∂L/∂w = (2/n) × Σ(ŷᵢ - yᵢ) × xᵢ
        ∂L/∂b = (2/n) × Σ(ŷᵢ - yᵢ)
        
        推导过程（链式法则）：
          L = (1/n) × Σ(yᵢ - ŷᵢ)²
          ŷᵢ = wxᵢ + b
          
          ∂L/∂w = ∂L/∂ŷᵢ × ∂ŷᵢ/∂w
                 = (2/n) × Σ(yᵢ - ŷᵢ) × (-1) × xᵢ
                 = (2/n) × Σ(ŷᵢ - yᵢ) × xᵢ
        
        Args:
            X: 输入特征
            y: 真实标签
        Returns:
            dw: 损失对 w 的梯度
            db: 损失对 b 的梯度
        """
        n = len(X)
        predictions = self.predict(X)
        errors = predictions - y  # (ŷᵢ - yᵢ)
        
        # dw = (2/n) × Σ(ŷᵢ - yᵢ) × xᵢ
        dw = (2/n) * np.sum(errors * X)
        
        # db = (2/n) × Σ(ŷᵢ - yᵢ)
        db = (2/n) * np.sum(errors)
        
        return dw, db
    
    def fit(self, X, y, verbose=True):
        """
        训练模型 —— 核心循环
        
        算法流程（伪代码）：
          for epoch in range(n_iterations):
              1. 计算预测值 ŷ = wx + b
              2. 计算梯度 dw, db
              3. 更新参数 w = w - lr * dw, b = b - lr * db
              4. 记录损失
              5. 检查是否收敛
          
        Args:
            X: 训练数据特征
            y: 训练数据标签
            verbose: 是否打印训练过程
        Returns:
            self
        """
        n = len(X)
        
        for i in range(self.n_iter):
            # 步骤 1: 计算梯度
            dw, db = self.compute_gradients(X, y)
            
            # 步骤 2: 更新参数
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            # 步骤 3: 记录损失和参数历史
            loss = self.compute_loss(X, y)
            self.losses.append(loss)
            self.w_history.append(self.w)
            self.b_history.append(self.b)
            
            # 打印训练进度
            if verbose and (i % 200 == 0 or i == self.n_iter - 1):
                print(f"  Epoch {i:4d}: loss={loss:.4f}, w={self.w:.4f}, b={self.b:.4f}, dw={dw:.4f}, db={db:.4f}")
            
            # 收敛检查：如果两次损失变化非常小，提前停止
            if i > 0 and abs(self.losses[-2] - self.losses[-1]) < 1e-8:
                if verbose:
                    print(f"  [提前收敛] Epoch {i}: loss 变化 < 1e-8")
                break
        
        return self


print("\n--- 开始训练 ---")
model = LinearRegressionFromScratch(learning_rate=0.005, n_iterations=2000)
model.fit(X, y)

print(f"\n--- 训练结果 ---")
print(f"学到的参数: w={model.w:.4f}, b={model.b:.4f}")
print(f"真实参数:   w={y_true_w:.4f}, b={y_true_b:.4f}")
print(f"误差:       Δw={abs(model.w - y_true_w):.4f}, Δb={abs(model.b - y_true_b):.4f}")
print(f"最终损失:   {model.losses[-1]:.4f}")

# 验证结果
assert abs(model.w - y_true_w) < 0.3, f"w 偏差太大: {model.w}"
assert abs(model.b - y_true_b) < 1.0, f"b 偏差太大: {model.b}"
print("✓ 参数接近真实值，验证通过")

# 可视化训练过程
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. 数据 + 拟合直线
axes[0].scatter(X, y, alpha=0.5, s=20, label='数据点')
axes[0].plot([0, 10], [y_true_b, y_true_w*10 + y_true_b], 'r--', alpha=0.7, label='真实直线')
axes[0].plot([0, 10], [model.b, model.w*10 + model.b], 'g-', linewidth=2, label='拟合直线')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title(f'拟合结果: y = {model.w:.2f}x + {model.b:.2f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. 损失曲线
axes[1].plot(model.losses, 'b-', linewidth=1.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MSE Loss')
axes[1].set_title('损失下降曲线')
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3)

# 3. 参数收敛
epochs_plot = range(len(model.w_history))
axes[2].plot(epochs_plot, model.w_history, 'b-', label=f'w (最终={model.w:.3f})')
axes[2].plot(epochs_plot, model.b_history, 'r-', label=f'b (最终={model.b:.3f})')
axes[2].axhline(y=y_true_w, color='b', linestyle='--', alpha=0.5)
axes[2].axhline(y=y_true_b, color='r', linestyle='--', alpha=0.5)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('参数值')
axes[2].set_title('参数收敛过程')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

save_plot(fig, 'training_process.png')

print("""
【训练过程解读】

上面的三张图展示了训练的三个维度：

左图（拟合结果）：
  - 蓝色散点 = 训练数据
  - 红色虚线 = 真实的数据生成直线
  - 绿色实线 = 我们学到的拟合直线
  - 两条线越接近，说明学得越好

中图（损失曲线）：
  - 损失从一开始很大，快速下降，然后趋于平缓
  - 对数坐标下，初期下降几乎是一条直线（指数衰减）
  - 最终趋近一个常数（噪声的方差，无法消除）

右图（参数收敛）：
  - w 和 b 从 0 开始，逐步接近真实值
  - 变化速度越来越慢 → 梯度越来越小 → 接近最优值

注意：最终损失不会为 0！因为数据中有噪声。
最优的 MSE = 噪声方差 = 1.5² = 2.25。
模型只能消除"系统性误差"，不能消除"随机噪声"。
""")

# ============================================================
# 5. 正规方程（解析解）
# ============================================================
print("\n" + "=" * 70)
print("5. 正规方程（解析解）—— 一步到位")
print("=" * 70)

print("""
【5.1 正规方程是什么？】

梯度下降是一个"迭代"方法：一步一步逼近最优解。
但对于线性回归，有一个"解析"方法：一步直接得到精确解！

这个精确解叫"正规方程"（Normal Equation）：

  θ = (XᵀX)⁻¹ Xᵀy

其中：
  θ = [w, b]ᵀ（所有参数组成的向量）
  X = 设计矩阵（每行一个样本，每列一个特征，第一列全为1对应偏置）
  y = 目标向量

【5.2 正规方程的推导（选读，但强烈推荐）】

目标：最小化 L = (1/n)(Xθ - y)ᵀ(Xθ - y)

步骤 1：展开
  L = (1/n)(θᵀXᵀXθ - θᵀXᵀy - yᵀXθ + yᵀy)
  L = (1/n)(θᵀXᵀXθ - 2θᵀXᵀy + yᵀy)
  （因为 θᵀXᵀy = yᵀXθ，两者都是标量且相等）

步骤 2：对 θ 求导
  ∂L/∂θ = (2/n)(XᵀXθ - Xᵀy)

步骤 3：令导数为 0（求最小值点）
  (2/n)(XᵀXθ - Xᵀy) = 0
  XᵀXθ = Xᵀy
  θ = (XᵀX)⁻¹ Xᵀy  ← 这就是正规方程！

【5.3 正规方程 vs 梯度下降】

┌────────────┬──────────────────┬──────────────────────┐
│ 特性       │ 正规方程         │ 梯度下降             │
├────────────┼──────────────────┼──────────────────────┤
│ 求解方式   │ 一步精确解       │ 逐步逼近             │
│ 时间复杂度 │ O(n·d² + d³)     │ O(k·n·d)             │
│ 需要调参   │ 不需要           │ 学习率、迭代次数     │
│ 数据量限制 │ d 大时很慢       │ 可处理大规模数据     │
│ 适用场景   │ 小数据(d<10000)  │ 大数据、深度学习     │
│ 特征缩放   │ 不需要           │ 需要（加速收敛）     │
└────────────┴──────────────────┴──────────────────────┘

其中 n = 样本数, d = 特征数, k = 迭代次数。

注意：(XᵀX)⁻¹ 要求 XᵀX 可逆。如果特征之间有共线性
（比如 x₁ 和 x₂ 完全线性相关），XᵀX 不可逆，正规方程失效。
此时可以用正则化（Ridge）或去掉共线特征。

【5.4 代码实现】
""")

# 构造设计矩阵（加一列 1 表示偏置）
X_design = np.column_stack([X, np.ones(n_samples)])
print(f"设计矩阵形状: {X_design.shape}  (100个样本 × 2列: [x, 1])")
print(f"前5行:\n{X_design[:5]}")

# 正规方程求解
theta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
w_analytical, b_analytical = theta[0], theta[1]

print(f"\n正规方程结果: w={w_analytical:.4f}, b={b_analytical:.4f}")
print(f"梯度下降结果: w={model.w:.4f}, b={model.b:.4f}")
print(f"真实值:       w={y_true_w:.4f}, b={y_true_b:.4f}")
print(f"差异:         Δw={abs(w_analytical - model.w):.6f}, Δb={abs(b_analytical - model.b):.6f}")

assert abs(w_analytical - y_true_w) < 0.3
print("✓ 正规方程验证通过")

# 可视化对比
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, y, alpha=0.4, s=20, label='数据点')
ax.plot([0, 10], [y_true_b, y_true_w*10 + y_true_b], 'r--', linewidth=2, label='真实直线')
ax.plot([0, 10], [b_analytical, w_analytical*10 + b_analytical], 'g-', linewidth=3, label='正规方程拟合')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('正规方程 vs 真实直线')
ax.legend()
ax.grid(True, alpha=0.3)
save_plot(fig, 'normal_equation_fit.png')

print("""
正规方程得到的是一个精确解（在数值精度范围内）。
与梯度下降相比：
  - 正规方程：一步到位，没有误差
  - 梯度下降：需要调参，但结果非常接近
  - 两者都应该接近真实参数（在噪声范围内）
""")

# ============================================================
# 6. 模型评估指标
# ============================================================
print("\n" + "=" * 70)
print("6. 模型评估指标 —— 怎么判断模型好不好？")
print("=" * 70)

print("""
【6.1 为什么需要评估指标？】

训练完模型后，我们得到一个损失值。但损失值本身不够直观：
  MSE = 2.34  → 这个是好还是差？没有参考系。

我们需要更直观的指标来评估模型。

【6.2 常用评估指标】

1. MSE（均方误差 Mean Squared Error）
   MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
   单位：y 的单位的平方
   越小越好，但没有上限，不方便跨数据集比较

2. RMSE（均方根误差 Root Mean Squared Error）
   RMSE = √MSE
   单位：与 y 相同
   解释："平均来说，预测偏离真实值大约 RMSE 个单位"
   
   例: 预测房价，RMSE = 5万元 → 平均偏离约 5 万元

3. MAE（平均绝对误差 Mean Absolute Error）
   MAE = (1/n) × Σ|yᵢ - ŷᵢ|
   单位：与 y 相同
   解释：与 RMSE 类似，但不放大大的误差

4. R²（决定系数 R-squared / Coefficient of Determination）
   R² = 1 - SS_res / SS_tot
   其中：
     SS_res = Σ(yᵢ - ŷᵢ)²  （残差平方和）
     SS_tot = Σ(yᵢ - ȳ)²  （总平方和，ȳ 是 y 的均值）
   
   取值范围：(-∞, 1]
   R² = 1：完美拟合
   R² = 0：和直接猜均值一样差
   R² < 0：比猜均值还差（模型完全不行）
   
   解释："模型解释了 y 的 R²×100% 的变化"
   例: R² = 0.85 → 模型解释了 85% 的数据变化

【6.3 为什么用 R² 而不是 MSE？】

MSE 的问题：
  - 没有标准化，不同数据集的 MSE 无法比较
  - 单位是平方，解释性差

R² 的优势：
  - 标准化到 [-∞, 1]，跨数据集可比
  - 有直观的百分比解释
  - 不需要调参或选择阈值

【6.4 代码演示】
""")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# sklearn 需要 2D 输入
X_sklearn = X.reshape(-1, 1)

# 训练 sklearn 模型
sk_model = LinearRegression()
sk_model.fit(X_sklearn, y)

# 预测
y_pred = sk_model.predict(X_sklearn)

# 计算各项指标
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

y_mean = np.mean(y)
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y_mean) ** 2)
r2_manual = 1 - ss_res / ss_tot

print(f"模型: ŷ = {sk_model.coef_[0]:.4f}x + {sk_model.intercept_:.4f}")
print(f"\n评估指标:")
print(f"  MSE  (均方误差):      {mse:.4f}")
print(f"  RMSE (均方根误差):     {rmse:.4f}  → 平均偏离 {rmse:.2f} 个单位")
print(f"  MAE  (平均绝对误差):   {mae:.4f}")
print(f"  R²   (决定系数):       {r2:.4f}  → 模型解释了 {r2*100:.1f}% 的变化")
print(f"  R²   (手动计算):       {r2_manual:.4f}  → 验证一致")

# 残差图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

residuals = y - y_pred

axes[0].scatter(y_pred, residuals, alpha=0.6, s=30)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('预测值')
axes[0].set_ylabel('残差 (真实 - 预测)')
axes[0].set_title('残差图（理想情况：随机分布在0附近）')
axes[0].grid(True, alpha=0.3)

# 预测值 vs 真实值
axes[1].scatter(y, y_pred, alpha=0.6, s=30)
max_val = max(y.max(), y_pred.max())
min_val = min(y.min(), y_pred.min())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测线')
axes[1].set_xlabel('真实值')
axes[1].set_ylabel('预测值')
axes[1].set_title('预测值 vs 真实值（越靠近红线越好）')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

save_plot(fig, 'evaluation_metrics.png')

print("""
【6.5 残差图解读】

残差图是诊断模型问题的利器：

  - 残差随机分布在 0 附近（无明显模式）→ 模型拟合良好 ✓
  - 残差呈现某种模式（如 U 形）→ 模型没有捕捉到非线性关系
  - 残差的方差随预测值增大 → 异方差性（需要加权最小二乘）
  - 某些残差异常大 → 可能是异常值

上面的残差图应该是随机分布的，说明线性回归模型对线性数据拟合良好。
""")

# ============================================================
# 7. 多元线性回归（矩阵形式）
# ============================================================
print("\n" + "=" * 70)
print("7. 多元线性回归 —— 多个特征的线性回归")
print("=" * 70)

print("""
【7.1 从一元到多元】

现实世界的问题很少有单一特征。例如预测房价：
  房价 = f(面积, 房间数, 楼龄, 地段, 学区, ...)

多元线性回归的公式：
  ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

向量形式：
  ŷ = w·x + b

矩阵形式（n 个样本）：
  ŷ = Xw + b

其中：
  X 是 n×d 的矩阵（n 个样本，d 个特征）
  w 是 d 维向量
  ŷ 是 n 维向量

【7.2 矩阵形式的梯度下降】

梯度公式从标量扩展到矩阵：
  dw = (2/n) × Xᵀ(ŷ - y)    ← 从 (2/n)×Σ(ŷᵢ-yᵢ)×xᵢ 扩展而来
  db = (2/n) × Σ(ŷᵢ - yᵢ)   ← 不变

这里的 Xᵀ(ŷ - y) 是一个矩阵乘法，等价于对每个特征的梯度求和。

【7.3 矩阵形式的正规方程】

θ = (XᵀX)⁻¹ Xᵀy

其中 X 的第一列全为 1（对应偏置项）。

【7.4 代码实现】
""")

# 生成多特征数据
np.random.seed(42)
n = 200
# 3个特征
X_multi = np.random.randn(n, 3)
true_weights = np.array([3.0, -1.5, 2.0])
true_bias = 5.0
y_multi = X_multi @ true_weights + true_bias + np.random.randn(n) * 0.5

print(f"数据生成: y = 3x₁ - 1.5x₂ + 2x₃ + 5 + N(0, 0.5²)")
print(f"特征矩阵形状: {X_multi.shape}  (200样本 × 3特征)")
print(f"特征1均值: {X_multi[:, 0].mean():.3f}, 特征2均值: {X_multi[:, 1].mean():.3f}, 特征3均值: {X_multi[:, 2].mean():.3f}")


class MultiLinearRegression:
    """多元线性回归 - 从零实现（矩阵运算）"""
    
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
    
    def fit(self, X, y):
        """
        训练多元线性回归模型
        
        矩阵运算 vs 标量运算：
        
        标量版（单变量）：
          dw = (2/n) * Σ(ŷᵢ - yᵢ) * xᵢ   ← 需要循环
          
        矩阵版（多变量）：
          dw = (2/n) * Xᵀ @ (ŷ - y)      ← 一行代码搞定！
          
        矩阵版的好处：
          1. 代码更简洁（没有 for 循环）
          2. 计算更快（NumPy 底层用 C/Fortran 优化）
          3. 自动处理任意数量的特征
        """
        n_samples, n_features = X.shape
        
        # 初始化权重为 0
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.losses = []
        
        for i in range(self.n_iter):
            # 前向传播
            y_pred = X @ self.w + self.b  # 矩阵乘法
            
            # 计算梯度（矩阵形式）
            errors = y_pred - y
            dw = (2/n_samples) * (X.T @ errors)  # Xᵀ @ errors
            db = (2/n_samples) * np.sum(errors)
            
            # 更新参数
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            # 记录损失
            loss = np.mean(errors ** 2)
            self.losses.append(loss)
        
        return self
    
    def predict(self, X):
        return X @ self.w + self.b
    
    def score(self, X, y):
        """计算 R² 分数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


# 训练
multi_model = MultiLinearRegression(lr=0.05, n_iter=1000)
multi_model.fit(X_multi, y_multi)

print(f"\n训练结果:")
print(f"  学到的权重: {multi_model.w.round(4)}")
print(f"  真实权重:   {true_weights}")
print(f"  权重误差:   {np.abs(multi_model.w - true_weights).round(4)}")
print(f"  学到的偏置: {multi_model.b:.4f}")
print(f"  真实偏置:   {true_bias}")
print(f"  偏置误差:   {abs(multi_model.b - true_bias):.4f}")
print(f"  最终损失:   {multi_model.losses[-1]:.4f}")
print(f"  R² 分数:    {multi_model.score(X_multi, y_multi):.4f}")

assert np.allclose(multi_model.w, true_weights, atol=0.2)
assert abs(multi_model.b - true_bias) < 0.5
print("✓ 多元线性回归验证通过")

# 特征重要性可视化
fig, ax = plt.subplots(figsize=(8, 4))
feature_names = ['x₁', 'x₂', 'x₃']
x_pos = np.arange(len(feature_names))

ax.bar(x_pos - 0.2, true_weights, 0.35, label='真实权重', alpha=0.7, color='steelblue')
ax.bar(x_pos + 0.2, multi_model.w, 0.35, label='学到的权重', alpha=0.7, color='coral')
ax.set_xticks(x_pos)
ax.set_xticklabels(feature_names)
ax.set_ylabel('权重值')
ax.set_title('特征权重对比：真实 vs 学到')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 零线
ax.axhline(y=0, color='black', linewidth=0.5)

save_plot(fig, 'feature_importance_multi.png')

print("""
【7.5 多元线性回归的关键要点】

1. 公式完全相同，只是从标量变成了矩阵
2. 每个特征有自己的权重 wᵢ，但共享一个偏置 b
3. 权重的正负表示特征与目标的关系方向：
   - w > 0：正相关（特征增大 → 预测增大）
   - w < 0：负相关（特征增大 → 预测减小）
   - w ≈ 0：几乎无影响
4. 权重的大小不能直接比较（因为特征的尺度可能不同）
   → 需要标准化后才能比较特征重要性
""")

# ============================================================
# 8. 特征缩放 —— 标准化 vs 归一化
# ============================================================
print("\n" + "=" * 70)
print("8. 特征缩放 —— 让梯度下降跑得更快")
print("=" * 70)

print("""
【8.1 为什么需要特征缩放？】

想象你要预测房价，特征包括：
  - 面积：50 ~ 200 m²（范围 150）
  - 房间数：1 ~ 5（范围 4）
  - 楼龄：0 ~ 50 年（范围 50）

面积的变化范围远大于房间数。这意味着：
  - 面积对应的梯度可能远大于房间数对应的梯度
  - 梯度下降的"碗"变得非常扁长（椭圆）
  - 在扁长的碗中，梯度下降会来回震荡，走很多弯路

特征缩放后，所有特征都在相似的尺度上：
  - 梯度下降的"碗"变得接近圆形
  - 可以更快地到达碗底

【8.2 两种常用方法】

1. 标准化（Standardization / Z-score Normalization）
   x' = (x - μ) / σ
   
   其中 μ 是均值，σ 是标准差
   
   结果：均值=0，标准差=1
   优点：对异常值相对稳健
   适用：大多数情况，特别是特征服从近似正态分布时

2. 归一化（Min-Max Normalization）
   x' = (x - x_min) / (x_max - x_min)
   
   结果：范围 [0, 1]
   优点：结果有明确范围
   缺点：对异常值敏感（一个极端值会压缩所有数据）
   适用：特征有明确的上下界时

【8.3 代码演示】
""")

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 创建不同尺度的特征
np.random.seed(42)
n_feat = 100
X_unscaled = np.column_stack([
    np.random.uniform(1000, 10000, n_feat),   # 房价：1000-10000
    np.random.uniform(0, 100, n_feat),        # 面积：0-100
    np.random.uniform(0, 1, n_feat),          # 评分：0-1
    np.random.uniform(1900, 2024, n_feat),    # 建造年份
])

print("缩放前:")
print(f"  特征0（房价）:  均值={X_unscaled[:, 0].mean():.0f}, 范围=[{X_unscaled[:, 0].min():.0f}, {X_unscaled[:, 0].max():.0f}]")
print(f"  特征1（面积）:  均值={X_unscaled[:, 1].mean():.1f}, 范围=[{X_unscaled[:, 1].min():.1f}, {X_unscaled[:, 1].max():.1f}]")
print(f"  特征2（评分）:  均值={X_unscaled[:, 2].mean():.2f}, 范围=[{X_unscaled[:, 2].min():.2f}, {X_unscaled[:, 2].max():.2f}]")
print(f"  特征3（年份）:  均值={X_unscaled[:, 3].mean():.0f}, 范围=[{X_unscaled[:, 3].min():.0f}, {X_unscaled[:, 3].max():.0f}]")

# 标准化
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_unscaled)

# 归一化
minmax = MinMaxScaler()
X_normalized = minmax.fit_transform(X_unscaled)

print("\n标准化后 (均值=0, 标准差=1):")
for i in range(4):
    print(f"  特征{i}: 均值={X_standardized[:, i].mean():.4f}, 标准差={X_standardized[:, i].std():.4f}")

print("\n归一化后 (范围=[0, 1]):")
for i in range(4):
    print(f"  特征{i}: 最小值={X_normalized[:, i].min():.4f}, 最大值={X_normalized[:, i].max():.4f}")

# 可视化特征缩放的效果
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 未缩放
for i in range(4):
    axes[0].hist(X_unscaled[:, i], alpha=0.5, label=f'特征{i}', bins=20)
axes[0].set_title('未缩放')
axes[0].legend()

# 标准化
for i in range(4):
    axes[1].hist(X_standardized[:, i], alpha=0.5, label=f'特征{i}', bins=20)
axes[1].set_title('标准化 (Z-score)')
axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axes[1].legend()

# 归一化
for i in range(4):
    axes[2].hist(X_normalized[:, i], alpha=0.5, label=f'特征{i}', bins=20)
axes[2].set_title('归一化 (Min-Max)')
axes[2].legend()

save_plot(fig, 'feature_scaling_comparison.png')

print("""
【8.4 重要提醒】

1. 训练集和测试集必须用相同的缩放参数！
   → 先在训练集上 fit，然后 transform 训练集和测试集
   → 不能分别 fit！否则数据会不一致

2. 正则化（Ridge/Lasso）对特征尺度非常敏感
   → 必须缩放！否则某些特征会被不公正地惩罚

3. 正规方程不需要特征缩放
   → 因为它是精确解，不受梯度影响

4. 梯度下降强烈建议特征缩放
   → 可以大幅提高收敛速度
""")

# ============================================================
# 9. 梯度下降的三种变体
# ============================================================
print("\n" + "=" * 70)
print("9. 梯度下降的三种变体")
print("=" * 70)

print("""
【9.1 批量梯度下降（Batch Gradient Descent, BGD）】

我们之前实现的就是 BGD：
  每次迭代使用全部训练数据计算梯度
  
  dw = (2/n) × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ) × xᵢ
                    ↑ 所有 n 个样本

优点：
  - 梯度稳定，收敛路径平滑
  - 保证收敛到全局最小（对凸函数）

缺点：
  - 数据量大时每次迭代都很慢
  - 需要所有数据都在内存中

【9.2 随机梯度下降（Stochastic Gradient Descent, SGD）】

每次只用一个样本来计算梯度：
  dw = 2 × (ŷᵢ - yᵢ) × xᵢ  （只用第 i 个样本）
  
  每次随机选一个样本，更新一次参数。

优点：
  - 每次迭代非常快
  - 可以在线学习（来一个新数据就更新一次）
  - 随机性有助于跳出局部最小值

缺点：
  - 梯度非常不稳定，损失曲线会震荡
  - 永远不会精确收敛（在最小值附近来回跳）
  - 需要逐渐减小学习率（学习率衰减）

【9.3 小批量梯度下降（Mini-batch Gradient Descent）】

折中方案：每次用一小批（如 32、64、128 个）样本。

  dw = (2/b) × Σᵢ∈batch (ŷᵢ - yᵢ) × xᵢ
                     ↑ 只算 batch_size 个样本

这是深度学习中最常用的方法（也是 scikit-learn 默认使用的方法）。

优点：
  - 兼顾速度和稳定性
  - 可以利用 GPU 的并行计算
  - batch_size 是可调节的超参数

【9.4 三种方法的对比】

┌────────────┬──────────┬────────┬──────────────┐
│ 方法       │ 样本数   │ 速度   │ 稳定性       │
├────────────┼──────────┼────────┼──────────────┤
│ BGD        │ 全部 n   │ 慢     │ 最稳定       │
│ SGD        │ 1        │ 最快   │ 最不稳定     │
│ Mini-batch │ b(32-256)│ 中等   │ 较稳定       │
└────────────┴──────────┴────────┴──────────────┘

【9.5 代码演示】
""")

# 实现三种梯度下降变体
def batch_gd(X, y, lr=0.001, n_iter=1000):
    """批量梯度下降"""
    n = len(X)
    w, b = 0.0, 0.0
    losses = []
    for _ in range(n_iter):
        y_pred = w * X + b
        dw = (2/n) * np.sum((y_pred - y) * X)
        db = (2/n) * np.sum(y_pred - y)
        w -= lr * dw
        b -= lr * db
        losses.append(np.mean((y_pred - y) ** 2))
    return w, b, losses

def sgd(X, y, lr=0.001, n_iter=1000):
    """随机梯度下降"""
    n = len(X)
    w, b = 0.0, 0.0
    losses = []
    np.random.seed(42)
    for _ in range(n_iter):
        # 随机选一个样本
        i = np.random.randint(0, n)
        y_pred = w * X[i] + b
        dw = 2 * (y_pred - y[i]) * X[i]
        db = 2 * (y_pred - y[i])
        w -= lr * dw
        b -= lr * db
        # 用全部数据计算损失（用于可视化）
        all_pred = w * X + b
        losses.append(np.mean((all_pred - y) ** 2))
    return w, b, losses

def mini_batch_gd(X, y, lr=0.001, n_iter=1000, batch_size=16):
    """小批量梯度下降"""
    n = len(X)
    w, b = 0.0, 0.0
    losses = []
    np.random.seed(42)
    for _ in range(n_iter):
        # 随机选一个 batch
        indices = np.random.choice(n, batch_size, replace=False)
        X_batch, y_batch = X[indices], y[indices]
        y_pred = w * X_batch + b
        dw = (2/batch_size) * np.sum((y_pred - y_batch) * X_batch)
        db = (2/batch_size) * np.sum(y_pred - y_batch)
        w -= lr * dw
        b -= lr * db
        all_pred = w * X + b
        losses.append(np.mean((all_pred - y) ** 2))
    return w, b, losses

# 用较小的数据集来演示
np.random.seed(42)
n_gd = 200
X_gd = np.random.uniform(0, 10, n_gd)
y_gd = 3 * X_gd + 2 + np.random.randn(n_gd) * 1.5

w_bgd, b_bgd, loss_bgd = batch_gd(X_gd, y_gd, lr=0.001, n_iter=500)
w_sgd, b_sgd, loss_sgd = sgd(X_gd, y_gd, lr=0.001, n_iter=500)
w_mb, b_mb, loss_mb = mini_batch_gd(X_gd, y_gd, lr=0.001, n_iter=500, batch_size=16)

print(f"批量梯度下降:  w={w_bgd:.4f}, b={b_bgd:.4f}, loss={loss_bgd[-1]:.4f}")
print(f"随机梯度下降:  w={w_sgd:.4f}, b={b_sgd:.4f}, loss={loss_sgd[-1]:.4f}")
print(f"小批量梯度下降: w={w_mb:.4f}, b={b_mb:.4f}, loss={loss_mb[-1]:.4f}")
print(f"真实值:       w=3.0000, b=2.0000")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
axes[0].plot(loss_bgd, 'b-', linewidth=1.5, label='BGD (批量)', alpha=0.8)
axes[0].plot(loss_sgd, 'r-', linewidth=1, label='SGD (随机)', alpha=0.7)
axes[0].plot(loss_mb, 'g-', linewidth=1.2, label='Mini-batch (小批量)', alpha=0.8)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE')
axes[0].set_title('三种梯度下降的损失曲线对比')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# 拟合直线
axes[1].scatter(X_gd, y_gd, alpha=0.3, s=15, label='数据点')
axes[1].plot([0, 10], [b_bgd, w_bgd*10+b_bgd], 'b-', linewidth=2, label=f'BGD: y={w_bgd:.2f}x+{b_bgd:.2f}')
axes[1].plot([0, 10], [b_sgd, w_sgd*10+b_sgd], 'r-', linewidth=2, label=f'SGD: y={w_sgd:.2f}x+{b_sgd:.2f}')
axes[1].plot([0, 10], [b_mb, w_mb*10+b_mb], 'g-', linewidth=2, label=f'Mini: y={w_mb:.2f}x+{b_mb:.2f}')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('三种方法的拟合直线')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

save_plot(fig, 'gd_variants_comparison.png')

print("""
观察结果：
  - BGD 的曲线最平滑，稳定下降
  - SGD 的曲线震荡最明显（每次都跳来跳去）
  - Mini-batch 介于两者之间

注意：SGD 的震荡不是问题！在深度学习中，这种随机性有时能帮助跳出局部最小值。
""")

# ============================================================
# 10. 过拟合、欠拟合与正则化
# ============================================================
print("\n" + "=" * 70)
print("10. 过拟合、欠拟合与正则化深度剖析")
print("=" * 70)

print("""
【10.1 什么是欠拟合 (Underfitting)？】

模型太简单，连训练数据都拟合不好。

比喻：
  用一条水平线去拟合一个明显的线性趋势
  → 模型"学不会"，训练误差大，测试误差也大

原因：
  - 模型复杂度太低（线性模型拟合非线性数据）
  - 特征太少
  - 训练不足

【10.2 什么是过拟合 (Overfitting)？】

模型太复杂，把训练数据中的噪声也"学"进去了。

比喻：
  学生背答案而不是理解原理。考试题目变了就不会了。
  
  或者：裁缝给一个人量身体裁衣，做得完全合身。
  但另一个人穿就不合身了。

表现：
  - 训练误差很小（甚至为 0）
  - 测试误差很大
  - 模型在训练集和测试集上的表现差距很大

【10.3 偏差-方差的直觉】

训练误差 = 偏差² + 方差 + 不可约误差

  偏差（Bias）：模型的系统性误差（模型太简单导致的误差）
  方差（Variance）：模型对训练数据的敏感度（模型太灵活导致的波动）

欠拟合 → 高偏差（模型太死板，拟合不了数据）
过拟合 → 高方差（模型太灵活，对每个噪声都反应过度）

【10.4 可视化：多项式拟合的例子】
""")

# 创建多项式拟合的数据
np.random.seed(42)
n_poly = 30
X_poly = np.sort(np.random.uniform(0, 1, n_poly))
y_poly = np.sin(2 * np.pi * X_poly) + np.random.randn(n_poly) * 0.2

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

degrees = [1, 3, 15]
titles = ['欠拟合 (1次多项式)\n高偏差，低方差', '适中 (3次多项式)\n偏差和方差均衡', '过拟合 (15次多项式)\n低偏差，高方差']

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

X_test_poly = np.linspace(0, 1, 100)

for idx, (degree, title) in enumerate(zip(degrees, titles)):
    ax = axes[idx]
    
    # 创建多项式特征
    poly = PolynomialFeatures(degree=degree)
    X_poly_feat = poly.fit_transform(X_poly.reshape(-1, 1))
    
    # 训练线性回归
    lr_model = LinearRegression()
    lr_model.fit(X_poly_feat, y_poly)
    
    # 预测
    X_test_feat = poly.transform(X_test_poly.reshape(-1, 1))
    y_pred_test = lr_model.predict(X_test_feat)
    
    # 计算误差
    train_pred = lr_model.predict(X_poly_feat)
    train_mse = np.mean((y_poly - train_pred) ** 2)
    
    ax.scatter(X_poly, y_poly, c='black', s=40, alpha=0.7, zorder=5, label='训练数据')
    ax.plot(X_test_poly, np.sin(2 * np.pi * X_test_poly), 'g--', linewidth=2, label='真实函数 sin(2πx)')
    ax.plot(X_test_poly, y_pred_test, 'r-', linewidth=2, label=f'拟合曲线 (degree={degree})')
    ax.set_title(title + f'\n训练 MSE = {train_mse:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, 1.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

save_plot(fig, 'underfit_vs_overfit.png')

print("""
从左到右看：

左图（欠拟合）：
  一条直线去拟合 sin 曲线 → 太简单了
  训练 MSE 很大，模型根本没学会数据的形状

中图（适中）：
  3 次多项式 → 刚刚好
  拟合了数据的总体趋势，没有被噪声带偏

右图（过拟合）：
  15 次多项式 → 太复杂了
  曲线经过每一个点（训练 MSE 接近 0），但弯弯曲曲
  → 对新数据（没见过的 x）会做出非常错误的预测

【10.5 怎么解决过拟合？】

方法 1：减少特征数量
  - 去掉不重要的特征
  - 特征选择

方法 2：正则化（Regularization） ★
  - 在损失函数中加入惩罚项，限制权重不能太大
  - L2 正则化（Ridge）
  - L1 正则化（Lasso）
  - Elastic Net（L1 + L2）

方法 3：增加训练数据
  - 更多的数据让模型难以"记住"噪声

方法 4：交叉验证
  - 用验证集来调参，避免只看训练误差

【10.6 正则化的数学原理】

L2 正则化（Ridge Regression）：
  Loss = MSE + λ × Σwᵢ²
  
  λ > 0 是正则化强度
  
  效果：
    - 权重趋向于小但不为零
    - 所有特征都参与，但影响被削弱
    - 也叫"权重衰减"（Weight Decay）

L1 正则化（Lasso Regression）：
  Loss = MSE + λ × Σ|wᵢ|
  
  效果：
    - 某些权重直接变为零
    - 自动特征选择！
    - 产生稀疏解

为什么 L1 会产生稀疏解？
  想象你在山上，L1 的"惩罚地形"像一个金字塔（尖的）。
  当你沿着山坡往下走时，更容易走到尖角（某些维度=0）。
  
  而 L2 的"惩罚地形"像一个碗（圆的）。
  你走到碗底时，所有维度都是小的正数，不太可能正好是 0。

【10.7 代码：正则化效果演示】
""")

from sklearn.linear_model import Ridge, Lasso, ElasticNet

# 划分训练集和测试集
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly.reshape(-1, 1), y_poly, test_size=0.3, random_state=42)

# 15 次多项式
poly_15 = PolynomialFeatures(degree=15)
X_train_15 = poly_15.fit_transform(X_train_p)
X_test_15 = poly_15.transform(X_test_p)

print("在 15 次多项式（极易过拟合）上比较不同正则化：\n")

# 无正则化
model_no_reg = LinearRegression()
model_no_reg.fit(X_train_15, y_train_p)
train_r2 = model_no_reg.score(X_train_15, y_train_p)
test_r2 = model_no_reg.score(X_test_15, y_test_p)
print(f"无正则化:")
print(f"  训练 R²: {train_r2:.4f}  ← 接近 1（完美拟合训练数据）")
print(f"  测试 R²: {test_r2:.4f}  ← 很差（过拟合了！）")
print(f"  训练-测试差距: {train_r2 - test_r2:.4f}\n")

# L2 正则化 (Ridge)
for alpha in [0.001, 0.01, 0.1, 1.0]:
    model_ridge = Ridge(alpha=alpha)
    model_ridge.fit(X_train_15, y_train_p)
    train_r2_r = model_ridge.score(X_train_15, y_train_p)
    test_r2_r = model_ridge.score(X_test_15, y_test_p)
    n_large = np.sum(np.abs(model_ridge.coef_) > 0.01)
    print(f"Ridge (α={alpha}): 训练 R²={train_r2_r:.4f}, 测试 R²={test_r2_r:.4f}, 非零权重={n_large}/15")

print()

# L1 正则化 (Lasso)
for alpha in [0.001, 0.01, 0.1]:
    model_lasso = Lasso(alpha=alpha, max_iter=10000)
    model_lasso.fit(X_train_15, y_train_p)
    train_r2_l = model_lasso.score(X_train_15, y_train_p)
    test_r2_l = model_lasso.score(X_test_15, y_test_p)
    n_zero = np.sum(np.abs(model_lasso.coef_) < 0.001)
    print(f"Lasso (α={alpha}): 训练 R²={train_r2_l:.4f}, 测试 R²={test_r2_l:.4f}, 零权重={n_zero}/15 (自动特征选择!)")

# 可视化正则化效果
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

models = [
    ('无正则化', LinearRegression(), 'blue'),
    ('Ridge α=0.01', Ridge(alpha=0.01), 'green'),
    ('Lasso α=0.01', Lasso(alpha=0.01, max_iter=10000), 'orange'),
]

for idx, (name, mdl, color) in enumerate(models):
    ax = axes[idx]
    mdl.fit(X_train_15, y_train_p)
    
    y_pred_test = mdl.predict(X_test_15)
    train_r2_m = mdl.score(X_train_15, y_train_p)
    test_r2_m = mdl.score(X_test_15, y_test_p)
    
    ax.scatter(X_train_p, y_train_p, c='black', s=30, alpha=0.7, label='训练数据')
    ax.scatter(X_test_p, y_test_p, c='gray', s=30, alpha=0.5, marker='x', label='测试数据')
    ax.plot(X_test_poly, np.sin(2 * np.pi * X_test_poly), 'g--', linewidth=1.5, label='真实函数')
    ax.plot(X_test_poly, mdl.predict(poly_15.transform(X_test_poly.reshape(-1, 1))), 
            color=color, linewidth=2, label=f'拟合曲线')
    ax.set_title(f'{name}\n训练 R²={train_r2_m:.3f}, 测试 R²={test_r2_m:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, 1.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

save_plot(fig, 'regularization_comparison.png')

print("""
【10.8 正则化的关键要点】

1. α（正则化强度）是关键超参数：
   - α = 0：完全不过则化（就是普通线性回归）
   - α 很小：轻微正则化，可能还是过拟合
   - α 适中：最佳平衡点
   - α 太大：欠拟合（权重被压得太狠）

2. 如何选择 α？
   - 交叉验证（Cross-Validation）
   - 试不同的 α，选验证集上最好的

3. Ridge vs Lasso 的选择：
   - Ridge：所有特征都有用，只是削弱它们的权重
   - Lasso：你想做特征选择（哪些特征可以完全不要？）
   - Elastic Net：结合两者优点（当特征之间有相关性时）
""")

# ============================================================
# 11. 偏差-方差权衡（Bias-Variance Tradeoff）
# ============================================================
print("\n" + "=" * 70)
print("11. 偏差-方差权衡（Bias-Variance Tradeoff）")
print("=" * 70)

print("""
【11.1 误差的分解】

任何模型的期望泛化误差可以分解为三项：

  E[(y - ŷ)²] = Bias² + Variance + σ²

其中：
  Bias²（偏差的平方）：
    E[ŷ] - y_true 的平方
    → 模型预测的平均值与真实值的差距
    → 高偏差 = 模型太简单，学不会

  Variance（方差）：
    E[(ŷ - E[ŷ])²]
    → 模型预测的波动程度
    → 高方差 = 模型对训练数据太敏感

  σ²（不可约误差）：
    数据本身的噪声，无法消除

【11.2 直觉理解：打靶的例子】

想象你练习打靶：

  高偏差 + 低方差 = 所有子弹都打在同一个地方，但偏离靶心
  → 模型太简单，系统性偏差
  
  低偏差 + 高方差 = 子弹散布很广，但平均在靶心附近
  → 模型太灵活，每次训练结果差异很大
  
  低偏差 + 低方差 = 所有子弹都密集地打在靶心
  → 理想状态！

【11.3 偏差-方差曲线】

随着模型复杂度增加：
  - 偏差逐渐降低（模型越来越能拟合数据）
  - 方差逐渐增加（模型越来越敏感）
  - 总误差先降后升，最低点 = 最佳模型复杂度

     误差
      │
      │         ╱
      │       ╱  ← 测试误差
      │     ╱    （先降后升）
      │   ╱  ╲
      │ ╱      ╲
      │╱        ╲
      │  ╲      ╱
      │    ╲  ╱   ← 训练误差
      │      ╲     （持续下降）
      └────────────── 模型复杂度
                ↑
            最佳复杂度
""")

# 绘制偏差-方差曲线
fig, ax = plt.subplots(figsize=(8, 5))

complexity = np.linspace(0.1, 2.0, 100)
bias_squared = 4 * np.exp(-2 * complexity) + 0.1
variance = 0.05 * np.exp(2.5 * complexity)
irreducible = 0.5 * np.ones_like(complexity)
total_error = bias_squared + variance + irreducible

ax.plot(complexity, bias_squared, 'b-', linewidth=2, label='偏差² (Bias²)', alpha=0.7)
ax.plot(complexity, variance, 'r-', linewidth=2, label='方差 (Variance)', alpha=0.7)
ax.plot(complexity, total_error, 'k-', linewidth=3, label='总误差')
ax.axhline(y=irreducible[0], color='g', linestyle='--', alpha=0.5, label='不可约误差 σ²')

# 标记最佳点
optimal_idx = np.argmin(total_error)
ax.axvline(x=complexity[optimal_idx], color='purple', linestyle=':', alpha=0.7)
ax.plot(complexity[optimal_idx], total_error[optimal_idx], 'o', color='purple', markersize=10, zorder=5)
ax.text(complexity[optimal_idx] + 0.05, total_error[optimal_idx] + 0.1, '最佳模型复杂度', fontsize=10, color='purple')

# 标注欠拟合和过拟合区域
ax.text(0.3, 3.5, '← 欠拟合\n(高偏差)', fontsize=10, color='blue')
ax.text(1.5, 3.5, '过拟合 →\n(高方差)', fontsize=10, color='red')

ax.set_xlabel('模型复杂度')
ax.set_ylabel('误差')
ax.set_title('偏差-方差权衡 (Bias-Variance Tradeoff)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 2.1)
ax.set_ylim(0, 4.5)

save_plot(fig, 'bias_variance_tradeoff.png')

print("""
【11.4 实际应用中的意义】

偏差-方差权衡是机器学习中最重要的概念之一：

1. 选择模型时：
   - 数据少 → 用简单模型（避免高方差）
   - 数据多 → 可以用复杂模型（有足够数据约束方差）

2. 调参时：
   - 训练误差高 → 欠拟合 → 增加模型复杂度 / 减少正则化
   - 训练误差低但测试误差高 → 过拟合 → 减少模型复杂度 / 增加正则化 / 增加数据

3. 集成方法（如随机森林）：
   - 通过"平均"多个模型来降低方差

记住：你不可能同时消除偏差和方差，只能在两者之间找到一个平衡点。
""")

# ============================================================
# 12. 完整项目实战
# ============================================================
print("\n" + "=" * 70)
print("12. 完整项目实战 —— 波士顿房价预测（简化版）")
print("=" * 70)

print("""
现在我们用一个真实场景来串起所有知识。

场景：用多个特征预测房价

步骤：
  1. 生成模拟数据（包含多种特征）
  2. 探索性数据分析（EDA）
  3. 数据预处理（特征缩放）
  4. 划分训练集/测试集
  5. 训练模型
  6. 评估模型
  7. 正则化调参
  8. 残差诊断
""")

# 步骤 1: 生成模拟数据
np.random.seed(42)
n_houses = 500

# 特征
data = {
    '面积(m²)': np.random.uniform(30, 200, n_houses),
    '房间数': np.random.randint(1, 8, n_houses).astype(float),
    '楼龄(年)': np.random.uniform(0, 50, n_houses),
    '距地铁(km)': np.random.uniform(0.1, 15, n_houses),
    '绿化率(%)': np.random.uniform(10, 60, n_houses),
}

X_house = np.column_stack([v for v in data.values()])

# 真实关系（加上噪声）
# 房价 = 2万/m² × 面积 + 1万/房间 × 房间数 - 0.1万/年 × 楼龄
#        - 0.5万/km × 距地铁 + 0.05万/% × 绿化率 + 30万
true_prices = (
    2.0 * data['面积(m²)'] + 
    1.0 * data['房间数'] - 
    0.1 * data['楼龄(年)'] - 
    0.5 * data['距地铁(km)'] + 
    0.05 * data['绿化率(%)'] + 
    30.0
)
noise = np.random.randn(n_houses) * 5  # 5万元的噪声
y_house = true_prices + noise

print(f"数据集: {n_houses} 套房子, 5 个特征")
print(f"房价范围: {y_house.min():.1f} ~ {y_house.max():.1f} 万元")
print(f"房价均值: {y_house.mean():.1f} 万元")
print(f"\n特征列表:")
for i, name in enumerate(data.keys()):
    print(f"  特征{i}: {name}  [均值={X_house[:, i].mean():.1f}, 范围={X_house[:, i].min():.1f}~{X_house[:, i].max():.1f}]")

# 步骤 2: EDA
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
feature_names = list(data.keys())

for i, (name, ax) in enumerate(zip(feature_names, axes.flat)):
    ax.scatter(X_house[:, i], y_house, alpha=0.3, s=10)
    # 计算相关系数
    corr = np.corrcoef(X_house[:, i], y_house)[0, 1]
    ax.set_xlabel(name)
    ax.set_ylabel('房价(万元)')
    ax.set_title(f'{name}\n相关系数 r = {corr:.3f}')
    ax.grid(True, alpha=0.3)

# 隐藏多余的子图
if len(feature_names) < 6:
    axes.flat[5].set_visible(False)

save_plot(fig, 'eda_scatter_plots.png')

print("\n\n相关性解读：")
for i, name in enumerate(feature_names):
    corr = np.corrcoef(X_house[:, i], y_house)[0, 1]
    direction = "正相关" if corr > 0 else "负相关"
    strength = "强" if abs(corr) > 0.5 else ("中等" if abs(corr) > 0.3 else "弱")
    print(f"  {name}: r={corr:.3f} ({strength}{direction})")

# 步骤 3: 数据预处理
print("\n--- 数据预处理 ---")

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42)

print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")

# 特征缩放
scaler_house = StandardScaler()
X_train_scaled = scaler_house.fit_transform(X_train)
X_test_scaled = scaler_house.transform(X_test)

print("特征缩放完成 ✓")

# 步骤 4-5: 训练模型
print("\n--- 训练线性回归模型 ---")

# 普通线性回归
lr_house = LinearRegression()
lr_house.fit(X_train_scaled, y_train)

print(f"\n普通线性回归结果:")
print(f"  训练 R²: {lr_house.score(X_train_scaled, y_train):.4f}")
print(f"  测试 R²: {lr_house.score(X_test_scaled, y_test):.4f}")

# 特征权重
print(f"\n各特征权重:")
for name, weight in zip(feature_names, lr_house.coef_):
    print(f"  {name:>12s}: {weight:+.4f}")
print(f"  偏置 b      : {lr_house.intercept_:+.4f}")

# 步骤 6: 评估
y_pred_house = lr_house.predict(X_test_scaled)
mse_house = mean_squared_error(y_test, y_pred_house)
rmse_house = np.sqrt(mse_house)
mae_house = mean_absolute_error(y_test, y_pred_house)
r2_house = r2_score(y_test, y_pred_house)

print(f"\n评估指标:")
print(f"  MSE  : {mse_house:.4f}")
print(f"  RMSE : {rmse_house:.4f} 万元 (平均偏离约 {rmse_house:.1f} 万元)")
print(f"  MAE  : {mae_house:.4f} 万元")
print(f"  R²   : {r2_house:.4f} (解释了 {r2_house*100:.1f}% 的变化)")

# 步骤 7: 正则化对比
print("\n--- 正则化对比 ---")

alphas = [0.01, 0.1, 1.0, 10.0]
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    train_r2 = ridge.score(X_train_scaled, y_train)
    test_r2 = ridge.score(X_test_scaled, y_test)
    print(f"Ridge α={alpha:5.2f}: 训练 R²={train_r2:.4f}, 测试 R²={test_r2:.4f}")

# 步骤 8: 残差诊断
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 残差图
residuals_house = y_test - y_pred_house
axes[0].scatter(y_pred_house, residuals_house, alpha=0.5, s=20)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('预测值(万元)')
axes[0].set_ylabel('残差(万元)')
axes[0].set_title('残差 vs 预测值')
axes[0].grid(True, alpha=0.3)

# 残差分布
axes[1].hist(residuals_house, bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--')
axes[1].set_xlabel('残差(万元)')
axes[1].set_ylabel('频数')
axes[1].set_title('残差分布')
axes[1].grid(True, alpha=0.3)

# Q-Q 图（正态性检验）
from scipy import stats
stats.probplot(residuals_house, dist="norm", plot=axes[2])
axes[2].set_title('Q-Q 图（检验残差是否正态分布）')
axes[2].grid(True, alpha=0.3)

save_plot(fig, 'house_price_residuals.png')

print("""
【项目总结】

1. 这是一个标准的机器学习项目流程
2. 线性回归在这个任务上表现不错（R² ≈ 0.95+）
3. 残差大致正态分布，随机分布在 0 附近 → 模型拟合良好
4. 正则化对 R² 影响不大（线性回归已经拟合得很好了）
5. 如果 R² 较低，可以尝试：
   - 增加更多特征
   - 尝试非线性模型（决策树、随机森林）
   - 特征工程（创建组合特征）
""")

# ============================================================
# 13. 总结
# ============================================================
print("\n" + "=" * 70)
print("13. 本章总结与知识图谱")
print("=" * 70)

print("""
═══════════════════════════════════════════════════════════════
                    线性回归知识图谱
═══════════════════════════════════════════════════════════════

一、核心概念
├── 什么是线性回归？
│   ├── 预测连续数值的监督学习算法
│   ├── 模型：ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
│   └── "线性"指模型关于参数是线性的
│
├── 损失函数（衡量模型有多差）
│   ├── MSE = (1/n) × Σ(yᵢ - ŷᵢ)²  ← 最常用
│   ├── MAE = (1/n) × Σ|yᵢ - ŷᵢ|  ← 对异常值稳健
│   └── 选择依据：可微性、凸性、统计意义
│
├── 梯度下降（优化算法）
│   ├── 原理：沿着梯度的反方向更新参数
│   ├── ∂L/∂w = (2/n) × Σ(ŷᵢ - yᵢ) × xᵢ
│   ├── ∂L/∂b = (2/n) × Σ(ŷᵢ - yᵢ)
│   ├── 学习率：太大发散，太小太慢
│   └── 三种变体：BGD / SGD / Mini-batch
│
└── 正规方程（解析解）
    ├── θ = (XᵀX)⁻¹ Xᵀy
    ├── 一步精确解
    └── 适合小数据，大数据用梯度下降

二、评估指标
├── MSE / RMSE：预测值偏离真实值的程度
├── MAE：平均绝对偏离
└── R²：模型解释了多少数据变化（-∞ 到 1）

三、模型问题诊断
├── 欠拟合（Underfitting）
│   ├── 表现：训练误差大
│   └── 解决：增加模型复杂度 / 增加特征 / 减少正则化
│
├── 过拟合（Overfitting）
│   ├── 表现：训练误差小，测试误差大
│   └── 解决：正则化 / 减少特征 / 增加数据
│
└── 偏差-方差权衡
    ├── 偏差高 → 欠拟合
    ├── 方差高 → 过拟合
    └── 目标：找到平衡点

四、实用技巧
├── 特征缩放：标准化 / 归一化
├── 正则化：Ridge (L2) / Lasso (L1) / ElasticNet
├── 数据划分：训练集 / 验证集 / 测试集
└── 残差分析：检查模型假设是否成立

═══════════════════════════════════════════════════════════════
""")

print("""
【思考题】

1. 如果数据中有异常值（outliers），用 MSE 还是 MAE 更好？为什么？

2. 正规方程中 XᵀX 不可逆怎么办？有哪些解决方案？

3. 为什么 Lasso (L1) 会产生稀疏解而 Ridge (L2) 不会？

4. 如果两个特征高度相关（共线性），会对模型产生什么影响？

5. 学习率从 0.1 改为 0.001，训练过程会有什么变化？

6. 如果 R² = -0.5，说明什么问题？

7. 为什么梯度下降需要特征缩放而正规方程不需要？

═══════════════════════════════════════════════════════════════

  下一节：3.2 逻辑回归 → 从预测数值到预测类别
  
  预告：
  - 为什么线性回归不能做分类？
  - Sigmoid 函数是什么？
  - 交叉熵损失函数
  - 从线性回归到逻辑回归的自然过渡
  - 多分类：Softmax 回归

═══════════════════════════════════════════════════════════════
""")

print(f"\n本节共生成 {len([f for f in os.listdir(PLOT_DIR) if f.startswith(('original_data', 'loss_function', 'loss_landscape', 'learning_rate', 'loss_curve', 'training_process', 'normal_equation', 'evaluation_metrics', 'feature_importance', 'feature_scaling', 'gd_variants', 'underfit_vs_overfit', 'regularization_comparison', 'bias_variance', 'eda_scatter', 'house_price_residuals'))])} 张图表")
print("所有图表已保存到: " + PLOT_DIR)
print("\n✓ 第三章 3.1 线性回归（深度版）学习完毕！")
