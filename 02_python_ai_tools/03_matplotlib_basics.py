"""
第二章 2.3：Matplotlib 可视化基础
================================

数据可视化在 AI 开发中的用途：
- 探索数据分布和特征关系
- 监控训练过程（损失曲线、准确率曲线）
- 展示模型结果（混淆矩阵、ROC曲线等）
- 调试模型（可视化权重、中间结果）

本节内容：
1. 基本绑图
2. 常用图表类型
3. AI 中常用的可视化
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI环境使用
import matplotlib.pyplot as plt

print("=" * 60)
print("第二章 2.3：Matplotlib 可视化基础")
print("=" * 60)

# ============================================================
# 1. 基本绑图
# ============================================================
print("\n" + "=" * 60)
print("1. 基本绑图")
print("=" * 60)

# 简单折线图
x = np.linspace(0, 2*np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.figure(figsize=(10, 4))
plt.plot(x, y_sin, label='sin(x)', color='blue')
plt.plot(x, y_cos, label='cos(x)', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trigonometric Functions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/home/gfhong/testcode/ai_tutorial/02_python_ai_tools/plot_basic.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ 基本折线图已保存: plot_basic.png")

# ============================================================
# 2. AI 中常用的可视化
# ============================================================
print("\n" + "=" * 60)
print("2. AI 中常用的可视化")
print("=" * 60)

# 2.1 训练损失曲线
print("\n--- 2.1 模拟训练损失曲线 ---")
np.random.seed(42)
epochs = 50
train_loss = 2.0 * np.exp(-0.08 * np.arange(epochs)) + 0.1 + np.random.randn(epochs) * 0.05
val_loss = 2.0 * np.exp(-0.06 * np.arange(epochs)) + 0.2 + np.random.randn(epochs) * 0.08
# 模拟过拟合：验证损失后期上升
val_loss[35:] += np.linspace(0, 0.3, 15)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss', color='blue')
plt.plot(val_loss, label='Val Loss', color='red')
plt.axvline(x=35, color='green', linestyle=':', label='Best Model (epoch 35)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 2.2 准确率曲线
train_acc = 1 - 0.5 * np.exp(-0.1 * np.arange(epochs)) + np.random.randn(epochs) * 0.02
val_acc = 1 - 0.6 * np.exp(-0.08 * np.arange(epochs)) + np.random.randn(epochs) * 0.03
val_acc[35:] -= np.linspace(0, 0.1, 15)

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Train Accuracy', color='blue')
plt.plot(val_acc, label='Val Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/gfhong/testcode/ai_tutorial/02_python_ai_tools/plot_training.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ 训练曲线已保存: plot_training.png")
print("  注意: epoch 35 后验证损失上升 = 过拟合的信号!")

# 2.3 数据分布可视化
print("\n--- 2.2 数据分布 ---")
np.random.seed(42)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 直方图
data = np.random.normal(170, 10, 1000)
axes[0, 0].hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Height Distribution (Normal)')
axes[0, 0].set_xlabel('Height (cm)')

# 散点图
x_data = np.random.randn(100)
y_data = 2 * x_data + 1 + np.random.randn(100) * 0.5
axes[0, 1].scatter(x_data, y_data, alpha=0.6, c='coral')
axes[0, 1].set_title('Scatter: Linear Relationship')
axes[0, 1].set_xlabel('Feature X')
axes[0, 1].set_ylabel('Target Y')

# 分类数据散点图
from sklearn.datasets import make_blobs
X_blobs, y_blobs = make_blobs(n_samples=200, centers=3, random_state=42)
scatter = axes[1, 0].scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, cmap='viridis', alpha=0.6)
axes[1, 0].set_title('Classification Data (3 classes)')
plt.colorbar(scatter, ax=axes[1, 0])

# 混淆矩阵
confusion = np.array([[45, 5, 2],
                       [3, 40, 7],
                       [1, 4, 43]])
im = axes[1, 1].imshow(confusion, cmap='Blues')
axes[1, 1].set_title('Confusion Matrix')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')
for i in range(3):
    for j in range(3):
        axes[1, 1].text(j, i, str(confusion[i, j]), ha='center', va='center')
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('/home/gfhong/testcode/ai_tutorial/02_python_ai_tools/plot_distributions.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ 数据分布图已保存: plot_distributions.png")

# 2.4 激活函数可视化
print("\n--- 2.3 激活函数可视化 ---")
x = np.linspace(-5, 5, 200)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def relu(x): return np.maximum(0, x)
def leaky_relu(x): return np.where(x > 0, x, 0.01 * x)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
funcs = [(sigmoid, 'Sigmoid', 'blue'), (tanh, 'Tanh', 'red'),
         (relu, 'ReLU', 'green'), (leaky_relu, 'Leaky ReLU', 'purple')]

for ax, (func, name, color) in zip(axes.flat, funcs):
    y = func(x)
    ax.plot(x, y, color=color, linewidth=2)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_title(f'{name} Activation')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)

plt.tight_layout()
plt.savefig('/home/gfhong/testcode/ai_tutorial/02_python_ai_tools/plot_activations.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ 激活函数图已保存: plot_activations.png")

# 2.5 梯度下降可视化
print("\n--- 2.4 梯度下降可视化 ---")
def f(x, y): return x**2 + 3*y**2

x_range = np.linspace(-4, 4, 100)
y_range = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)

# 模拟梯度下降路径
path_x, path_y = [3.5], [3.5]
lr = 0.1
for _ in range(20):
    gx = 2 * path_x[-1]
    gy = 6 * path_y[-1]
    path_x.append(path_x[-1] - lr * gx)
    path_y.append(path_y[-1] - lr * gy)

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=20, cmap='coolwarm', alpha=0.8)
plt.plot(path_x, path_y, 'k.-', markersize=8, linewidth=1.5, label='GD Path')
plt.plot(path_x[0], path_y[0], 'go', markersize=12, label='Start')
plt.plot(path_x[-1], path_y[-1], 'r*', markersize=15, label='End')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent on f(x,y) = x² + 3y²')
plt.legend()
plt.colorbar(label='f(x,y)')
plt.savefig('/home/gfhong/testcode/ai_tutorial/02_python_ai_tools/plot_gradient_descent.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ 梯度下降可视化已保存: plot_gradient_descent.png")

print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点：
1. plt.plot() - 折线图（训练曲线）
2. plt.scatter() - 散点图（数据分布、聚类）
3. plt.hist() - 直方图（特征分布）
4. plt.imshow() - 热力图（混淆矩阵、注意力权重）
5. plt.contour() - 等高线（损失曲面、梯度下降）

AI 开发中的可视化习惯：
  - 训练时一定要画损失曲线（监控训练/过拟合）
  - 特征工程前先看数据分布
  - 模型评估时画混淆矩阵、ROC曲线
  - 调试时可视化中间结果

所有图片已保存到当前目录，可以查看。

下一章：机器学习基础 → 从线性回归开始
""")
