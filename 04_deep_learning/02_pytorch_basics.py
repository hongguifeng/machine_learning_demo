"""
第四章 4.2：PyTorch 入门
========================

PyTorch 是目前最流行的深度学习框架（尤其在研究领域）。
它提供：
  1. 自动求导（不用手写反向传播！）
  2. GPU 加速
  3. 丰富的预定义层和优化器

本节内容：
1. 张量 (Tensor) 基础
2. 自动求导 (Autograd)
3. 用 PyTorch 重写神经网络
4. nn.Module 和优化器
5. 完整训练流程
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("=" * 60)
print("第四章 4.2：PyTorch 入门")
print("=" * 60)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

# ============================================================
# 1. 张量基础
# ============================================================
print("\n" + "=" * 60)
print("1. 张量 (Tensor) 基础")
print("=" * 60)

print("""
【Tensor vs NumPy Array】
PyTorch 的 Tensor 和 NumPy 的 Array 很像，但有两个关键区别：
  1. Tensor 可以在 GPU 上运算（快几十倍）
  2. Tensor 支持自动求导（自动计算梯度）
""")

# 创建张量
print("--- 创建张量 ---")
# 从列表
t1 = torch.tensor([1.0, 2.0, 3.0])
print(f"从列表: {t1}, dtype={t1.dtype}")

# 从 NumPy
np_arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
t2 = torch.from_numpy(np_arr)
print(f"从 NumPy:\n{t2}")

# 常用创建
print(f"\nzeros: {torch.zeros(2, 3)}")
print(f"ones: {torch.ones(2, 3)}")
print(f"rand (均匀): {torch.rand(2, 3)}")
print(f"randn (正态): {torch.randn(2, 3)}")

# 形状操作
t3 = torch.arange(12).reshape(3, 4)
print(f"\narange + reshape:\n{t3}")
print(f"shape: {t3.shape}")
print(f"转置: {t3.T.shape}")

# 运算
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(f"\n--- 张量运算 ---")
print(f"a + b = {a + b}")
print(f"a * b = {a * b} (逐元素)")
print(f"a @ b = {a @ b} (点积)")

# 矩阵乘法
A = torch.randn(3, 4)
B = torch.randn(4, 2)
C = A @ B  # 或 torch.matmul(A, B)
print(f"\n矩阵乘法: ({list(A.shape)}) @ ({list(B.shape)}) → {list(C.shape)}")


# ============================================================
# 2. 自动求导 (Autograd)
# ============================================================
print("\n" + "=" * 60)
print("2. 自动求导 (Autograd) — PyTorch 的核心魔法")
print("=" * 60)

print("""
【自动求导】
设置 requires_grad=True 的张量，PyTorch 会记录所有操作，
然后自动计算梯度。这就是为什么不用手写反向传播！

原理：
  PyTorch 内部构建一个"计算图"，记录所有操作。
  调用 .backward() 时，自动沿计算图反向传播。
""")

# 示例 1: 简单求导
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2  # y = x²
y.backward()  # 计算 dy/dx
print(f"y = x², x=3")
print(f"dy/dx = 2x = 2×3 = 6")
print(f"PyTorch 计算: x.grad = {x.grad}")
assert x.grad == 6.0
print("✓ 验证通过")

# 示例 2: 复合函数
x = torch.tensor(2.0, requires_grad=True)
y = (2*x + 1) ** 3  # y = (2x+1)³
y.backward()
# dy/dx = 3(2x+1)² × 2 = 6(2x+1)² = 6×25 = 150
print(f"\ny = (2x+1)³, x=2")
print(f"dy/dx = 6(2x+1)² = 6×5² = 150")
print(f"PyTorch 计算: {x.grad}")
assert x.grad == 150.0
print("✓ 验证通过")

# 示例 3: 多变量
x = torch.tensor(1.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)

# 模拟一个前向传播
z = w * x + b
a = torch.sigmoid(z)
loss = (a - 1.0) ** 2  # MSE loss with target=1

loss.backward()

print(f"\nz = w*x + b = {z.item():.4f}")
print(f"a = sigmoid(z) = {a.item():.4f}")
print(f"loss = (a-1)² = {loss.item():.6f}")
print(f"\n自动计算的梯度:")
print(f"  ∂loss/∂w = {w.grad:.6f}")
print(f"  ∂loss/∂b = {b.grad:.6f}")
print(f"  ∂loss/∂x = {x.grad:.6f}")

# 数值验证
eps = 1e-5
w_val = 2.0
z_plus = (w_val + eps) * 1.0 + 0.5
a_plus = 1 / (1 + np.exp(-z_plus))
loss_plus = (a_plus - 1.0) ** 2

z_minus = (w_val - eps) * 1.0 + 0.5
a_minus = 1 / (1 + np.exp(-z_minus))
loss_minus = (a_minus - 1.0) ** 2

numerical_grad_w = (loss_plus - loss_minus) / (2 * eps)
print(f"\n数值验证 ∂loss/∂w: {numerical_grad_w:.6f}")
assert abs(w.grad.item() - numerical_grad_w) < 1e-4
print("✓ 自动求导结果正确")


# ============================================================
# 3. nn.Module —— 构建神经网络
# ============================================================
print("\n" + "=" * 60)
print("3. nn.Module — 构建神经网络")
print("=" * 60)

print("""
【nn.Module 是所有网络层的基类】

定义网络的标准方式：
  1. 继承 nn.Module
  2. 在 __init__ 中定义各层
  3. 在 forward 中定义前向传播
  
反向传播自动完成！(thanks to autograd)
""")

class SimpleNet(nn.Module):
    """简单的两层神经网络"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)  # 全连接层
        self.relu = nn.ReLU()                              # 激活函数
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)     # 线性变换
        x = self.relu(x)       # ReLU 激活
        x = self.layer2(x)     # 线性变换
        x = self.sigmoid(x)    # Sigmoid 输出概率
        return x

# 创建模型
model = SimpleNet(input_size=2, hidden_size=16, output_size=1)
print(f"模型结构:\n{model}")

# 查看参数
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {total_params}")
for name, param in model.named_parameters():
    print(f"  {name}: shape={list(param.shape)}, 参数量={param.numel()}")


# ============================================================
# 4. 完整训练流程
# ============================================================
print("\n" + "=" * 60)
print("4. 完整训练流程")
print("=" * 60)

print("""
【PyTorch 训练标准模板】

for epoch in range(n_epochs):
    # 1. 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # 2. 反向传播
    optimizer.zero_grad()   # 清空上一步的梯度！(重要)
    loss.backward()         # 计算梯度
    optimizer.step()        # 更新参数
""")

# 准备数据
X_moon, y_moon = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_moon, y_moon, test_size=0.2, random_state=42)

# 转为 PyTorch 张量
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).reshape(-1, 1)

# 创建 DataLoader（自动分批）
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义模型、损失函数、优化器
model = SimpleNet(input_size=2, hidden_size=32, output_size=1)
criterion = nn.BCELoss()        # 二分类交叉熵
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练循环
print("\n--- 开始训练 ---")
n_epochs = 100

for epoch in range(n_epochs):
    model.train()  # 训练模式
    epoch_loss = 0
    
    for batch_X, batch_y in train_loader:
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数
        
        epoch_loss += loss.item()
    
    if epoch % 20 == 0 or epoch == n_epochs - 1:
        # 评估
        model.eval()  # 评估模式
        with torch.no_grad():  # 不计算梯度（节省内存）
            train_pred = (model(X_train_t) >= 0.5).float()
            test_pred = (model(X_test_t) >= 0.5).float()
            train_acc = (train_pred == y_train_t).float().mean()
            test_acc = (test_pred == y_test_t).float().mean()
        print(f"  Epoch {epoch:3d}: loss={epoch_loss/len(train_loader):.4f}, "
              f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

# 最终评估
model.eval()
with torch.no_grad():
    y_pred_final = (model(X_test_t) >= 0.5).numpy().flatten()
final_acc = accuracy_score(y_test, y_pred_final)
print(f"\n最终测试准确率: {final_acc:.4f}")
assert final_acc > 0.85
print("✓ 验证通过")


# ============================================================
# 5. 更优雅的网络定义方式
# ============================================================
print("\n" + "=" * 60)
print("5. Sequential 和其他技巧")
print("=" * 60)

# 方式1: nn.Sequential（简单网络推荐）
model_seq = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Dropout(0.2),      # Dropout 正则化（随机丢弃 20% 的神经元）
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(16, 1),
    nn.Sigmoid()
)
print(f"Sequential 模型:\n{model_seq}")

# BatchNorm (Batch Normalization)
print("""
\n【常用技巧】

1. Dropout (随机失活):
   - 训练时随机关闭部分神经元
   - 防止过拟合（让网络不依赖任何单个神经元）
   - 测试时不使用 dropout

2. Batch Normalization:
   - 对每一层的输出做标准化
   - 加速训练，允许更大学习率
   - 轻微的正则化效果

3. 权重衰减 (Weight Decay):
   - 在优化器中加入: optimizer = Adam(lr=0.01, weight_decay=1e-4)
   - 等价于 L2 正则化
""")

# 训练 Sequential 模型
optimizer_seq = optim.Adam(model_seq.parameters(), lr=0.01, weight_decay=1e-4)
criterion_seq = nn.BCELoss()

model_seq.train()
for epoch in range(100):
    for batch_X, batch_y in train_loader:
        outputs = model_seq(batch_X)
        loss = criterion_seq(outputs, batch_y)
        optimizer_seq.zero_grad()
        loss.backward()
        optimizer_seq.step()

model_seq.eval()
with torch.no_grad():
    y_pred_seq = (model_seq(X_test_t) >= 0.5).numpy().flatten()
acc_seq = accuracy_score(y_test, y_pred_seq)
print(f"\nSequential 模型测试准确率: {acc_seq:.4f}")
assert acc_seq > 0.85
print("✓ 验证通过")


# ============================================================
# 6. 保存和加载模型
# ============================================================
print("\n" + "=" * 60)
print("6. 保存和加载模型")
print("=" * 60)

# 保存
save_path = '/home/gfhong/testcode/ai_tutorial/04_deep_learning/model_demo.pth'
torch.save(model.state_dict(), save_path)
print(f"模型已保存: {save_path}")

# 加载
loaded_model = SimpleNet(input_size=2, hidden_size=32, output_size=1)
loaded_model.load_state_dict(torch.load(save_path, weights_only=True))
loaded_model.eval()

# 验证加载的模型
with torch.no_grad():
    y_pred_loaded = (loaded_model(X_test_t) >= 0.5).numpy().flatten()
assert np.array_equal(y_pred_final, y_pred_loaded)
print("✓ 加载的模型预测结果与原模型一致")

print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点：
1. PyTorch Tensor ≈ NumPy Array + GPU支持 + 自动求导
2. requires_grad=True + .backward() → 自动计算所有梯度
3. nn.Module 是构建网络的基础类
4. 训练模板: forward → loss → zero_grad → backward → step
5. model.train() 训练模式 / model.eval() 评估模式
6. torch.no_grad() 推理时关闭梯度计算

PyTorch 的优势：
  - 动态计算图（方便调试，print 就能看中间结果）
  - 代码直觉（就像写 NumPy）
  - 社区活跃，论文实现多用 PyTorch

下一节：卷积神经网络(CNN) → 图像处理的利器
""")
