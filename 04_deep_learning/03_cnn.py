"""
第四章 4.3：卷积神经网络 (CNN)
==============================

CNN 是处理图像的经典深度学习架构。
通过"卷积"操作，自动学习图像中的特征（边缘、纹理、形状等）。

本节内容：
1. 卷积操作的直觉
2. CNN 的核心组件
3. 从零实现卷积
4. 用 PyTorch 构建 CNN
5. 在 MNIST 手写数字上训练
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms
    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False

print("=" * 60)
print("第四章 4.3：卷积神经网络 (CNN)")
print("=" * 60)

# ============================================================
# 1. 卷积操作的直觉
# ============================================================
print("\n" + "=" * 60)
print("1. 卷积操作的直觉")
print("=" * 60)

print("""
【为什么图像不用全连接网络？】

一张 28×28 的灰度图 = 784 个像素
如果隐藏层有 256 个神经元：参数量 = 784×256 = 200,704

一张 224×224×3 的彩色图 = 150,528 个像素
如果隐藏层有 1000 个神经元：参数量 = 150,528,000 = 1.5 亿！
→ 参数太多，容易过拟合，计算也太慢

【卷积的核心思想】
1. 局部连接: 一个神经元只看图像的一小块区域
   (猫的耳朵在哪里是局部特征)

2. 参数共享: 同一个"滤波器"在整张图上滑动
   (检测"竖直边缘"的滤波器，不管边缘在左边还是右边都能检测到)

3. 平移不变性: 猫在图片左边还是右边，都能被识别

【卷积操作】
用一个小的"卷积核"(如 3×3) 在输入图像上滑动：
  - 在每个位置，核与图像对应区域做逐元素乘法再求和
  - 产生一个输出值
  - 所有位置的输出组成"特征图" (feature map)

例: 3×3 的边缘检测核:
  [[-1, -1, -1],
   [ 0,  0,  0],
   [ 1,  1,  1]]
  → 检测水平边缘（上下像素差异大的地方）
""")

# ============================================================
# 2. 从零实现卷积
# ============================================================
print("\n" + "=" * 60)
print("2. 从零实现卷积操作")
print("=" * 60)

def conv2d_naive(input_image, kernel, stride=1, padding=0):
    """
    朴素卷积实现（用于理解原理，实际不会这么慢）
    
    input_image: (H, W)
    kernel: (kH, kW)
    """
    # 添加 padding
    if padding > 0:
        input_image = np.pad(input_image, padding, mode='constant')
    
    H, W = input_image.shape
    kH, kW = kernel.shape
    
    # 输出尺寸公式: out = (in + 2*padding - kernel) / stride + 1
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    
    output = np.zeros((out_H, out_W))
    
    for i in range(out_H):
        for j in range(out_W):
            # 取输入的局部区域
            region = input_image[i*stride:i*stride+kH, j*stride:j*stride+kW]
            # 逐元素乘法再求和
            output[i, j] = np.sum(region * kernel)
    
    return output

# 示例：5×5 的图像
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
], dtype=float)

# 边缘检测核
edge_kernel = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=float)

output = conv2d_naive(image, edge_kernel)
print("输入 (5×5):")
print(image)
print(f"\n水平边缘检测核 (3×3):")
print(edge_kernel)
print(f"\n卷积输出 (3×3):")
print(output)
print(f"\n输出尺寸公式: (5 - 3)/1 + 1 = 3")
print(f"解读: 正值=上方边缘, 负值=下方边缘, 0=无边缘")

# 验证 PyTorch 结果一致
image_t = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)  # (1,1,5,5)
kernel_t = torch.FloatTensor(edge_kernel).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)
output_t = torch.nn.functional.conv2d(image_t, kernel_t).squeeze().numpy()
assert np.allclose(output, output_t)
print("\n✓ 与 PyTorch 结果一致")


# ============================================================
# 3. CNN 的核心组件
# ============================================================
print("\n" + "=" * 60)
print("3. CNN 的核心组件")
print("=" * 60)

print("""
【CNN 典型结构】
  输入图像 → [卷积 → 激活 → 池化] × N → 展平 → 全连接 → 输出

各组件说明：

1. 卷积层 (Conv2d):
   - 输入: (batch, channels, H, W)
   - 参数: kernel_size, stride, padding, out_channels
   - 输出通道数 = 滤波器数量（每个滤波器学一种模式）
   
2. 池化层 (MaxPool2d / AvgPool2d):
   - 降低分辨率（减少参数和计算量）
   - MaxPool: 取局部最大值（保留最强特征）
   - 通常 kernel_size=2, stride=2 → 尺寸减半
   
3. Batch Normalization:
   - 标准化每层输出，加速训练
   
4. 展平 (Flatten):
   - 把 (batch, C, H, W) 变成 (batch, C×H×W)
   - 然后接全连接层做分类

【输出尺寸计算】
  out_size = (in_size + 2×padding - kernel_size) / stride + 1
  
  常见配置:
  - kernel=3, stride=1, padding=1 → 尺寸不变
  - MaxPool(2, 2) → 尺寸减半
""")

# 演示各层的尺寸变化
print("\n--- 各层尺寸变化示例 ---")
x = torch.randn(1, 1, 28, 28)  # 1张 1通道 28×28 的图
print(f"输入: {list(x.shape)}")

conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
x1 = conv1(x)
print(f"Conv2d(1→16, k=3, p=1): {list(x1.shape)}")

pool = nn.MaxPool2d(2, 2)
x2 = pool(x1)
print(f"MaxPool2d(2): {list(x2.shape)}")

conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
x3 = conv2(x2)
print(f"Conv2d(16→32, k=3, p=1): {list(x3.shape)}")

x4 = pool(x3)
print(f"MaxPool2d(2): {list(x4.shape)}")

x5 = x4.flatten(1)
print(f"Flatten: {list(x5.shape)}")

fc = nn.Linear(32 * 7 * 7, 10)
x6 = fc(x5)
print(f"Linear(→10): {list(x6.shape)} (10个类别)")


# ============================================================
# 4. 构建 CNN 分类器
# ============================================================
print("\n" + "=" * 60)
print("4. 构建 CNN 分类器 (MNIST 手写数字)")
print("=" * 60)

class MNISTNet(nn.Module):
    """用于 MNIST 手写数字识别的 CNN"""
    
    def __init__(self):
        super().__init__()
        # 卷积特征提取部分
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (1,28,28) → (16,28,28)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # (16,28,28) → (16,14,14)
            
            # 第二个卷积块
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (16,14,14) → (32,14,14)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # (32,14,14) → (32,7,7)
        )
        
        # 分类部分
        self.classifier = nn.Sequential(
            nn.Flatten(),                                   # (32,7,7) → (1568)
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)                             # 10 个数字类别
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = MNISTNet()
print(f"CNN 模型结构:")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {total_params:,}")


# ============================================================
# 5. 在 MNIST 上训练
# ============================================================
print("\n" + "=" * 60)
print("5. 训练 MNIST 手写数字识别")
print("=" * 60)

# 下载数据（第一次运行会下载）
print("加载 MNIST 数据集...")
try:
    if not HAS_TORCHVISION:
        raise RuntimeError("torchvision 不可用或与当前 PyTorch 版本不兼容")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),           # 转为张量并归一化到 [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # 为了演示速度，只用部分数据
    train_subset = torch.utils.data.Subset(train_dataset, range(5000))
    test_subset = torch.utils.data.Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    print(f"训练集: {len(train_subset)} 样本")
    print(f"测试集: {len(test_subset)} 样本")
    print(f"图像大小: 28×28 灰度")
    print(f"类别数: 10 (数字 0-9)")
    
    # 训练
    model = MNISTNet()
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵（内含 softmax）
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n--- 开始训练 ---")
    n_epochs = 5
    
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        train_acc = correct / total
        
        # 测试
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                _, predicted = outputs.max(1)
                test_total += batch_y.size(0)
                test_correct += predicted.eq(batch_y).sum().item()
        
        test_acc = test_correct / test_total
        print(f"  Epoch {epoch+1}/{n_epochs}: loss={running_loss/len(train_loader):.4f}, "
              f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
    
    print(f"\n最终测试准确率: {test_acc:.4f}")
    assert test_acc > 0.90, f"准确率太低: {test_acc}"
    print("✓ CNN 在 MNIST 上达到 >90% 准确率")
    
    # 展示一些预测
    model.eval()
    with torch.no_grad():
        sample_X, sample_y = next(iter(test_loader))
        outputs = model(sample_X[:10])
        _, predictions = outputs.max(1)
    
    print(f"\n前10个测试样本:")
    print(f"  真实标签: {sample_y[:10].tolist()}")
    print(f"  模型预测: {predictions[:10].tolist()}")

except Exception as e:
    print(f"注意: MNIST 数据加载失败 ({e})")
    print("这可能是网络问题。代码逻辑正确，可以在有网络时重试。")
    
    # 用随机数据演示模型可以正常前向传播
    print("\n使用随机数据验证模型结构正确性...")
    model = MNISTNet()
    dummy_input = torch.randn(4, 1, 28, 28)
    dummy_output = model(dummy_input)
    print(f"  输入形状: {list(dummy_input.shape)}")
    print(f"  输出形状: {list(dummy_output.shape)}")
    assert dummy_output.shape == (4, 10)
    print("✓ 模型结构正确")

print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点：
1. CNN 通过卷积实现 局部连接 + 参数共享 → 大幅减少参数
2. 卷积核自动学习图像特征（边缘→纹理→形状→物体）
3. 池化降低分辨率，增加平移不变性
4. 典型结构: Conv-BN-ReLU-Pool 重复N次 → Flatten → FC
5. 输出尺寸: (in + 2*pad - kernel) / stride + 1

CNN 的演进（了解即可）：
  LeNet (1998) → AlexNet (2012) → VGG (2014) → 
  ResNet (2015) → EfficientNet (2019) → Vision Transformer (2020)

下一节：循环神经网络(RNN) → 处理序列数据
""")
