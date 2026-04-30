"""
第一章 1.3：概率与统计基础
========================

概率与统计在 AI 中无处不在：
- 分类模型输出的是"属于某类的概率"
- 训练数据本身就是从某个分布中采样的
- 贝叶斯方法是很多 AI 算法的理论基础
- 损失函数很多来源于概率论（交叉熵、最大似然）

本节内容：
1. 概率基础概念
2. 条件概率与贝叶斯定理
3. 常见概率分布
4. 期望、方差、标准差
5. 最大似然估计
6. 信息论基础（熵、交叉熵）
"""

import numpy as np

print("=" * 60)
print("第一章 1.3：概率与统计基础")
print("=" * 60)

# ============================================================
# 1. 概率基础
# ============================================================
print("\n" + "=" * 60)
print("1. 概率基础概念")
print("=" * 60)

print("""
【概率的含义】
概率 P(A) 表示事件 A 发生的可能性，范围 [0, 1]
  P(A) = 0 → 不可能发生
  P(A) = 1 → 一定发生
  P(A) = 0.7 → 70% 的可能性发生

【基本规则】
  P(A或B) = P(A) + P(B) - P(A且B)   （加法规则）
  P(A且B) = P(A) × P(B|A)            （乘法规则）

  如果 A 和 B 互相独立：
  P(A且B) = P(A) × P(B)
""")

# 用模拟验证概率
print("--- 用模拟验证概率 ---")
np.random.seed(42)
n_trials = 100000

# 模拟掷骰子
dice_rolls = np.random.randint(1, 7, size=n_trials)
prob_6 = np.mean(dice_rolls == 6)
print(f"掷骰子得到6的概率：")
print(f"  理论值: 1/6 = {1/6:.4f}")
print(f"  模拟值: {prob_6:.4f} (模拟{n_trials}次)")

# 模拟两个骰子之和
dice1 = np.random.randint(1, 7, size=n_trials)
dice2 = np.random.randint(1, 7, size=n_trials)
prob_sum7 = np.mean((dice1 + dice2) == 7)
print(f"\n两个骰子之和为7的概率：")
print(f"  理论值: 6/36 = {6/36:.4f}")
print(f"  模拟值: {prob_sum7:.4f}")


# ============================================================
# 2. 条件概率与贝叶斯定理
# ============================================================
print("\n" + "=" * 60)
print("2. 条件概率与贝叶斯定理")
print("=" * 60)

print("""
【条件概率】
P(A|B) = 在 B 已经发生的条件下，A 发生的概率
P(A|B) = P(A且B) / P(B)

例子：
  已知一封邮件包含"免费"这个词，它是垃圾邮件的概率是多少？
  这就是条件概率：P(垃圾邮件 | 包含"免费")

【贝叶斯定理】
  P(A|B) = P(B|A) × P(A) / P(B)

直觉："用新证据更新我们的信念"

  P(A)     = 先验概率（在看到证据 B 之前，A 的概率）
  P(A|B)   = 后验概率（看到证据 B 之后，A 的概率）
  P(B|A)   = 似然（如果 A 为真，看到 B 的概率）

在 AI 中：
  - 朴素贝叶斯分类器直接使用贝叶斯定理
  - 贝叶斯网络
  - 很多模型的训练本质上是最大化后验概率
""")

# 经典例子：医学检测
print("--- 贝叶斯定理实际例子：医学检测 ---")
print("""
场景：
  - 某疾病的患病率为 1%（先验概率）
  - 检测的准确率：
    - 真正患病的人，99% 概率检测为阳性（灵敏度）
    - 没有患病的人，95% 概率检测为阴性（特异度）
  
问题：如果一个人检测为阳性，他真的患病的概率是多少？

很多人直觉会说 99%，但实际上...
""")

P_disease = 0.01          # P(患病)
P_positive_given_disease = 0.99   # P(阳性|患病)
P_negative_given_healthy = 0.95   # P(阴性|健康)
P_positive_given_healthy = 1 - P_negative_given_healthy  # P(阳性|健康) = 0.05

# 贝叶斯定理
# P(患病|阳性) = P(阳性|患病) × P(患病) / P(阳性)
# P(阳性) = P(阳性|患病)×P(患病) + P(阳性|健康)×P(健康)
P_positive = P_positive_given_disease * P_disease + P_positive_given_healthy * (1 - P_disease)
P_disease_given_positive = P_positive_given_disease * P_disease / P_positive

print(f"计算过程：")
print(f"  P(阳性) = P(阳性|患病)×P(患病) + P(阳性|健康)×P(健康)")
print(f"          = {P_positive_given_disease}×{P_disease} + {P_positive_given_healthy}×{1-P_disease}")
print(f"          = {P_positive:.4f}")
print(f"")
print(f"  P(患病|阳性) = P(阳性|患病)×P(患病) / P(阳性)")
print(f"               = {P_positive_given_disease}×{P_disease} / {P_positive:.4f}")
print(f"               = {P_disease_given_positive:.4f}")
print(f"")
print(f"结果：检测为阳性，实际患病的概率只有 {P_disease_given_positive*100:.1f}%！")
print(f"原因：因为患病率太低(1%)，大量健康人中误检(5%)的绝对数量远超真正患病者。")

# 用模拟验证
population = 100000
is_sick = np.random.random(population) < P_disease
test_positive = np.where(
    is_sick,
    np.random.random(population) < P_positive_given_disease,
    np.random.random(population) < P_positive_given_healthy
)
simulated = np.mean(is_sick[test_positive]) 
print(f"\n模拟验证 ({population}人):")
print(f"  P(患病|阳性) ≈ {simulated:.4f}")
assert abs(simulated - P_disease_given_positive) < 0.02
print("✓ 验证通过")


# ============================================================
# 3. 常见概率分布
# ============================================================
print("\n" + "=" * 60)
print("3. 常见概率分布")
print("=" * 60)

print("""
【均匀分布 Uniform】
  所有值出现的概率相等
  例: 掷骰子，每个面的概率都是 1/6

【正态分布 (高斯分布) Normal/Gaussian】 ← AI 中最重要！
  钟形曲线，由均值 μ 和标准差 σ 决定
  
  公式: f(x) = (1/√(2πσ²)) × e^(-(x-μ)²/(2σ²))
  
  特点：
  - 68% 的数据在 μ±σ 内
  - 95% 的数据在 μ±2σ 内
  - 99.7% 的数据在 μ±3σ 内
  
  在 AI 中：
  - 权重初始化通常用正态分布
  - 很多自然现象近似正态分布
  - 噪声通常建模为正态分布

【伯努利分布 Bernoulli】
  只有两个结果：成功(1)或失败(0)
  P(X=1) = p, P(X=0) = 1-p
  例: 抛硬币

【Softmax 分布】← 分类模型的输出！
  将任意实数向量转换为概率分布（所有值在0-1之间，且和为1）
  softmax(xᵢ) = eˣⁱ / Σⱼ(eˣʲ)
""")

# 正态分布示例
print("--- 正态分布验证 ---")
mu, sigma = 170, 10  # 身高: 均值170cm, 标准差10cm
samples = np.random.normal(mu, sigma, 100000)

within_1sigma = np.mean(np.abs(samples - mu) <= sigma) * 100
within_2sigma = np.mean(np.abs(samples - mu) <= 2*sigma) * 100
within_3sigma = np.mean(np.abs(samples - mu) <= 3*sigma) * 100

print(f"正态分布 N(μ={mu}, σ={sigma})（模拟身高数据）")
print(f"  μ±1σ ({mu-sigma}-{mu+sigma}cm) 内: {within_1sigma:.1f}% (理论 68.3%)")
print(f"  μ±2σ ({mu-2*sigma}-{mu+2*sigma}cm) 内: {within_2sigma:.1f}% (理论 95.4%)")
print(f"  μ±3σ ({mu-3*sigma}-{mu+3*sigma}cm) 内: {within_3sigma:.1f}% (理论 99.7%)")

# Softmax 实现
print("\n--- Softmax 函数 ---")
def softmax(x):
    """将任意向量转换为概率分布"""
    # 减去最大值防止数值溢出（结果不变，这是一个重要的工程技巧）
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# 假设模型输出3个类别的"原始分数"
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)

print(f"模型原始输出 (logits): {logits}")
print(f"Softmax 后 (概率):     {probs}")
print(f"概率之和: {probs.sum():.6f} (应该是 1.0)")
print(f"\n解读: 模型认为")
print(f"  类别0的概率: {probs[0]*100:.1f}%")
print(f"  类别1的概率: {probs[1]*100:.1f}%")
print(f"  类别2的概率: {probs[2]*100:.1f}%")
assert abs(probs.sum() - 1.0) < 1e-10
print("✓ 概率之和为1，验证通过")


# ============================================================
# 4. 期望、方差、标准差
# ============================================================
print("\n" + "=" * 60)
print("4. 期望、方差、标准差")
print("=" * 60)

print("""
【期望 (均值) E[X]】
  = 所有可能值的加权平均
  = Σ xᵢ × P(xᵢ)
  
  直觉: 如果重复实验无穷多次，平均结果是多少

【方差 Var(X)】
  = 数据分散程度的度量
  = E[(X - E[X])²]
  = 每个值与均值之差的平方的期望
  
  方差大 → 数据分散 → 不确定性大
  方差小 → 数据集中 → 不确定性小

【标准差 σ】
  = √方差
  = 和原始数据同一单位，更直觉
  
在 AI 中：
  - 数据标准化: x_new = (x - μ) / σ （使数据均值0方差1）
  - Batch Normalization 用到均值和方差
  - 模型输出的不确定性估计
""")

# 计算示例
data = np.array([85, 90, 78, 92, 88, 76, 95, 89, 84, 91])
print(f"学生成绩: {data}")
print(f"  均值 (期望): {np.mean(data):.2f}")
print(f"  方差: {np.var(data):.2f}")
print(f"  标准差: {np.std(data):.2f}")
print(f"  最高分与均值差: {np.max(data) - np.mean(data):.2f} = {(np.max(data) - np.mean(data))/np.std(data):.2f} 个标准差")

# 数据标准化
print("\n--- 数据标准化 (Z-Score) ---")
print("公式: z = (x - μ) / σ")
print("效果: 变换后均值=0, 标准差=1")
print("为什么重要: 不同特征的量纲不同(身高cm, 体重kg, 年龄年)，标准化后可公平比较")

data_normalized = (data - np.mean(data)) / np.std(data)
print(f"\n原始数据: {data}")
print(f"标准化后: {np.round(data_normalized, 3)}")
print(f"  新均值: {np.mean(data_normalized):.10f} (≈0)")
print(f"  新标准差: {np.std(data_normalized):.10f} (≈1)")
assert abs(np.mean(data_normalized)) < 1e-10
assert abs(np.std(data_normalized) - 1.0) < 1e-10
print("✓ 验证通过")


# ============================================================
# 5. 最大似然估计 (MLE)
# ============================================================
print("\n" + "=" * 60)
print("5. 最大似然估计 (Maximum Likelihood Estimation)")
print("=" * 60)

print("""
【什么是最大似然估计？】

"给定观测到的数据，哪个参数值最有可能产生这些数据？"

举例：
  你抛了一枚硬币 10 次，得到 7 次正面。
  这枚硬币正面概率 p 最可能是多少？
  
  直觉：p = 7/10 = 0.7 （这就是最大似然估计！）

数学过程：
  似然函数 L(p) = P(数据|参数p)
  对于 n 次独立实验，k 次成功：
  L(p) = C(n,k) × p^k × (1-p)^(n-k)
  
  取对数（便于计算，因为乘法变加法）：
  log L(p) = k×log(p) + (n-k)×log(1-p) + const
  
  对 p 求导令其为 0：
  d/dp [log L] = k/p - (n-k)/(1-p) = 0
  → p = k/n

为什么 AI 关心这个？
  训练神经网络 = 最大化模型参数在训练数据上的似然
  = 最小化负对数似然
  = 很多损失函数(如交叉熵)的本质！
""")

# 模拟最大似然估计
print("--- MLE 示例：估计硬币正面概率 ---")
true_p = 0.7
n_flips = 1000
flips = np.random.binomial(1, true_p, n_flips)
k = flips.sum()

mle_p = k / n_flips
print(f"真实概率: {true_p}")
print(f"抛 {n_flips} 次，正面 {k} 次")
print(f"MLE 估计: p̂ = {k}/{n_flips} = {mle_p:.4f}")
print(f"误差: {abs(mle_p - true_p):.4f}")

# 绘制似然函数
p_values = np.linspace(0.01, 0.99, 100)
log_likelihood = k * np.log(p_values) + (n_flips - k) * np.log(1 - p_values)
best_p_idx = np.argmax(log_likelihood)
print(f"\n似然函数最大值处的 p: {p_values[best_p_idx]:.4f}")
print(f"与直接计算 k/n = {mle_p:.4f} 一致")
assert abs(p_values[best_p_idx] - mle_p) < 0.02
print("✓ 验证通过")


# ============================================================
# 6. 信息论基础（熵和交叉熵）
# ============================================================
print("\n" + "=" * 60)
print("6. 信息论基础：熵和交叉熵")
print("=" * 60)

print("""
【信息量】
  一个事件的信息量 = -log₂(P(事件))
  
  直觉：
  - 越不可能发生的事件，发生时包含的信息量越大
  - "太阳从东方升起" → 概率≈1，信息量≈0（不意外）
  - "彩票中了500万" → 概率极小，信息量极大（太意外了！）

【熵 (Entropy)】
  H(P) = -Σ P(x) × log₂(P(x))
  
  = 一个概率分布的"平均信息量"或"不确定性"
  
  熵高 → 不确定性大 → 分布均匀（不知道会出什么）
  熵低 → 不确定性小 → 分布集中（很确定会出什么）

例如：
  - 公平硬币(0.5, 0.5)：熵 = 1 bit（最大不确定性）
  - 偏硬币(0.99, 0.01)：熵 ≈ 0.08 bit（几乎确定正面）

【交叉熵 (Cross-Entropy)】← AI 中最常用的损失函数！
  H(P, Q) = -Σ P(x) × log(Q(x))
  
  P = 真实分布, Q = 模型预测的分布
  
  交叉熵衡量：用分布 Q 来编码来自分布 P 的数据，需要多少信息。
  P 和 Q 越接近，交叉熵越小。
  
  在分类任务中：
  P = [0, 0, 1, 0] (真实标签：第3类)
  Q = [0.1, 0.1, 0.7, 0.1] (模型预测)
  交叉熵 = -(0×log(0.1) + 0×log(0.1) + 1×log(0.7) + 0×log(0.1))
          = -log(0.7) ≈ 0.357
  
  模型越确信正确答案，交叉熵越小！
""")

def entropy(probs):
    """计算熵"""
    probs = np.array(probs)
    # 过滤掉 0 (因为 0*log(0) 定义为 0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def cross_entropy(p_true, q_pred):
    """计算交叉熵"""
    p_true = np.array(p_true, dtype=float)
    q_pred = np.array(q_pred, dtype=float)
    # 避免 log(0)
    q_pred = np.clip(q_pred, 1e-15, 1.0)
    return -np.sum(p_true * np.log(q_pred))

# 熵的例子
print("--- 熵的计算 ---")
fair_coin = [0.5, 0.5]
biased_coin = [0.99, 0.01]
uniform_dice = [1/6] * 6

print(f"公平硬币 {fair_coin}: 熵 = {entropy(fair_coin):.4f} bit")
print(f"偏硬币 {biased_coin}: 熵 = {entropy(biased_coin):.4f} bit")
print(f"公平骰子: 熵 = {entropy(uniform_dice):.4f} bit")
print(f"\n→ 越均匀，熵越大（不确定性越高）")

# 交叉熵的例子（分类）
print("\n--- 交叉熵（分类损失函数）---")
y_true = [0, 0, 1, 0]  # 真实标签：第3类

# 好的预测
q_good = [0.05, 0.05, 0.85, 0.05]
# 差的预测
q_bad = [0.25, 0.25, 0.25, 0.25]
# 错误的预测
q_wrong = [0.7, 0.1, 0.1, 0.1]

ce_good = cross_entropy(y_true, q_good)
ce_bad = cross_entropy(y_true, q_bad)
ce_wrong = cross_entropy(y_true, q_wrong)

print(f"真实标签: {y_true} (第3类)")
print(f"")
print(f"好的预测 {q_good}")
print(f"  交叉熵 = -log(0.85) = {ce_good:.4f} ← 小(好)")
print(f"")
print(f"差的预测 {q_bad}")
print(f"  交叉熵 = -log(0.25) = {ce_bad:.4f} ← 大(差)")
print(f"")
print(f"错误预测 {q_wrong}")
print(f"  交叉熵 = -log(0.1) = {ce_wrong:.4f} ← 很大(很差)")
print(f"")
print(f"→ 预测越准确(给正确类别更高概率)，交叉熵越小")
assert ce_good < ce_bad < ce_wrong
print("✓ 验证通过：好预测 < 差预测 < 错误预测")

# 二分类交叉熵 (Binary Cross-Entropy)
print("\n--- 二分类交叉熵 (BCE) ---")
print("""
对于二分类 (标签 y ∈ {0, 1})：
  BCE = -[y×log(p) + (1-y)×log(1-p)]

  其中 p 是模型预测为正类的概率
""")

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 5 个样本
y = np.array([1, 0, 1, 1, 0])
p_good = np.array([0.9, 0.1, 0.8, 0.95, 0.05])  # 好的预测
p_bad = np.array([0.5, 0.5, 0.5, 0.5, 0.5])     # 差的预测

bce_good = binary_cross_entropy(y, p_good)
bce_bad = binary_cross_entropy(y, p_bad)

print(f"真实标签:  {y}")
print(f"好的预测:  {p_good} → BCE = {bce_good:.4f}")
print(f"差的预测:  {p_bad} → BCE = {bce_bad:.4f}")
assert bce_good < bce_bad
print("✓ 好的预测损失更小，验证通过")

print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点：
1. 条件概率与贝叶斯定理 → 用证据更新信念
2. 正态分布 → 数据和噪声的默认假设
3. Softmax → 将模型输出转为概率分布
4. 标准化 → 让不同特征可以公平比较
5. 最大似然估计 → 训练模型的数学原理
6. 交叉熵 → 分类任务最常用的损失函数

核心直觉：训练 AI 模型 = 最小化交叉熵 = 让模型的预测分布接近真实分布

下一节：优化基础 → 梯度下降的各种改进
""")
