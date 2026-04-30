"""
第三章 3.4：模型评估
==================

"如果你不能衡量它，你就不能改进它。"

本节内容：
1. 训练集/验证集/测试集划分
2. 交叉验证
3. 分类指标（精确率、召回率、F1）
4. 混淆矩阵
5. 偏差-方差权衡
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)

print("=" * 60)
print("第三章 3.4：模型评估")
print("=" * 60)

# ============================================================
# 1. 数据集划分
# ============================================================
print("\n" + "=" * 60)
print("1. 训练集/验证集/测试集")
print("=" * 60)

print("""
【为什么要分割数据？】

训练集: 用来训练模型（学习参数）
验证集: 用来调超参数（选最好的模型配置）
测试集: 最终评估模型效果（模拟真实场景）

如果用训练数据来评估模型 → 看到的是"背书"能力，不是"理解"能力

常见比例: 训练 60% + 验证 20% + 测试 20%
简单场景: 训练 80% + 测试 20%

【数据泄露 (Data Leakage)】— 最常见的错误！
  测试集的信息不能泄露到训练过程中！
  
  错误示例：
    1. 先在全部数据上做标准化，再分割 ← 错！
    2. 用全部数据选特征，再训练 ← 错！
  
  正确做法：
    1. 先分割数据
    2. 只在训练集上计算统计量（均值、标准差等）
    3. 用训练集的统计量转换测试集
""")

np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                          random_state=42)

# 正确的分割方式
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

print(f"总数据: {len(X)} 样本")
print(f"训练集: {len(X_train)} 样本 ({len(X_train)/len(X)*100:.0f}%)")
print(f"验证集: {len(X_val)} 样本 ({len(X_val)/len(X)*100:.0f}%)")
print(f"测试集: {len(X_test)} 样本 ({len(X_test)/len(X)*100:.0f}%)")

# 正确的标准化流程
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
X_train_norm = (X_train - train_mean) / train_std
X_val_norm = (X_val - train_mean) / train_std  # 用训练集的统计量!
X_test_norm = (X_test - train_mean) / train_std  # 用训练集的统计量!
print("\n✓ 标准化：使用训练集的 mean/std 转换所有数据集")


# ============================================================
# 2. 交叉验证
# ============================================================
print("\n" + "=" * 60)
print("2. 交叉验证 (Cross-Validation)")
print("=" * 60)

print("""
【K折交叉验证】
数据量不大时，单次分割可能不稳定（运气好坏影响结果）。

K折交叉验证：
  1. 将数据分成 K 份（通常 K=5 或 10）
  2. 每次用 1 份做验证，其余 K-1 份做训练
  3. 重复 K 次，得到 K 个分数
  4. 取平均作为最终评估

优点：充分利用数据，评估更稳定
缺点：计算量 × K
""")

# 手动实现 K 折交叉验证
def manual_cross_validation(model_class, X, y, k=5, **model_params):
    """手动实现 K 折交叉验证"""
    n = len(X)
    fold_size = n // k
    indices = np.random.permutation(n)
    scores = []
    
    for i in range(k):
        # 划分当前折
        val_idx = indices[i*fold_size : (i+1)*fold_size]
        train_idx = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
        
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # 训练和评估
        model = model_class(**model_params)
        model.fit(X_train_fold, y_train_fold)
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)
    
    return np.array(scores)

# 手动实现
scores_manual = manual_cross_validation(LogisticRegression, X, y, k=5, max_iter=1000)
print(f"手动5折交叉验证: {scores_manual.round(4)}")
print(f"  平均: {scores_manual.mean():.4f} ± {scores_manual.std():.4f}")

# sklearn 实现
scores_sklearn = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5)
print(f"\nsklearn 5折交叉验证: {scores_sklearn.round(4)}")
print(f"  平均: {scores_sklearn.mean():.4f} ± {scores_sklearn.std():.4f}")


# ============================================================
# 3. 分类指标
# ============================================================
print("\n" + "=" * 60)
print("3. 分类指标详解")
print("=" * 60)

print("""
【为什么准确率不够？】

场景：信用卡欺诈检测
  - 99% 的交易是正常的，1% 是欺诈
  - 如果模型永远预测"正常"，准确率 = 99%！
  - 但它从没抓到过一个骗子！

所以需要更细粒度的指标：

【混淆矩阵】
                    预测正  预测负
  实际正 (1)  →   TP      FN
  实际负 (0)  →   FP      TN

  TP (True Positive):  正确预测为正 → 抓到了真骗子
  FP (False Positive): 错误预测为正 → 冤枉好人
  FN (False Negative): 错误预测为负 → 放走了骗子
  TN (True Negative):  正确预测为负 → 好人没被冤枉

【精确率 (Precision)】
  P = TP / (TP + FP)
  "预测为正的里面，有多少是真的正？"
  关心的是：别冤枉好人

【召回率 (Recall)】
  R = TP / (TP + FN)
  "实际为正的里面，有多少被找出来了？"
  关心的是：别漏掉坏人

【F1 Score】
  F1 = 2 × P × R / (P + R)
  精确率和召回率的调和平均（平衡两者）
""")

# 模拟不平衡数据
np.random.seed(42)
X_imb, y_imb = make_classification(n_samples=1000, n_features=10,
                                    weights=[0.95, 0.05],  # 95:5 不平衡
                                    random_state=42)
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_imb, y_imb, test_size=0.3, random_state=42)

print(f"\n不平衡数据集:")
print(f"  正类比例: {y_imb.mean():.2%}")

# 训练模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train_i, y_train_i)
y_pred = model.predict(X_test_i)

# 各种指标
acc = accuracy_score(y_test_i, y_pred)
prec = precision_score(y_test_i, y_pred)
rec = recall_score(y_test_i, y_pred)
f1 = f1_score(y_test_i, y_pred)

print(f"\n模型评估:")
print(f"  准确率 (Accuracy):  {acc:.4f}")
print(f"  精确率 (Precision): {prec:.4f}")
print(f"  召回率 (Recall):    {rec:.4f}")
print(f"  F1 Score:           {f1:.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test_i, y_pred)
print(f"\n混淆矩阵:")
print(f"              预测负  预测正")
print(f"  实际负 (0): {cm[0][0]:4d}  {cm[0][1]:4d}")
print(f"  实际正 (1): {cm[1][0]:4d}  {cm[1][1]:4d}")

# "笨"模型 - 永远预测负类
y_pred_dumb = np.zeros_like(y_test_i)
acc_dumb = accuracy_score(y_test_i, y_pred_dumb)
print(f"\n'永远预测负类'的笨模型:")
print(f"  准确率: {acc_dumb:.4f} (看起来很高!)")
print(f"  但召回率: {recall_score(y_test_i, y_pred_dumb):.4f} (一个正类都没找到)")
print(f"\n→ 这就是为什么不能只看准确率!")

# 完整分类报告
print(f"\n--- sklearn classification_report ---")
print(classification_report(y_test_i, y_pred, target_names=['负类', '正类']))


# ============================================================
# 4. 偏差-方差权衡
# ============================================================
print("\n" + "=" * 60)
print("4. 偏差-方差权衡 (Bias-Variance Tradeoff)")
print("=" * 60)

print("""
【模型误差的组成】
  总误差 = 偏差² + 方差 + 不可约误差(噪声)

【偏差 (Bias)】
  模型的假设太简单，无法捕捉数据的真实模式
  → 欠拟合 (Underfitting)
  例: 用直线拟合曲线数据

【方差 (Variance)】
  模型对训练数据太敏感，换一组数据就得到很不同的模型
  → 过拟合 (Overfitting)
  例: 用 100 次多项式拟合 10 个数据点

【权衡】
  简单模型: 高偏差，低方差（欠拟合）
  复杂模型: 低偏差，高方差（过拟合）
  
  最佳模型: 在两者之间找到平衡

【实际诊断方法】
  训练误差高 + 测试误差高 → 欠拟合 → 需要更复杂的模型
  训练误差低 + 测试误差高 → 过拟合 → 需要正则化/更多数据
  训练误差低 + 测试误差低 → Just right! ✓
""")

# 演示偏差-方差权衡
from sklearn.tree import DecisionTreeClassifier

print("\n--- 用决策树深度演示偏差-方差权衡 ---")
np.random.seed(42)
X_bv, y_bv = make_classification(n_samples=500, n_features=10, 
                                  n_informative=5, random_state=42)
X_train_bv, X_test_bv, y_train_bv, y_test_bv = train_test_split(
    X_bv, y_bv, test_size=0.3, random_state=42)

print(f"{'树深度':<8} {'训练准确率':<12} {'测试准确率':<12} {'诊断'}")
print("-" * 50)

for depth in [1, 2, 3, 5, 10, 20, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train_bv, y_train_bv)
    train_acc = tree.score(X_train_bv, y_train_bv)
    test_acc = tree.score(X_test_bv, y_test_bv)
    
    depth_str = str(depth) if depth else "无限"
    gap = train_acc - test_acc
    if train_acc < 0.85:
        diag = "欠拟合"
    elif gap > 0.1:
        diag = "过拟合"
    else:
        diag = "✓ 合适"
    
    print(f"{depth_str:<8} {train_acc:<12.4f} {test_acc:<12.4f} {diag}")

print("""
→ 深度太浅: 欠拟合（训练和测试都不好）
→ 深度太深: 过拟合（训练完美但测试变差）
→ 中间深度: 刚好（训练和测试都不错）
""")

print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点：
1. 永远用测试集评估，不要用训练集自己评估自己
2. 交叉验证给出更稳定的评估结果
3. 不平衡数据时，用 Precision/Recall/F1，不要只看准确率
4. 偏差-方差权衡：模型不能太简单也不能太复杂
5. 诊断方法：比较训练误差和测试误差的差距

下一章：深度学习 → 用神经网络解决复杂问题
""")
