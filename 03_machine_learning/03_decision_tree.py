"""
第三章 3.3：决策树与随机森林
===========================

决策树是一种直觉上非常容易理解的算法：
通过一系列"if-else"条件来做决策。

本节内容：
1. 决策树的直觉
2. 信息增益与基尼系数
3. 从零实现决策树
4. 随机森林（集成学习）
5. 使用 sklearn 实现
"""

import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

print("=" * 60)
print("第三章 3.3：决策树与随机森林")
print("=" * 60)

# ============================================================
# 1. 决策树的直觉
# ============================================================
print("\n" + "=" * 60)
print("1. 决策树的直觉")
print("=" * 60)

print("""
【决策树就像玩"20个问题"游戏】

例: 判断一种动物是什么
  Q1: 会飞吗？
    是 → Q2: 有羽毛吗？
      是 → 鸟
      否 → 蝙蝠
    否 → Q3: 有四条腿吗？
      是 → 狗或猫
      否 → 蛇

【决策树的构建过程】
1. 遍历所有特征和所有可能的分割点
2. 找到"最好的"分割（让分割后的子集最"纯"）
3. 对每个子集递归重复，直到满足停止条件

【什么是"最好的分割"？】
= 分割后，每个子集中的样本尽可能属于同一类别

衡量"不纯度"的指标：
  - 基尼系数 (Gini): 随机选一个样本，把它错误分类的概率
  - 信息增益 (Entropy): 分割前后信息熵的减少量
""")


# ============================================================
# 2. 基尼系数与信息增益
# ============================================================
print("\n" + "=" * 60)
print("2. 基尼系数与信息增益")
print("=" * 60)

print("""
【基尼系数 (Gini Impurity)】
  Gini = 1 - Σ pᵢ²
  
  其中 pᵢ 是第 i 类的比例
  
  Gini = 0 → 完全纯（只有一个类别）
  Gini = 0.5 → 最不纯（二分类，两类各 50%）

例:
  [10猫, 0狗]: Gini = 1 - (1.0² + 0²) = 0    (完全纯)
  [5猫, 5狗]:  Gini = 1 - (0.5² + 0.5²) = 0.5 (最不纯)
  [8猫, 2狗]:  Gini = 1 - (0.8² + 0.2²) = 0.32

【信息增益 (Information Gain)】
  IG = H(parent) - Σ(|child|/|parent|) × H(child)
  
  其中 H 是熵: H = -Σ pᵢ × log₂(pᵢ)
  
  信息增益越大 → 分割越好
""")

def gini_impurity(y):
    """计算基尼系数"""
    if len(y) == 0:
        return 0
    counts = Counter(y)
    probs = [count/len(y) for count in counts.values()]
    return 1 - sum(p**2 for p in probs)

def entropy(y):
    """计算信息熵"""
    if len(y) == 0:
        return 0
    counts = Counter(y)
    probs = [count/len(y) for count in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)

# 验证
print("--- 验证基尼系数 ---")
y_pure = [0, 0, 0, 0, 0]
y_mixed = [0, 0, 0, 1, 1]
y_even = [0, 0, 1, 1]

print(f"纯净 [全0]:  Gini = {gini_impurity(y_pure):.4f} (应为 0)")
print(f"混合 [3:2]:  Gini = {gini_impurity(y_mixed):.4f}")
print(f"均匀 [2:2]:  Gini = {gini_impurity(y_even):.4f} (应为 0.5)")
assert gini_impurity(y_pure) == 0
assert gini_impurity(y_even) == 0.5
print("✓ 验证通过")

# 演示信息增益
print("\n--- 信息增益示例 ---")
y_parent = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 5:5

# 分割方案1: [0,0,0,0,1] 和 [0,1,1,1,1] → 不太好
left1 = [0, 0, 0, 0, 1]
right1 = [0, 1, 1, 1, 1]
ig1 = entropy(y_parent) - (len(left1)/len(y_parent) * entropy(left1) + 
                            len(right1)/len(y_parent) * entropy(right1))

# 分割方案2: [0,0,0,0,0] 和 [1,1,1,1,1] → 完美！
left2 = [0, 0, 0, 0, 0]
right2 = [1, 1, 1, 1, 1]
ig2 = entropy(y_parent) - (len(left2)/len(y_parent) * entropy(left2) + 
                            len(right2)/len(y_parent) * entropy(right2))

print(f"父节点 (5:5): 熵 = {entropy(y_parent):.4f}")
print(f"\n方案1 分割为 [4:1] 和 [1:4]:")
print(f"  信息增益 = {ig1:.4f}")
print(f"\n方案2 分割为 [5:0] 和 [0:5] (完美分割):")
print(f"  信息增益 = {ig2:.4f}")
print(f"\n→ 方案2 的信息增益更大，所以选方案2！")


# ============================================================
# 3. 从零实现决策树
# ============================================================
print("\n" + "=" * 60)
print("3. 从零实现决策树")
print("=" * 60)

class DecisionTreeNode:
    """决策树节点"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        # 分割的特征索引
        self.threshold = threshold    # 分割阈值
        self.left = left             # 左子树 (≤ threshold)
        self.right = right           # 右子树 (> threshold)
        self.value = value           # 叶节点的预测值

class SimpleDecisionTree:
    """简单决策树分类器"""
    
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X, y, depth):
        n_samples = len(y)
        
        # 停止条件
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            # 叶节点：返回最多的类
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)
        
        # 找最佳分割
        best_feature, best_threshold, best_gini = None, None, float('inf')
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                
                # 加权基尼系数
                gini = (left_mask.sum() / n_samples * gini_impurity(y[left_mask]) +
                       right_mask.sum() / n_samples * gini_impurity(y[right_mask]))
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        
        if best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)
        
        # 递归构建子树
        left_mask = X[:, best_feature] <= best_threshold
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return DecisionTreeNode(feature=best_feature, threshold=best_threshold,
                               left=left_tree, right=right_tree)
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])
    
    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

# 使用鸢尾花数据集
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42)

print("--- 在鸢尾花数据集上训练 ---")
print(f"特征: {iris.feature_names}")
print(f"类别: {list(iris.target_names)}")
print(f"训练集: {len(X_train_i)} 样本, 测试集: {len(X_test_i)} 样本")

tree = SimpleDecisionTree(max_depth=4)
tree.fit(X_train_i, y_train_i)
y_pred_tree = tree.predict(X_test_i)
acc_tree = accuracy_score(y_test_i, y_pred_tree)
print(f"\n我们的决策树准确率: {acc_tree:.4f}")
assert acc_tree > 0.85
print("✓ 验证通过")


# ============================================================
# 4. 随机森林
# ============================================================
print("\n" + "=" * 60)
print("4. 随机森林（集成学习）")
print("=" * 60)

print("""
【集成学习的思想】
"三个臭皮匠，赛过诸葛亮"

一棵决策树容易过拟合，但很多棵"不完美"的树投票，
结果往往比一棵"完美"的树更好。

【随机森林 = 很多决策树 + 投票】

训练每棵树时的"随机性":
  1. Bootstrap 采样: 从训练集中有放回地随机抽取样本
     (每棵树看到的数据不同 → 树之间有差异)
  2. 随机特征子集: 每次分割时只考虑部分特征
     (进一步增加树之间的差异)

预测时：
  - 分类: 所有树投票，少数服从多数
  - 回归: 所有树预测值取平均

【为什么有效？】
  - 每棵树的误差是不同的（不同数据、不同特征）
  - 多棵树平均后，各自的随机误差相互抵消
  - 只有系统性的正确判断被保留
""")

class SimpleRandomForest:
    """简单随机森林"""
    
    def __init__(self, n_trees=10, max_depth=4, sample_ratio=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_ratio = sample_ratio
        self.trees = []
    
    def fit(self, X, y):
        n_samples = len(X)
        
        for _ in range(self.n_trees):
            # Bootstrap 采样
            indices = np.random.choice(n_samples, int(n_samples * self.sample_ratio), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            # 训练一棵树
            tree = SimpleDecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        # 所有树投票
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # 每个样本取众数
        final_pred = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            final_pred.append(Counter(votes).most_common(1)[0][0])
        return np.array(final_pred)

# 训练随机森林
print("\n--- 随机森林 vs 单棵决策树 ---")
np.random.seed(42)
rf = SimpleRandomForest(n_trees=20, max_depth=4)
rf.fit(X_train_i, y_train_i)
y_pred_rf = rf.predict(X_test_i)
acc_rf = accuracy_score(y_test_i, y_pred_rf)

print(f"单棵决策树准确率: {acc_tree:.4f}")
print(f"随机森林(20棵树)准确率: {acc_rf:.4f}")
print(f"提升: {(acc_rf - acc_tree)*100:.1f}%")


# ============================================================
# 5. sklearn 实现
# ============================================================
print("\n" + "=" * 60)
print("5. sklearn 实现")
print("=" * 60)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# sklearn 决策树
sk_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
sk_tree.fit(X_train_i, y_train_i)
sk_tree_acc = sk_tree.score(X_test_i, y_test_i)

# sklearn 随机森林
sk_rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
sk_rf.fit(X_train_i, y_train_i)
sk_rf_acc = sk_rf.score(X_test_i, y_test_i)

print(f"sklearn 决策树: {sk_tree_acc:.4f}")
print(f"sklearn 随机森林 (100棵): {sk_rf_acc:.4f}")

# 特征重要性
print(f"\n--- 特征重要性 ---")
importances = sk_rf.feature_importances_
for name, imp in sorted(zip(iris.feature_names, importances), key=lambda x: -x[1]):
    bar = '█' * int(imp * 40)
    print(f"  {name:<20s}: {imp:.4f} {bar}")

print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点：
1. 决策树通过 if-else 条件分割数据，直觉清晰
2. 分割标准：基尼系数或信息增益
3. 单棵树容易过拟合
4. 随机森林 = 多棵树 + 投票，效果更好更稳定
5. 随机森林不太需要调参，是很好的 baseline 模型

决策树 vs 神经网络：
  决策树: 可解释性强，适合表格数据
  神经网络: 效果更强，适合图像/文本/复杂数据

下一节：模型评估 → 如何正确评价一个模型
""")
