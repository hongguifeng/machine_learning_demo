"""
第八章 8.1：特征工程与数据处理实战
===================================

"数据和特征决定了模型的上限，算法只是逼近这个上限"

本节内容：
1. 特征类型与编码
2. 特征缩放
3. 特征选择
4. 处理缺失值和不平衡数据
5. 实战: 完整的数据管道
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder, 
    OneHotEncoder, PolynomialFeatures
)
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer

print("=" * 60)
print("第八章 8.1：特征工程与数据处理实战")
print("=" * 60)

# ============================================================
# 1. 特征类型与编码
# ============================================================
print("\n" + "=" * 60)
print("1. 特征类型与编码")
print("=" * 60)

print("""
【特征类型】
1. 数值特征: 年龄(23), 收入(50000), 温度(25.3)
2. 分类特征: 性别(男/女), 颜色(红/绿/蓝), 城市(北京/上海)
3. 有序特征: 教育(小学<中学<大学), 评分(1-5星)
4. 文本特征: "这个产品很好"
5. 时间特征: 2024-01-15 14:30:00

【编码方法】
- 标签编码 (Label Encoding): 类别→数字 (北京=0, 上海=1, 广州=2)
  ⚠ 只适合有序特征! 否则模型会误认为"广州>上海>北京"

- 独热编码 (One-Hot Encoding): 每个类别一列
  北京=[1,0,0], 上海=[0,1,0], 广州=[0,0,1]
  ✓ 适合无序分类特征

- 目标编码 (Target Encoding): 用该类别的目标均值替换
  适合高基数分类特征 (如城市有上百个)
""")

# 示例数据
np.random.seed(42)
n = 200
data = pd.DataFrame({
    'age': np.random.randint(18, 65, n),
    'income': np.random.normal(50000, 15000, n).astype(int),
    'city': np.random.choice(['北京', '上海', '广州', '深圳'], n),
    'education': np.random.choice(['高中', '本科', '硕士', '博士'], n),
    'purchased': np.random.choice([0, 1], n, p=[0.6, 0.4])
})

print("\n--- 原始数据 ---")
print(data.head())
print(f"\n数据类型:")
print(f"  age: 数值")
print(f"  income: 数值")
print(f"  city: 分类(无序)")
print(f"  education: 有序")

# 独热编码 city
city_dummies = pd.get_dummies(data['city'], prefix='city')
print(f"\n独热编码 'city':")
print(city_dummies.head())

# 有序编码 education
edu_order = {'高中': 0, '本科': 1, '硕士': 2, '博士': 3}
data['education_encoded'] = data['education'].map(edu_order)
print(f"\n有序编码 'education':")
print(data[['education', 'education_encoded']].drop_duplicates().sort_values('education_encoded'))


# ============================================================
# 2. 特征缩放
# ============================================================
print("\n" + "=" * 60)
print("2. 特征缩放")
print("=" * 60)

print("""
【为什么要缩放？】
  age: 18~65
  income: 20000~80000
  
  如果不缩放，income 会主导距离计算和梯度更新。
  很多算法对特征尺度敏感: KNN, SVM, 线性回归, 神经网络
  不敏感的: 决策树, 随机森林

【常用方法】
  StandardScaler: z = (x - μ) / σ  → 均值0, 标准差1
  MinMaxScaler:   x' = (x - min) / (max - min) → [0, 1]
  
  选择:
    - 正态分布数据 → StandardScaler
    - 有界数据/神经网络 → MinMaxScaler
""")

X_num = data[['age', 'income']].values

# StandardScaler
ss = StandardScaler()
X_standard = ss.fit_transform(X_num)

# MinMaxScaler
mms = MinMaxScaler()
X_minmax = mms.fit_transform(X_num)

print(f"原始数据统计:")
print(f"  age:    均值={X_num[:,0].mean():.1f}, 范围=[{X_num[:,0].min()}, {X_num[:,0].max()}]")
print(f"  income: 均值={X_num[:,1].mean():.0f}, 范围=[{X_num[:,1].min()}, {X_num[:,1].max()}]")

print(f"\nStandardScaler 后:")
print(f"  age:    均值={X_standard[:,0].mean():.4f}, 标准差={X_standard[:,0].std():.4f}")
print(f"  income: 均值={X_standard[:,1].mean():.4f}, 标准差={X_standard[:,1].std():.4f}")

print(f"\nMinMaxScaler 后:")
print(f"  age:    范围=[{X_minmax[:,0].min():.2f}, {X_minmax[:,0].max():.2f}]")
print(f"  income: 范围=[{X_minmax[:,1].min():.2f}, {X_minmax[:,1].max():.2f}]")
print("✓ 缩放后特征在同一量级")


# ============================================================
# 3. 特征选择
# ============================================================
print("\n" + "=" * 60)
print("3. 特征选择")
print("=" * 60)

print("""
【为什么要特征选择？】
  - 去除无关特征 → 减少过拟合
  - 减少计算量
  - 提高可解释性

【方法】
1. 过滤法: 根据统计指标打分 (互信息、相关系数)
2. 包裹法: 用模型评估特征子集 (递归特征消除)
3. 嵌入法: 模型自带特征重要性 (随机森林, L1正则化)
""")

# 构造完整特征矩阵
X = np.hstack([
    X_num,
    city_dummies.values,
    data[['education_encoded']].values,
    np.random.randn(n, 3)  # 故意加3个无关特征
])
feature_names = (['age', 'income'] + 
                 list(city_dummies.columns) + 
                 ['education'] + 
                 ['noise_1', 'noise_2', 'noise_3'])
y = data['purchased'].values

# 互信息评分
mi_scores = mutual_info_classif(X, y, random_state=42)
print(f"\n互信息评分 (越高越相关):")
for name, score in sorted(zip(feature_names, mi_scores), key=lambda x: -x[1]):
    bar = '█' * int(score * 50)
    print(f"  {name:15s}: {score:.4f} {bar}")

# 随机森林特征重要性
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
print(f"\n随机森林特征重要性:")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])[:5]:
    bar = '█' * int(imp * 100)
    print(f"  {name:15s}: {imp:.4f} {bar}")

print("\n→ 噪声特征重要性低，可以安全删除")


# ============================================================
# 4. 处理缺失值和不平衡数据
# ============================================================
print("\n" + "=" * 60)
print("4. 处理缺失值和不平衡数据")
print("=" * 60)

print("""
【缺失值处理】
  - 删除: 缺失比例高的特征或样本直接删
  - 填充: 均值/中位数/众数/前后值
  - 模型预测: 用其他特征预测缺失值
  - 标记: 加一个"是否缺失"的二值特征

【不平衡数据】(如欺诈检测: 99%正常, 1%欺诈)
  问题: 模型倾向于预测多数类 (全预测"正常"就有99%准确率)
  
  解决方法:
  - 过采样少数类: SMOTE (合成新样本)
  - 欠采样多数类: 随机删除部分多数类
  - 类别权重: class_weight='balanced'
  - 阈值调整: 降低正类的判断阈值
  - 评估指标: 用 F1/AUC 而不是准确率
""")

# 模拟缺失值
data_with_missing = data.copy()
mask = np.random.random(n) < 0.15
data_with_missing.loc[mask, 'income'] = np.nan
print(f"模拟缺失值: income 列有 {mask.sum()} 个缺失")

# 填充策略
imputer_mean = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')

income_col = data_with_missing[['income']].values
filled_mean = imputer_mean.fit_transform(income_col)
filled_median = imputer_median.fit_transform(income_col)

print(f"  原始均值: {data['income'].mean():.0f}")
print(f"  均值填充后均值: {filled_mean.mean():.0f}")
print(f"  中位数填充后均值: {filled_median.mean():.0f}")

# 不平衡数据演示
print(f"\n--- 不平衡数据处理 ---")
np.random.seed(42)
X_imb = np.random.randn(1000, 5)
y_imb = np.zeros(1000)
y_imb[:50] = 1  # 只有5%是正类
X_imb[y_imb == 1] += 1.0  # 正类稍微偏移

X_tr, X_te, y_tr, y_te = train_test_split(X_imb, y_imb, test_size=0.3, random_state=42)

# 不处理
lr_basic = LogisticRegression(random_state=42)
lr_basic.fit(X_tr, y_tr)
f1_basic = f1_score(y_te, lr_basic.predict(X_te))

# 加权
lr_weighted = LogisticRegression(class_weight='balanced', random_state=42)
lr_weighted.fit(X_tr, y_tr)
f1_weighted = f1_score(y_te, lr_weighted.predict(X_te))

print(f"正类比例: {y_imb.mean():.1%}")
print(f"  不处理 F1: {f1_basic:.3f}")
print(f"  加权后 F1: {f1_weighted:.3f}")
if f1_weighted >= f1_basic:
    print("✓ class_weight='balanced' 改善了少数类识别")
else:
    print("  (本次随机数据中差异不大，实际不平衡场景效果显著)")
print("✓ 演示完成")


# ============================================================
# 5. 完整数据管道
# ============================================================
print("\n" + "=" * 60)
print("5. 完整数据管道示例")
print("=" * 60)

print("""
【实际项目的数据处理流程】
  1. 探索性数据分析 (EDA)
  2. 处理缺失值
  3. 特征编码 (分类→数值)
  4. 特征缩放
  5. 特征工程 (创建新特征)
  6. 特征选择
  7. 划分数据集
  
  ⚠ 重要: 缩放/编码只在训练集上 fit，然后 transform 测试集！
  否则有信息泄露 (data leakage)
""")

# 完整管道示例
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

print("使用 sklearn Pipeline 构建标准化流程:")
print("""
  pipeline = Pipeline([
      ('preprocessor', ColumnTransformer([
          ('num', StandardScaler(), numerical_features),
          ('cat', OneHotEncoder(), categorical_features),
      ])),
      ('classifier', LogisticRegression())
  ])
  
  pipeline.fit(X_train, y_train)      # 一步完成预处理+训练
  pipeline.predict(X_test)            # 自动对测试集做相同预处理
""")

# 实际演示
X_full = data[['age', 'income', 'education_encoded']].values
y_full = data['purchased'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42))
])

pipe.fit(X_train, y_train)
acc = pipe.score(X_test, y_test)
print(f"\nPipeline 结果: 准确率={acc:.3f}")
print("✓ Pipeline 确保预处理步骤不会泄露测试集信息")


print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点:
1. 特征编码: 无序用OneHot, 有序用数值映射
2. 特征缩放: StandardScaler/MinMaxScaler, 对距离型算法必须
3. 特征选择: 互信息/随机森林重要性, 去掉噪声特征
4. 缺失值: 均值/中位数填充, 或加"是否缺失"标记
5. 不平衡: class_weight='balanced' 或 SMOTE
6. Pipeline: 防止数据泄露, 确保训练/测试一致

实战经验:
  - 特征工程通常比模型选择更重要
  - 始终检查是否有数据泄露
  - 先用简单模型建立 baseline, 再逐步改进
""")
