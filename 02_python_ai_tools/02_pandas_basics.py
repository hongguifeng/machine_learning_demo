"""
第二章 2.2：Pandas 数据处理基础
============================

Pandas 是 Python 中处理表格数据的利器。
在机器学习中，数据预处理通常占工作量的 60-80%，Pandas 是必备工具。

本节内容：
1. DataFrame 基础
2. 数据加载与查看
3. 数据选择与过滤
4. 缺失值处理
5. 数据转换
6. 实战：为机器学习准备数据
"""

import numpy as np
import pandas as pd

print("=" * 60)
print("第二章 2.2：Pandas 数据处理基础")
print("=" * 60)

# ============================================================
# 1. DataFrame 基础
# ============================================================
print("\n" + "=" * 60)
print("1. DataFrame 基础")
print("=" * 60)

print("""
【核心概念】
- Series: 一维数据（一列），相当于带标签的数组
- DataFrame: 二维表格数据（多列），AI 中最常用

可以把 DataFrame 想象成 Excel 表格或 SQL 表。
""")

# 创建 DataFrame
data = {
    '姓名': ['张三', '李四', '王五', '赵六', '钱七'],
    '年龄': [25, 30, 35, 28, 32],
    '身高': [175, 168, 182, 170, 178],
    '体重': [70, 55, 85, 65, 75],
    '城市': ['北京', '上海', '北京', '广州', '上海']
}
df = pd.DataFrame(data)
print("创建的 DataFrame:")
print(df)
print(f"\n形状: {df.shape} (5行5列)")
print(f"列名: {list(df.columns)}")
print(f"数据类型:\n{df.dtypes}")

# ============================================================
# 2. 数据查看
# ============================================================
print("\n" + "=" * 60)
print("2. 数据查看")
print("=" * 60)

# 生成更大的模拟数据集
np.random.seed(42)
n = 100
large_df = pd.DataFrame({
    '年龄': np.random.randint(18, 65, n),
    '收入': np.random.normal(50000, 15000, n).astype(int),
    '教育年限': np.random.randint(9, 22, n),
    '工作经验': np.random.randint(0, 30, n),
    '满意度': np.random.choice(['高', '中', '低'], n),
})
# 添加一些缺失值
large_df.loc[np.random.choice(n, 5), '收入'] = np.nan
large_df.loc[np.random.choice(n, 3), '教育年限'] = np.nan

print("large_df.head() - 前5行:")
print(large_df.head())

print("\nlarge_df.tail(3) - 后3行:")
print(large_df.tail(3))

print("\nlarge_df.info():")
print(large_df.info())

print("\nlarge_df.describe() - 统计摘要:")
print(large_df.describe())

# ============================================================
# 3. 数据选择与过滤
# ============================================================
print("\n" + "=" * 60)
print("3. 数据选择与过滤")
print("=" * 60)

print("--- 选择列 ---")
print(f"df['年龄']:\n{df['年龄'].values}")
print(f"\ndf[['姓名', '年龄']]:")
print(df[['姓名', '年龄']])

print("\n--- 选择行 ---")
print(f"df.iloc[0] (按位置选第0行):\n{df.iloc[0].values}")
print(f"\ndf.iloc[1:3] (第1-2行):")
print(df.iloc[1:3])

print("\n--- 条件过滤 ---")
young = df[df['年龄'] < 30]
print(f"年龄 < 30 的人:")
print(young)

tall_heavy = df[(df['身高'] > 175) & (df['体重'] > 70)]
print(f"\n身高>175 且 体重>70:")
print(tall_heavy)

beijing = df[df['城市'] == '北京']
print(f"\n北京的人:")
print(beijing)


# ============================================================
# 4. 缺失值处理
# ============================================================
print("\n" + "=" * 60)
print("4. 缺失值处理")
print("=" * 60)

print("""
【缺失值在 AI 中的重要性】
真实数据几乎总有缺失值！
处理方式：
  1. 删除含缺失值的行 → 简单但可能丢失太多数据
  2. 填充（均值/中位数/众数）→ 最常用
  3. 用模型预测缺失值 → 复杂但效果好
""")

print(f"缺失值统计:")
print(large_df.isnull().sum())

# 方法1: 删除
df_dropped = large_df.dropna()
print(f"\n删除缺失值后: {len(df_dropped)} 行 (原来 {len(large_df)} 行)")

# 方法2: 填充
df_filled = large_df.copy()
df_filled['收入'] = df_filled['收入'].fillna(df_filled['收入'].median())
df_filled['教育年限'] = df_filled['教育年限'].fillna(df_filled['教育年限'].mean())
print(f"\n填充后缺失值:")
print(df_filled.isnull().sum())
assert df_filled.isnull().sum().sum() == 0
print("✓ 所有缺失值已处理")


# ============================================================
# 5. 数据转换
# ============================================================
print("\n" + "=" * 60)
print("5. 数据转换")
print("=" * 60)

print("--- 新增列 ---")
df['BMI'] = df['体重'] / (df['身高']/100) ** 2
print(df[['姓名', '身高', '体重', 'BMI']].round(1))

print("\n--- 类别编码（One-Hot Encoding）---")
print("""
机器学习模型需要数字输入，不能直接处理文字。
One-Hot 编码：每个类别变成一列 (0/1)

  城市='北京' → [1, 0, 0]
  城市='上海' → [0, 1, 0]
  城市='广州' → [0, 0, 1]
""")
city_dummies = pd.get_dummies(df['城市'], prefix='城市')
print(city_dummies)

print("\n--- 分箱（Binning）---")
df_filled['收入等级'] = pd.cut(df_filled['收入'], 
                              bins=[0, 30000, 50000, 70000, float('inf')],
                              labels=['低', '中', '中高', '高'])
print(df_filled['收入等级'].value_counts())

print("\n--- Apply 自定义函数 ---")
df['体重类别'] = df['BMI'].apply(lambda x: '偏瘦' if x < 18.5 else ('正常' if x < 25 else '偏胖'))
print(df[['姓名', 'BMI', '体重类别']].round(1))


# ============================================================
# 6. 实战：为机器学习准备数据
# ============================================================
print("\n" + "=" * 60)
print("6. 实战：为机器学习准备数据")
print("=" * 60)

print("""
完整的数据准备流程：
1. 加载数据 → 查看概况
2. 处理缺失值
3. 特征工程（创建/转换特征）
4. 编码分类变量
5. 标准化/归一化数值特征
6. 分割训练集/测试集
""")

# 创建一个模拟的"是否购买"数据集
np.random.seed(42)
n = 200
dataset = pd.DataFrame({
    '年龄': np.random.randint(18, 60, n),
    '月收入': np.random.normal(8000, 3000, n).astype(int),
    '网站浏览时间_分钟': np.random.exponential(10, n).round(1),
    '历史购买次数': np.random.poisson(3, n),
    '性别': np.random.choice(['男', '女'], n),
    '会员等级': np.random.choice(['普通', '银卡', '金卡'], n, p=[0.6, 0.3, 0.1]),
})
# 创建目标变量（是否购买）- 基于特征的逻辑
buy_prob = 1 / (1 + np.exp(-(
    0.02 * (dataset['年龄'] - 30) + 
    0.0003 * (dataset['月收入'] - 8000) + 
    0.05 * dataset['网站浏览时间_分钟'] +
    0.1 * dataset['历史购买次数'] - 1
)))
dataset['是否购买'] = (np.random.random(n) < buy_prob).astype(int)

print("原始数据集前5行:")
print(dataset.head())
print(f"\n数据形状: {dataset.shape}")
print(f"目标变量分布:\n{dataset['是否购买'].value_counts()}")

# 步骤 1: 分离特征和标签
y = dataset['是否购买'].values
X_df = dataset.drop('是否购买', axis=1)

# 步骤 2: 编码分类变量
X_encoded = pd.get_dummies(X_df, columns=['性别', '会员等级'])
print(f"\nOne-Hot 编码后的列: {list(X_encoded.columns)}")

# 步骤 3: 标准化数值特征
numeric_cols = ['年龄', '月收入', '网站浏览时间_分钟', '历史购买次数']
X_final = X_encoded.copy()
for col in numeric_cols:
    mean = X_final[col].mean()
    std = X_final[col].std()
    X_final[col] = (X_final[col] - mean) / std

print(f"\n标准化后的数值特征统计:")
print(X_final[numeric_cols].describe().round(3))

# 步骤 4: 转为 NumPy 数组
X = X_final.values.astype(np.float64)
print(f"\n最终特征矩阵形状: {X.shape}")
print(f"标签数组形状: {y.shape}")
print(f"特征示例 (第一行): {X[0].round(3)}")

# 步骤 5: 分割训练集/测试集
def train_test_split(X, y, test_ratio=0.2, random_seed=42):
    """手动实现训练集/测试集分割"""
    np.random.seed(random_seed)
    n = len(X)
    indices = np.random.permutation(n)
    test_size = int(n * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split(X, y)
print(f"\n训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")
print(f"训练集正例比例: {y_train.mean():.2%}")
print(f"测试集正例比例: {y_test.mean():.2%}")

print("\n✓ 数据准备完成！可以输入机器学习模型了。")

print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点：
1. DataFrame 是处理表格数据的核心工具
2. 真实数据需要处理缺失值、异常值
3. 分类变量需要编码为数字（One-Hot）
4. 数值特征需要标准化（不同量纲影响模型）
5. 数据要分割为训练集和测试集

实际工作中的比例：
  数据准备: 60-80% 的时间
  模型训练: 10-20% 的时间
  模型评估: 10-20% 的时间

下一节：Matplotlib 可视化 → 直观理解数据和模型
""")
