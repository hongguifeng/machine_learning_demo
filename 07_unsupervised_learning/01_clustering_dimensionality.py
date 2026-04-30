"""
第七章 7.1：无监督学习
====================

无监督学习 = 从无标注数据中发现隐藏结构

应用: 聚类分析、降维可视化、异常检测、数据压缩

本节内容：
1. 聚类 (K-Means, DBSCAN)
2. 降维 (PCA, t-SNE)
3. 异常检测
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

print("=" * 60)
print("第七章 7.1：无监督学习")
print("=" * 60)

# ============================================================
# 1. K-Means 聚类
# ============================================================
print("\n" + "=" * 60)
print("1. K-Means 聚类")
print("=" * 60)

print("""
【什么是聚类？】
把数据分成若干组，组内相似、组间不同。
不需要标签！纯粹根据数据的结构来分。

【K-Means 算法】
  输入: 数据点, K (簇数)
  
  1. 随机初始化 K 个聚类中心
  2. 重复直到收敛:
     a. 分配: 每个点归属最近的中心
     b. 更新: 重新计算每个簇的中心 (均值)
  
  收敛条件: 中心不再移动

【K-Means 的数学目标】
  最小化簇内平方和 (Inertia):
  J = Σᵢ Σ_{x∈Cᵢ} ||x - μᵢ||²

【选择 K 值: 肘部法 (Elbow Method)】
  尝试不同的 K，画 K vs Inertia 曲线
  找到"拐点" = 合适的 K
""")

# 从零实现 K-Means
def kmeans_from_scratch(X, k, max_iters=100, seed=42):
    """从零实现 K-Means"""
    np.random.seed(seed)
    n_samples = X.shape[0]
    
    # 随机初始化中心
    indices = np.random.choice(n_samples, k, replace=False)
    centers = X[indices].copy()
    
    for iteration in range(max_iters):
        # 分配: 计算每个点到每个中心的距离
        distances = np.sqrt(((X[:, np.newaxis] - centers[np.newaxis, :]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        
        # 更新: 计算新中心
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # 收敛检查
        if np.allclose(centers, new_centers):
            print(f"  K-Means 在第 {iteration+1} 次迭代收敛")
            break
        centers = new_centers
    
    return labels, centers

# 生成测试数据
np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

# 从零实现
labels_scratch, centers_scratch = kmeans_from_scratch(X, k=3)

# sklearn 实现
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_sklearn = kmeans.fit_predict(X)

# 评估
ari_scratch = adjusted_rand_score(y_true, labels_scratch)
ari_sklearn = adjusted_rand_score(y_true, labels_sklearn)
sil_score = silhouette_score(X, labels_sklearn)

print(f"\n--- K-Means 聚类结果 ---")
print(f"数据: 300个点, 3个真实簇")
print(f"从零实现 ARI: {ari_scratch:.3f}")
print(f"sklearn ARI:  {ari_sklearn:.3f}")
print(f"轮廓系数:     {sil_score:.3f} (越接近1越好)")
assert ari_sklearn > 0.8
print("✓ 聚类效果良好")

# 肘部法选 K
print(f"\n--- 肘部法选择 K ---")
inertias = []
for k in range(1, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
    print(f"  K={k}: Inertia={km.inertia_:.1f}")
print("→ K=3 处有明显拐点 (Inertia 下降变缓)")


# ============================================================
# 2. DBSCAN
# ============================================================
print("\n" + "=" * 60)
print("2. DBSCAN (密度聚类)")
print("=" * 60)

print("""
【K-Means 的局限】
  - 需要预先指定 K
  - 只能发现球形簇
  - 对异常点敏感

【DBSCAN: 基于密度的聚类】
  核心思想: "密集区域是一个簇，稀疏区域是分隔"
  
  参数:
    eps: 邻域半径
    min_samples: 成为核心点所需的最少邻居数
  
  优点:
    - 不需要指定簇数
    - 能发现任意形状的簇
    - 自动识别异常点 (噪声)
""")

# 月牙形数据 (K-Means 处理不好的)
X_moon, y_moon = make_moons(n_samples=200, noise=0.1, random_state=42)

# K-Means 失败
km_moon = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_km = km_moon.fit_predict(X_moon)
ari_km = adjusted_rand_score(y_moon, labels_km)

# DBSCAN 成功
db = DBSCAN(eps=0.2, min_samples=5)
labels_db = db.fit_predict(X_moon)
ari_db = adjusted_rand_score(y_moon, labels_db)

print(f"\n月牙形数据聚类对比:")
print(f"  K-Means ARI: {ari_km:.3f} (效果差)")
print(f"  DBSCAN ARI:  {ari_db:.3f} (效果好!)")
print(f"  DBSCAN 发现的簇数: {len(set(labels_db) - {-1})}")
print(f"  噪声点数: {(labels_db == -1).sum()}")
assert ari_db > ari_km
print("✓ DBSCAN 在非球形数据上优于 K-Means")


# ============================================================
# 3. PCA 降维
# ============================================================
print("\n" + "=" * 60)
print("3. PCA (主成分分析)")
print("=" * 60)

print("""
【为什么要降维？】
  - 高维数据难以可视化
  - 减少噪声和冗余
  - 加速后续算法
  - 发现数据的主要结构

【PCA 的原理】
  找到数据方差最大的方向（主成分）
  
  步骤:
  1. 中心化数据 (减去均值)
  2. 计算协方差矩阵
  3. 求特征值和特征向量
  4. 取最大的 k 个特征值对应的特征向量
  5. 投影: X_new = X @ W_k

  数学:
    Cov = (1/n) × Xᵀ × X
    特征分解: Cov × v = λ × v
    λ (特征值) = 该方向上的方差
    v (特征向量) = 主成分方向
""")

# 从零实现 PCA
def pca_from_scratch(X, n_components):
    """从零实现 PCA"""
    # 1. 中心化
    mean = X.mean(axis=0)
    X_centered = X - mean
    
    # 2. 协方差矩阵
    cov_matrix = np.cov(X_centered.T)
    
    # 3. 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. 按特征值从大到小排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 5. 取前 k 个
    W = eigenvectors[:, :n_components]
    
    # 6. 投影
    X_projected = X_centered @ W
    
    # 方差解释比例
    explained_variance_ratio = eigenvalues[:n_components] / eigenvalues.sum()
    
    return X_projected, explained_variance_ratio

# 测试: 5维数据降到2维
np.random.seed(42)
X_5d = np.random.randn(200, 5)
# 让前2维有更多信息
X_5d[:, 0] *= 5
X_5d[:, 1] *= 3

X_2d, var_ratio = pca_from_scratch(X_5d, n_components=2)

print(f"\n--- PCA 降维结果 ---")
print(f"原始维度: {X_5d.shape[1]}")
print(f"降维后:   {X_2d.shape[1]}")
print(f"各主成分解释的方差比例: {var_ratio.round(3)}")
print(f"前2个主成分累积解释: {var_ratio.sum():.1%}")

# 与 sklearn 对比
pca_sk = PCA(n_components=2)
X_2d_sk = pca_sk.fit_transform(X_5d)
print(f"\nsklearn PCA 解释方差: {pca_sk.explained_variance_ratio_.round(3)}")
assert abs(var_ratio[0] - pca_sk.explained_variance_ratio_[0]) < 0.01
print("✓ 从零实现与 sklearn 结果一致")


# ============================================================
# 4. 异常检测
# ============================================================
print("\n" + "=" * 60)
print("4. 异常检测")
print("=" * 60)

print("""
【异常检测的方法】

1. 基于统计: 假设正常数据符合某个分布，偏离的就是异常
2. 基于距离: 离大多数点远的就是异常 (如 KNN 距离)
3. 基于密度: 密度比邻居低的就是异常 (如 Local Outlier Factor)
4. 基于重构: 自编码器重构误差大的就是异常

这里演示简单的基于统计方法 (Z-score)
""")

# 生成正常数据 + 注入异常
np.random.seed(42)
normal_data = np.random.randn(200, 2)  # 正常: 标准正态
anomalies = np.random.randn(10, 2) * 0.5 + 4  # 异常: 偏移
X_all = np.vstack([normal_data, anomalies])
y_all = np.array([0]*200 + [1]*10)  # 0=正常, 1=异常

# 方法1: Z-score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)
z_scores = np.sqrt((X_scaled ** 2).sum(axis=1))
threshold = 3.0  # Z-score > 3 视为异常
predictions = (z_scores > threshold).astype(int)

# 评估
from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_all, predictions)
recall = recall_score(y_all, predictions)

print(f"\n--- Z-score 异常检测 ---")
print(f"数据: 200正常 + 10异常")
print(f"阈值: Z-score > {threshold}")
print(f"检出异常数: {predictions.sum()}")
print(f"精确率: {precision:.3f}")
print(f"召回率: {recall:.3f}")
assert recall > 0.5
print("✓ 异常检测有效")


print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点:
1. K-Means: 简单高效, 球形簇, 需指定K, 用肘部法选K
2. DBSCAN: 基于密度, 任意形状, 自动识别噪声
3. PCA: 找方差最大的方向, 降维+去噪+可视化
4. 异常检测: 偏离正常分布的点

无监督学习的应用场景:
  - 客户分群 (聚类)
  - 数据可视化 (降维)
  - 欺诈检测 (异常检测)
  - 推荐系统 (发现用户/物品的潜在结构)

下一节: 生成模型 → 自编码器和 GAN
""")
