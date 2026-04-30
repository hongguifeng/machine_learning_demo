"""
第八章 8.2：超参数调优与模型部署
=================================

调出好模型 + 把模型用起来

本节内容：
1. 超参数调优方法
2. 模型保存与加载
3. 模型部署基础 (推理优化)
4. MLOps 核心概念
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import json
import os
import time

print("=" * 60)
print("第八章 8.2：超参数调优与模型部署")
print("=" * 60)

# ============================================================
# 1. 超参数调优
# ============================================================
print("\n" + "=" * 60)
print("1. 超参数调优")
print("=" * 60)

print("""
【超参数 vs 模型参数】
  模型参数: 训练过程中学到的 (权重 w, 偏置 b)
  超参数:   训练前人工设定的 (学习率, 层数, 正则化强度)

【调优方法】

1. 网格搜索 (Grid Search):
   穷举所有超参数组合
   + 保证找到搜索范围内最优
   - 组合爆炸，太慢

2. 随机搜索 (Random Search):
   随机采样超参数组合
   + 效率高，特别是某些参数不重要时
   + 相同预算下通常比网格搜索好

3. 贝叶斯优化 (Bayesian Optimization):
   用高斯过程建模"超参数→性能"的关系
   根据已有结果，智能选择下一个要尝试的组合
   + 最高效
   - 实现复杂 (用 optuna 库)

4. 经验法则:
   - 学习率: 从 0.001 开始
   - batch_size: 32, 64, 128
   - 神经网络层数: 从小到大试
   - 正则化: L2=0.01~0.0001
""")

# 生成数据
X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 网格搜索
print("\n--- 网格搜索 (Grid Search) ---")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

start = time.time()
grid_search.fit(X_train, y_train)
grid_time = time.time() - start

print(f"搜索空间: {3*4*3} = 36 种组合")
print(f"最优参数: {grid_search.best_params_}")
print(f"最优CV分数: {grid_search.best_score_:.4f}")
print(f"测试集分数: {grid_search.score(X_test, y_test):.4f}")
print(f"耗时: {grid_time:.1f}s")

# 随机搜索
print("\n--- 随机搜索 (Random Search) ---")
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=20,  # 只试20次
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

start = time.time()
random_search.fit(X_train, y_train)
rand_time = time.time() - start

print(f"搜索次数: 20 (从无限空间中采样)")
print(f"最优参数: {random_search.best_params_}")
print(f"最优CV分数: {random_search.best_score_:.4f}")
print(f"测试集分数: {random_search.score(X_test, y_test):.4f}")
print(f"耗时: {rand_time:.1f}s")

print(f"\n对比: 随机搜索用更少的尝试达到了相近(甚至更好)的结果")


# ============================================================
# 2. 模型保存与加载
# ============================================================
print("\n" + "=" * 60)
print("2. 模型保存与加载")
print("=" * 60)

print("""
【为什么要保存模型？】
  训练一个模型可能要几小时/天/周
  不能每次用的时候都重新训练！

【常见格式】
  - pickle/joblib: sklearn 模型
  - .pt/.pth: PyTorch 模型
  - ONNX: 跨框架标准格式
  - SavedModel: TensorFlow

【PyTorch 模型保存最佳实践】
  推荐保存 state_dict (只保存参数):
    torch.save(model.state_dict(), 'model.pth')
    
  加载时需要先创建模型结构:
    model = MyModel()
    model.load_state_dict(torch.load('model.pth'))
""")

# 保存 sklearn 模型
import pickle

best_model = grid_search.best_estimator_
model_path = '/tmp/rf_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

# 加载并验证
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

pred_original = best_model.predict(X_test)
pred_loaded = loaded_model.predict(X_test)
assert np.array_equal(pred_original, pred_loaded)
print("✓ sklearn 模型保存/加载: 预测结果一致")

# 保存 PyTorch 模型
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = SimpleNN()
model_pt_path = '/tmp/nn_model.pth'

# 保存
torch.save(model.state_dict(), model_pt_path)
print(f"模型文件大小: {os.path.getsize(model_pt_path)} bytes")

# 加载
model_loaded = SimpleNN()
model_loaded.load_state_dict(torch.load(model_pt_path, weights_only=True))
model_loaded.eval()

# 验证
x_test_tensor = torch.randn(5, 20)
with torch.no_grad():
    out1 = model(x_test_tensor)
    out2 = model_loaded(x_test_tensor)
assert torch.allclose(out1, out2)
print("✓ PyTorch 模型保存/加载: 输出一致")

# 清理临时文件
os.remove(model_path)
os.remove(model_pt_path)


# ============================================================
# 3. 推理优化
# ============================================================
print("\n" + "=" * 60)
print("3. 推理优化")
print("=" * 60)

print("""
【推理优化技术】

1. 量化 (Quantization):
   FP32 → INT8: 模型大小减少4倍，速度提升2-4倍
   精度损失通常可接受 (<1%)
   
   PyTorch:
     model_int8 = torch.quantization.quantize_dynamic(
         model, {nn.Linear}, dtype=torch.qint8)

2. 知识蒸馏 (Knowledge Distillation):
   大模型(Teacher) → 训练小模型(Student)
   Student 学习 Teacher 的"软标签"(概率分布)
   小模型获得接近大模型的性能

3. 剪枝 (Pruning):
   去除接近零的权重
   结构化剪枝: 去除整个神经元/通道
   非结构化剪枝: 去除单个权重

4. ONNX 导出:
   统一模型格式，部署到各种推理引擎
   torch.onnx.export(model, dummy_input, "model.onnx")

5. 批处理推理:
   一次预测多个样本，充分利用 GPU 并行性
""")

# 量化示例
model_fp32 = SimpleNN()
model_fp32.eval()

model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {nn.Linear}, dtype=torch.qint8
)

# 比较大小
import io
def get_model_size(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()

size_fp32 = get_model_size(model_fp32)
size_int8 = get_model_size(model_int8)

print(f"\n--- 动态量化示例 ---")
print(f"FP32 模型大小: {size_fp32:,} bytes")
print(f"INT8 模型大小: {size_int8:,} bytes")
print(f"压缩比: {size_fp32/size_int8:.1f}x")

# 验证精度
x_sample = torch.randn(100, 20)
with torch.no_grad():
    out_fp32 = model_fp32(x_sample)
    out_int8 = model_int8(x_sample)
    diff = (out_fp32 - out_int8).abs().mean().item()
print(f"平均输出差异: {diff:.6f} (非常小)")
print("✓ 量化显著减小模型大小，精度损失极小")


# ============================================================
# 4. MLOps 核心概念
# ============================================================
print("\n" + "=" * 60)
print("4. MLOps 核心概念")
print("=" * 60)

print("""
【什么是 MLOps？】
MLOps = Machine Learning + DevOps
让机器学习模型可靠地部署和运维

【ML 项目生命周期】
  数据收集 → 数据处理 → 特征工程 → 模型训练 
  → 模型评估 → 模型部署 → 监控 → 迭代

【核心实践】

1. 实验追踪 (Experiment Tracking):
   记录每次实验的超参数、指标、代码版本
   工具: MLflow, Weights & Biases (W&B), TensorBoard

2. 数据版本控制:
   数据和代码一样需要版本管理
   工具: DVC (Data Version Control)

3. 模型注册 (Model Registry):
   管理模型的不同版本，标记哪个是生产版
   工具: MLflow Model Registry

4. CI/CD for ML:
   代码变更 → 自动训练 → 自动评估 → 自动部署
   包含数据验证和模型质量门控

5. 模型监控 (Model Monitoring):
   - 数据漂移 (Data Drift): 输入分布变了
   - 概念漂移 (Concept Drift): 输入-输出关系变了
   - 性能下降: 模型准确率在下降

6. A/B 测试:
   新旧模型同时上线，对比真实效果

【模型部署方式】
  - REST API: Flask/FastAPI 包装模型
  - 批处理: 定时跑一批预测
  - 嵌入式: 模型打包到移动端
  - Serverless: AWS Lambda / Cloud Functions
""")

# 模拟实验追踪
print("\n--- 模拟实验追踪 ---")
experiments = []

for n_est in [50, 100, 200]:
    for max_d in [5, 10, None]:
        model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        
        exp = {
            'n_estimators': n_est,
            'max_depth': max_d,
            'cv_mean': round(scores.mean(), 4),
            'cv_std': round(scores.std(), 4)
        }
        experiments.append(exp)

# 保存实验记录
exp_path = '/tmp/experiments.json'
with open(exp_path, 'w') as f:
    json.dump(experiments, f, indent=2)

# 找最优
best_exp = max(experiments, key=lambda x: x['cv_mean'])
print(f"实验总数: {len(experiments)}")
print(f"最优配置: n_estimators={best_exp['n_estimators']}, max_depth={best_exp['max_depth']}")
print(f"最优CV分数: {best_exp['cv_mean']:.4f} ± {best_exp['cv_std']:.4f}")
print(f"\n实验记录已保存 (生产环境用 MLflow/W&B)")

os.remove(exp_path)

# 模拟简单 API 部署
print(f"\n--- 模型 API 部署示例 (FastAPI) ---")
print("""
# model_api.py
from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# 加载模型
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(features: list[float]):
    X = np.array(features).reshape(1, -1)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0].tolist()
    return {
        "prediction": int(prediction),
        "probability": probability
    }

# 运行: uvicorn model_api:app --host 0.0.0.0 --port 8000
# 调用: curl -X POST "http://localhost:8000/predict" \\
#        -H "Content-Type: application/json" \\
#        -d '{"features": [1.0, 2.0, ...]}'
""")
print("✓ 以上是一个完整的模型部署 API 示例")


print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点:
1. 超参数调优: 随机搜索 > 网格搜索, 贝叶斯优化最优
2. 模型保存: sklearn用pickle, PyTorch用state_dict
3. 推理优化: 量化(4x压缩), 蒸馏, 剪枝, ONNX
4. MLOps: 实验追踪 + 版本控制 + 监控 + CI/CD

工程师必知:
  - 训练完模型只是开始，部署和维护才是大头
  - 模型会退化，需要持续监控和更新
  - 用 Pipeline 确保训练和推理预处理一致
  - 记录每个实验的完整配置（可复现性）
""")
