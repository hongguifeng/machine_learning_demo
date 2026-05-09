# AI 开发入门教程

## 面向软件工程师的 AI 学习路径

本教程从零开始，系统性地教授机器学习、深度学习和大语言模型的开发。

## 目录结构

### 第一章：数学基础
- [1.1 线性代数基础](01_math_foundations/01_linear_algebra.py) - 向量、矩阵、运算
- [1.2 微积分基础](01_math_foundations/02_calculus.py) - 导数、偏导数、链式法则
- [1.3 概率与统计](01_math_foundations/03_probability_statistics.py) - 概率分布、贝叶斯定理
- [1.4 优化基础](01_math_foundations/04_optimization.py) - 梯度下降、损失函数

### 第二章：Python AI 工具库
- [2.1 NumPy 基础](02_python_ai_tools/01_numpy_basics.py) - 数组操作、广播机制
- [2.2 Pandas 数据处理](02_python_ai_tools/02_pandas_basics.py) - 数据加载、清洗、分析
- [2.3 Matplotlib 可视化](02_python_ai_tools/03_matplotlib_basics.py) - 数据可视化

### 第三章：机器学习基础
- [3.1 线性回归](03_machine_learning/01_linear_regression.py) - 从零实现 + sklearn
- [3.2 逻辑回归](03_machine_learning/02_logistic_regression.py) - 二分类问题
- [3.3 决策树与随机森林](03_machine_learning/03_decision_tree.py) - 树模型
- [3.4 模型评估](03_machine_learning/04_model_evaluation.py) - 交叉验证、指标

### 第四章：深度学习基础
- [4.1 感知机与神经网络](04_deep_learning/01_perceptron_nn.py) - 从零实现神经网络
- [4.2 PyTorch 入门](04_deep_learning/02_pytorch_basics.py) - 张量、自动求导
- [4.3 卷积神经网络](04_deep_learning/03_cnn.py) - 图像分类
- [4.4 循环神经网络](04_deep_learning/04_rnn.py) - 序列处理

### 第五章：大语言模型
- [5.1 词嵌入与 Word2Vec](05_llm/01_word_embedding.py) - 词向量表示
- [5.2 注意力机制](05_llm/02_attention_mechanism.py) - Self-Attention 详解
- [5.3 Transformer 架构](05_llm/03_transformer.py) - 从零实现 Transformer
- [5.4 使用预训练模型](05_llm/04_pretrained_models.py) - Hugging Face 实践

### 第六章：强化学习
- [6.1 强化学习基础](06_reinforcement_learning/01_rl_basics.py) - MDP、价值迭代、Q-Learning
- [6.2 深度强化学习](06_reinforcement_learning/02_deep_rl.py) - DQN、策略梯度、PPO、RLHF

### 第七章：无监督学习与生成模型
- [7.1 聚类与降维](07_unsupervised_learning/01_clustering_dimensionality.py) - K-Means、DBSCAN、PCA、异常检测
- [7.2 生成模型](07_unsupervised_learning/02_generative_models.py) - 自编码器、VAE、GAN

### 第八章：ML 工程实践
- [8.1 特征工程](08_practical_engineering/01_feature_engineering.py) - 编码、缩放、选择、缺失值、不平衡
- [8.2 调优与部署](08_practical_engineering/02_tuning_deployment.py) - 超参数搜索、模型保存、量化、MLOps

## 环境准备

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 学习建议

1. 按顺序学习，每章内容都依赖前面的知识
2. 每个 .py 文件都可以直接运行，建议边看边跑
3. 修改代码中的参数，观察结果变化，加深理解
