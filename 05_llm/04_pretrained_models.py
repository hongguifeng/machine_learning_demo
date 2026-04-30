"""
第五章 5.4：使用预训练模型 (Hugging Face)
=========================================

在实际工作中，我们很少从零训练大模型。
而是使用预训练好的模型进行微调或直接推理。

Hugging Face 是最流行的预训练模型平台。

本节内容：
1. Hugging Face 生态介绍
2. Tokenizer 详解
3. 使用预训练模型做文本分类
4. 文本生成
5. 微调 (Fine-tuning) 基础
"""

import numpy as np
import torch
import torch.nn as nn

print("=" * 60)
print("第五章 5.4：使用预训练模型 (Hugging Face)")
print("=" * 60)

# ============================================================
# 1. Hugging Face 生态
# ============================================================
print("\n" + "=" * 60)
print("1. Hugging Face 生态介绍")
print("=" * 60)

print("""
【Hugging Face 核心组件】

1. transformers 库:
   - 提供数千个预训练模型
   - 统一的 API: AutoModel, AutoTokenizer
   - 支持 PyTorch 和 TensorFlow

2. datasets 库:
   - 大量现成的数据集
   - 高效的数据加载

3. Model Hub (huggingface.co):
   - 社区分享的模型
   - 模型卡片、使用示例

【常用模型家族】
- BERT: 双向编码器，擅长理解任务
- GPT-2/3: 单向解码器，擅长生成
- T5: 编码器-解码器，万能模型
- LLaMA: Meta 的开源大模型

【使用方式】
  from transformers import AutoTokenizer, AutoModel
  
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  model = AutoModel.from_pretrained("bert-base-uncased")
""")


# ============================================================
# 2. Tokenizer 详解
# ============================================================
print("\n" + "=" * 60)
print("2. Tokenizer (分词器) 详解")
print("=" * 60)

print("""
【为什么需要 Tokenizer？】
模型只能处理数字，不能直接处理文字。
Tokenizer 的任务: 文本 → 数字序列

【分词策略的演变】
1. 按词分词: "I love AI" → ["I", "love", "AI"]
   问题: 词表太大，无法处理未见过的词

2. 按字符分词: "love" → ["l", "o", "v", "e"]
   问题: 序列太长，丢失词级信息

3. 子词分词 (Subword): 当前主流！
   "unhappiness" → ["un", "happiness"]
   "playing" → ["play", "ing"]
   
   BPE (Byte Pair Encoding): GPT 使用
   WordPiece: BERT 使用
   SentencePiece: LLaMA/T5 使用
""")

# 简单的 BPE Tokenizer 演示
print("\n--- 简单 BPE 演示 ---")

class SimpleBPE:
    """简化版 BPE Tokenizer"""
    
    def __init__(self):
        self.vocab = {}
        self.merges = []
    
    def train(self, text, num_merges=10):
        """训练 BPE"""
        # 初始化: 每个字符是一个 token
        words = text.split()
        # 用空格分隔字符，加上 </w> 表示词尾
        word_freqs = {}
        for word in words:
            chars = ' '.join(list(word)) + ' </w>'
            word_freqs[chars] = word_freqs.get(chars, 0) + 1
        
        print(f"初始词表:")
        for word, freq in list(word_freqs.items())[:5]:
            print(f"  '{word}': {freq}")
        
        for i in range(num_merges):
            # 统计所有相邻 pair 的频率
            pairs = {}
            for word, freq in word_freqs.items():
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pair = (symbols[j], symbols[j+1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            
            if not pairs:
                break
            
            # 找最频繁的 pair
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            
            # 合并该 pair
            new_word_freqs = {}
            bigram = ' '.join(best_pair)
            replacement = ''.join(best_pair)
            
            for word, freq in word_freqs.items():
                new_word = word.replace(bigram, replacement)
                new_word_freqs[new_word] = freq
            
            word_freqs = new_word_freqs
            
            if i < 5:
                print(f"\n合并 #{i+1}: '{best_pair[0]}' + '{best_pair[1]}' → '{replacement}'")
        
        print(f"\n最终词表示例:")
        for word, freq in list(word_freqs.items())[:5]:
            print(f"  '{word}': {freq}")
        
        return word_freqs

# 训练 BPE
text = "low lower lowest new newer newest"
text = (text + " ") * 10  # 重复增加频率
print(f"训练文本: 'low lower lowest new newer newest' (重复10次)")
bpe = SimpleBPE()
result = bpe.train(text, num_merges=8)

print(f"\n【关键理解】")
print(f"BPE 通过反复合并最频繁的相邻字符对来构建子词词表")
print(f"'low', 'lower', 'lowest' 共享前缀 'low'")
print(f"→ 模型可以利用子词之间的共享语义！")


# ============================================================
# 3. 使用预训练模型
# ============================================================
print("\n" + "=" * 60)
print("3. 使用预训练模型 (代码示例)")
print("=" * 60)

print("""
【注意】以下代码需要安装 transformers 库并下载模型
如果网络环境允许，可以实际运行

--- 文本分类 (情感分析) ---

from transformers import pipeline

# 最简单的方式: 使用 pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love this tutorial! It's amazing.")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

--- 文本生成 ---

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)
print(result[0]['generated_text'])

--- 更细粒度的控制 ---

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 编码
input_ids = tokenizer.encode("Hello, how are", return_tensors="pt")

# 生成
output = model.generate(input_ids, max_length=20, temperature=0.7)

# 解码
text = tokenizer.decode(output[0])
print(text)
""")

# 我们可以用之前实现的 MiniGPT 来模拟这个流程
print("\n--- 用我们的 MiniGPT 模拟 Hugging Face 的使用流程 ---")

# 模拟一个简单的 Tokenizer
class SimpleTokenizer:
    """模拟 Hugging Face Tokenizer 的接口"""
    
    def __init__(self, vocab):
        self.vocab = vocab  # word -> id
        self.id_to_word = {v: k for k, v in vocab.items()}
        self.vocab_size = len(vocab)
    
    def encode(self, text):
        """文本 → token ids"""
        tokens = text.lower().split()
        ids = [self.vocab.get(t, self.vocab.get('<unk>')) for t in tokens]
        return ids
    
    def decode(self, ids):
        """token ids → 文本"""
        words = [self.id_to_word.get(i, '<unk>') for i in ids]
        return ' '.join(words)

# 创建一个玩具词表
vocab = {
    '<pad>': 0, '<unk>': 1, '<eos>': 2,
    'the': 3, 'cat': 4, 'sat': 5, 'on': 6, 'mat': 7,
    'dog': 8, 'ran': 9, 'in': 10, 'park': 11,
    'a': 12, 'big': 13, 'small': 14, 'happy': 15
}

tokenizer = SimpleTokenizer(vocab)

# 测试
text = "the cat sat on the mat"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)
print(f"原文: '{text}'")
print(f"编码: {ids}")
print(f"解码: '{decoded}'")
assert text == decoded, "编码解码不一致!"
print("✓ Tokenizer 正确")


# ============================================================
# 4. 微调 (Fine-tuning) 的概念
# ============================================================
print("\n" + "=" * 60)
print("4. 微调 (Fine-tuning)")
print("=" * 60)

print("""
【什么是微调？】

预训练模型已经学会了语言的通用知识。
微调 = 在你的特定任务数据上继续训练，让模型适应你的需求。

【微调的类型】

1. 全量微调 (Full Fine-tuning):
   更新所有参数
   - 效果最好
   - 但需要大量显存和计算

2. 冻结部分层:
   只更新最后几层
   - 更快，显存更少
   - 底层的通用知识保持不变

3. LoRA (Low-Rank Adaptation):
   只训练小的附加矩阵
   - 参数效率极高
   - 现在最流行的方法！
   
   原理: W_new = W_old + A × B
   W_old: 冻结的原始权重 (d×d)
   A: (d×r), B: (r×d), r << d
   只训练 A 和 B，参数量从 d² 减少到 2dr

4. Prompt Tuning:
   只优化输入的 prompt 向量
   - 模型完全冻结
   - 参数量最少
""")

# 演示微调过程
print("\n--- 微调演示 ---")

class SimpleClassifier(nn.Module):
    """在预训练模型上加分类头"""
    
    def __init__(self, pretrained_model, d_model, n_classes):
        super().__init__()
        self.backbone = pretrained_model  # 预训练的 Transformer
        self.classifier = nn.Linear(d_model, n_classes)  # 新加的分类头
    
    def forward(self, input_ids):
        # 通过预训练模型获取表示
        features = self.backbone(input_ids)  # (batch, seq, d_model) 这里是 logits 但我们模拟
        # 取 [CLS] 位置 (第一个 token) 的表示做分类
        cls_feature = features[:, 0, :]  # 简化: 用第一个 token
        logits = self.classifier(cls_feature)
        return logits

print("""
微调步骤:
1. 加载预训练模型
2. 冻结或部分冻结模型参数
3. 添加任务特定的头（如分类头）
4. 在任务数据上训练
5. 评估

代码示例 (Hugging Face):

  from transformers import AutoModelForSequenceClassification, Trainer
  
  # 加载预训练模型 + 分类头
  model = AutoModelForSequenceClassification.from_pretrained(
      "bert-base-uncased", 
      num_labels=2  # 二分类
  )
  
  # 冻结底层（可选）
  for param in model.bert.encoder.layer[:8].parameters():
      param.requires_grad = False
  
  # 训练
  trainer = Trainer(model=model, train_dataset=train_dataset, ...)
  trainer.train()
""")


# ============================================================
# 5. LoRA 的简单实现
# ============================================================
print("\n" + "=" * 60)
print("5. LoRA 简单实现")
print("=" * 60)

class LoRALayer(nn.Module):
    """LoRA: Low-Rank Adaptation"""
    
    def __init__(self, original_layer, rank=4):
        super().__init__()
        d_in = original_layer.in_features
        d_out = original_layer.out_features
        
        # 冻结原始权重
        self.original = original_layer
        for param in self.original.parameters():
            param.requires_grad = False
        
        # 低秩分解矩阵
        self.A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, d_out))
        # B 初始化为 0，所以初始时 LoRA 不改变原始行为
    
    def forward(self, x):
        # W_new = W_original + A × B
        original_output = self.original(x)
        lora_output = x @ self.A @ self.B
        return original_output + lora_output

# 演示 LoRA
print("--- LoRA 参数效率 ---")
d_model = 512
original = nn.Linear(d_model, d_model)
lora = LoRALayer(original, rank=8)

original_params = d_model * d_model
lora_params = d_model * 8 + 8 * d_model
trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)

print(f"原始层参数: {original_params:,} ({d_model}×{d_model})")
print(f"LoRA 可训练参数: {trainable:,} ({d_model}×8 + 8×{d_model})")
print(f"参数比例: {trainable/original_params*100:.1f}%")
print(f"\n→ 只需训练 {trainable/original_params*100:.1f}% 的参数就能适应新任务！")

# 验证初始时 LoRA 不改变输出
x = torch.randn(1, d_model)
with torch.no_grad():
    orig_out = original(x)
    lora_out = lora(x)
    diff = (orig_out - lora_out).abs().max().item()
print(f"\n初始时 LoRA 输出与原始差异: {diff:.10f}")
print("✓ B 初始化为 0，所以 LoRA 初始不改变模型行为")


# ============================================================
# 6. 总结和学习路线图
# ============================================================
print("\n" + "=" * 60)
print("6. 总结和下一步学习路线")
print("=" * 60)

print("""
🎉 恭喜！你已经完成了从零开始的 AI 学习之旅！

【回顾我们学到了什么】

第1章 数学基础:
  ✓ 线性代数 (向量、矩阵、特征值)
  ✓ 微积分 (导数、梯度、反向传播)
  ✓ 概率统计 (贝叶斯、分布、信息论)
  ✓ 优化方法 (SGD、Adam)

第2章 Python 工具:
  ✓ NumPy (高效数值计算)
  ✓ Pandas (数据处理)
  ✓ Matplotlib (可视化)

第3章 机器学习:
  ✓ 线性回归 & 逻辑回归
  ✓ 决策树 & 随机森林
  ✓ 模型评估 (交叉验证、偏差-方差)

第4章 深度学习:
  ✓ 从零实现神经网络
  ✓ PyTorch 框架
  ✓ CNN (卷积神经网络)
  ✓ RNN/LSTM (循环神经网络)

第5章 大语言模型:
  ✓ 词嵌入 (Word2Vec)
  ✓ 注意力机制
  ✓ Transformer 架构
  ✓ 预训练模型和微调

【下一步学习建议】

1. 实践项目:
   - 用 Hugging Face 做一个文本分类项目
   - 用 PyTorch 训练一个小型 CNN 做图像分类
   - 尝试微调一个小型语言模型

2. 深入学习:
   - 读 "Attention Is All You Need" 论文
   - 学习 RLHF (人类反馈强化学习)
   - 了解 RAG (检索增强生成)
   - 学习 Agent 和 Tool Use

3. 工程实践:
   - 模型部署 (ONNX, TensorRT)
   - 分布式训练
   - 模型量化和加速

4. 推荐资源:
   - fast.ai 课程 (实践导向)
   - Stanford CS229/CS231n/CS224n
   - Andrej Karpathy 的 YouTube 频道
   - Hugging Face 官方教程
""")
