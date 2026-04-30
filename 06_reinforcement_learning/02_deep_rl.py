"""
第六章 6.2：深度强化学习
========================

当状态空间太大（如图像），表格法无法存储所有 Q 值。
解决方案：用神经网络来近似 Q 函数或策略函数。

本节内容：
1. DQN (Deep Q-Network)
2. 策略梯度 (Policy Gradient / REINFORCE)
3. PPO (Proximal Policy Optimization)
4. RLHF: 强化学习在 LLM 中的应用
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

print("=" * 60)
print("第六章 6.2：深度强化学习")
print("=" * 60)

# ============================================================
# 1. DQN
# ============================================================
print("\n" + "=" * 60)
print("1. DQN (Deep Q-Network)")
print("=" * 60)

print("""
【从 Q-Learning 到 DQN】

Q-Learning: Q 表 (|S| × |A| 的二维数组)
  - 状态少时可行 (如 4×4 网格 = 16 个状态)
  - 状态多时不可行 (如 Atari 游戏, 状态=图像像素, 无穷多)

DQN: 用神经网络近似 Q 函数
  Q(s, a) ≈ Q_θ(s, a)   (θ 是网络参数)
  
  输入: 状态 s
  输出: 所有动作的 Q 值 [Q(s,a₁), Q(s,a₂), ...]

【DQN 的两个关键技术】

1. 经验回放 (Experience Replay):
   - 把 (s, a, r, s', done) 存到一个缓冲区
   - 随机采样 batch 来训练
   - 打破数据的时间相关性，稳定训练

2. 目标网络 (Target Network):
   - 用一个"旧"的网络计算 target
   - 每隔 N 步才更新目标网络
   - 避免"追逐一个不断移动的目标"

【DQN 训练流程】
  1. 当前网络预测: Q(s, a; θ)
  2. 目标网络计算 target: y = r + γ · max_a' Q(s', a'; θ⁻)
  3. 损失: L = (Q(s,a;θ) - y)²
  4. 更新 θ，定期复制 θ → θ⁻
""")

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """深度 Q 网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """DQN 智能体"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 当前网络和目标网络
        self.q_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state_t)
            return q_values.argmax(dim=1).item()
    
    def update(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)
        
        # 当前 Q 值
        current_q = self.q_net(states_t).gather(1, actions_t)
        
        # 目标 Q 值
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1, keepdim=True)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)
        
        # 更新
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss.item()
    
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# 简单环境测试 DQN
print("\n--- DQN 在简单环境中测试 ---")

class SimpleEnv:
    """简单环境: 1D 走廊，目标在右端"""
    def __init__(self, size=10):
        self.size = size
        self.pos = 0
    
    def reset(self):
        self.pos = 0
        return np.array([self.pos / self.size], dtype=np.float32)
    
    def step(self, action):
        # action 0=左, 1=右
        if action == 1:
            self.pos = min(self.size - 1, self.pos + 1)
        else:
            self.pos = max(0, self.pos - 1)
        
        state = np.array([self.pos / self.size], dtype=np.float32)
        done = (self.pos == self.size - 1)
        reward = 1.0 if done else -0.01
        return state, reward, done

env = SimpleEnv(size=10)
agent = DQNAgent(state_dim=1, action_dim=2, lr=0.01, gamma=0.95)

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

episode_rewards = []
for ep in range(300):
    state = env.reset()
    total_reward = 0
    for step in range(50):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.buffer.push(state, action, reward, next_state, float(done))
        agent.update(batch_size=32)
        total_reward += reward
        state = next_state
        if done:
            break
    episode_rewards.append(total_reward)
    if ep % 50 == 0:
        agent.update_target()

print(f"训练 300 episodes:")
print(f"  前50 episodes 平均奖励: {np.mean(episode_rewards[:50]):.3f}")
print(f"  后50 episodes 平均奖励: {np.mean(episode_rewards[-50:]):.3f}")
assert np.mean(episode_rewards[-50:]) > np.mean(episode_rewards[:50])
print("✓ DQN 学会了向右走到达目标！")


# ============================================================
# 2. 策略梯度 (REINFORCE)
# ============================================================
print("\n" + "=" * 60)
print("2. 策略梯度 (Policy Gradient / REINFORCE)")
print("=" * 60)

print("""
【基于价值 vs 基于策略】

DQN (基于价值):
  学习 Q(s,a)，然后选 Q 最大的动作
  适合离散动作空间

策略梯度 (基于策略):
  直接学习策略 π_θ(a|s) — 输出动作的概率分布
  适合连续动作空间 (如机器人关节角度)

【REINFORCE 算法】

核心思想: "如果一个动作导致了好的结果(高回报)，增加它的概率"

  目标: 最大化期望回报 J(θ) = E[G]
  梯度: ∇J(θ) = E[G_t · ∇log π_θ(a_t|s_t)]

  更新规则:
    θ ← θ + α · G_t · ∇log π_θ(a_t|s_t)

  直觉:
    - G_t 大 (好结果) → 增大该动作概率
    - G_t 小 (坏结果) → 减小该动作概率
    - ∇log π 是让概率增大的方向
""")

class PolicyNetwork(nn.Module):
    """策略网络: 输出动作概率"""
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)


def reinforce(env, n_episodes=500, gamma=0.95, lr=0.01):
    """REINFORCE 算法"""
    policy = PolicyNetwork(state_dim=1, action_dim=2)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    rewards_history = []
    
    for ep in range(n_episodes):
        states, actions, rewards = [], [], []
        state = env.reset()
        
        # 收集一个 episode
        for _ in range(50):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            next_state, reward, done = env.step(action.item())
            
            states.append(state_t)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            if done:
                break
        
        # 计算回报 G_t
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        # 标准化减少方差
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 策略梯度更新
        loss = 0
        for state_t, action, G in zip(states, actions, returns):
            probs = policy(state_t)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)
            loss -= log_prob * G  # 负号因为我们用的是梯度下降
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        rewards_history.append(sum(rewards))
    
    return policy, rewards_history

torch.manual_seed(42)
env2 = SimpleEnv(size=10)
policy, rewards_hist = reinforce(env2, n_episodes=500)

print(f"\nREINFORCE 训练结果:")
print(f"  前50 episodes 平均奖励: {np.mean(rewards_hist[:50]):.3f}")
print(f"  后50 episodes 平均奖励: {np.mean(rewards_hist[-50:]):.3f}")
print("✓ 策略梯度方法也学会了到达目标")


# ============================================================
# 3. PPO 概述
# ============================================================
print("\n" + "=" * 60)
print("3. PPO (Proximal Policy Optimization)")
print("=" * 60)

print("""
【为什么需要 PPO？】

REINFORCE 的问题:
  - 高方差: 每个 episode 的回报波动大
  - 数据效率低: 每个 episode 的数据只用一次
  - 更新不稳定: 步长太大会导致策略突变

PPO 的改进:
  - 使用 Actor-Critic 架构减少方差
  - 允许多次使用同一批数据 (数据高效)
  - 限制策略更新幅度 (稳定训练)

【PPO 的核心思想: 截断的替代目标】

  L(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]

  其中:
    r_t(θ) = π_θ(a|s) / π_θ_old(a|s)   (新旧策略的概率比)
    A_t = 优势函数 (这个动作比平均好多少)
    ε ≈ 0.2 (截断范围)

  直觉:
    - 如果新策略比旧策略好很多 (r_t 很大)，clip 防止走太远
    - 确保每次更新不会"跳"太远，像是给策略加了"安全绳"

【Actor-Critic 架构】
  Actor (策略网络): 输出动作概率 π(a|s)
  Critic (价值网络): 估计状态价值 V(s)
  
  优势 A(s,a) = R + γ·V(s') - V(s)
    (实际获得的 vs 预期的，差额就是"优势")
""")

class ActorCritic(nn.Module):
    """Actor-Critic 网络 (PPO 的基础)"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        # Actor: 输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic: 输出状态价值
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

# 演示 Actor-Critic 结构
ac = ActorCritic(state_dim=4, action_dim=2)
dummy_state = torch.randn(1, 4)
probs, value = ac(dummy_state)
print(f"\nActor-Critic 示例:")
print(f"  输入状态: 4维向量")
print(f"  Actor 输出 (动作概率): {probs.detach().numpy().round(3)}")
print(f"  Critic 输出 (状态价值): {value.item():.3f}")
print(f"\n  训练时:")
print(f"    Actor 被奖励信号(优势函数)引导")
print(f"    Critic 学习预测每个状态的价值")


# ============================================================
# 4. RLHF: RL 在大语言模型中的应用
# ============================================================
print("\n" + "=" * 60)
print("4. RLHF: 强化学习对齐大语言模型")
print("=" * 60)

print("""
【RLHF = Reinforcement Learning from Human Feedback】

ChatGPT 为什么比纯 GPT 好用？因为经过了 RLHF！

三步训练流程:

步骤 1: 监督微调 (SFT)
  在人工编写的高质量对话数据上微调基础模型
  → 模型学会"像人类助手一样回答问题"

步骤 2: 训练奖励模型 (Reward Model)
  给同一个问题生成多个回答
  人类标注员对回答排序（哪个更好）
  训练一个模型来打分: reward = RM(prompt, response)
  → 模型学会"什么样的回答是人类喜欢的"

步骤 3: PPO 强化学习
  状态 = prompt + 已生成的文本
  动作 = 下一个 token
  奖励 = 奖励模型的打分
  用 PPO 优化策略（语言模型），最大化奖励
  + KL 惩罚: 防止偏离原始模型太远

  目标: max E[RM(prompt, response)] - β·KL(π_θ || π_ref)

【为什么 RLHF 有效？】
  - 监督学习只能模仿数据中的行为
  - RLHF 可以让模型学到"超越数据"的行为
  - 人类判断"好不好"比"写出好回答"容易得多

【替代方案: DPO (Direct Preference Optimization)】
  绕过奖励模型，直接从人类偏好数据优化策略
  数学上等价于 RLHF，但实现更简单
  loss = -log σ(β · (log π(y_w|x) - log π(y_l|x)))
  y_w = 人类偏好的回答, y_l = 不偏好的回答
""")

# 简单模拟 RLHF 的概念
print("\n--- RLHF 概念模拟 ---")

class SimpleRewardModel(nn.Module):
    """模拟奖励模型"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# 模拟: 奖励模型学习人类偏好
torch.manual_seed(42)
dim = 8
rm = SimpleRewardModel(dim)
optimizer = optim.Adam(rm.parameters(), lr=0.01)

# 模拟偏好数据: response_good > response_bad
n_pairs = 200
data_good = torch.randn(n_pairs, dim) + 0.5  # "好回答"偏正
data_bad = torch.randn(n_pairs, dim) - 0.5   # "坏回答"偏负

for epoch in range(100):
    reward_good = rm(data_good)
    reward_bad = rm(data_bad)
    # 奖励模型损失: 好回答的分数应该高于坏回答
    loss = -torch.log(torch.sigmoid(reward_good - reward_bad)).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 验证
with torch.no_grad():
    r_good = rm(data_good[:10]).mean().item()
    r_bad = rm(data_bad[:10]).mean().item()

print(f"奖励模型学习结果:")
print(f"  好回答平均奖励: {r_good:.3f}")
print(f"  坏回答平均奖励: {r_bad:.3f}")
assert r_good > r_bad
print("✓ 奖励模型学会了区分好坏回答")


print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点:
1. DQN: 用神经网络近似 Q 函数 + 经验回放 + 目标网络
2. 策略梯度: 直接优化策略, 适合连续动作
3. PPO: 截断替代目标 + Actor-Critic, 当前最流行的 RL 算法
4. RLHF: SFT → 奖励模型 → PPO, 让 LLM 对齐人类偏好

RL 在 AI 中的重要应用:
  - 游戏: AlphaGo, AlphaStar, OpenAI Five
  - 机器人: 运动控制, 抓取
  - LLM 对齐: ChatGPT, Claude 都用了 RLHF/类似方法
  - 推荐系统: 序列推荐
  - 自动驾驶: 决策规划
""")
