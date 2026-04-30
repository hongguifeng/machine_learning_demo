"""
第六章 6.1：强化学习基础
========================

强化学习 (Reinforcement Learning, RL) 是机器学习的第三大范式：
  - 监督学习: 从标注数据中学习
  - 无监督学习: 从无标注数据中发现结构
  - 强化学习: 通过与环境交互，从奖励信号中学习

应用场景: 游戏AI、机器人控制、推荐系统、自动驾驶、RLHF(LLM对齐)

本节内容：
1. RL 的核心概念
2. 马尔可夫决策过程 (MDP)
3. 价值函数与贝尔曼方程
4. 动态规划求解
5. Q-Learning (表格法)
6. 实战: 网格世界
"""

import numpy as np

print("=" * 60)
print("第六章 6.1：强化学习基础")
print("=" * 60)

# ============================================================
# 1. 核心概念
# ============================================================
print("\n" + "=" * 60)
print("1. 强化学习的核心概念")
print("=" * 60)

print("""
【强化学习 vs 监督学习】

监督学习:
  数据集 → {(输入, 正确答案)} → 模型学习映射
  "老师给你正确答案"

强化学习:
  智能体(Agent) 在环境(Environment) 中行动
  环境给出奖励(Reward)信号
  智能体学习最大化长期累积奖励的策略
  "没人告诉你正确答案，你通过试错来学习"

【核心要素】

  ┌─────────────────────────────────────┐
  │           环境 (Environment)         │
  │                                      │
  │  状态 s ──→ 智能体(Agent) ──→ 动作 a │
  │       ↑                         │    │
  │       └── 奖励 r, 新状态 s' ←──┘    │
  └─────────────────────────────────────┘

  - 状态 (State, s): 环境当前的情况
  - 动作 (Action, a): 智能体可以做的选择
  - 奖励 (Reward, r): 环境对动作的即时反馈
  - 策略 (Policy, π): 状态 → 动作的映射规则
  - 目标: 找到最优策略 π*，最大化累积奖励

【例子：下棋】
  状态 = 棋盘局面
  动作 = 落子位置
  奖励 = 赢了+1, 输了-1, 其他0
  策略 = 看到某个局面，决定下哪里

【挑战】
  1. 延迟奖励: 当前动作的好坏可能要很久以后才知道
  2. 探索 vs 利用: 是尝试新动作，还是用已知的好动作？
  3. 信用分配: 一局棋赢了，到底是哪步棋下得好？
""")


# ============================================================
# 2. 马尔可夫决策过程 (MDP)
# ============================================================
print("\n" + "=" * 60)
print("2. 马尔可夫决策过程 (MDP)")
print("=" * 60)

print("""
【MDP 是 RL 的数学框架】

定义: MDP = (S, A, P, R, γ)
  S: 状态空间 (所有可能的状态)
  A: 动作空间 (所有可能的动作)
  P: 状态转移概率 P(s'|s, a) (做了动作后到哪个新状态)
  R: 奖励函数 R(s, a, s') (获得多少奖励)
  γ: 折扣因子 (0≤γ≤1, 未来奖励的衰减)

【马尔可夫性质】
  下一个状态只取决于当前状态和动作，与历史无关。
  P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)

【折扣因子 γ 的意义】
  累积奖励 G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
  
  γ = 0: 只看当前奖励（短视）
  γ = 1: 未来奖励和当前同等重要
  γ = 0.99: 常用值，稍微偏好近期奖励

  为什么要折扣？
  1. 数学上保证累积奖励有限
  2. 现实中，未来有不确定性，近期奖励更可靠
""")

# 定义一个简单的网格世界
print("\n--- 简单网格世界 ---")
print("""
4×4 网格世界:
  ┌───┬───┬───┬───┐
  │ S │   │   │ G │   S=起点, G=终点(奖励+1)
  ├───┼───┼───┼───┤
  │   │ X │   │   │   X=陷阱(奖励-1)
  ├───┼───┼───┼───┤
  │   │   │   │ X │
  ├───┼───┼───┼───┤
  │   │   │   │   │
  └───┴───┴───┴───┘

动作: 上(0), 下(1), 左(2), 右(3)
到达 G 或 X 则结束。每步奖励 -0.04（鼓励快速到达目标）
""")


class GridWorld:
    """简单的网格世界环境"""
    
    def __init__(self, size=4):
        self.size = size
        self.start = (0, 0)
        self.goal = (0, 3)       # 目标: +1
        self.traps = [(1, 1), (2, 3)]  # 陷阱: -1
        self.state = self.start
        
        # 动作: 上下左右
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_names = ['上', '下', '左', '右']
        self.n_actions = 4
        self.n_states = size * size
    
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        """执行动作，返回 (新状态, 奖励, 是否结束)"""
        dy, dx = self.actions[action]
        new_y = max(0, min(self.size - 1, self.state[0] + dy))
        new_x = max(0, min(self.size - 1, self.state[1] + dx))
        self.state = (new_y, new_x)
        
        if self.state == self.goal:
            return self.state, 1.0, True
        elif self.state in self.traps:
            return self.state, -1.0, True
        else:
            return self.state, -0.04, False
    
    def state_to_idx(self, state):
        return state[0] * self.size + state[1]
    
    def idx_to_state(self, idx):
        return (idx // self.size, idx % self.size)


env = GridWorld()
print(f"状态空间大小: {env.n_states}")
print(f"动作空间大小: {env.n_actions}")


# ============================================================
# 3. 价值函数
# ============================================================
print("\n" + "=" * 60)
print("3. 价值函数与贝尔曼方程")
print("=" * 60)

print("""
【价值函数: 评估"一个状态有多好"】

状态价值函数 V^π(s):
  从状态 s 出发，按照策略 π 行动，期望的累积奖励
  V^π(s) = E[r_t + γ·r_{t+1} + γ²·r_{t+2} + ... | s_t = s, π]

动作价值函数 Q^π(s, a):
  在状态 s 执行动作 a，然后按策略 π 行动的期望累积奖励
  Q^π(s, a) = E[r_t + γ·r_{t+1} + ... | s_t = s, a_t = a, π]

【贝尔曼方程 (递归关系)】

  V^π(s) = Σ_a π(a|s) · Σ_{s'} P(s'|s,a) · [R(s,a,s') + γ·V^π(s')]
  
  直觉: 当前状态的价值 = 即时奖励 + 折扣后的下一个状态的价值

  这是一个递推公式，可以通过迭代求解！

【最优价值函数】
  V*(s) = max_a Σ_{s'} P(s'|s,a) · [R(s,a,s') + γ·V*(s')]
  
  在每个状态都选最优动作时的价值。
""")


# ============================================================
# 4. 价值迭代 (Value Iteration)
# ============================================================
print("\n" + "=" * 60)
print("4. 价值迭代 (Value Iteration)")
print("=" * 60)

print("""
【算法】
  1. 初始化 V(s) = 0 对所有 s
  2. 重复直到收敛:
     对每个状态 s:
       V(s) = max_a [R(s,a) + γ · V(s')]
  3. 从最优价值提取策略:
     π*(s) = argmax_a [R(s,a) + γ · V(s')]
""")

def value_iteration(env, gamma=0.9, threshold=1e-6):
    """价值迭代算法"""
    V = np.zeros(env.n_states)
    
    for iteration in range(1000):
        V_new = np.zeros(env.n_states)
        
        for s_idx in range(env.n_states):
            state = env.idx_to_state(s_idx)
            
            # 终止状态价值为 0
            if state == env.goal or state in env.traps:
                continue
            
            # 对每个动作计算价值
            action_values = []
            for a in range(env.n_actions):
                env.state = state
                next_state, reward, done = env.step(a)
                next_idx = env.state_to_idx(next_state)
                
                if done:
                    action_values.append(reward)
                else:
                    action_values.append(reward + gamma * V[next_idx])
            
            V_new[s_idx] = max(action_values)
        
        # 检查收敛
        if np.max(np.abs(V_new - V)) < threshold:
            print(f"  价值迭代在第 {iteration+1} 次收敛")
            break
        V = V_new
    
    # 提取策略
    policy = np.zeros(env.n_states, dtype=int)
    for s_idx in range(env.n_states):
        state = env.idx_to_state(s_idx)
        if state == env.goal or state in env.traps:
            continue
        
        action_values = []
        for a in range(env.n_actions):
            env.state = state
            next_state, reward, done = env.step(a)
            next_idx = env.state_to_idx(next_state)
            if done:
                action_values.append(reward)
            else:
                action_values.append(reward + gamma * V[next_idx])
        
        policy[s_idx] = np.argmax(action_values)
    
    return V, policy

V_opt, policy_opt = value_iteration(env)

# 显示结果
print(f"\n最优状态价值 V*(s):")
V_grid = V_opt.reshape(4, 4)
for i in range(4):
    row = ""
    for j in range(4):
        row += f"{V_grid[i,j]:6.2f} "
    print(f"  {row}")

print(f"\n最优策略 (↑↓←→):")
arrows = ['↑', '↓', '←', '→']
for i in range(4):
    row = ""
    for j in range(4):
        state = (i, j)
        if state == env.goal:
            row += "  G   "
        elif state in env.traps:
            row += "  X   "
        else:
            idx = env.state_to_idx(state)
            row += f"  {arrows[policy_opt[idx]]}   "
    print(f"  {row}")


# ============================================================
# 5. Q-Learning
# ============================================================
print("\n" + "=" * 60)
print("5. Q-Learning (无模型强化学习)")
print("=" * 60)

print("""
【为什么需要 Q-Learning？】
价值迭代需要知道环境的转移概率 P(s'|s,a) → 需要"模型"
但很多时候我们不知道环境的具体规则！

Q-Learning: 通过与环境交互来学习，不需要知道环境模型。
这是"无模型" (Model-Free) 方法。

【Q-Learning 算法】
  初始化 Q(s, a) = 0
  重复:
    1. 在状态 s，选择动作 a (ε-贪心)
    2. 执行 a，观察奖励 r 和新状态 s'
    3. 更新:
       Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') - Q(s, a)]
    4. s ← s'

  α: 学习率
  γ: 折扣因子
  ε: 探索率 (以 ε 的概率随机探索，1-ε 的概率利用已知最优)

【ε-贪心策略 (Exploration vs Exploitation)】
  以概率 ε 随机选动作（探索新的可能）
  以概率 1-ε 选 Q 值最大的动作（利用已知信息）
  
  通常 ε 随训练逐渐减小（开始多探索，后来多利用）
""")

def q_learning(env, n_episodes=1000, alpha=0.1, gamma=0.9, 
               epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    """Q-Learning 算法"""
    Q = np.zeros((env.n_states, env.n_actions))
    epsilon = epsilon_start
    
    rewards_history = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(100):  # 最多100步
            s_idx = env.state_to_idx(state)
            
            # ε-贪心选择动作
            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[s_idx])
            
            # 执行动作
            next_state, reward, done = env.step(action)
            next_idx = env.state_to_idx(next_state)
            total_reward += reward
            
            # Q-Learning 更新
            best_next_q = np.max(Q[next_idx]) if not done else 0
            td_target = reward + gamma * best_next_q
            td_error = td_target - Q[s_idx, action]
            Q[s_idx, action] += alpha * td_error
            
            if done:
                break
            state = next_state
        
        rewards_history.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
    return Q, rewards_history

np.random.seed(42)
Q, rewards = q_learning(env, n_episodes=2000)

# 显示学到的 Q 值和策略
print(f"\n--- Q-Learning 结果 ---")
print(f"训练 2000 个 episode")
print(f"最后100个episode平均奖励: {np.mean(rewards[-100:]):.3f}")

# 提取策略
q_policy = np.argmax(Q, axis=1)
print(f"\n学到的策略:")
arrows = ['↑', '↓', '←', '→']
for i in range(4):
    row = ""
    for j in range(4):
        state = (i, j)
        if state == env.goal:
            row += "  G   "
        elif state in env.traps:
            row += "  X   "
        else:
            idx = env.state_to_idx(state)
            row += f"  {arrows[q_policy[idx]]}   "
    print(f"  {row}")

# 验证策略：从起点走到终点
print(f"\n--- 用学到的策略走一遍 ---")
state = env.reset()
path = [state]
for step in range(20):
    s_idx = env.state_to_idx(state)
    action = np.argmax(Q[s_idx])
    state, reward, done = env.step(action)
    path.append(state)
    if done:
        break

print(f"路径: {' → '.join([str(s) for s in path])}")
if state == env.goal:
    print(f"✓ 成功到达目标！步数: {len(path)-1}")
else:
    print(f"到达: {state}")

# 验证Q值合理性
assert Q[env.state_to_idx((0, 2)), 3] > 0, "靠近目标的Q值应该为正"
print("✓ Q-Learning 学到了合理的策略")


# ============================================================
# 6. 总结
# ============================================================
print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点：
1. RL = 智能体通过与环境交互、从奖励中学习最优行为
2. MDP = (状态, 动作, 转移概率, 奖励, 折扣因子) 
3. 价值函数 V(s)/Q(s,a) 评估状态/动作的长期价值
4. 贝尔曼方程: 当前价值 = 即时奖励 + 折扣×未来价值
5. Q-Learning: 无模型、通过试错学习 Q 值
6. ε-贪心: 平衡探索与利用

RL 方法分类:
  - 基于价值: Q-Learning, DQN
  - 基于策略: REINFORCE, PPO
  - Actor-Critic: A2C, A3C, SAC

下一节: 深度强化学习 → DQN、策略梯度、PPO
""")
