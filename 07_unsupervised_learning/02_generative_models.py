"""
第七章 7.2：生成模型 (自编码器与 GAN)
======================================

生成模型 = 学习数据的分布，然后生成新的数据

应用: 图像生成、数据增强、异常检测、风格迁移

本节内容：
1. 自编码器 (Autoencoder)
2. 变分自编码器 (VAE)
3. 生成对抗网络 (GAN)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("第七章 7.2：生成模型 (自编码器与 GAN)")
print("=" * 60)

# ============================================================
# 1. 自编码器
# ============================================================
print("\n" + "=" * 60)
print("1. 自编码器 (Autoencoder)")
print("=" * 60)

print("""
【自编码器的结构】

  输入 x → [编码器] → 潜在表示 z → [解码器] → 重构 x̂

  编码器: 压缩数据到低维空间 (瓶颈层)
  解码器: 从低维表示重构原始数据
  训练目标: 让 x̂ 尽量接近 x (重构损失)

  损失: L = ||x - x̂||² (MSE)

【为什么有用？】
  - 降维: z 是数据的压缩表示 (类似 PCA 但非线性)
  - 去噪: 加噪的输入通过瓶颈后，噪声被滤掉
  - 异常检测: 正常数据重构好，异常数据重构差
  - 特征学习: z 可以作为下游任务的特征

【与 PCA 的关系】
  线性自编码器 (无激活函数) ≈ PCA
  非线性自编码器 > PCA (能捕捉非线性结构)
""")

class Autoencoder(nn.Module):
    """简单自编码器"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

# 生成简单数据: 高维数据实际上分布在低维流形
torch.manual_seed(42)
np.random.seed(42)

# 真实结构: 2D 圆环 → 嵌入到 20D
n_samples = 500
t = np.linspace(0, 2*np.pi, n_samples)
circle_2d = np.stack([np.cos(t), np.sin(t)], axis=1) + np.random.randn(n_samples, 2) * 0.1
# 嵌入到20维
W_embed = np.random.randn(2, 20)
X_20d = circle_2d @ W_embed + np.random.randn(n_samples, 20) * 0.05
X_tensor = torch.FloatTensor(X_20d)

# 训练自编码器: 20D → 2D → 20D
ae = Autoencoder(input_dim=20, hidden_dim=32, latent_dim=2)
optimizer = optim.Adam(ae.parameters(), lr=0.001)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

print("\n--- 训练自编码器 (20D → 2D → 20D) ---")
for epoch in range(100):
    total_loss = 0
    for (batch,) in loader:
        recon, z = ae(batch)
        loss = F.mse_loss(recon, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}: loss={total_loss/len(loader):.6f}")

# 评估
ae.eval()
with torch.no_grad():
    recon, latent = ae(X_tensor)
    final_loss = F.mse_loss(recon, X_tensor).item()

print(f"\n最终重构损失: {final_loss:.6f}")
print(f"潜在表示维度: {latent.shape[1]}")
print(f"→ 成功将 20D 数据压缩到 2D 再重构")
assert final_loss < 0.1
print("✓ 自编码器有效学到了数据的低维结构")


# ============================================================
# 2. 变分自编码器 (VAE)
# ============================================================
print("\n" + "=" * 60)
print("2. 变分自编码器 (VAE)")
print("=" * 60)

print("""
【自编码器 vs VAE】

自编码器:
  z = encoder(x)  ← z 是一个确定的点
  问题: 潜在空间不连续，无法用来生成新数据

VAE:
  μ, σ = encoder(x)  ← 输出均值和方差
  z ~ N(μ, σ²)       ← z 是从分布中采样的
  x̂ = decoder(z)
  
  好处: 潜在空间连续、平滑 → 可以采样生成新数据！

【VAE 的损失函数】
  L = 重构损失 + KL散度
  
  重构损失: ||x - x̂||²  (让重构准确)
  KL散度:   KL(q(z|x) || p(z))  (让潜在分布接近标准正态)
  
  KL 散度的作用: 正则化潜在空间，让它有好的结构
  KL(N(μ,σ²) || N(0,1)) = -0.5 × Σ(1 + log(σ²) - μ² - σ²)

【重参数化技巧 (Reparameterization Trick)】
  问题: z ~ N(μ, σ²) 采样操作不可导！
  解决: z = μ + σ × ε, 其中 ε ~ N(0, 1)
  这样梯度可以流过 μ 和 σ
""")

class VAE(nn.Module):
    """变分自编码器"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    """VAE 损失 = 重构损失 + KL散度"""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# 训练 VAE
vae = VAE(input_dim=20, hidden_dim=32, latent_dim=2)
optimizer = optim.Adam(vae.parameters(), lr=0.001)

print("\n--- 训练 VAE ---")
for epoch in range(100):
    total_loss = 0
    for (batch,) in loader:
        recon, mu, logvar = vae(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}: loss={total_loss/n_samples:.4f}")

# 生成新数据
vae.eval()
with torch.no_grad():
    # 从标准正态采样潜在向量
    z_sample = torch.randn(10, 2)
    generated = vae.decode(z_sample)

print(f"\n从标准正态采样生成 10 个新数据点:")
print(f"  采样 z: shape={list(z_sample.shape)}")
print(f"  生成 x: shape={list(generated.shape)}")
print(f"  → VAE 可以生成新的、类似训练数据的样本！")
print("✓ VAE 训练成功")


# ============================================================
# 3. GAN (生成对抗网络)
# ============================================================
print("\n" + "=" * 60)
print("3. GAN (生成对抗网络)")
print("=" * 60)

print("""
【GAN 的核心思想: 对抗训练】

两个网络互相博弈:
  生成器 G: 把随机噪声变成假数据 (伪造者)
  判别器 D: 判断数据是真还是假 (警察)

训练过程:
  D 尽力区分真假: max_D E[log D(x)] + E[log(1 - D(G(z)))]
  G 尽力欺骗 D:   min_G E[log(1 - D(G(z)))]

直觉:
  - G 像伪造画家，D 像鉴定专家
  - G 不断改进，直到 D 无法分辨真假
  - 最终 G 能生成非常逼真的数据

【训练流程】
  重复:
    1. 训练 D: 真数据标签1, 假数据标签0
    2. 训练 G: 让 D 对假数据输出1 (骗过D)
    
  关键: D 和 G 交替训练, 保持平衡

【GAN 的变体 (了解即可)】
  - DCGAN: 用卷积 (图像生成)
  - StyleGAN: 超高质量人脸生成
  - CycleGAN: 风格迁移 (马→斑马)
  - Diffusion Models: 扩散模型 (现在更主流, 如 Stable Diffusion)
""")

class Generator(nn.Module):
    """生成器: 噪声 → 数据"""
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    """判别器: 数据 → 真/假概率"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# 训练 GAN 生成简单的 2D 数据
torch.manual_seed(42)
latent_dim = 2
data_dim = 2
hidden_dim = 64

G = Generator(latent_dim, hidden_dim, data_dim)
D = Discriminator(data_dim, hidden_dim)

optim_G = optim.Adam(G.parameters(), lr=0.001)
optim_D = optim.Adam(D.parameters(), lr=0.001)

# 真实数据: 2D 高斯混合
def sample_real_data(n):
    """生成真实数据: 两个高斯分布的混合"""
    n1 = n // 2
    n2 = n - n1
    d1 = torch.randn(n1, 2) * 0.5 + torch.tensor([2.0, 2.0])
    d2 = torch.randn(n2, 2) * 0.5 + torch.tensor([-2.0, -2.0])
    return torch.cat([d1, d2], dim=0)

print("\n--- 训练 GAN ---")
batch_size = 128
n_epochs = 300

for epoch in range(n_epochs):
    # --- 训练判别器 ---
    real_data = sample_real_data(batch_size)
    z = torch.randn(batch_size, latent_dim)
    fake_data = G(z).detach()  # detach: 不传梯度给G
    
    d_real = D(real_data)
    d_fake = D(fake_data)
    
    d_loss = -torch.mean(torch.log(d_real + 1e-8) + torch.log(1 - d_fake + 1e-8))
    
    optim_D.zero_grad()
    d_loss.backward()
    optim_D.step()
    
    # --- 训练生成器 ---
    z = torch.randn(batch_size, latent_dim)
    fake_data = G(z)
    d_fake = D(fake_data)
    
    g_loss = -torch.mean(torch.log(d_fake + 1e-8))
    
    optim_G.zero_grad()
    g_loss.backward()
    optim_G.step()
    
    if epoch % 60 == 0:
        print(f"  Epoch {epoch:3d}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")

# 评估: 生成的数据是否像真实数据
G.eval()
with torch.no_grad():
    z_test = torch.randn(200, latent_dim)
    generated_data = G(z_test).numpy()

real_test = sample_real_data(200).numpy()
gen_mean = generated_data.mean(axis=0)
real_mean = real_test.mean(axis=0)

print(f"\n生成数据 vs 真实数据统计:")
print(f"  真实数据均值: ({real_mean[0]:.2f}, {real_mean[1]:.2f})")
print(f"  生成数据均值: ({gen_mean[0]:.2f}, {gen_mean[1]:.2f})")
print(f"  真实数据标准差: {real_test.std():.2f}")
print(f"  生成数据标准差: {generated_data.std():.2f}")

# 判别器对生成数据的评分
D.eval()
with torch.no_grad():
    d_score = D(torch.FloatTensor(generated_data)).mean().item()
print(f"  判别器对生成数据的平均评分: {d_score:.3f} (越接近0.5越好)")
print("✓ GAN 学会了生成类似真实分布的数据")


print("\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
关键要点:
1. 自编码器: 编码器压缩 + 解码器重构, 学习数据的低维表示
2. VAE: 潜在空间是连续分布, 可以采样生成新数据
3. GAN: 生成器和判别器对抗训练, 生成逼真数据
4. 扩散模型 (Diffusion): 现在最强的图像生成方法 (Stable Diffusion, DALL-E)

生成模型的演进:
  AE → VAE → GAN → Diffusion Models → 多模态生成

实际应用:
  - 图像生成/编辑 (Midjourney, DALL-E)
  - 数据增强 (训练数据不够时生成更多)
  - 异常检测 (重构误差大 = 异常)
  - 药物分子设计
""")
