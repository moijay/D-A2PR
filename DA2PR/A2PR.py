import copy
import numpy as np
import torch
import torch.nn.functional as F
from core.network.actor import A2PR_Actor
from core.network.critic import Critic, Value
import torch.nn as nn
from utils.helpers import SinusoidalPosEmb, vp_beta_schedule

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4, head_dim=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * head_dim
        
        self.qkv = nn.Linear(dim, self.total_dim * 3)
        self.proj = nn.Linear(self.total_dim, dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        qkv = self.qkv(x).reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Split into q, k, v
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(batch_size, -1, self.total_dim)
        x = self.proj(x)
        return x

class AttentionMLP(nn.Module):
    def __init__(self, state_dim, action_dim, device, t_dim=16, hidden_dim=512):
        super(AttentionMLP, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        # Input processing
        input_dim = state_dim + action_dim + t_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention layers
        self.attention1 = MultiHeadAttention(hidden_dim)
        self.attention2 = MultiHeadAttention(hidden_dim)
        
        # Feed-forward layers
        self.ff_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Output layer
        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time, state):
        # Time embedding
        t = self.time_mlp(time)
        
        # Combine inputs
        x = torch.cat([x, t, state], dim=1)
        x = self.input_layer(x)
        
        # Reshape for attention (batch_size, seq_len=1, hidden_dim)
        x = x.unsqueeze(1)
        
        # First attention block
        residual = x
        x = self.norm1(x)
        x = self.attention1(x)
        x = x + residual
        
        # Second attention block
        residual = x
        x = self.norm2(x)
        x = self.attention2(x)
        x = x + residual
        
        # Feed-forward block
        residual = x
        x = self.norm3(x)
        x = self.ff_network(x)
        x = x + residual
        
        # Reshape and output
        x = x.squeeze(1)
        return self.final_layer(x)

class DiffusionModel(nn.Module):
    """扩散模型实现"""
    def __init__(self, state_dim, action_dim, max_action, device, num_timesteps=30):
        super(DiffusionModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device
        self.num_timesteps = num_timesteps
        
        # 使用带注意力的MLP模型作为噪声预测网络
        self.model = AttentionMLP(state_dim, action_dim, device)
        
        # 使用helpers.py中的vp_beta_schedule
        self.betas = vp_beta_schedule(num_timesteps, dtype=torch.float32).to(device)
        
        # 计算alphas相关系数
        self.alphas = (1. - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        
        # 预计算并移动到正确的设备
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(device)
        
    def forward(self, state, action, t=None):
        """前向扩散过程"""
        if t is None:
            t = torch.randint(0, self.num_timesteps, (state.shape[0],), device=self.device).long()
        else:
            t = t.to(self.device)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        action = action.to(self.device)
        noise = torch.randn_like(action, device=self.device)
        noisy_action = sqrt_alpha_cumprod_t * action + sqrt_one_minus_alpha_cumprod_t * noise
        
        predicted_noise = self.model(noisy_action, t.float(), state)
        
        return predicted_noise, noise, noisy_action
    
    def sample(self, state):
        """从噪声中生成动作样本的反向扩散过程"""
        state = state.to(self.device)
        x_t = torch.randn((state.shape[0], self.action_dim), device=self.device)
        
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((state.shape[0],), t, device=self.device, dtype=torch.long)
            predicted_noise = self.model(x_t, t_tensor.float(), state)
            
            alpha_t = self.alphas[t].to(self.device)
            alpha_cumprod_t = self.alphas_cumprod[t].to(self.device)
            beta_t = self.betas[t].to(self.device)

            if t > 0:
                noise = torch.randn_like(x_t, device=self.device)
                alpha_cumprod_prev = self.alphas_cumprod[t-1].to(self.device)
            else:
                noise = torch.zeros_like(x_t, device=self.device)
                alpha_cumprod_prev = torch.tensor(1.0, device=self.device)
                
            sqrt_alpha_t = torch.sqrt(alpha_t)
            one_minus_alpha_t = 1.0 - alpha_t
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
            
            x_t = (1. / sqrt_alpha_t) * (
                x_t - ((one_minus_alpha_t) / sqrt_one_minus_alpha_cumprod_t) * predicted_noise
            ) + torch.sqrt(beta_t) * (1-alpha_cumprod_prev)/(1-alpha_cumprod_t) * noise
        
        return self.max_action * torch.tanh(x_t)

class A2PR(object):
    def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, policy_noise=0.2,
                noise_clip=0.5, policy_freq=2, actor_lr=3e-4, critic_lr=3e-4, alpha=2.5, mask=1.0, diffusion_weight=1.0):
        self.device = device
        self.actor = A2PR_Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.value = Value(state_dim).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3e-4)
        self.advantage_list = []
        self.total_it = 0
        
        # 初始化带注意力的扩散模型
        self.diffusion = DiffusionModel(state_dim, action_dim, max_action, device).to(self.device)
        self.diffusion_optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=3e-4)
        self.mask = mask
        self.diffusion_weight = diffusion_weight

        self.models = {
            "actor": self.actor,
            "critic": self.critic,
            "actor_target": self.actor_target,
            "critic_target": self.critic_target,
            "actor_optimizer": self.actor_optimizer,
            "critic_optimizer": self.critic_optimizer,
            "value": self.value,
            "value_optimizer": self.value_optimizer,
            "diffusion": self.diffusion,
            "diffusion_optimizer": self.diffusion_optimizer,
        }

        print("state_dim:", state_dim, "action_dim:", action_dim)

    # 其余方法保持不变...

    @torch.no_grad()
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        tb_statics = dict()

        # Sample replay buffer
        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
            next_v = self.value(next_state)

        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        tb_statics.update({"critic_loss": critic_loss.item()})

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 计算价值函数
        with torch.no_grad():
            next_current_Q1, next_current_Q2 = self.critic(next_state, next_action)
            next_Q = torch.min(next_current_Q1, next_current_Q2)
            targ_Q = reward + not_done * self.discount * next_Q
            t_Q = torch.min(targ_Q, target_Q)
        
        value = self.value(state)
        value_loss = F.mse_loss(value, t_Q)
        

        # 优化价值网络
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 训练扩散模型
        predicted_noise, noise, noisy_action = self.diffusion(state, action)
        Q_diffusion = self.critic.Q1(state, action)
        value = self.value(state)
        embedding_diffusion = (Q_diffusion > value)
        diffusion_loss = (embedding_diffusion * torch.square(noise - predicted_noise)).mean()
        
        
        
        # 优化扩散模型
        self.diffusion_optimizer.zero_grad()
        diffusion_loss.backward()
        self.diffusion_optimizer.step()

        # 记录扩散模型的损失
        tb_statics.update({"diffusion_loss": diffusion_loss.item()})

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # 计算演员损失
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha / Q.abs().mean().detach()
            actor_loss = -lmbda * Q.mean()

            # 使用扩散模型生成动作
            denoised_action = self.diffusion.sample(state)
            Q_denoised = self.critic.Q1(state, denoised_action)
            
            # 计算优势值
            advantage_action = Q - value
            advantage_denoised = Q_denoised - value
            
            # 基于优势值创建动作嵌入
            embedding_denoised = denoised_action * (advantage_denoised >= advantage_action)
            embedding_action = action * (advantage_action > advantage_denoised)
            embedding = embedding_denoised + embedding_action
            
            # 计算优势标志
            adv_denoised = (Q_denoised.detach() >= value.detach())
            adv_action = (Q.detach() > value.detach())
            adv = self.mask * (adv_denoised + adv_action)
            
            # 计算行为克隆损失
            bc_loss = (torch.square(pi - embedding) * adv).mean()

            # 优化演员网络
            combined_loss = actor_loss + self.diffusion_weight * bc_loss
            self.actor_optimizer.zero_grad()
            combined_loss.backward()
            self.actor_optimizer.step()

            # 更新统计信息
            tb_statics.update({
                "bc_loss": bc_loss.item(),
                "actor_loss": actor_loss.item(),
                "combined_loss": combined_loss.item(),
                "Q_value": torch.mean(Q).item(),
                "targ_Q": torch.mean(targ_Q).item(),
                "target_Q": torch.mean(target_Q).item(),
                "lmbda": lmbda,
            })

            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return tb_statics

    def compute_advantage(self, gamma, lmbda, td_delta):
        """计算广义优势估计（GAE）"""
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage_list = np.array(advantage_list)
        return torch.tensor(advantage_list, dtype=torch.float)

    def save(self, model_path):
        """保存模型"""
        state_dict = dict()
        for model_name, model in self.models.items():
            state_dict[model_name] = model.state_dict()
        torch.save(state_dict, model_path)

    def load(self, model_path):
        """加载模型"""
        state_dict = torch.load(model_path)
        for model_name, model in self.models.items():
            model.load_state_dict(state_dict[model_name])

    def add_data(self, new_data):
        """添加新的优势数据到列表中"""
        if len(self.advantage_list) >= 10:
            self.advantage_list.pop(0)
        self.advantage_list.append(new_data)
