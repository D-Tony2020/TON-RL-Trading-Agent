"""
REINFORCE with Baseline — 策略梯度 Agent

双独立网络架构：
- PolicyNetwork: 策略网络，输出动作概率分布
- ValueNetwork: 价值网络（baseline），输出状态价值估计

与 DQNAgent 保持相同接口：select_action, select_greedy_action, save/load_checkpoint
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.config import (
    REINFORCE_PARAMS,
    CHECKPOINT_DIR,
    PROGRESS_CONFIG,
)


class PolicyNetwork(nn.Module):
    """策略网络：两层 MLP → logits（不含 Softmax，由 Categorical 处理）"""

    def __init__(self, state_dim=8, n_actions=5, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)  # 输出 logits，不是 probs


class ValueNetwork(nn.Module):
    """价值网络（Baseline）：两层 MLP → 标量状态价值"""

    def __init__(self, state_dim=8, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class REINFORCEAgent:
    """REINFORCE with Baseline Agent"""

    def __init__(self, **kwargs):
        # 合并默认参数和自定义参数
        params = {**REINFORCE_PARAMS, **kwargs}
        self.state_dim = params["state_dim"]
        self.n_actions = params["n_actions"]
        self.gamma = params["gamma"]
        self.gradient_clip = params["gradient_clip"]
        self.entropy_coeff = params["entropy_coeff"]

        # 设备检测
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化网络
        self.policy_net = PolicyNetwork(
            self.state_dim, self.n_actions, params["hidden_dim"]
        ).to(self.device)
        self.value_net = ValueNetwork(
            self.state_dim, params["hidden_dim"]
        ).to(self.device)

        # 两个独立的优化器
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=params["lr_policy"]
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=params["lr_value"]
        )

        # Episode buffer（每 episode 清空）
        self.saved_states = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []

        # 兼容性属性（REINFORCE 无 epsilon，但保留接口）
        self.epsilon = None

    def select_action(self, state):
        """
        根据策略网络采样动作

        Args:
            state: np.ndarray, shape (state_dim,)
        Returns:
            int: 选中的动作
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        logits = self.policy_net(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample()

        # 存入 episode buffer（state 用于 finish_episode 中重新计算 value）
        self.saved_states.append(state_t.detach())
        self.log_probs.append(dist.log_prob(action))
        self.entropies.append(dist.entropy())

        return action.item()

    def select_greedy_action(self, state):
        """
        确定性贪心动作（用于回测/评估）

        Args:
            state: np.ndarray, shape (state_dim,)
        Returns:
            int: 概率最大的动作
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy_net(state_t)
        return int(torch.argmax(logits, dim=1).item())

    def store_reward(self, reward):
        """存储当前步的奖励"""
        self.rewards.append(reward)

    def finish_episode(self, accumulate=False):
        """
        Episode 结束后执行 REINFORCE 更新

        Args:
            accumulate: 如果 True，只累积梯度不更新（用于 batch update）

        Returns:
            tuple: (policy_loss, value_loss, mean_entropy)
        """
        if len(self.rewards) == 0:
            return 0.0, 0.0, 0.0

        # 计算 discounted returns G_t（从后向前累积）
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns_t = torch.FloatTensor(returns).to(self.device)

        log_probs_t = torch.stack(self.log_probs)
        entropies_t = torch.stack(self.entropies)

        # 重新计算 value（需要梯度以更新 value_net）
        states_t = torch.cat(self.saved_states, dim=0)  # (T, state_dim)
        values_t = self.value_net(states_t).squeeze(-1)  # (T,)

        # Advantage = G_t - V(s_t)（不做标准化，保留原始信号强度）
        advantages = returns_t - values_t.detach()

        # Policy loss: -E[log_prob * advantage] - entropy_coeff * entropy
        policy_loss = -(log_probs_t * advantages).mean() - self.entropy_coeff * entropies_t.mean()

        # Value loss: MSE(V(s_t), G_t)
        value_loss = nn.functional.mse_loss(values_t, returns_t)

        # 反向传播（累积梯度）
        policy_loss.backward()
        value_loss.backward()

        if not accumulate:
            # 立即更新
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
            self.policy_optimizer.step()
            self.policy_optimizer.zero_grad()

            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.gradient_clip)
            self.value_optimizer.step()
            self.value_optimizer.zero_grad()

        # 记录指标
        p_loss = policy_loss.item()
        v_loss = value_loss.item()
        mean_ent = entropies_t.mean().item()

        # 清空 episode buffer
        self.saved_states = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []

        return p_loss, v_loss, mean_ent

    def flush_gradients(self):
        """Batch update: 裁剪并应用累积的梯度，然后清零"""
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad()

        nn.utils.clip_grad_norm_(self.value_net.parameters(), self.gradient_clip)
        self.value_optimizer.step()
        self.value_optimizer.zero_grad()

    def decay_epsilon(self):
        """No-op，REINFORCE 无 epsilon，保留接口兼容"""
        pass

    def save_checkpoint(self, filepath):
        """保存训练状态到 checkpoint"""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, filepath):
        """从 checkpoint 恢复训练状态"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])


def train_reinforce(env, agent, n_episodes=None, episode_length=None,
                    checkpoint_interval=None, verbose=True):
    """
    训练 REINFORCE Agent

    Args:
        env: CryptoTradingEnv (mode='continuous')
        agent: REINFORCEAgent
        n_episodes: 训练轮数（默认从 config 读取）
        episode_length: 每轮步数（默认从 config 读取）
        checkpoint_interval: 每 N 轮保存 checkpoint
        verbose: 是否打印进度

    Returns:
        dict: 训练历史记录
    """
    params = REINFORCE_PARAMS
    n_episodes = n_episodes or params["n_episodes"]
    episode_length = episode_length or params["episode_length"]
    checkpoint_interval = checkpoint_interval or params["checkpoint_interval"]
    print_interval = PROGRESS_CONFIG["print_interval_reinforce"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"REINFORCE 训练开始")
        print(f"设备: {agent.device}")
        if agent.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Episodes: {n_episodes}, Steps/ep: {episode_length}")
        print(f"lr_policy={params['lr_policy']}, lr_value={params['lr_value']}, "
              f"gamma={agent.gamma}, entropy_coeff={agent.entropy_coeff}")
        print(f"{'='*60}\n")

    history = {
        "episode_rewards": [],
        "episode_portfolio_values": [],
        "policy_losses": [],
        "value_losses": [],
        "entropy_history": [],
        "training_time": 0.0,
    }

    start_time = time.time()

    for ep in range(n_episodes):
        ep_start = time.time()
        state = env.reset(episode_length=episode_length)
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_reward(reward)
            ep_reward += reward
            state = next_state

        # Episode 结束，执行 REINFORCE 更新
        p_loss, v_loss, mean_ent = agent.finish_episode()

        # 记录历史
        history["episode_rewards"].append(ep_reward)
        history["episode_portfolio_values"].append(info["portfolio_value"])
        history["policy_losses"].append(p_loss)
        history["value_losses"].append(v_loss)
        history["entropy_history"].append(mean_ent)

        # 打印进度
        if verbose and (ep + 1) % print_interval == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (ep + 1) / elapsed
            remaining = (n_episodes - ep - 1) / eps_per_sec if eps_per_sec > 0 else 0

            # 最近 N 轮的平均指标
            recent = min(print_interval, len(history["episode_rewards"]))
            avg_reward = np.mean(history["episode_rewards"][-recent:])
            avg_pv = np.mean(history["episode_portfolio_values"][-recent:])

            print(
                f"  Ep {ep+1:>4d}/{n_episodes} | "
                f"Reward: {avg_reward:>8.2f} | "
                f"PV: ${avg_pv:>10,.0f} | "
                f"P_loss: {p_loss:>8.4f} | "
                f"V_loss: {v_loss:>8.4f} | "
                f"Entropy: {mean_ent:>.4f} | "
                f"Elapsed: {elapsed:>5.0f}s | "
                f"ETA: {remaining:>5.0f}s"
            )

        # 保存 checkpoint
        if (ep + 1) % checkpoint_interval == 0:
            ckpt_path = CHECKPOINT_DIR / f"reinforce_ep{ep+1}.pt"
            agent.save_checkpoint(str(ckpt_path))
            if verbose:
                print(f"    → Checkpoint 保存: {ckpt_path.name}")

    history["training_time"] = time.time() - start_time

    if verbose:
        print(f"\n训练完成！总时间: {history['training_time']:.1f}s")
        print(f"最终 Portfolio: ${history['episode_portfolio_values'][-1]:,.0f}")

    return history
