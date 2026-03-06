"""
Dueling Double DQN + Prioritized Experience Replay Agent
- Dueling 架构：分离 V(s) 和 A(s,a) 流
- Double DQN：online 网络选动作，target 网络评估
- PER：按 TD error 优先级采样经验
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path
from src.config import DQN_PARAMS, CHECKPOINT_DIR, PROGRESS_CONFIG


# ============================================================
# Dueling DQN 网络
# ============================================================
class DuelingDQN(nn.Module):
    """
    Dueling DQN 网络架构
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

    Args:
        state_dim: 输入状态维度
        n_actions: 输出动作数量
        hidden_dim: 共享隐藏层维度
        v_dim: V 流隐藏层维度
        a_dim: A 流隐藏层维度
    """

    def __init__(self, state_dim=7, n_actions=3, hidden_dim=128, v_dim=64, a_dim=64):
        super().__init__()

        # 共享特征层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # V 流（状态价值）
        self.v_stream = nn.Sequential(
            nn.Linear(hidden_dim, v_dim),
            nn.ReLU(),
            nn.Linear(v_dim, 1),
        )

        # A 流（动作优势）
        self.a_stream = nn.Sequential(
            nn.Linear(hidden_dim, a_dim),
            nn.ReLU(),
            nn.Linear(a_dim, n_actions),
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: shape (batch, state_dim) 的状态张量
        Returns:
            shape (batch, n_actions) 的 Q 值张量
        """
        features = self.feature(x)
        v = self.v_stream(features)           # (batch, 1)
        a = self.a_stream(features)           # (batch, n_actions)
        # Q = V + (A - mean(A))，保证可识别性
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


# ============================================================
# SumTree（PER 的核心数据结构）
# ============================================================
class SumTree:
    """
    SumTree 数据结构，用于 O(log N) 的优先级采样和更新

    内部结构：
    - 叶子节点存储 transition 的优先级
    - 父节点存储子节点之和
    - 总优先级 = 根节点值
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # 完全二叉树
        self.data = [None] * capacity             # 叶子节点对应的数据
        self.write_idx = 0
        self.size = 0

    def add(self, priority, data):
        """添加新 transition"""
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(tree_idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx, priority):
        """更新指定叶子的优先级，并向上传播"""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, s):
        """
        根据累积优先级 s 采样一个 transition
        返回 (tree_idx, priority, data)
        """
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]


# ============================================================
# Prioritized Experience Replay Buffer
# ============================================================
class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区

    Args:
        capacity: 缓冲区大小
        alpha: 优先级指数（0=均匀采样，1=完全按优先级）
        beta_start: 重要性采样权重初始值
        beta_end: 重要性采样权重最终值
        beta_frames: beta 退火总步数
        epsilon: 最小优先级（确保零 TD error 也能被采样）
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0,
                 beta_frames=100000, epsilon=1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 0
        self.max_priority = 1.0

    @property
    def beta(self):
        """当前的 beta 值（线性退火）"""
        fraction = min(self.frame / max(self.beta_frames, 1), 1.0)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def push(self, state, action, reward, next_state, done):
        """添加 transition，初始优先级为当前最大优先级"""
        transition = (state, action, reward, next_state, done)
        priority = self.max_priority  # 不再重复应用 alpha（update_priorities 已含）
        self.tree.add(priority, transition)

    def sample(self, batch_size):
        """
        按优先级采样一个 batch

        Returns:
            (batch, indices, weights)
            batch: (states, actions, rewards, next_states, dones) 各为张量
            indices: tree 索引，用于后续更新优先级
            weights: 重要性采样权重
        """
        indices = []
        priorities = []
        batch = []
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            if data is None:
                # 缓冲区未满时可能采到空位，重新采样
                idx, priority, data = self.tree.get(np.random.uniform(0, self.tree.total_priority))
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        self.frame += 1

        # 计算重要性采样权重
        priorities = np.array(priorities, dtype=np.float32)
        probs = priorities / (self.tree.total_priority + 1e-10)
        weights = (self.tree.size * probs + 1e-10) ** (-self.beta)
        weights = weights / (weights.max() + 1e-10)  # 归一化

        # 拆分 batch
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        ), indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        """用新的 TD error 更新优先级"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.size


# ============================================================
# DQN Agent
# ============================================================
class DQNAgent:
    """
    Dueling Double DQN + PER Agent

    Args:
        state_dim: 状态维度
        n_actions: 动作数量
        lr: 学习率
        gamma: 折扣因子
        buffer_size: 回放缓冲区大小
        batch_size: 训练批量
        target_update: 目标网络更新频率
        epsilon/epsilon_min/epsilon_decay: 探索率参数
        gradient_clip: 梯度裁剪阈值
        per_alpha/per_beta_start/per_beta_end/per_epsilon: PER 参数
        device: 计算设备
    """

    def __init__(self, **kwargs):
        # 合并默认参数和自定义参数
        p = {**DQN_PARAMS, **kwargs}

        self.state_dim = p["state_dim"]
        self.n_actions = p["n_actions"]
        self.gamma = p["gamma"]
        self.batch_size = p["batch_size"]
        self.target_update_freq = p["target_update"]
        self.epsilon = p["epsilon"]
        self.epsilon_min = p["epsilon_min"]
        self.epsilon_decay = p["epsilon_decay"]
        self.gradient_clip = p["gradient_clip"]
        self.min_buffer_size = p["min_buffer_size"]
        self.soft_update_enabled = p.get("soft_update", False)
        self.tau = p.get("tau", 0.005)

        # 设备（自动检测 GPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 网络
        self.online_net = DuelingDQN(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            hidden_dim=p["hidden_dim"],
            v_dim=p["v_stream_dim"],
            a_dim=p["a_stream_dim"],
        ).to(self.device)

        self.target_net = DuelingDQN(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            hidden_dim=p["hidden_dim"],
            v_dim=p["v_stream_dim"],
            a_dim=p["a_stream_dim"],
        ).to(self.device)

        # 初始化目标网络 = 在线网络
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # 优化器 + Huber Loss
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=p["lr"])
        self.loss_fn = nn.SmoothL1Loss(reduction="none")  # Huber Loss，逐元素

        # 学习率调度（可选）
        self.lr_schedule_enabled = p.get("lr_schedule", False)
        self.scheduler = None
        if self.lr_schedule_enabled:
            total_steps = p["n_episodes"] * p["episode_length"]
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=p.get("lr_end", 3e-5)
            )

        # PER 缓冲区
        beta_frames = p["n_episodes"] * p["episode_length"]
        self.buffer = PrioritizedReplayBuffer(
            capacity=p["buffer_size"],
            alpha=p["per_alpha"],
            beta_start=p["per_beta_start"],
            beta_end=p["per_beta_end"],
            beta_frames=beta_frames,
            epsilon=p["per_epsilon"],
        )

        # 训练步计数
        self.train_steps = 0

    def select_action(self, state):
        """
        ε-greedy 动作选择

        Args:
            state: np.ndarray shape (state_dim,)
        Returns:
            int: 动作索引
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)
            return q_values.argmax(dim=1).item()

    def select_greedy_action(self, state):
        """纯贪心动作选择（评估/回测用）"""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """存入 PER 缓冲区"""
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        执行一步 Double DQN + PER 训练

        Returns:
            (loss, mean_q): 损失值和平均 Q 值，或 (None, None) 如果缓冲区不够
        """
        if len(self.buffer) < self.min_buffer_size:
            return None, None

        # 从 PER 采样
        batch, indices, weights = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch

        # 转为张量
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        # 当前 Q 值
        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN 目标计算
        with torch.no_grad():
            # online 网络选动作
            best_actions = self.online_net(next_states_t).argmax(dim=1)
            # target 网络评估
            next_q = self.target_net(next_states_t).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        # TD error（用于更新 PER 优先级）
        td_errors = (current_q - target_q).detach().cpu().numpy()

        # 加权 Huber Loss
        element_loss = self.loss_fn(current_q, target_q)
        loss = (element_loss * weights_t).mean()

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.gradient_clip)
        self.optimizer.step()

        # 更新 PER 优先级
        self.buffer.update_priorities(indices, td_errors)

        # 学习率调度
        if self.scheduler is not None:
            self.scheduler.step()

        # 更新目标网络
        self.train_steps += 1
        if self.soft_update_enabled:
            self._soft_update_target()
        elif self.train_steps % self.target_update_freq == 0:
            self._hard_update_target()

        return loss.item(), current_q.mean().item()

    def _hard_update_target(self):
        """硬更新目标网络：θ⁻ ← θ"""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _soft_update_target(self):
        """Polyak 软更新目标网络：θ⁻ ← τθ + (1-τ)θ⁻"""
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def update_target(self):
        """公共接口（向后兼容）"""
        if self.soft_update_enabled:
            self._soft_update_target()
        else:
            self._hard_update_target()

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_checkpoint(self, filepath):
        """保存完整 checkpoint"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
            "buffer_size": len(self.buffer),
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """加载 checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.train_steps = checkpoint["train_steps"]


def train_dqn(env, agent, n_episodes=None, episode_length=None,
              checkpoint_interval=None, verbose=True):
    """
    DQN 训练循环
    含进度监控、checkpoint 保存、GPU 状态打印

    Args:
        env: CryptoTradingEnv（mode='continuous'）
        agent: DQNAgent
        n_episodes: 训练轮数
        episode_length: 每轮步数
        checkpoint_interval: 保存间隔
        verbose: 是否打印进度
    Returns:
        dict: 训练记录
    """
    p = DQN_PARAMS
    n_episodes = n_episodes or p["n_episodes"]
    episode_length = episode_length or p["episode_length"]
    checkpoint_interval = checkpoint_interval or p["checkpoint_interval"]
    print_interval = PROGRESS_CONFIG["print_interval_dqn"]

    if verbose:
        print(f"  DQN 训练设备: {agent.device}")
        if agent.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 训练记录
    history = {
        "episode_rewards": [],
        "episode_portfolio_values": [],
        "losses": [],
        "q_values_mean": [],
        "epsilon_history": [],
    }

    start_time = time.time()
    total_steps = 0

    for episode in range(n_episodes):
        state = env.reset(episode_length=episode_length)
        total_reward = 0.0
        episode_losses = []
        episode_q_values = []

        for step in range(episode_length):
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 存入缓冲区
            agent.store_transition(state, action, reward, next_state, done)

            # 训练一步
            loss, mean_q = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
                episode_q_values.append(mean_q)

            total_reward += reward
            state = next_state
            total_steps += 1

            if done:
                break

        # 衰减 epsilon
        agent.decay_epsilon()

        # 记录
        history["episode_rewards"].append(total_reward)
        history["episode_portfolio_values"].append(info["portfolio_value"])
        history["losses"].append(np.mean(episode_losses) if episode_losses else 0)
        history["q_values_mean"].append(np.mean(episode_q_values) if episode_q_values else 0)
        history["epsilon_history"].append(agent.epsilon)

        # 进度打印
        if verbose and (episode + 1) % print_interval == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed
            eta = (n_episodes - episode - 1) / eps_per_sec if eps_per_sec > 0 else 0

            avg_reward = np.mean(history["episode_rewards"][-print_interval:])
            avg_loss = np.mean(history["losses"][-print_interval:])
            avg_q = np.mean(history["q_values_mean"][-print_interval:])

            print(
                f"  Episode {episode+1:>4d}/{n_episodes} | "
                f"Reward: {avg_reward:>8.4f} | "
                f"Portfolio: ${info['portfolio_value']:>10.2f} | "
                f"Loss: {avg_loss:.6f} | "
                f"Q: {avg_q:.4f} | "
                f"ε: {agent.epsilon:.4f} | "
                f"Buffer: {len(agent.buffer):>6d} | "
                f"ETA: {eta:.0f}s"
            )

        # Checkpoint 保存
        if checkpoint_interval > 0 and (episode + 1) % checkpoint_interval == 0:
            ckpt_path = CHECKPOINT_DIR / f"dqn_ep{episode+1}.pt"
            agent.save_checkpoint(ckpt_path)

    total_time = time.time() - start_time
    if verbose:
        print(f"\n  DQN 训练完成: {total_time:.1f}s, "
              f"总步数: {total_steps}, Buffer: {len(agent.buffer)}")

    history["training_time"] = total_time
    return history
