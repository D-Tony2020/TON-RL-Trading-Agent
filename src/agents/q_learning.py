"""
Q-Learning Agent — 表格型 TD 方法
更新公式：Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
"""
import numpy as np
import pickle
import time
from collections import defaultdict
from pathlib import Path
from src.config import QLEARNING_PARAMS, CHECKPOINT_DIR, PROGRESS_CONFIG


class QLearningAgent:
    """
    Q-Learning Agent（Off-policy TD Control）

    Args:
        n_actions: 动作数量（默认3：BUY/HOLD/SELL）
        alpha: 学习率
        gamma: 折扣因子
        epsilon: 初始探索率
        epsilon_min: 最小探索率
        epsilon_decay: 探索率衰减系数
    """

    def __init__(
        self,
        n_actions=None,
        alpha=None,
        gamma=None,
        epsilon=None,
        epsilon_min=None,
        epsilon_decay=None,
    ):
        p = QLEARNING_PARAMS
        self.n_actions = n_actions if n_actions is not None else p.get("n_actions", 5)
        self.alpha = alpha if alpha is not None else p["alpha"]
        self.gamma = gamma if gamma is not None else p["gamma"]
        self.epsilon = epsilon if epsilon is not None else p["epsilon"]
        self.epsilon_min = epsilon_min if epsilon_min is not None else p["epsilon_min"]
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else p["epsilon_decay"]

        # Q 表：defaultdict 自动为新状态初始化零向量
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))

    def select_action(self, state):
        """
        ε-greedy 动作选择

        Args:
            state: 离散状态 tuple
        Returns:
            int: 动作索引
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.q_table[state]
            # 如果有多个最大值，随机选一个（打破平局）
            max_q = np.max(q_values)
            max_actions = np.where(q_values == max_q)[0]
            return np.random.choice(max_actions)

    def select_greedy_action(self, state):
        """纯贪心动作选择（用于评估/回测）
        当多个动作 Q 值相同时, 选索引最小的（确定性, 保证可复现）"""
        q_values = self.q_table[state]
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning 更新：Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        Returns:
            float: TD error（供监控用）
        """
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        td_error = target - current_q
        self.q_table[state][action] = current_q + self.alpha * td_error

        return td_error

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_checkpoint(self, filepath):
        """
        保存 checkpoint（Q表 + 超参数状态）

        Args:
            filepath: 保存路径
        """
        checkpoint = {
            "q_table": dict(self.q_table),  # 转为普通 dict 以便 pickle
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "n_actions": self.n_actions,
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, filepath):
        """
        加载 checkpoint

        Args:
            filepath: checkpoint 路径
        """
        with open(filepath, "rb") as f:
            checkpoint = pickle.load(f)

        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        self.q_table.update(checkpoint["q_table"])
        self.epsilon = checkpoint["epsilon"]

    @property
    def q_table_size(self):
        """已访问的状态数"""
        return len(self.q_table)


def train_qlearning(env, agent, n_episodes=None, episode_length=None,
                    checkpoint_interval=None, verbose=True):
    """
    Q-Learning 训练循环
    含进度监控、checkpoint 保存

    Args:
        env: CryptoTradingEnv（mode='discrete'）
        agent: QLearningAgent
        n_episodes: 训练轮数
        episode_length: 每轮步数
        checkpoint_interval: 保存间隔
        verbose: 是否打印进度
    Returns:
        dict: 训练记录
    """
    p = QLEARNING_PARAMS
    n_episodes = n_episodes or p["n_episodes"]
    episode_length = episode_length or p["episode_length"]
    checkpoint_interval = checkpoint_interval or p["checkpoint_interval"]
    print_interval = PROGRESS_CONFIG["print_interval_qlearning"]

    # 训练记录
    history = {
        "episode_rewards": [],
        "episode_portfolio_values": [],
        "q_table_sizes": [],
        "epsilon_history": [],
        "td_errors": [],
    }

    start_time = time.time()

    for episode in range(n_episodes):
        state = env.reset(episode_length=episode_length)
        total_reward = 0.0
        episode_td_errors = []

        for step in range(episode_length):
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 更新 Q 表
            td_error = agent.update(state, action, reward, next_state, done)
            episode_td_errors.append(abs(td_error))

            total_reward += reward
            state = next_state

            if done:
                break

        # 衰减 epsilon
        agent.decay_epsilon()

        # 记录
        history["episode_rewards"].append(total_reward)
        history["episode_portfolio_values"].append(info["portfolio_value"])
        history["q_table_sizes"].append(agent.q_table_size)
        history["epsilon_history"].append(agent.epsilon)
        history["td_errors"].append(np.mean(episode_td_errors) if episode_td_errors else 0)

        # 进度打印
        if verbose and (episode + 1) % print_interval == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed
            eta = (n_episodes - episode - 1) / eps_per_sec if eps_per_sec > 0 else 0

            avg_reward = np.mean(history["episode_rewards"][-print_interval:])
            avg_td = np.mean(history["td_errors"][-print_interval:])

            print(
                f"  Episode {episode+1:>4d}/{n_episodes} | "
                f"Reward: {avg_reward:>8.4f} | "
                f"Portfolio: ${info['portfolio_value']:>10.2f} | "
                f"Q-states: {agent.q_table_size:>5d} | "
                f"ε: {agent.epsilon:.4f} | "
                f"TD: {avg_td:.6f} | "
                f"ETA: {eta:.0f}s"
            )

        # Checkpoint 保存
        if checkpoint_interval > 0 and (episode + 1) % checkpoint_interval == 0:
            ckpt_path = CHECKPOINT_DIR / f"qlearning_ep{episode+1}.pkl"
            agent.save_checkpoint(ckpt_path)

    total_time = time.time() - start_time
    if verbose:
        print(f"\n  Q-Learning 训练完成: {total_time:.1f}s, "
              f"Q 表大小: {agent.q_table_size} states")

    history["training_time"] = total_time
    return history
