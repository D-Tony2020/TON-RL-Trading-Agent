"""
Agent 测试 — 验证 Q-Learning 和 DQN Agent 的核心逻辑
"""
import numpy as np
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.q_learning import QLearningAgent, train_qlearning
from src.agents.dqn import DuelingDQN, PrioritizedReplayBuffer, DQNAgent, train_dqn
from src.data_pipeline import load_and_prepare_ton
from src.environment import CryptoTradingEnv, BUY, HOLD, SELL, SHORT, COVER


@pytest.fixture(scope="module")
def train_data():
    """加载训练数据"""
    _, train_df, _ = load_and_prepare_ton()
    return train_df


# ============================================================
# Q-Learning Agent 测试
# ============================================================
class TestQLearningAgent:

    def test_select_action_returns_valid(self):
        """动作应在 [0, n_actions) 范围内"""
        agent = QLearningAgent(n_actions=3)
        state = (3, 2, 1, 2, 1, 0)
        for _ in range(100):
            action = agent.select_action(state)
            assert 0 <= action < 3

    def test_update_changes_q_value(self):
        """update 应改变 Q 值"""
        agent = QLearningAgent(n_actions=3, alpha=0.1, gamma=0.97)
        state = (0, 0, 0, 0, 0, 0)
        action = 1
        reward = 1.0
        next_state = (0, 0, 0, 0, 0, 1)

        q_before = agent.q_table[state][action]
        agent.update(state, action, reward, next_state, done=False)
        q_after = agent.q_table[state][action]

        assert q_after != q_before, "Q 值未更新"

    def test_update_formula_correctness(self):
        """验证 Q-Learning 更新公式：Q ← Q + α(r + γ·maxQ' - Q)"""
        agent = QLearningAgent(n_actions=3, alpha=0.1, gamma=0.9)
        state = (1, 1, 1, 1, 1, 0)
        action = 0
        reward = 0.5
        next_state = (2, 2, 2, 2, 2, 1)

        # 手动设置 next_state 的 Q 值
        agent.q_table[next_state] = np.array([0.3, 0.8, 0.1])

        q_before = agent.q_table[state][action]  # 0.0
        agent.update(state, action, reward, next_state, done=False)
        q_after = agent.q_table[state][action]

        # 预期：Q = 0 + 0.1 * (0.5 + 0.9 * 0.8 - 0) = 0.1 * 1.22 = 0.122
        expected = q_before + 0.1 * (reward + 0.9 * 0.8 - q_before)
        assert abs(q_after - expected) < 1e-6, f"Q={q_after}, 预期={expected}"

    def test_epsilon_decay(self):
        """epsilon 应按衰减系数递减"""
        agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.9)
        agent.decay_epsilon()
        assert abs(agent.epsilon - 0.9) < 1e-6

    def test_epsilon_min_floor(self):
        """epsilon 不应低于 epsilon_min"""
        agent = QLearningAgent(epsilon=0.06, epsilon_min=0.05, epsilon_decay=0.5)
        agent.decay_epsilon()
        assert agent.epsilon >= 0.05

    def test_q_table_size_grows(self):
        """update 新状态后 Q 表应增长"""
        agent = QLearningAgent()
        states = [(i, 0, 0, 0, 0, 0) for i in range(5)]
        next_s = (0, 0, 0, 0, 0, 1)
        for s in states:
            agent.update(s, 0, 0.1, next_s, done=False)
        assert agent.q_table_size >= 5

    def test_checkpoint_save_load(self, tmp_path):
        """checkpoint 保存和加载应保持 Q 表一致"""
        agent = QLearningAgent()
        state = (1, 2, 3, 1, 2, 0)
        agent.q_table[state] = np.array([0.1, 0.5, -0.3])
        agent.epsilon = 0.42

        filepath = tmp_path / "test_ckpt.pkl"
        agent.save_checkpoint(filepath)

        agent2 = QLearningAgent()
        agent2.load_checkpoint(filepath)

        assert np.allclose(agent2.q_table[state], np.array([0.1, 0.5, -0.3]))
        assert abs(agent2.epsilon - 0.42) < 1e-6


class TestQLearningTraining:
    """Q-Learning 短训练冒烟测试"""

    def test_smoke_train(self, train_data):
        """10 个 episode 的短训练不应崩溃"""
        env = CryptoTradingEnv(train_data, mode="discrete", reward_mode="simple")
        agent = QLearningAgent()
        history = train_qlearning(env, agent, n_episodes=10, episode_length=100,
                                  checkpoint_interval=0, verbose=False)

        assert len(history["episode_rewards"]) == 10
        assert agent.q_table_size > 0
        assert all(not np.isnan(r) for r in history["episode_rewards"])


# ============================================================
# DQN Agent 测试
# ============================================================
class TestDuelingDQN:

    def test_forward_shape(self):
        """网络输出 shape 应为 (batch, n_actions)"""
        net = DuelingDQN(state_dim=7, n_actions=3)
        x = torch.randn(32, 7)
        out = net(x)
        assert out.shape == (32, 3), f"预期 (32,3)，实际 {out.shape}"

    def test_single_input(self):
        """单个输入也应工作"""
        net = DuelingDQN(state_dim=7, n_actions=3)
        x = torch.randn(1, 7)
        out = net(x)
        assert out.shape == (1, 3)

    def test_dueling_decomposition(self):
        """V + A - mean(A) = Q，验证 Dueling 分解"""
        net = DuelingDQN(state_dim=7, n_actions=3)
        x = torch.randn(4, 7)

        # 获取中间结果
        features = net.feature(x)
        v = net.v_stream(features)
        a = net.a_stream(features)
        expected_q = v + (a - a.mean(dim=1, keepdim=True))

        actual_q = net(x)
        assert torch.allclose(actual_q, expected_q, atol=1e-6)


class TestPER:

    def test_push_and_size(self):
        """push 后 buffer 大小应增加"""
        buffer = PrioritizedReplayBuffer(capacity=100)
        state = np.zeros(7, dtype=np.float32)
        buffer.push(state, 0, 1.0, state, False)
        assert len(buffer) == 1

    def test_sample_batch(self):
        """采样应返回正确格式"""
        buffer = PrioritizedReplayBuffer(capacity=100)
        for _ in range(20):
            s = np.random.randn(7).astype(np.float32)
            buffer.push(s, np.random.randint(3), np.random.randn(), s, False)

        batch, indices, weights = buffer.sample(8)
        states, actions, rewards, next_states, dones = batch

        assert states.shape == (8, 7)
        assert actions.shape == (8,)
        assert rewards.shape == (8,)
        assert next_states.shape == (8, 7)
        assert dones.shape == (8,)
        assert len(indices) == 8
        assert weights.shape == (8,)

    def test_priority_update(self):
        """高 TD error 的 transition 应被更频繁采样"""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=1.0)

        # 添加10个低优先级 + 1个高优先级
        low_state = np.zeros(7, dtype=np.float32)
        high_state = np.ones(7, dtype=np.float32)

        for _ in range(10):
            buffer.push(low_state, 0, 0.0, low_state, False)
        buffer.push(high_state, 1, 1.0, high_state, False)

        # 更新优先级：让最后一个有极高 TD error
        last_idx = buffer.tree.write_idx - 1 + buffer.tree.capacity - 1
        buffer.tree.update(last_idx, 100.0)

        # 采样100次，高优先级应被频繁采样
        high_count = 0
        for _ in range(100):
            batch, _, _ = buffer.sample(1)
            if np.allclose(batch[0][0], high_state):
                high_count += 1

        # 高优先级应占多数
        assert high_count > 50, f"高优先级被采样 {high_count}/100 次，太少"


class TestDQNAgent:

    def test_select_action_valid(self):
        """动作应在 [0, n_actions) 范围内"""
        agent = DQNAgent(state_dim=7, n_actions=3)
        state = np.random.randn(7).astype(np.float32)
        for _ in range(50):
            action = agent.select_action(state)
            assert 0 <= action < 3

    def test_double_dqn_target(self):
        """验证 Double DQN：online 选动作，target 评估"""
        agent = DQNAgent(state_dim=7, n_actions=3)

        # 填充缓冲区
        for _ in range(100):
            s = np.random.randn(7).astype(np.float32)
            a = np.random.randint(3)
            r = np.random.randn()
            s2 = np.random.randn(7).astype(np.float32)
            agent.store_transition(s, a, r, s2, False)

        agent.min_buffer_size = 50
        loss, q_val = agent.train_step()

        # 应能成功训练（非 None）
        assert loss is not None
        assert q_val is not None
        assert not np.isnan(loss)
        assert not np.isnan(q_val)

    def test_device_detection(self):
        """设备应为 cuda 或 cpu"""
        agent = DQNAgent()
        assert agent.device.type in ["cuda", "cpu"]

    def test_checkpoint_save_load(self, tmp_path):
        """checkpoint 保存和加载后网络参数应一致"""
        agent = DQNAgent(state_dim=7, n_actions=3)
        agent.epsilon = 0.33
        agent.train_steps = 42

        filepath = tmp_path / "test_dqn.pt"
        agent.save_checkpoint(filepath)

        agent2 = DQNAgent(state_dim=7, n_actions=3)
        agent2.load_checkpoint(filepath)

        assert abs(agent2.epsilon - 0.33) < 1e-6
        assert agent2.train_steps == 42

        # 网络参数应一致
        for p1, p2 in zip(agent.online_net.parameters(), agent2.online_net.parameters()):
            assert torch.allclose(p1, p2)


class TestDQNTraining:
    """DQN 短训练冒烟测试"""

    def test_smoke_train(self, train_data):
        """5 个 episode 的短训练不应崩溃"""
        env = CryptoTradingEnv(train_data, mode="continuous", reward_mode="simple")
        agent = DQNAgent(state_dim=7, n_actions=5, min_buffer_size=200)
        history = train_dqn(env, agent, n_episodes=5, episode_length=100,
                            checkpoint_interval=0, verbose=False)

        assert len(history["episode_rewards"]) == 5
        assert all(not np.isnan(r) for r in history["episode_rewards"])

        # Loss 不应全为 NaN（至少后几个 episode 有值）
        non_zero_losses = [l for l in history["losses"] if l > 0]
        assert len(non_zero_losses) > 0, "所有 episode 的 loss 都是 0（缓冲区可能太小）"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
