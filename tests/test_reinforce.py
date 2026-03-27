"""
REINFORCE Agent 测试
"""
import os
import sys
import tempfile
import numpy as np
import torch
import pytest

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.reinforce import PolicyNetwork, ValueNetwork, REINFORCEAgent, train_reinforce
from src.environment import CryptoTradingEnv
from src.data_pipeline import load_and_prepare_ton


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture(scope="module")
def data():
    """加载数据（模块级缓存）"""
    full_df, train_df, test_df = load_and_prepare_ton()
    return full_df, train_df, test_df


# ============================================================
# PolicyNetwork 测试
# ============================================================
class TestPolicyNetwork:
    def test_output_shape(self):
        """输出 shape 正确"""
        net = PolicyNetwork(state_dim=8, n_actions=5, hidden_dim=128)
        x = torch.randn(4, 8)  # batch=4
        out = net(x)
        assert out.shape == (4, 5)

    def test_logits_output(self):
        """输出为 logits（可正可负，不受 Softmax 约束）"""
        net = PolicyNetwork(state_dim=8, n_actions=5)
        x = torch.randn(10, 8)
        logits = net(x)
        assert logits.shape == (10, 5)
        # logits 不需要全正或和为 1，Categorical(logits=) 内部处理

    def test_categorical_compatible(self):
        """logits 可以直接用于 Categorical 分布"""
        net = PolicyNetwork(state_dim=8, n_actions=5)
        x = torch.randn(1, 8)
        logits = net(x)
        from torch.distributions import Categorical
        dist = Categorical(logits=logits)
        action = dist.sample()
        assert 0 <= action.item() < 5


# ============================================================
# ValueNetwork 测试
# ============================================================
class TestValueNetwork:
    def test_output_shape(self):
        """输出 shape 正确"""
        net = ValueNetwork(state_dim=8, hidden_dim=128)
        x = torch.randn(4, 8)
        out = net(x)
        assert out.shape == (4, 1)

    def test_scalar_output(self):
        """单个输入返回标量"""
        net = ValueNetwork(state_dim=8)
        x = torch.randn(1, 8)
        out = net(x)
        assert out.numel() == 1


# ============================================================
# REINFORCEAgent 测试
# ============================================================
class TestREINFORCEAgent:
    def test_select_action_range(self):
        """select_action 返回值在合法范围内"""
        agent = REINFORCEAgent()
        state = np.random.randn(8).astype(np.float32)
        for _ in range(50):
            action = agent.select_action(state)
            assert 0 <= action < 5
        # 清空 buffer 防止影响其他测试
        agent.saved_states = []
        agent.log_probs = []
        agent.rewards = []
        agent.entropies = []

    def test_select_greedy_deterministic(self):
        """select_greedy_action 确定性"""
        agent = REINFORCEAgent()
        state = np.random.randn(8).astype(np.float32)
        actions = [agent.select_greedy_action(state) for _ in range(20)]
        assert len(set(actions)) == 1, "贪心动作应完全确定"

    def test_finish_episode(self):
        """finish_episode 正常执行，无 NaN"""
        agent = REINFORCEAgent()
        state = np.random.randn(8).astype(np.float32)
        # 模拟 5 步 episode
        for _ in range(5):
            agent.select_action(state)
            agent.store_reward(np.random.randn())
        p_loss, v_loss, entropy = agent.finish_episode()
        assert not np.isnan(p_loss), "Policy loss 不应为 NaN"
        assert not np.isnan(v_loss), "Value loss 不应为 NaN"
        assert not np.isnan(entropy), "Entropy 不应为 NaN"

    def test_checkpoint_roundtrip(self):
        """save/load checkpoint 保持参数一致"""
        agent = REINFORCEAgent()
        # 先做一些更新，确保参数非初始值
        state = np.random.randn(8).astype(np.float32)
        for _ in range(5):
            agent.select_action(state)
            agent.store_reward(1.0)
        agent.finish_episode()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name

        try:
            agent.save_checkpoint(ckpt_path)

            # 创建新 agent 并加载
            agent2 = REINFORCEAgent()
            agent2.load_checkpoint(ckpt_path)

            # 验证参数一致
            for p1, p2 in zip(agent.policy_net.parameters(),
                              agent2.policy_net.parameters()):
                assert torch.allclose(p1, p2), "Policy net 参数应一致"
            for p1, p2 in zip(agent.value_net.parameters(),
                              agent2.value_net.parameters()):
                assert torch.allclose(p1, p2), "Value net 参数应一致"
        finally:
            os.unlink(ckpt_path)

    def test_decay_epsilon_noop(self):
        """decay_epsilon 不报错"""
        agent = REINFORCEAgent()
        agent.decay_epsilon()  # 不应抛异常


# ============================================================
# 冒烟测试：训练 5 个 episode
# ============================================================
class TestREINFORCETraining:
    def test_smoke_train(self, data):
        """5 episode 冒烟测试：不崩溃、无 NaN"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="continuous", reward_mode="simple")
        agent = REINFORCEAgent()

        history = train_reinforce(
            env, agent,
            n_episodes=5,
            episode_length=100,
            checkpoint_interval=999,  # 不保存
            verbose=False,
        )

        assert len(history["episode_rewards"]) == 5
        assert len(history["policy_losses"]) == 5
        assert len(history["value_losses"]) == 5
        assert all(not np.isnan(r) for r in history["episode_rewards"]), "Rewards 不应含 NaN"
        assert all(not np.isnan(l) for l in history["policy_losses"]), "Policy losses 不应含 NaN"
        assert history["training_time"] > 0


# ============================================================
# 环境奖励模式测试
# ============================================================
class TestTraderRewardModes:
    @pytest.mark.parametrize("mode", ["arbitrageur", "manipulator", "retail"])
    def test_reward_mode_runs(self, data, mode):
        """三种 trader 奖励模式不崩溃"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="continuous", reward_mode=mode)
        state = env.reset(episode_length=50)
        for _ in range(10):
            action = np.random.randint(0, 5)
            state, reward, done, info = env.step(action)
            assert not np.isnan(reward), f"{mode} 模式 reward 不应为 NaN"
            if done:
                break
