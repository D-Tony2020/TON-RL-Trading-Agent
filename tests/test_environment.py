"""
交易环境测试 -- 验证 CryptoTradingEnv 的核心逻辑（含做空）
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import load_and_prepare_ton
from src.environment import CryptoTradingEnv, BUY, HOLD, SELL, SHORT, COVER


@pytest.fixture(scope="module")
def data():
    """加载数据（module scope 避免重复加载）"""
    full_df, train_df, test_df = load_and_prepare_ton()
    return full_df, train_df, test_df


class TestDiscreteEnv:
    """离散模式环境测试（Q-Learning 用）"""

    def test_reset_state_is_tuple(self, data):
        """reset 应返回 tuple 类型的离散状态"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete")
        state = env.reset(episode_length=100)
        assert isinstance(state, tuple), f"预期 tuple，实际 {type(state)}"

    def test_reset_state_dimension(self, data):
        """离散状态应为 6 维 tuple"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete")
        state = env.reset(episode_length=100)
        assert len(state) == 6, f"预期 6 维，实际 {len(state)}"

    def test_step_returns_correct_format(self, data):
        """step 应返回 (state, reward, done, info) 四元组"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete")
        env.reset(episode_length=100)
        result = env.step(HOLD)
        assert len(result) == 4
        state, reward, done, info = result
        assert isinstance(state, tuple)
        assert isinstance(reward, (int, float))
        assert isinstance(done, (bool, np.bool_))
        assert isinstance(info, dict)


class TestContinuousEnv:
    """连续模式环境测试（DQN 用）"""

    def test_reset_state_is_array(self, data):
        """reset 应返回 numpy 数组"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="continuous")
        state = env.reset(episode_length=100)
        assert isinstance(state, np.ndarray)

    def test_reset_state_shape(self, data):
        """连续状态应为 shape (7,)"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="continuous")
        state = env.reset(episode_length=100)
        assert state.shape == (7,), f"预期 (7,)，实际 {state.shape}"

    def test_state_in_range(self, data):
        """连续状态值应在 [0, 1] 范围内"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="continuous")
        state = env.reset(episode_length=100)
        assert (state >= 0.0).all() and (state <= 1.0).all(), f"状态值越界: {state}"


class TestTrading:
    """交易逻辑测试（多头 + 空头）"""

    def test_buy_changes_position(self, data):
        """空仓 BUY -> position 变为 1"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete")
        env.reset(episode_length=100)
        assert env.position == 0  # 初始空仓
        env.step(BUY)
        assert env.position == 1

    def test_sell_changes_position(self, data):
        """满仓 SELL -> position 变为 0"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete")
        env.reset(episode_length=100)
        env.step(BUY)  # 先买入
        assert env.position == 1
        env.step(SELL)
        assert env.position == 0

    def test_short_changes_position(self, data):
        """空仓 SHORT -> position 变为 -1"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete")
        env.reset(episode_length=100)
        assert env.position == 0
        env.step(SHORT)
        assert env.position == -1

    def test_cover_changes_position(self, data):
        """空头 COVER -> position 变为 0"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete")
        env.reset(episode_length=100)
        env.step(SHORT)  # 先做空
        assert env.position == -1
        env.step(COVER)
        assert env.position == 0

    def test_illegal_sell_when_empty(self, data):
        """空仓 SELL -> 映射为 HOLD，position 不变"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete")
        env.reset(episode_length=100)
        assert env.position == 0
        _, _, _, info = env.step(SELL)
        assert env.position == 0
        assert info["action_taken"] == HOLD

    def test_illegal_buy_when_full(self, data):
        """满仓 BUY -> 映射为 HOLD，position 不变"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete")
        env.reset(episode_length=100)
        env.step(BUY)
        assert env.position == 1
        _, _, _, info = env.step(BUY)
        assert env.position == 1
        assert info["action_taken"] == HOLD

    def test_illegal_short_when_long(self, data):
        """多头 SHORT -> 映射为 HOLD"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete")
        env.reset(episode_length=100)
        env.step(BUY)
        assert env.position == 1
        _, _, _, info = env.step(SHORT)
        assert env.position == 1
        assert info["action_taken"] == HOLD

    def test_illegal_cover_when_not_short(self, data):
        """非空头 COVER -> 映射为 HOLD"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete")
        env.reset(episode_length=100)
        assert env.position == 0
        _, _, _, info = env.step(COVER)
        assert env.position == 0
        assert info["action_taken"] == HOLD

    def test_illegal_buy_when_short(self, data):
        """空头 BUY -> 映射为 HOLD（需先 COVER 再 BUY）"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete")
        env.reset(episode_length=100)
        env.step(SHORT)
        assert env.position == -1
        _, _, _, info = env.step(BUY)
        assert env.position == -1
        assert info["action_taken"] == HOLD

    def test_transaction_cost_on_trade(self, data):
        """BUY/SELL/SHORT/COVER 时应扣除交易成本"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete", cost_rate=0.001)
        env.reset(episode_length=100)
        _, _, _, info = env.step(BUY)
        assert info["transaction_cost"] > 0

    def test_transaction_cost_on_short(self, data):
        """SHORT 时应扣除交易成本"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete", cost_rate=0.001)
        env.reset(episode_length=100)
        _, _, _, info = env.step(SHORT)
        assert info["transaction_cost"] > 0

    def test_hold_no_cost(self, data):
        """HOLD 不应有交易成本"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete")
        env.reset(episode_length=100)
        _, _, _, info = env.step(HOLD)
        assert info["transaction_cost"] == 0.0

    def test_episode_termination(self, data):
        """episode 应在指定步数后终止"""
        _, train_df, _ = data
        episode_length = 50
        env = CryptoTradingEnv(train_df, mode="discrete")
        env.reset(episode_length=episode_length)

        for i in range(episode_length - 1):
            _, _, done, _ = env.step(HOLD)
            assert not done, f"在第 {i+1} 步提前终止"

        _, _, done, _ = env.step(HOLD)
        assert done, "在最后一步未终止"


class TestPortfolioConservation:
    """资金守恒测试"""

    def test_hold_empty_conservation(self, data):
        """空仓全 HOLD：portfolio_value 应保持不变（=初始资金）"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete", initial_balance=10000)
        env.reset(episode_length=50)

        for _ in range(50):
            _, _, done, info = env.step(HOLD)
            assert abs(info["portfolio_value"] - 10000.0) < 0.01, (
                f"空仓 HOLD 后资金变化: {info['portfolio_value']}"
            )
            if done:
                break

    def test_buy_hold_tracks_price(self, data):
        """买入后全 HOLD：portfolio 应跟随价格变化"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete", initial_balance=10000, cost_rate=0)
        env.reset(start_idx=0, episode_length=10)

        # 买入
        env.step(BUY)

        # HOLD 几步
        for _ in range(5):
            _, _, _, info = env.step(HOLD)

        # portfolio value 应约等于 holdings * current_price
        expected_value = env.holdings * info["current_price"]
        assert abs(info["portfolio_value"] - expected_value) < 0.01

    def test_short_hold_inverse_price(self, data):
        """做空后全 HOLD：portfolio 应反向跟随价格"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete", initial_balance=10000, cost_rate=0)
        env.reset(start_idx=0, episode_length=10)

        # 做空
        env.step(SHORT)
        entry_price = env.short_entry_price
        short_qty = env.short_holdings

        # HOLD 几步
        for _ in range(5):
            _, _, _, info = env.step(HOLD)

        # portfolio = short_qty * entry_price + short_qty * (entry_price - current_price)
        current_price = info["current_price"]
        expected_value = short_qty * entry_price + short_qty * (entry_price - current_price)
        assert abs(info["portfolio_value"] - expected_value) < 0.01


class TestRewardModes:
    """奖励模式测试"""

    def test_simple_reward_returns_float(self, data):
        """simple 模式应返回 float 类型奖励"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete", reward_mode="simple")
        env.reset(episode_length=50)
        _, reward, _, _ = env.step(HOLD)
        assert isinstance(reward, (int, float))

    def test_sharpe_reward_returns_float(self, data):
        """sharpe 模式应返回 float 类型奖励"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="continuous", reward_mode="sharpe")
        env.reset(episode_length=50)
        _, reward, _, _ = env.step(HOLD)
        assert isinstance(reward, (int, float))

    def test_simple_reward_hold_empty_is_zero(self, data):
        """simple 模式下空仓 HOLD 奖励应为 0"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete", reward_mode="simple")
        env.reset(episode_length=50)
        _, reward, _, _ = env.step(HOLD)
        assert reward == 0.0, f"空仓 HOLD 奖励不为 0: {reward}"

    def test_reward_no_nan(self, data):
        """运行一个完整 episode，包含全部 5 种动作，奖励不应有 NaN"""
        _, train_df, _ = data
        for reward_mode in ["simple", "sharpe"]:
            env = CryptoTradingEnv(train_df, mode="continuous", reward_mode=reward_mode)
            env.reset(episode_length=100)
            for _ in range(100):
                action = np.random.choice([BUY, HOLD, SELL, SHORT, COVER])
                _, reward, done, _ = env.step(action)
                assert not np.isnan(reward), f"reward_mode={reward_mode} 产生 NaN 奖励"
                if done:
                    break


class TestInfoDict:
    """info 字典完整性测试"""

    def test_info_keys(self, data):
        """info 应包含所有必要字段"""
        _, train_df, _ = data
        env = CryptoTradingEnv(train_df, mode="discrete")
        env.reset(episode_length=10)
        _, _, _, info = env.step(HOLD)

        required_keys = [
            "portfolio_value", "position", "action_taken",
            "transaction_cost", "current_price", "balance", "holdings",
            "short_holdings",
        ]
        for key in required_keys:
            assert key in info, f"info 缺少字段: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
