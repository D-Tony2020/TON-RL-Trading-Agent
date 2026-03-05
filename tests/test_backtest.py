"""
回测引擎测试
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest import (
    compute_metrics, count_trades, backtest_buy_and_hold,
    backtest_random, backtest_rsi_rule,
)
from src.data_pipeline import load_and_prepare_ton
from src.environment import CryptoTradingEnv, BUY, HOLD, SELL, SHORT, COVER


@pytest.fixture(scope="module")
def test_data():
    _, _, test_df = load_and_prepare_ton()
    return test_df


class TestComputeMetrics:

    def test_perfect_return(self):
        """从10000涨到20000，收益率应为100%"""
        pv = np.linspace(10000, 20000, 100)
        metrics = compute_metrics(pv)
        assert abs(metrics["total_return"] - 1.0) < 0.01

    def test_zero_return(self):
        """恒定10000，收益率应为0%"""
        pv = np.full(100, 10000.0)
        metrics = compute_metrics(pv)
        assert abs(metrics["total_return"]) < 1e-6

    def test_negative_return(self):
        """从10000跌到5000，收益率应为-50%"""
        pv = np.linspace(10000, 5000, 100)
        metrics = compute_metrics(pv)
        assert abs(metrics["total_return"] - (-0.5)) < 0.01

    def test_max_drawdown_range(self):
        """最大回撤应在 [0, 1] 范围内"""
        pv = np.array([10000, 12000, 8000, 11000, 7000, 9000])
        metrics = compute_metrics(pv)
        assert 0 <= metrics["max_drawdown"] <= 1

    def test_max_drawdown_computation(self):
        """手算最大回撤"""
        pv = np.array([100, 120, 80, 110])
        metrics = compute_metrics(pv)
        # 峰值120，谷值80，回撤 = (120-80)/120 = 33.3%
        assert abs(metrics["max_drawdown"] - (120 - 80) / 120) < 0.01

    def test_sharpe_ratio_positive_for_uptrend(self):
        """稳定上涨的 Sharpe 应为正"""
        pv = np.cumsum(np.ones(1000)) + 10000
        metrics = compute_metrics(pv)
        assert metrics["annualized_sharpe"] > 0

    def test_win_rate_range(self):
        """胜率应在 [0, 1]"""
        pv = np.random.randn(100).cumsum() + 10000
        metrics = compute_metrics(pv)
        assert 0 <= metrics["win_rate"] <= 1


class TestCountTrades:

    def test_all_hold(self):
        """全 HOLD 应为 0 笔交易"""
        actions = np.full(100, HOLD)
        assert count_trades(actions) == 0

    def test_buy_sell_pair(self):
        """一次 BUY + 一次 SELL = 2 笔交易"""
        actions = np.array([BUY, HOLD, HOLD, SELL, HOLD])
        assert count_trades(actions) == 2

    def test_short_cover_pair(self):
        """一次 SHORT + 一次 COVER = 2 笔交易"""
        actions = np.array([SHORT, HOLD, HOLD, COVER, HOLD])
        assert count_trades(actions) == 2

    def test_all_actions_counted(self):
        """BUY + SELL + SHORT + COVER = 4 笔交易"""
        actions = np.array([BUY, SELL, SHORT, COVER, HOLD])
        assert count_trades(actions) == 4


class TestBaselines:

    def test_buy_and_hold_runs(self, test_data):
        """Buy & Hold 应在测试集上完整运行"""
        env = CryptoTradingEnv(test_data, mode="discrete")
        result = backtest_buy_and_hold(env)
        assert len(result["portfolio_values"]) > 100

    def test_buy_and_hold_has_position(self, test_data):
        """Buy & Hold 在买入后应始终持仓"""
        env = CryptoTradingEnv(test_data, mode="discrete")
        result = backtest_buy_and_hold(env)
        # 第一步之后全部应为 position=1
        assert all(p == 1 for p in result["positions"][1:])

    def test_random_runs(self, test_data):
        """Random 策略应在测试集上完整运行"""
        env = CryptoTradingEnv(test_data, mode="discrete")
        result = backtest_random(env)
        assert len(result["portfolio_values"]) > 100

    def test_rsi_rule_runs(self, test_data):
        """RSI Rule 应在测试集上完整运行"""
        env = CryptoTradingEnv(test_data, mode="discrete")
        result = backtest_rsi_rule(env)
        assert len(result["portfolio_values"]) > 100

    def test_random_has_trades(self, test_data):
        """Random 策略应有交易（非全 HOLD）"""
        env = CryptoTradingEnv(test_data, mode="discrete")
        result = backtest_random(env)
        trades = count_trades(result["actions"])
        assert trades > 0, "Random 策略居然零交易"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
