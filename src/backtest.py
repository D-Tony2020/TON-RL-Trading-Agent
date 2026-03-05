"""
回测引擎 + 评估指标 + Baseline 策略
支持 5 动作: BUY/HOLD/SELL/SHORT/COVER
"""
import numpy as np
from src.environment import CryptoTradingEnv, BUY, HOLD, SELL, SHORT, COVER, ACTION_NAMES
from src.config import HOURS_PER_YEAR, RSI_RULE_BUY, RSI_RULE_SELL


def backtest(env, agent, agent_type="qlearning"):
    """
    在测试集上执行贪心回测

    Args:
        env: CryptoTradingEnv（用测试集数据初始化）
        agent: 训练好的 agent（QLearningAgent 或 DQNAgent）
        agent_type: 'qlearning' 或 'dqn'
    Returns:
        dict: 回测结果
    """
    # 用整个测试集做一个 episode
    state = env.reset(start_idx=0, episode_length=len(env.df) - 1)

    portfolio_values = [env.initial_balance]
    actions = []
    positions = []
    regimes = []
    rewards = []

    done = False
    while not done:
        # 贪心动作选择
        if agent_type == "qlearning":
            action = agent.select_greedy_action(state)
        else:
            action = agent.select_greedy_action(state)

        state, reward, done, info = env.step(action)

        portfolio_values.append(info["portfolio_value"])
        actions.append(info["action_taken"])
        positions.append(info["position"])
        regimes.append(env.get_current_regime())
        rewards.append(reward)

    return {
        "portfolio_values": np.array(portfolio_values),
        "actions": np.array(actions),
        "positions": np.array(positions),
        "regimes": np.array(regimes),
        "rewards": np.array(rewards),
    }


def backtest_buy_and_hold(env):
    """
    Buy & Hold 策略回测：t=0 买入，最后卖出

    Args:
        env: CryptoTradingEnv
    Returns:
        dict: 回测结果
    """
    state = env.reset(start_idx=0, episode_length=len(env.df) - 1)

    portfolio_values = [env.initial_balance]
    actions = []
    positions = []

    # 第一步买入
    state, _, done, info = env.step(BUY)
    portfolio_values.append(info["portfolio_value"])
    actions.append(BUY)
    positions.append(info["position"])

    # 之后全部 HOLD
    while not done:
        state, _, done, info = env.step(HOLD)
        portfolio_values.append(info["portfolio_value"])
        actions.append(HOLD)
        positions.append(info["position"])

    return {
        "portfolio_values": np.array(portfolio_values),
        "actions": np.array(actions),
        "positions": np.array(positions),
    }


def backtest_random(env, seed=42):
    """
    随机策略回测：每步等概率 5 个动作

    Args:
        env: CryptoTradingEnv
        seed: 随机种子
    Returns:
        dict: 回测结果
    """
    rng = np.random.RandomState(seed)
    state = env.reset(start_idx=0, episode_length=len(env.df) - 1)

    portfolio_values = [env.initial_balance]
    actions = []
    positions = []

    done = False
    while not done:
        action = rng.randint(5)  # 5 个动作
        state, _, done, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        actions.append(info["action_taken"])
        positions.append(info["position"])

    return {
        "portfolio_values": np.array(portfolio_values),
        "actions": np.array(actions),
        "positions": np.array(positions),
    }


def backtest_rsi_rule(env):
    """
    RSI 规则策略回测（含做空）：
    - RSI < 30 且空仓 -> BUY
    - RSI > 70 且多头 -> SELL
    - RSI > 70 且空仓 -> SHORT
    - RSI < 30 且空头 -> COVER
    - 其余 -> HOLD

    Args:
        env: CryptoTradingEnv
    Returns:
        dict: 回测结果
    """
    state = env.reset(start_idx=0, episode_length=len(env.df) - 1)

    portfolio_values = [env.initial_balance]
    actions = []
    positions = []

    done = False
    while not done:
        rsi = env.df.iloc[env.current_step]["rsi_14"]
        if rsi < RSI_RULE_BUY and env.position == 0:
            action = BUY
        elif rsi < RSI_RULE_BUY and env.position == -1:
            action = COVER
        elif rsi > RSI_RULE_SELL and env.position == 1:
            action = SELL
        elif rsi > RSI_RULE_SELL and env.position == 0:
            action = SHORT
        else:
            action = HOLD

        state, _, done, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        actions.append(info["action_taken"])
        positions.append(info["position"])

    return {
        "portfolio_values": np.array(portfolio_values),
        "actions": np.array(actions),
        "positions": np.array(positions),
    }


def compute_metrics(portfolio_values):
    """
    计算回测性能指标

    Args:
        portfolio_values: 每步的 portfolio 价值序列
    Returns:
        dict: 性能指标
    """
    pv = np.array(portfolio_values, dtype=np.float64)

    # 总收益率
    total_return = (pv[-1] - pv[0]) / pv[0] if pv[0] > 0 else 0.0

    # 每步收益率
    returns = np.diff(pv) / (pv[:-1] + 1e-10)

    # 年化 Sharpe Ratio
    if len(returns) > 1 and np.std(returns) > 0:
        annualized_sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(HOURS_PER_YEAR)
    else:
        annualized_sharpe = 0.0

    # 最大回撤
    cummax = np.maximum.accumulate(pv)
    drawdowns = (cummax - pv) / (cummax + 1e-10)
    max_drawdown = np.max(drawdowns)

    # 胜率（正收益步数占比）
    win_rate = np.mean(returns > 0) if len(returns) > 0 else 0.0

    return {
        "total_return": total_return,
        "annualized_sharpe": annualized_sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "final_value": pv[-1],
        "initial_value": pv[0],
    }


def count_trades(actions):
    """统计交易次数（所有非 HOLD 动作）"""
    return int(np.sum(
        (actions == BUY) | (actions == SELL) |
        (actions == SHORT) | (actions == COVER)
    ))


def analyze_by_regime(backtest_result):
    """
    按 Regime 分析策略行为

    Args:
        backtest_result: backtest() 的返回值（需含 regimes 字段）
    Returns:
        dict: 每种 regime 的指标和动作分布
    """
    regimes = backtest_result.get("regimes", np.array([]))
    actions = backtest_result.get("actions", np.array([]))
    pv = backtest_result.get("portfolio_values", np.array([]))

    if len(regimes) == 0:
        return {}

    analysis = {}
    for regime in ["rise", "decline", "steady"]:
        mask = regimes == regime
        if mask.sum() == 0:
            continue

        # 动作分布（5 个动作）
        regime_actions = actions[mask]
        action_dist = {
            "BUY": float(np.mean(regime_actions == BUY)),
            "HOLD": float(np.mean(regime_actions == HOLD)),
            "SELL": float(np.mean(regime_actions == SELL)),
            "SHORT": float(np.mean(regime_actions == SHORT)),
            "COVER": float(np.mean(regime_actions == COVER)),
        }

        # 该 regime 期间的收益
        regime_indices = np.where(mask)[0]
        if len(regime_indices) > 0:
            pv_in_regime = pv[regime_indices]
            if len(pv_in_regime) > 1:
                regime_return = (pv_in_regime[-1] - pv_in_regime[0]) / (pv_in_regime[0] + 1e-10)
            else:
                regime_return = 0.0
        else:
            regime_return = 0.0

        analysis[regime] = {
            "count": int(mask.sum()),
            "action_distribution": action_dist,
            "regime_return": regime_return,
        }

    return analysis


def run_all_backtests(test_df, q_agent, dqn_agent, reward_mode="simple",
                      initial_balance=10000, cost_rate=0.001):
    """
    一站式运行所有策略的回测

    Args:
        test_df: 测试集 DataFrame
        q_agent: 训练好的 QLearningAgent
        dqn_agent: 训练好的 DQNAgent
        reward_mode: 奖励模式
        initial_balance: 初始资金
        cost_rate: 交易成本
    Returns:
        dict: {策略名: (backtest_result, metrics)}
    """
    results = {}

    # Q-Learning
    env_q = CryptoTradingEnv(test_df, mode="discrete", reward_mode=reward_mode,
                             initial_balance=initial_balance, cost_rate=cost_rate)
    bt_q = backtest(env_q, q_agent, agent_type="qlearning")
    results["Q-Learning"] = (bt_q, compute_metrics(bt_q["portfolio_values"]))

    # DQN
    env_dqn = CryptoTradingEnv(test_df, mode="continuous", reward_mode=reward_mode,
                               initial_balance=initial_balance, cost_rate=cost_rate)
    bt_dqn = backtest(env_dqn, dqn_agent, agent_type="dqn")
    results["DQN"] = (bt_dqn, compute_metrics(bt_dqn["portfolio_values"]))

    # Buy & Hold
    env_bh = CryptoTradingEnv(test_df, mode="discrete", reward_mode="simple",
                              initial_balance=initial_balance, cost_rate=cost_rate)
    bt_bh = backtest_buy_and_hold(env_bh)
    results["Buy & Hold"] = (bt_bh, compute_metrics(bt_bh["portfolio_values"]))

    # Random
    env_rand = CryptoTradingEnv(test_df, mode="discrete", reward_mode="simple",
                                initial_balance=initial_balance, cost_rate=cost_rate)
    bt_rand = backtest_random(env_rand)
    results["Random"] = (bt_rand, compute_metrics(bt_rand["portfolio_values"]))

    # RSI Rule
    env_rsi = CryptoTradingEnv(test_df, mode="discrete", reward_mode="simple",
                               initial_balance=initial_balance, cost_rate=cost_rate)
    bt_rsi = backtest_rsi_rule(env_rsi)
    results["RSI Rule"] = (bt_rsi, compute_metrics(bt_rsi["portfolio_values"]))

    return results


def format_results_table(results):
    """
    格式化结果为对比表

    Args:
        results: run_all_backtests() 的返回值
    Returns:
        str: Markdown 格式的表格
    """
    header = "| Strategy | Total Return | Sharpe | Max Drawdown | Win Rate | Final Value | Trades |"
    separator = "|----------|-------------|--------|-------------|----------|-------------|--------|"
    rows = [header, separator]

    for name, (bt_result, metrics) in results.items():
        trades = count_trades(bt_result["actions"])
        row = (
            f"| {name:<10s} | "
            f"{metrics['total_return']:>+10.2%} | "
            f"{metrics['annualized_sharpe']:>6.2f} | "
            f"{metrics['max_drawdown']:>11.2%} | "
            f"{metrics['win_rate']:>8.2%} | "
            f"${metrics['final_value']:>10,.2f} | "
            f"{trades:>6d} |"
        )
        rows.append(row)

    return "\n".join(rows)
