"""
可视化模块 — 生成报告所需的全部图表和表格
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from datetime import datetime
from src.config import FIGURES_DIR
from src.environment import BUY, HOLD, SELL, SHORT, COVER


# 全局绘图设置
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# 颜色方案
REGIME_COLORS = {"rise": "#2ecc71", "decline": "#e74c3c", "steady": "#95a5a6"}
STRATEGY_COLORS = {
    "Q-Learning": "#3498db",
    "DQN": "#e74c3c",
    "Buy & Hold": "#f39c12",
    "Random": "#95a5a6",
    "RSI Rule": "#9b59b6",
}


def _save_fig(fig, name, timestamp=True):
    """保存图表到 output/figures/"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = FIGURES_DIR / f"{name}_{ts}.png"
    else:
        filepath = FIGURES_DIR / f"{name}.png"
    fig.savefig(filepath, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return filepath


def plot_price_with_regimes(df, title="TON Price with Market Regimes"):
    """
    图1: TON 价格走势 + Regime 着色标注

    Args:
        df: 含 close 和 regime 列的 DataFrame
    Returns:
        保存的文件路径
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df.index, df["close"], color="black", linewidth=0.8, alpha=0.8)

    # Regime 着色
    for regime, color in REGIME_COLORS.items():
        mask = df["regime"] == regime
        if mask.any():
            ax.fill_between(
                df.index, df["close"].min() * 0.9, df["close"].max() * 1.1,
                where=mask, alpha=0.15, color=color, label=regime.capitalize(),
            )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    return _save_fig(fig, "price_regimes")


def plot_training_curves(history, agent_name="Agent"):
    """
    图2: 训练曲线（episode reward, loss, Q-value, epsilon）

    Args:
        history: 训练返回的 history dict
        agent_name: 'Q-Learning' 或 'DQN'
    Returns:
        保存的文件路径
    """
    n_plots = 2
    has_loss = "losses" in history and any(l > 0 for l in history["losses"])
    has_q = "q_values_mean" in history and any(q != 0 for q in history["q_values_mean"])
    if has_loss:
        n_plots += 1
    if has_q:
        n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    idx = 0

    # Episode Reward
    rewards = history["episode_rewards"]
    axes[idx].plot(rewards, alpha=0.3, color="blue")
    # 滑动平均
    window = min(20, len(rewards) // 3) if len(rewards) > 3 else 1
    if window > 1:
        ma = pd.Series(rewards).rolling(window).mean()
        axes[idx].plot(ma, color="blue", linewidth=2, label=f"MA-{window}")
        axes[idx].legend()
    axes[idx].set_title(f"{agent_name} - Episode Reward")
    axes[idx].set_xlabel("Episode")
    axes[idx].set_ylabel("Total Reward")
    idx += 1

    # Epsilon
    axes[idx].plot(history["epsilon_history"], color="orange")
    axes[idx].set_title(f"{agent_name} - Epsilon Decay")
    axes[idx].set_xlabel("Episode")
    axes[idx].set_ylabel("Epsilon")
    idx += 1

    # Loss（仅 DQN）
    if has_loss:
        losses = history["losses"]
        axes[idx].plot(losses, alpha=0.3, color="red")
        if window > 1:
            ma = pd.Series(losses).rolling(window).mean()
            axes[idx].plot(ma, color="red", linewidth=2)
        axes[idx].set_title(f"{agent_name} - Loss")
        axes[idx].set_xlabel("Episode")
        axes[idx].set_ylabel("Huber Loss")
        idx += 1

    # Mean Q-value（仅 DQN）
    if has_q:
        q_vals = history["q_values_mean"]
        axes[idx].plot(q_vals, color="green")
        axes[idx].set_title(f"{agent_name} - Mean Q-Value")
        axes[idx].set_xlabel("Episode")
        axes[idx].set_ylabel("Q-Value")

    fig.suptitle(f"{agent_name} Training Curves", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save_fig(fig, f"training_{agent_name.lower().replace(' ', '_')}")


def plot_backtest_comparison(results, title="Backtest Portfolio Value Comparison"):
    """
    图3: 回测 portfolio value 对比图

    Args:
        results: run_all_backtests() 的返回值
    Returns:
        保存的文件路径
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    for name, (bt_result, metrics) in results.items():
        pv = bt_result["portfolio_values"]
        color = STRATEGY_COLORS.get(name, "gray")
        label = f"{name} ({metrics['total_return']:+.1%})"
        ax.plot(pv, color=color, linewidth=1.5, label=label, alpha=0.85)

    ax.axhline(y=10000, color="black", linestyle="--", alpha=0.3, label="Initial $10,000")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time Steps (hours)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="best")
    fig.tight_layout()
    return _save_fig(fig, "backtest_comparison")


def plot_regime_action_distribution(regime_analysis, agent_name="Agent"):
    """
    图4: Regime 下的动作分布柱状图

    Args:
        regime_analysis: analyze_by_regime() 的返回值
        agent_name: 策略名称
    Returns:
        保存的文件路径
    """
    if not regime_analysis:
        return None

    regimes = list(regime_analysis.keys())
    actions = ["BUY", "HOLD", "SELL", "SHORT", "COVER"]
    action_colors = {
        "BUY": "#2ecc71", "HOLD": "#3498db", "SELL": "#e74c3c",
        "SHORT": "#9b59b6", "COVER": "#f39c12",
    }

    fig, axes = plt.subplots(1, len(regimes), figsize=(5 * len(regimes), 4))
    if len(regimes) == 1:
        axes = [axes]

    for i, regime in enumerate(regimes):
        dist = regime_analysis[regime]["action_distribution"]
        values = [dist[a] for a in actions]
        colors = [action_colors[a] for a in actions]
        axes[i].bar(actions, values, color=colors, alpha=0.8)
        axes[i].set_title(f"{regime.capitalize()} (n={regime_analysis[regime]['count']})")
        axes[i].set_ylim(0, 1)
        axes[i].set_ylabel("Proportion")

    fig.suptitle(f"{agent_name} - Action Distribution by Regime", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save_fig(fig, f"regime_actions_{agent_name.lower().replace(' ', '_')}")


def plot_correlation_heatmap(corr_matrix, title="Asset Correlation Matrix"):
    """
    图5: 相关性热力图

    Args:
        corr_matrix: pd.DataFrame 相关性矩阵
    Returns:
        保存的文件路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".3f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True, ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return _save_fig(fig, "correlation_heatmap")


def plot_rolling_correlation(rolling_corr_df, title="TON Rolling Correlation (30-day)"):
    """
    图6: 滚动相关性时间序列

    Args:
        rolling_corr_df: rolling_correlation() 的返回值
    Returns:
        保存的文件路径
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    for col in rolling_corr_df.columns:
        ax.plot(rolling_corr_df.index, rolling_corr_df[col], label=col, alpha=0.7, linewidth=1)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation with TON")
    ax.legend(loc="best", ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save_fig(fig, "rolling_correlation")


def plot_intraday_effects(hourly_stats, title="TON Intraday Effects"):
    """
    图7: 日内效应图

    Args:
        hourly_stats: intraday_analysis() 的返回值
    Returns:
        保存的文件路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    hours = hourly_stats.index

    # 均值收益率
    axes[0].bar(hours, hourly_stats["mean_return"] * 100, color="#3498db", alpha=0.7)
    axes[0].set_title("Mean Hourly Return (%)")
    axes[0].set_xlabel("Hour (UTC)")
    axes[0].axhline(y=0, color="red", linestyle="--", alpha=0.5)

    # 波动率
    axes[1].bar(hours, hourly_stats["std_return"] * 100, color="#e74c3c", alpha=0.7)
    axes[1].set_title("Hourly Volatility (%)")
    axes[1].set_xlabel("Hour (UTC)")

    # 成交量
    axes[2].bar(hours, hourly_stats["mean_volume"], color="#2ecc71", alpha=0.7)
    axes[2].set_title("Mean Volume")
    axes[2].set_xlabel("Hour (UTC)")

    # 标注交易时段
    for ax in axes:
        ax.axvspan(14.5, 21, alpha=0.1, color="orange", label="US Market Hours")

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return _save_fig(fig, "intraday_effects")


def plot_agent_actions(test_df, bt_result, agent_name="Agent", metrics=None):
    """
    Agent 动作可视化：价格 + 动作标记 + 仓位 + Portfolio

    三行子图：
    1. 价格曲线 + BUY/SELL/SHORT/COVER 标记点 + Regime 背景色
    2. 仓位时间线 (position: -1=SHORT, 0=FLAT, 1=LONG)
    3. Portfolio Value 曲线

    Args:
        test_df: 测试集 DataFrame (含 close, regime 列)
        bt_result: backtest() 返回的 dict
        agent_name: 策略名称
        metrics: compute_metrics() 的结果 (可选, 用于标题)
    Returns:
        保存的文件路径
    """
    actions = bt_result["actions"]
    positions = bt_result["positions"]
    portfolio_values = bt_result["portfolio_values"]
    regimes = bt_result.get("regimes", np.array([]))

    n_steps = len(actions)
    # 测试集时间轴 (actions 从 step 1 开始, 对齐到 test_df.index[1:])
    time_index = test_df.index[1:n_steps + 1] if len(test_df) > n_steps else np.arange(n_steps)
    prices = test_df["close"].values[1:n_steps + 1] if len(test_df) > n_steps else np.arange(n_steps)

    # 动作标记配置
    action_markers = {
        BUY:   {"color": "#2ecc71", "marker": "^", "label": "BUY",   "size": 60},
        SELL:  {"color": "#e74c3c", "marker": "v", "label": "SELL",  "size": 60},
        SHORT: {"color": "#9b59b6", "marker": "v", "label": "SHORT", "size": 80},
        COVER: {"color": "#f39c12", "marker": "^", "label": "COVER", "size": 80},
    }

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), gridspec_kw={"height_ratios": [3, 1, 2]})
    fig.subplots_adjust(hspace=0.15)

    # ---- Panel 1: Price + Actions + Regime ----
    ax1 = axes[0]
    ax1.plot(time_index, prices, color="black", linewidth=0.7, alpha=0.8, zorder=2)

    # Regime 背景色
    if len(regimes) > 0:
        for regime, color in REGIME_COLORS.items():
            mask = regimes[:n_steps] == regime
            if mask.any():
                ax1.fill_between(
                    time_index, np.min(prices) * 0.95, np.max(prices) * 1.05,
                    where=mask, alpha=0.10, color=color, zorder=1,
                )

    # 动作标记
    for action_id, style in action_markers.items():
        mask = actions == action_id
        if mask.any():
            ax1.scatter(
                np.array(time_index)[mask], prices[mask],
                c=style["color"], marker=style["marker"], s=style["size"],
                label=f'{style["label"]} (n={mask.sum()})',
                alpha=0.85, edgecolors="white", linewidth=0.5, zorder=3,
            )

    # 标题: 含 metrics
    title_str = f"{agent_name} - Test Set Actions"
    if metrics:
        title_str += f"  |  Return: {metrics['total_return']:+.2%}  Sharpe: {metrics['annualized_sharpe']:.2f}  MaxDD: {metrics['max_drawdown']:.2%}"
    ax1.set_title(title_str, fontsize=13, fontweight="bold")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper right", fontsize=9, ncol=2)
    ax1.tick_params(labelbottom=False)

    # ---- Panel 2: Position Timeline ----
    ax2 = axes[1]
    pos_colors = np.where(positions == 1, "#2ecc71", np.where(positions == -1, "#9b59b6", "#95a5a6"))
    ax2.bar(time_index, positions, color=pos_colors, width=0.04, alpha=0.8)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_ylabel("Position")
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["SHORT", "FLAT", "LONG"])
    ax2.set_ylim(-1.3, 1.3)
    ax2.tick_params(labelbottom=False)

    # 计算各仓位占比
    long_pct = np.mean(positions == 1) * 100
    short_pct = np.mean(positions == -1) * 100
    flat_pct = np.mean(positions == 0) * 100
    ax2.set_title(f"Position  (LONG {long_pct:.1f}% | FLAT {flat_pct:.1f}% | SHORT {short_pct:.1f}%)",
                  fontsize=10, loc="left")

    # ---- Panel 3: Portfolio Value ----
    ax3 = axes[2]
    pv_time = test_df.index[:len(portfolio_values)] if len(test_df) >= len(portfolio_values) else np.arange(len(portfolio_values))
    ax3.plot(pv_time, portfolio_values, color=STRATEGY_COLORS.get(agent_name, "#3498db"),
             linewidth=1.5, label=agent_name)
    ax3.axhline(y=portfolio_values[0], color="black", linestyle="--", alpha=0.3, label=f"Initial ${portfolio_values[0]:,.0f}")
    ax3.fill_between(pv_time, portfolio_values[0], portfolio_values,
                     where=np.array(portfolio_values) >= portfolio_values[0],
                     alpha=0.15, color="#2ecc71")
    ax3.fill_between(pv_time, portfolio_values[0], portfolio_values,
                     where=np.array(portfolio_values) < portfolio_values[0],
                     alpha=0.15, color="#e74c3c")
    ax3.set_title(f"Portfolio Value  (${portfolio_values[0]:,.0f} -> ${portfolio_values[-1]:,.0f})",
                  fontsize=10, loc="left")
    ax3.set_ylabel("Portfolio ($)")
    ax3.set_xlabel("Date")
    ax3.legend(loc="best", fontsize=9)

    # X 轴日期格式
    import matplotlib.dates as mdates
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    fig.autofmt_xdate()
    fig.tight_layout()
    return _save_fig(fig, f"agent_actions_{agent_name.lower().replace(' ', '_').replace('-', '_')}")


def plot_actions_comparison(test_df, results):
    """
    双 Agent 动作对比图：Q-Learning vs DQN 并排

    Args:
        test_df: 测试集 DataFrame
        results: run_all_backtests() 的返回值
    Returns:
        保存的文件路径列表
    """
    saved = []
    for name in ["Q-Learning", "DQN"]:
        if name in results:
            bt_result, metrics = results[name]
            path = plot_agent_actions(test_df, bt_result, agent_name=name, metrics=metrics)
            saved.append(path)
    return saved


def plot_regime_correlation_comparison(regime_corrs):
    """
    图8: 分 Regime 相关性对比

    Args:
        regime_corrs: regime_correlation() 的返回值
    Returns:
        保存的文件路径
    """
    n_regimes = len(regime_corrs)
    if n_regimes == 0:
        return None

    fig, axes = plt.subplots(1, n_regimes, figsize=(7 * n_regimes, 5))
    if n_regimes == 1:
        axes = [axes]

    for i, (regime, corr_mat) in enumerate(regime_corrs.items()):
        # 只显示 TON 行（与其他资产的相关性）
        if "TON" in corr_mat.index:
            ton_corrs = corr_mat.loc["TON"].drop("TON")
            sns.heatmap(
                corr_mat, annot=True, fmt=".3f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=axes[i],
                linewidths=0.5, square=True,
            )
        axes[i].set_title(f"{regime.capitalize()} Regime", fontsize=12)

    fig.suptitle("Correlation by Market Regime", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save_fig(fig, "regime_correlation_comparison")
