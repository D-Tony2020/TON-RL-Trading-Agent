"""
监管分析模块 — 危机频率、稳定币稳定性、市场效率

3 张图：
1. Crisis Frequency: 按月统计 regime=='decline' 占比
2. Stablecoin Stability: 1 / (1 + rolling_std(TON-BTC correlation))
3. Market Efficiency: 滚动 lag-1 autocorrelation of hourly returns
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from src.config import FIGURES_DIR
from src.correlation import rolling_correlation
from src.data_pipeline import load_auxiliary_data, align_auxiliary_to_ton


def compute_crisis_frequency(df, freq="M"):
    """
    按时间窗口统计危机（decline regime）频率

    Args:
        df: 含 regime 列的 DataFrame
        freq: 聚合频率（'M'=月, 'W'=周）

    Returns:
        pd.Series: 每个时间窗口内 decline 占比
    """
    is_decline = (df["regime"] == "decline").astype(float)
    crisis_freq = is_decline.resample(freq).mean()
    return crisis_freq


def compute_stablecoin_stability(ton_df, window=720):
    """
    计算稳定币稳定性指标: 1 / (1 + rolling_std(TON-BTC correlation))

    较高值 = 更稳定的跨资产关系

    Args:
        ton_df: TON 数据 DataFrame
        window: 滚动窗口（默认720小时≈30天）

    Returns:
        pd.Series: 稳定性指标时间序列
    """
    # 加载 BTC 数据并计算滚动相关性
    aux_dfs = load_auxiliary_data()
    if "BTC" not in aux_dfs:
        raise ValueError("BTC 数据不可用，无法计算稳定性指标")

    btc_only = {"BTC": aux_dfs["BTC"]}
    rolling_corr = rolling_correlation(ton_df, btc_only, window=window)

    if "BTC" not in rolling_corr.columns:
        raise ValueError("无法计算 TON-BTC 滚动相关性")

    # 计算相关性的滚动标准差
    corr_rolling_std = rolling_corr["BTC"].rolling(window=window // 2, min_periods=window // 4).std()

    # 稳定性 = 1 / (1 + std)
    stability = 1.0 / (1.0 + corr_rolling_std)

    return stability


def compute_market_efficiency(df, window=720):
    """
    计算市场效率指标：滚动 lag-1 autocorrelation of hourly returns

    接近 0 = 更高效的市场（弱形式有效市场假说）

    Args:
        df: 含 hourly_return 列的 DataFrame
        window: 滚动窗口

    Returns:
        pd.Series: 滚动自相关时间序列
    """
    returns = df["hourly_return"].dropna()

    # 滚动 lag-1 自相关
    autocorr = returns.rolling(window=window, min_periods=window // 2).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan,
        raw=False,
    )

    return autocorr


def plot_regulatory_dashboard(df, ton_df=None):
    """
    绘制监管分析三合一仪表盘

    Args:
        df: 含 regime、hourly_return 列的完整数据 DataFrame
        ton_df: TON 原始数据（如果 None 则使用 df）

    Returns:
        保存的文件路径
    """
    if ton_df is None:
        ton_df = df

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # ---- Panel 1: Crisis Frequency ----
    ax1 = axes[0]
    crisis_freq = compute_crisis_frequency(df, freq="M")
    ax1.bar(crisis_freq.index, crisis_freq.values, width=20, color="#e74c3c", alpha=0.7)
    ax1.axhline(y=crisis_freq.mean(), color="black", linestyle="--", alpha=0.5,
                label=f"Mean: {crisis_freq.mean():.1%}")
    ax1.set_title("Crisis Frequency (Monthly Decline Regime Proportion)", fontsize=13)
    ax1.set_ylabel("Decline Proportion")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # ---- Panel 2: Stablecoin Stability ----
    ax2 = axes[1]
    try:
        stability = compute_stablecoin_stability(ton_df)
        ax2.plot(stability.index, stability.values, color="#3498db", linewidth=1, alpha=0.8)
        ax2.axhline(y=stability.mean(), color="black", linestyle="--", alpha=0.5,
                    label=f"Mean: {stability.mean():.3f}")
        ax2.set_title("Stablecoin Stability Index: 1/(1 + rolling_std(TON-BTC corr))", fontsize=13)
        ax2.set_ylabel("Stability Index")
        ax2.legend(loc="upper right")

        # 双 Y 轴：叠加 TON-BTC 相关性
        aux_dfs = load_auxiliary_data()
        if "BTC" in aux_dfs:
            btc_only = {"BTC": aux_dfs["BTC"]}
            rc = rolling_correlation(ton_df, btc_only)
            if "BTC" in rc.columns:
                ax2_twin = ax2.twinx()
                ax2_twin.plot(rc.index, rc["BTC"].values, color="#e74c3c",
                             linewidth=0.8, alpha=0.5, label="TON-BTC Corr")
                ax2_twin.set_ylabel("Correlation", color="#e74c3c")
                ax2_twin.legend(loc="lower right")
    except (ValueError, KeyError) as e:
        ax2.text(0.5, 0.5, f"BTC 数据不可用: {e}",
                transform=ax2.transAxes, ha="center", va="center", fontsize=12)
        ax2.set_title("Stablecoin Stability (N/A - missing BTC data)")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # ---- Panel 3: Market Efficiency ----
    ax3 = axes[2]
    autocorr = compute_market_efficiency(df)
    ax3.plot(autocorr.index, autocorr.values, color="#2ecc71", linewidth=1, alpha=0.8)
    ax3.axhline(y=0, color="black", linewidth=1, alpha=0.5)
    ax3.axhline(y=0.05, color="red", linestyle="--", alpha=0.3, label="+/- 0.05 ref")
    ax3.axhline(y=-0.05, color="red", linestyle="--", alpha=0.3)
    ax3.fill_between(autocorr.index, -0.05, 0.05, alpha=0.05, color="green",
                     label="Efficient zone")
    ax3.set_title("Market Efficiency: Rolling Lag-1 Autocorrelation of Hourly Returns", fontsize=13)
    ax3.set_ylabel("Autocorrelation")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper right")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    fig.suptitle("Regulatory Analysis Dashboard", fontsize=16, y=1.01)
    fig.autofmt_xdate()
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = FIGURES_DIR / f"regulatory_dashboard_{ts}.png"
    fig.savefig(filepath, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return filepath
