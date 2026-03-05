"""
跨资产相关性分析模块
回应作业要求："correlations with other assets and cryptos, correlations during market stress"
"""
import numpy as np
import pandas as pd
from src.data_pipeline import load_auxiliary_data, align_auxiliary_to_ton
from src.config import REGIME_PARAMS


def compute_hourly_returns(dfs_dict, min_data_points=100):
    """
    计算各资产的小时收益率

    Args:
        dfs_dict: {资产名: DataFrame}，每个 DataFrame 含 'close' 列
        min_data_points: 资产至少需要的数据行数，少于此数的资产被忽略
    Returns:
        pd.DataFrame: 各资产的小时收益率（列名=资产名）
    """
    returns = {}
    for name, df in dfs_dict.items():
        if len(df) >= min_data_points:
            returns[name] = df["close"].pct_change()

    # 使用 dropna(how='all') 而非 dropna()，
    # 保留至少有一个非 NaN 值的行，相关性计算会自动处理缺失值
    return pd.DataFrame(returns).dropna(how="all")


def compute_correlation_matrix(ton_df, aux_dfs, method="inner"):
    """
    计算整体 Pearson 相关性矩阵

    Args:
        ton_df: TON 数据 DataFrame
        aux_dfs: {资产名: DataFrame} 辅助资产
        method: 对齐方式（'inner' 或 'ffill'）
    Returns:
        pd.DataFrame: 相关性矩阵
    """
    # 对齐数据
    aligned = align_auxiliary_to_ton(ton_df, aux_dfs, method=method)

    # 构建收益率 DataFrame
    all_dfs = {"TON": ton_df, **aligned}
    returns_df = compute_hourly_returns(all_dfs)

    return returns_df.corr()


def rolling_correlation(ton_df, aux_dfs, window=720, method="inner"):
    """
    计算 TON 与各资产的滚动相关性

    Args:
        ton_df: TON 数据
        aux_dfs: 辅助资产数据
        window: 滚动窗口（默认720小时≈30天）
        method: 对齐方式
    Returns:
        pd.DataFrame: 滚动相关性（列=资产名，索引=时间）
    """
    aligned = align_auxiliary_to_ton(ton_df, aux_dfs, method=method)
    all_dfs = {"TON": ton_df, **aligned}
    returns_df = compute_hourly_returns(all_dfs)

    ton_returns = returns_df["TON"]
    rolling_corrs = {}

    for name in returns_df.columns:
        if name == "TON":
            continue
        rolling_corrs[name] = ton_returns.rolling(window, min_periods=window // 2).corr(returns_df[name])

    return pd.DataFrame(rolling_corrs).dropna(how="all")


def regime_correlation(ton_df, aux_dfs, method="inner"):
    """
    分 Regime 计算相关性矩阵

    Args:
        ton_df: TON 数据（需含 'regime' 列）
        aux_dfs: 辅助资产数据
        method: 对齐方式
    Returns:
        dict: {regime: 相关性矩阵 DataFrame}
    """
    if "regime" not in ton_df.columns:
        raise ValueError("ton_df 需要含 'regime' 列，请先调用 classify_regime()")

    aligned = align_auxiliary_to_ton(ton_df, aux_dfs, method=method)
    all_dfs = {"TON": ton_df, **aligned}
    returns_df = compute_hourly_returns(all_dfs)

    # 将 regime 标签对齐到 returns_df
    regime_series = ton_df["regime"].reindex(returns_df.index)

    regime_corrs = {}
    for regime in ["rise", "decline", "steady"]:
        mask = regime_series == regime
        if mask.sum() > 30:  # 至少需要30个样本
            regime_corrs[regime] = returns_df.loc[mask].corr()

    return regime_corrs


def intraday_analysis(ton_df):
    """
    日内效应分析：每小时的均值收益率、波动率、成交量

    Args:
        ton_df: TON 数据 DataFrame（需含 hourly_return 和 volume 列）
    Returns:
        pd.DataFrame: 按小时汇总的统计量
    """
    df = ton_df.copy()
    df["hour"] = df.index.hour

    hourly_stats = df.groupby("hour").agg(
        mean_return=("hourly_return", "mean"),
        std_return=("hourly_return", "std"),
        mean_volume=("volume", "mean"),
        median_volume=("volume", "median"),
        count=("hourly_return", "count"),
    )

    return hourly_stats


def session_analysis(ton_df):
    """
    交易时段分析：亚洲/欧洲/美国时段的差异

    时段定义（UTC）：
    - 亚洲：00:00-08:00
    - 欧洲：08:00-16:00
    - 美国：16:00-24:00（含14:30-21:00美股交易时段）

    Returns:
        pd.DataFrame: 各时段的统计量
    """
    df = ton_df.copy()
    df["hour"] = df.index.hour

    def classify_session(hour):
        if 0 <= hour < 8:
            return "Asia"
        elif 8 <= hour < 16:
            return "Europe"
        else:
            return "US"

    df["session"] = df["hour"].apply(classify_session)

    session_stats = df.groupby("session").agg(
        mean_return=("hourly_return", "mean"),
        std_return=("hourly_return", "std"),
        mean_volume=("volume", "mean"),
        count=("hourly_return", "count"),
    )

    return session_stats
