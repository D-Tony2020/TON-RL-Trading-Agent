"""
数据管道模块 — 数据加载、清洗、特征工程、Regime标注、数据切分
"""
import numpy as np
import pandas as pd
from src.config import (
    CSV_SKIPROWS, CSV_COLUMNS, FLASH_CRASH_DATETIME,
    FEATURE_PARAMS, TRAIN_RATIO, REGIME_PARAMS,
    TON_DATA_PATH, AUXILIARY_ASSETS, DISCRETIZE_BINS, NORMALIZE_RANGES,
)


def load_csv(path):
    """
    加载任意 CSV 数据文件
    所有 CSV 文件统一格式：3行 yfinance 多级表头，skiprows=2 后重命名列

    Returns:
        pd.DataFrame: 列为 ['datetime','close','high','low','open','volume']，datetime为索引
    """
    df = pd.read_csv(path, skiprows=CSV_SKIPROWS)
    df.columns = CSV_COLUMNS
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()

    # 价格列转为 float，volume 转为 float（处理可能的缺失值）
    for col in ["close", "high", "low", "open"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(float)

    return df


def clean_ton_data(df):
    """
    清洗 TON 数据：处理闪崩异常值
    2024-09-03 07:00: $5.24 → $0.31（-94%），下一小时弹回 $5.19
    用前后价格均值插值替换

    Args:
        df: 原始 TON DataFrame
    Returns:
        pd.DataFrame: 清洗后的 DataFrame（副本，不修改原始数据）
    """
    df = df.copy()
    crash_time = pd.Timestamp(FLASH_CRASH_DATETIME)

    if crash_time in df.index:
        idx_pos = df.index.get_loc(crash_time)

        if idx_pos > 0 and idx_pos < len(df) - 1:
            # 用前后行的均值插值替换 OHLC
            prev_row = df.iloc[idx_pos - 1]
            next_row = df.iloc[idx_pos + 1]

            for col in ["close", "high", "low", "open"]:
                df.iloc[idx_pos, df.columns.get_loc(col)] = (
                    prev_row[col] + next_row[col]
                ) / 2.0

            # volume 也用均值
            df.iloc[idx_pos, df.columns.get_loc("volume")] = (
                prev_row["volume"] + next_row["volume"]
            ) / 2.0

    return df


def compute_rsi(series, period=14):
    """
    计算 RSI (Relative Strength Index)，归一化到 [0, 1]

    Args:
        series: 价格序列
        period: RSI 周期
    Returns:
        pd.Series: RSI 值，范围 [0, 1]
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # 使用 EMA（指数移动平均）计算
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 1.0 - 1.0 / (1.0 + rs)  # 等价于 rs/(1+rs)，范围 [0, 1]

    return rsi


def compute_features(df):
    """
    从原始 OHLCV 数据计算全部状态特征

    计算的特征：
    1. price_change_24h: 过去24小时收益率
    2. price_change_4h:  过去4小时收益率
    3. volatility_24h:   过去24小时波动率（hourly return 的 rolling std）
    4. rsi_14:           14期RSI，归一化到[0,1]
    5. hour_of_day:      当前小时 (0-23)
    6. volume_ratio:     成交量 / 72h滚动均值（零成交量时设为1.0）

    Args:
        df: 包含 OHLCV 的 DataFrame（datetime 为索引）
    Returns:
        pd.DataFrame: 原始数据 + 新特征列，已丢弃 warmup 期的 NaN 行
    """
    df = df.copy()
    p = FEATURE_PARAMS

    # 1. 过去24小时收益率
    df["price_change_24h"] = df["close"].pct_change(p["price_change_24h_period"])

    # 2. 过去4小时收益率
    df["price_change_4h"] = df["close"].pct_change(p["price_change_4h_period"])

    # 3. 过去24小时波动率
    hourly_return = df["close"].pct_change()
    df["volatility_24h"] = hourly_return.rolling(p["volatility_window"]).std()

    # 4. RSI-14，归一化到 [0, 1]
    df["rsi_14"] = compute_rsi(df["close"], period=p["rsi_period"])

    # 5. 小时 (0-23)
    df["hour_of_day"] = df.index.hour

    # 6. 成交量比率
    # 只用非零成交量计算滚动均值，避免大量零值拉低均值
    vol_series = df["volume"].replace(0, np.nan)
    vol_rolling_mean = vol_series.rolling(
        p["volume_ratio_window"], min_periods=1
    ).mean()
    # 成交量为0或均值为NaN时，volume_ratio 设为 1.0（中性值）
    df["volume_ratio"] = np.where(
        (df["volume"] == 0) | (vol_rolling_mean == 0) | vol_rolling_mean.isna(),
        1.0,
        df["volume"] / vol_rolling_mean,
    )

    # 保留 hourly_return 供后续使用（reward计算、regime分类等）
    df["hourly_return"] = hourly_return

    # 丢弃 warmup 期（前24行有 NaN）
    warmup_period = max(
        p["price_change_24h_period"],
        p["volatility_window"],
        p["rsi_period"],
        p["volume_ratio_window"],
    )
    df = df.iloc[warmup_period:]

    # 确认无 NaN（特征列）
    feature_cols = [
        "price_change_24h", "price_change_4h", "volatility_24h",
        "rsi_14", "hour_of_day", "volume_ratio",
    ]
    assert df[feature_cols].isna().sum().sum() == 0, (
        f"特征中仍有 NaN: {df[feature_cols].isna().sum()}"
    )

    return df


def classify_regime(df):
    """
    基于72小时滚动收益率标注市场 Regime

    - rise:    72h_return > rise_threshold（急涨）
    - decline: 72h_return < decline_threshold（急跌）
    - steady:  其余（横盘）

    Args:
        df: 含有 close 列的 DataFrame
    Returns:
        pd.DataFrame: 添加了 'regime' 和 'rolling_72h_return' 列
    """
    df = df.copy()
    r = REGIME_PARAMS

    df["rolling_72h_return"] = df["close"].pct_change(r["rolling_window"])

    conditions = [
        df["rolling_72h_return"] > r["rise_threshold"],
        df["rolling_72h_return"] < r["decline_threshold"],
    ]
    choices = ["rise", "decline"]
    df["regime"] = np.select(conditions, choices, default="steady")

    return df


def prepare_dataset(df):
    """
    时间顺序 80/20 切分数据集
    严禁随机切分（防止未来数据泄露）

    Args:
        df: 完整的特征DataFrame
    Returns:
        (train_df, test_df): 训练集和测试集
    """
    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


def load_auxiliary_data():
    """
    加载全部辅助资产数据

    Returns:
        dict: {资产名: DataFrame}，每个 DataFrame 格式与 TON 一致
    """
    aux_data = {}
    for name, path in AUXILIARY_ASSETS.items():
        if path.exists():
            aux_data[name] = load_csv(path)
    return aux_data


def align_auxiliary_to_ton(ton_df, aux_dfs, method="inner", min_overlap=100):
    """
    将辅助资产数据对齐到 TON 的时间索引

    对于时间戳不在整点的资产（如 SPY 的 14:30, 15:30），
    先 resample 到整点再对齐。

    Args:
        ton_df: TON 数据 DataFrame
        aux_dfs: {资产名: DataFrame} 辅助资产字典
        method: 'inner'（仅共同时间段）或 'ffill'（前向填充）
        min_overlap: 最少重叠数据点数，少于此数的资产被跳过
    Returns:
        dict: {资产名: 对齐后的 DataFrame}
    """
    aligned = {}
    for name, df in aux_dfs.items():
        working_df = df.copy()

        # 检测是否有非整点时间戳，如果有则 resample 到整点
        non_round_mask = working_df.index.minute != 0
        if non_round_mask.sum() > len(working_df) * 0.5:
            # 多数时间戳不在整点 → resample 到1H（取最后一个值=收盘价逻辑）
            working_df = working_df.resample("1h").last().dropna(subset=["close"])

        if method == "inner":
            common_idx = ton_df.index.intersection(working_df.index)
            result = working_df.loc[common_idx]
        elif method == "ffill":
            result = working_df.reindex(ton_df.index, method="ffill")
        else:
            result = working_df

        # 跳过重叠太少的资产
        if len(result.dropna(subset=["close"])) >= min_overlap:
            aligned[name] = result

    return aligned


def discretize_state(features, bins_config=None):
    """
    将连续特征离散化为 Q-Learning 的状态元组
    使用 np.digitize 进行分箱

    Args:
        features: dict，{特征名: 特征值}
        bins_config: 分箱配置，默认使用 DISCRETIZE_BINS
    Returns:
        tuple: 离散状态元组，可作为 dict 的 key
    """
    if bins_config is None:
        bins_config = DISCRETIZE_BINS

    state_parts = []
    for feature_name in [
        "price_change_24h", "price_change_4h", "volatility_24h",
        "rsi_14", "hour_of_day",
    ]:
        value = features[feature_name]
        bins = bins_config[feature_name]
        state_parts.append(int(np.digitize(value, bins)))

    # position: {-1, 0, 1} -> 映射为 {0, 1, 2} 作为离散索引
    state_parts.append(int(features["position"]) + 1)

    return tuple(state_parts)


def normalize_state(features, ranges_config=None):
    """
    将连续特征归一化到 [0, 1] 或 [-1, 1] 范围，供 DQN 使用
    hour_of_day 使用 sin/cos 周期编码（2维），其余线性归一化

    Args:
        features: dict，{特征名: 特征值}
        ranges_config: 归一化范围配置，默认使用 NORMALIZE_RANGES
    Returns:
        np.ndarray: shape (8,) 的归一化状态向量
    """
    if ranges_config is None:
        ranges_config = NORMALIZE_RANGES

    state = []
    for feature_name in [
        "price_change_24h", "price_change_4h", "volatility_24h",
        "rsi_14",
    ]:
        value = features[feature_name]
        low, high = ranges_config[feature_name]
        clipped = np.clip(value, low, high)
        if high > low:
            normalized = (clipped - low) / (high - low)
        else:
            normalized = 0.0
        state.append(normalized)

    # hour_of_day: sin/cos 周期编码（保留周期连续性）
    hour = features["hour_of_day"]
    state.append(np.sin(2 * np.pi * hour / 24.0))  # hour_sin ∈ [-1, 1]
    state.append(np.cos(2 * np.pi * hour / 24.0))  # hour_cos ∈ [-1, 1]

    # position 和 volume_ratio 照常归一化
    for feature_name in ["position", "volume_ratio"]:
        value = features[feature_name]
        low, high = ranges_config[feature_name]
        clipped = np.clip(value, low, high)
        if high > low:
            normalized = (clipped - low) / (high - low)
        else:
            normalized = 0.0
        state.append(normalized)

    return np.array(state, dtype=np.float32)


def load_and_prepare_ton():
    """
    一站式加载并准备 TON 数据（加载 → 清洗 → 特征工程 → Regime标注 → 切分）

    Returns:
        (full_df, train_df, test_df): 完整数据集、训练集、测试集
    """
    # 1. 加载
    raw_df = load_csv(TON_DATA_PATH)

    # 2. 清洗异常值
    clean_df = clean_ton_data(raw_df)

    # 3. 特征工程
    featured_df = compute_features(clean_df)

    # 4. Regime 标注
    featured_df = classify_regime(featured_df)

    # 5. 时间顺序切分
    train_df, test_df = prepare_dataset(featured_df)

    return featured_df, train_df, test_df
