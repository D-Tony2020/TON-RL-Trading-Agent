"""
数据管道测试 — 验证数据加载、清洗、特征工程、Regime标注、数据切分
"""
import numpy as np
import pandas as pd
import pytest
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import (
    load_csv, clean_ton_data, compute_features, compute_rsi,
    classify_regime, prepare_dataset, load_auxiliary_data,
    discretize_state, normalize_state, load_and_prepare_ton,
)
from src.config import TON_DATA_PATH, AUXILIARY_ASSETS, FLASH_CRASH_DATETIME


class TestLoadCSV:
    """测试 CSV 数据加载"""

    def test_load_ton_shape(self):
        """TON 数据加载后应有 17,243 行"""
        df = load_csv(TON_DATA_PATH)
        assert len(df) == 17243, f"预期 17243 行，实际 {len(df)} 行"

    def test_load_ton_columns(self):
        """加载后应有正确的列名"""
        df = load_csv(TON_DATA_PATH)
        expected_cols = ["close", "high", "low", "open", "volume"]
        assert list(df.columns) == expected_cols

    def test_load_ton_no_nan_prices(self):
        """价格列不应有 NaN"""
        df = load_csv(TON_DATA_PATH)
        for col in ["close", "high", "low", "open"]:
            assert df[col].isna().sum() == 0, f"{col} 列有 NaN"

    def test_load_ton_datetime_index(self):
        """索引应为 DatetimeIndex"""
        df = load_csv(TON_DATA_PATH)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_load_ton_date_range(self):
        """日期范围应从 2024-03-06 到 2026-02-23"""
        df = load_csv(TON_DATA_PATH)
        assert df.index[0].strftime("%Y-%m-%d") == "2024-03-06"
        assert df.index[-1].strftime("%Y-%m-%d") == "2026-02-23"

    def test_load_auxiliary_files_exist(self):
        """所有辅助数据文件应存在"""
        for name, path in AUXILIARY_ASSETS.items():
            assert path.exists(), f"缺少辅助数据文件: {name} at {path}"


class TestCleanData:
    """测试数据清洗"""

    def test_flash_crash_interpolated(self):
        """闪崩异常值应被插值替换"""
        raw_df = load_csv(TON_DATA_PATH)
        clean_df = clean_ton_data(raw_df)

        crash_time = pd.Timestamp(FLASH_CRASH_DATETIME)
        crash_close = clean_df.loc[crash_time, "close"]

        # 清洗后价格应在正常范围（$4-$6），不是 $0.31
        assert crash_close > 4.0, f"闪崩点价格未被修复: {crash_close}"
        assert crash_close < 6.0, f"闪崩点价格异常高: {crash_close}"

    def test_clean_return_std(self):
        """清洗后 hourly return 标准差应约为 0.87%"""
        raw_df = load_csv(TON_DATA_PATH)
        clean_df = clean_ton_data(raw_df)

        hourly_return = clean_df["close"].pct_change().dropna()
        std = hourly_return.std()

        # 允许 0.5% - 1.5% 的范围
        assert 0.005 < std < 0.015, f"清洗后 hourly return std = {std:.4f}，不在合理范围"

    def test_clean_preserves_length(self):
        """清洗不应改变数据行数"""
        raw_df = load_csv(TON_DATA_PATH)
        clean_df = clean_ton_data(raw_df)
        assert len(clean_df) == len(raw_df)


class TestComputeFeatures:
    """测试特征工程"""

    @pytest.fixture
    def featured_df(self):
        raw_df = load_csv(TON_DATA_PATH)
        clean_df = clean_ton_data(raw_df)
        return compute_features(clean_df)

    def test_features_no_nan(self, featured_df):
        """warmup 期丢弃后，所有特征列不应有 NaN"""
        feature_cols = [
            "price_change_24h", "price_change_4h", "volatility_24h",
            "rsi_14", "hour_of_day", "volume_ratio",
        ]
        nan_counts = featured_df[feature_cols].isna().sum()
        assert nan_counts.sum() == 0, f"特征中有 NaN:\n{nan_counts}"

    def test_rsi_range(self, featured_df):
        """RSI 值应全部在 [0, 1] 范围内"""
        rsi = featured_df["rsi_14"]
        assert rsi.min() >= 0.0, f"RSI 最小值 {rsi.min()} < 0"
        assert rsi.max() <= 1.0, f"RSI 最大值 {rsi.max()} > 1"

    def test_hour_of_day_range(self, featured_df):
        """hour_of_day 应在 [0, 23]"""
        hour = featured_df["hour_of_day"]
        assert hour.min() >= 0
        assert hour.max() <= 23

    def test_volatility_positive(self, featured_df):
        """波动率应为非负"""
        vol = featured_df["volatility_24h"]
        assert (vol >= 0).all(), "存在负波动率"

    def test_warmup_dropped(self, featured_df):
        """warmup 期（前72行）应被丢弃"""
        # 特征工程需要最多72行的 warmup（volume_ratio_window=72）
        # 丢弃后数据量应小于原始的 17,243
        assert len(featured_df) < 17243
        assert len(featured_df) > 17000  # 但不应丢弃太多

    def test_volume_ratio_no_inf(self, featured_df):
        """volume_ratio 不应有 Inf"""
        vr = featured_df["volume_ratio"]
        assert not np.isinf(vr).any(), "volume_ratio 中有 Inf"

    def test_hourly_return_column_exists(self, featured_df):
        """应生成 hourly_return 列"""
        assert "hourly_return" in featured_df.columns


class TestRegimeClassification:
    """测试 Regime 标注"""

    @pytest.fixture
    def regime_df(self):
        raw_df = load_csv(TON_DATA_PATH)
        clean_df = clean_ton_data(raw_df)
        featured_df = compute_features(clean_df)
        return classify_regime(featured_df)

    def test_regime_column_exists(self, regime_df):
        """应有 regime 列"""
        assert "regime" in regime_df.columns

    def test_regime_values(self, regime_df):
        """regime 只应包含 rise/decline/steady"""
        valid_regimes = {"rise", "decline", "steady"}
        actual_regimes = set(regime_df["regime"].unique())
        assert actual_regimes.issubset(valid_regimes), f"意外的 regime 值: {actual_regimes}"

    def test_regime_coverage(self, regime_df):
        """三种 regime 都应有样本"""
        counts = regime_df["regime"].value_counts()
        for regime in ["rise", "decline", "steady"]:
            assert regime in counts.index, f"缺少 {regime} regime"
            assert counts[regime] > 100, f"{regime} regime 样本太少: {counts[regime]}"

    def test_regime_distribution_reasonable(self, regime_df):
        """每种 regime 占比应在 5%-70% 之间（steady通常占多数）"""
        counts = regime_df["regime"].value_counts(normalize=True)
        for regime in ["rise", "decline", "steady"]:
            pct = counts[regime]
            assert 0.05 < pct < 0.70, f"{regime} 占比 {pct:.1%} 不在合理范围"


class TestTrainTestSplit:
    """测试数据切分"""

    @pytest.fixture
    def split_data(self):
        raw_df = load_csv(TON_DATA_PATH)
        clean_df = clean_ton_data(raw_df)
        featured_df = compute_features(clean_df)
        return prepare_dataset(featured_df)

    def test_split_ratio(self, split_data):
        """训练集应约占 80%"""
        train_df, test_df = split_data
        total = len(train_df) + len(test_df)
        train_ratio = len(train_df) / total
        assert 0.79 < train_ratio < 0.81, f"训练集比例 {train_ratio:.2%} 不在预期范围"

    def test_no_data_leakage(self, split_data):
        """训练集最后时间应早于测试集最早时间"""
        train_df, test_df = split_data
        assert train_df.index[-1] < test_df.index[0], "存在数据泄露：训练集与测试集时间重叠"

    def test_split_completeness(self, split_data):
        """切分后总行数应等于原始行数"""
        train_df, test_df = split_data
        total = len(train_df) + len(test_df)
        # 特征工程丢弃 warmup 后的总行数
        raw_df = load_csv(TON_DATA_PATH)
        clean_df = clean_ton_data(raw_df)
        featured_df = compute_features(clean_df)
        assert total == len(featured_df)


class TestDiscretizeNormalize:
    """测试状态离散化和归一化"""

    def test_discretize_state_tuple(self):
        """离散化应返回 tuple"""
        features = {
            "price_change_24h": 0.01,
            "price_change_4h": -0.001,
            "volatility_24h": 0.006,
            "rsi_14": 0.55,
            "hour_of_day": 14,
            "position": 1,
        }
        state = discretize_state(features)
        assert isinstance(state, tuple)
        assert len(state) == 6  # 6维离散状态

    def test_discretize_state_deterministic(self):
        """相同输入应产生相同离散状态"""
        features = {
            "price_change_24h": 0.03,
            "price_change_4h": 0.01,
            "volatility_24h": 0.01,
            "rsi_14": 0.8,
            "hour_of_day": 20,
            "position": 0,
        }
        assert discretize_state(features) == discretize_state(features)

    def test_normalize_state_shape(self):
        """归一化应返回 shape (7,) 的 float32 数组"""
        features = {
            "price_change_24h": 0.01,
            "price_change_4h": -0.001,
            "volatility_24h": 0.006,
            "rsi_14": 0.55,
            "hour_of_day": 14,
            "position": 1,
            "volume_ratio": 1.5,
        }
        state = normalize_state(features)
        assert state.shape == (7,)
        assert state.dtype == np.float32

    def test_normalize_state_range(self):
        """归一化后所有值应在 [0, 1]"""
        features = {
            "price_change_24h": 0.01,
            "price_change_4h": -0.001,
            "volatility_24h": 0.006,
            "rsi_14": 0.55,
            "hour_of_day": 14,
            "position": 1,
            "volume_ratio": 1.5,
        }
        state = normalize_state(features)
        assert (state >= 0.0).all(), f"归一化值 < 0: {state}"
        assert (state <= 1.0).all(), f"归一化值 > 1: {state}"

    def test_normalize_state_clipping(self):
        """超出范围的值应被裁剪"""
        features = {
            "price_change_24h": 0.5,     # 远超 ±15% 范围
            "price_change_4h": -0.5,     # 远超范围
            "volatility_24h": 0.1,       # 超出 0.03
            "rsi_14": 0.55,
            "hour_of_day": 14,
            "position": 1,
            "volume_ratio": 100.0,       # 远超 5.0
        }
        state = normalize_state(features)
        assert (state >= 0.0).all() and (state <= 1.0).all()


class TestEndToEnd:
    """端到端测试"""

    def test_load_and_prepare_ton(self):
        """一站式函数应返回正确格式的数据"""
        full_df, train_df, test_df = load_and_prepare_ton()

        # 检查基本属性
        assert len(full_df) > 0
        assert len(train_df) > 0
        assert len(test_df) > 0
        assert len(train_df) + len(test_df) == len(full_df)

        # 检查必要列都在
        required_cols = [
            "close", "price_change_24h", "price_change_4h",
            "volatility_24h", "rsi_14", "hour_of_day", "volume_ratio",
            "hourly_return", "regime",
        ]
        for col in required_cols:
            assert col in full_df.columns, f"缺少列: {col}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
