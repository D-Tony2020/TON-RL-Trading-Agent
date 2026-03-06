"""
全局配置文件 — 路径、超参数默认值与搜索范围
所有超参数集中在此，代码中不硬编码
"""
import os
from pathlib import Path

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = PROJECT_ROOT / "data"
TON_DATA_PATH = DATA_DIR / "TON" / "TON11419-USD_DataHr.csv"
FUTURES_DIR = DATA_DIR / "futures"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# 辅助资产文件映射
AUXILIARY_ASSETS = {
    "BTC": FUTURES_DIR / "BTC-USD_DataHr.csv",
    "ETH": FUTURES_DIR / "ETH-USD_DataHr.csv",
    "SOL": FUTURES_DIR / "SOL-USD_DataHr.csv",
    "Gold": FUTURES_DIR / "GC=F_DataHr.csv",
    "Silver": FUTURES_DIR / "SI=F_DataHr.csv",
    "SP500": FUTURES_DIR / "ES=F_DataHr.csv",
    "USBond": FUTURES_DIR / "ZN=F_DataHr.csv",
    "SPY": FUTURES_DIR / "SPY_DataHr.csv",
}

# ============================================================
# 数据处理配置
# ============================================================
CSV_SKIPROWS = 2  # yfinance 多级表头，跳过前2行
CSV_COLUMNS = ["datetime", "close", "high", "low", "open", "volume"]

# 闪崩异常值时间点
FLASH_CRASH_DATETIME = "2024-09-03 07:00:00+00:00"

# 特征工程参数
FEATURE_PARAMS = {
    "price_change_24h_period": 24,
    "price_change_4h_period": 4,
    "volatility_window": 24,
    "rsi_period": 14,
    "volume_ratio_window": 72,
}

# 数据切分比例（时间顺序）
TRAIN_RATIO = 0.8

# Regime 分类阈值（基于72小时滚动收益率）
REGIME_PARAMS = {
    "rolling_window": 72,        # 滚动窗口（小时）
    "rise_threshold": 0.05,      # 72h收益率 > 5% 为急涨
    "decline_threshold": -0.05,  # 72h收益率 < -5% 为急跌
}

# ============================================================
# Q-Learning 离散化 bins
# ============================================================
DISCRETIZE_BINS = {
    "price_change_24h": [-0.05, -0.02, -0.005, 0.005, 0.02, 0.05],  # → 7 bins
    "price_change_4h": [-0.02, -0.005, 0.005, 0.02],                 # → 5 bins
    "volatility_24h": [0.004, 0.008, 0.015],                         # → 4 bins
    "rsi_14": [0.3, 0.5, 0.7],                                       # → 4 bins
    "hour_of_day": [6, 12, 18],                                       # → 4 bins
    # position: {-1, 0, 1} 不需要 bins，直接用 position+1 作为索引 (0,1,2)
}

# DQN 归一化范围（用于连续状态）
NORMALIZE_RANGES = {
    "price_change_24h": (-0.15, 0.15),   # 裁剪到 ±15%
    "price_change_4h": (-0.08, 0.08),    # 裁剪到 ±8%
    "volatility_24h": (0.0, 0.03),       # 波动率范围
    "rsi_14": (0.0, 1.0),                # RSI 已归一化
    "hour_of_day": (0, 23),              # 小时
    "position": (-1, 1),                   # 持仓状态 (-1=空头, 0=空仓, 1=多头)
    "volume_ratio": (0.0, 5.0),          # 成交量比率裁剪
}

# ============================================================
# 奖励函数配置
# ============================================================
REWARD_CONFIG = {
    "simple": {
        "cost_rate": 0.001,          # 单边交易成本 0.1%
    },
    "sharpe": {
        "cost_rate": 0.001,
        "lambda_risk": 0.1,          # 回撤惩罚权重
        "lambda_sharpe": 1.0,        # Sharpe 奖励权重
        "drawdown_threshold": 0.05,  # 回撤触发阈值
        "sharpe_eta": 0.01,          # Differential Sharpe 的指数衰减因子
    },
}

# ============================================================
# Q-Learning 超参数
# ============================================================
QLEARNING_PARAMS = {
    "n_actions": 5,          # 动作数量 (BUY/HOLD/SELL/SHORT/COVER)
    "alpha": 0.1,            # 学习率
    "gamma": 0.97,           # 折扣因子
    "epsilon": 1.0,          # 初始探索率
    "epsilon_min": 0.05,     # 最小探索率
    "epsilon_decay": 0.9925, # 探索率衰减（per episode: 0.9925^400 ≈ 0.05）
    "n_episodes": 400,       # 训练轮数
    "episode_length": 720,   # 每轮步数（≈30天）
    "checkpoint_interval": 50,  # 每50轮保存一次
}

# Q-Learning 超参数搜索范围
QLEARNING_SEARCH_SPACE = {
    "alpha": [0.01, 0.05, 0.1, 0.2, 0.5],
    "gamma": [0.9, 0.95, 0.97, 0.99],
    "epsilon_decay": [0.985, 0.99, 0.9925, 0.995],  # per episode
}

# ============================================================
# DQN 超参数
# ============================================================
DQN_PARAMS = {
    "state_dim": 8,           # 状态维度（Run#7: hour→sin/cos, 7→8）
    "n_actions": 5,           # 动作数量 (BUY/HOLD/SELL/SHORT/COVER)
    "lr": 0.0003,             # 学习率 (Adam)
    "gamma": 0.99,            # 折扣因子
    "batch_size": 32,         # 批量大小
    "buffer_size": 100_000,   # 经验回放缓冲区
    "target_update": 500,     # 目标网络硬更新频率（步）Run#6: 1000→500
    "soft_update": False,     # Run#6: 回退到硬更新（软更新导致Q值爆炸）
    "tau": 0.005,             # 软更新系数（仅 soft_update=True 时生效）
    "lr_schedule": True,      # Run#6: 学习率余弦退火
    "lr_end": 3e-5,           # 最终学习率
    "epsilon": 1.0,           # 初始探索率
    "epsilon_min": 0.05,      # 最小探索率
    "epsilon_decay": 0.994,   # 探索率衰减（per episode: 0.994^500 ≈ 0.05）
    "n_episodes": 500,        # 训练轮数（Run#5: 200→500）
    "episode_length": 720,    # 每轮步数
    "min_buffer_size": 5000,  # 开始训练前的最小缓冲区
    "gradient_clip": 1.0,     # 梯度裁剪
    "checkpoint_interval": 20,  # 每20轮保存一次
    # Dueling 网络架构
    "hidden_dim": 128,        # 隐藏层维度
    "v_stream_dim": 64,       # V 流维度
    "a_stream_dim": 64,       # A 流维度
    # PER 参数
    "per_alpha": 0.6,         # PER 优先级指数
    "per_beta_start": 0.4,    # PER 重要性采样初始 beta
    "per_beta_end": 1.0,      # PER 重要性采样最终 beta
    "per_epsilon": 1e-6,      # PER 最小优先级
}

# DQN 超参数搜索范围
DQN_SEARCH_SPACE = {
    "lr": [0.0001, 0.0003, 0.001],
    "gamma": [0.95, 0.97, 0.99],
    "target_update": [500, 1000, 2000, 5000],
    "per_alpha": [0.4, 0.6, 0.8],
}

# ============================================================
# 交易环境配置
# ============================================================
ENV_CONFIG = {
    "initial_balance": 10_000,  # 初始资金
    "cost_rate": 0.001,         # 交易成本
}

# ============================================================
# 评估指标配置
# ============================================================
HOURS_PER_YEAR = 365 * 24  # 年化用的小时数

# Baseline 策略 RSI 阈值
RSI_RULE_BUY = 0.3    # RSI < 30 买入（归一化后 0.3）
RSI_RULE_SELL = 0.7   # RSI > 70 卖出（归一化后 0.7）

# ============================================================
# 进度监控配置
# ============================================================
PROGRESS_CONFIG = {
    "print_interval_qlearning": 20,   # Q-Learning 每20轮打印一次
    "print_interval_dqn": 10,         # DQN 每10轮打印一次
}
