"""
交易环境模块 -- CryptoTradingEnv
OpenAI Gym 风格接口：reset() -> state, step(action) -> (next_state, reward, done, info)
通过 mode='discrete'/'continuous' 同时服务 Q-Learning 和 DQN
支持做空: position in {-1, 0, 1}
"""
import numpy as np
from src.data_pipeline import discretize_state, normalize_state
from src.config import REWARD_CONFIG, ENV_CONFIG


# 动作常量
BUY = 0       # 空仓 -> 多仓 (position: 0 -> 1)
HOLD = 1      # 维持当前仓位
SELL = 2      # 多仓 -> 空仓 (position: 1 -> 0)
SHORT = 3     # 空仓 -> 空头 (position: 0 -> -1)
COVER = 4     # 空头 -> 空仓 (position: -1 -> 0)

N_ACTIONS = 5
ACTION_NAMES = {BUY: "BUY", HOLD: "HOLD", SELL: "SELL", SHORT: "SHORT", COVER: "COVER"}


class CryptoTradingEnv:
    """
    加密货币交易环境

    动作空间: {BUY(0), HOLD(1), SELL(2), SHORT(3), COVER(4)}
    持仓状态: position in {-1(空头), 0(空仓), 1(多头)}
    非法动作一律映射为 HOLD

    Args:
        df: 含有特征列的 DataFrame（来自 data_pipeline.compute_features）
        mode: 'discrete'（Q-Learning用）或 'continuous'（DQN用）
        reward_mode: 'simple'（仅收益-成本）或 'sharpe'（含风险+Sharpe调整）
        initial_balance: 初始资金
        cost_rate: 单边交易成本率
        **reward_kwargs: 覆盖 REWARD_CONFIG 中的默认参数
    """

    def __init__(
        self,
        df,
        mode="discrete",
        reward_mode="simple",
        initial_balance=None,
        cost_rate=None,
        **reward_kwargs,
    ):
        self.df = df.reset_index(drop=True)  # 用整数索引方便随机采样
        self.mode = mode
        self.reward_mode = reward_mode
        self.initial_balance = initial_balance or ENV_CONFIG["initial_balance"]

        # 奖励参数：先从预设加载，再用 kwargs 覆盖
        self.reward_params = REWARD_CONFIG.get(reward_mode, REWARD_CONFIG["simple"]).copy()
        if cost_rate is not None:
            self.reward_params["cost_rate"] = cost_rate
        self.reward_params.update(reward_kwargs)

        self.cost_rate = self.reward_params["cost_rate"]

        # 数据长度
        self.data_length = len(self.df)

        # Differential Sharpe Ratio 的指数移动平均状态
        self._sharpe_A = 0.0  # E[R] 的 EMA
        self._sharpe_B = 0.0  # E[R^2] 的 EMA

        # 状态变量（在 reset 中初始化）
        self.current_step = 0
        self.end_step = 0
        self.position = 0          # -1=空头, 0=空仓, 1=多头
        self.balance = 0.0         # 现金（空仓时有值）
        self.holdings = 0.0        # 持有的 TON 数量（多头时有值）
        self.short_holdings = 0.0  # 做空的 TON 数量（空头时有值）
        self.short_entry_price = 0.0  # 做空入场价格
        self.portfolio_value = 0.0
        self.prev_portfolio_value = 0.0
        self.max_portfolio_value = 0.0  # 用于回撤计算

    def reset(self, start_idx=None, episode_length=720):
        """
        重置环境

        Args:
            start_idx: 起始数据索引（None则随机选取）
            episode_length: 每个 episode 的步数
        Returns:
            state: 初始状态
        """
        # 确定起始和结束位置
        max_start = self.data_length - episode_length - 1
        if max_start < 0:
            max_start = 0
            episode_length = self.data_length - 1

        if start_idx is None:
            self.current_step = np.random.randint(0, max(1, max_start))
        else:
            self.current_step = min(start_idx, max_start)

        self.end_step = self.current_step + episode_length

        # 初始化账户状态
        self.position = 0
        self.balance = self.initial_balance
        self.holdings = 0.0
        self.short_holdings = 0.0
        self.short_entry_price = 0.0
        self.portfolio_value = self.initial_balance
        self.prev_portfolio_value = self.initial_balance
        self.max_portfolio_value = self.initial_balance

        # 重置 Differential Sharpe 状态
        self._sharpe_A = 0.0
        self._sharpe_B = 0.0

        return self._get_state()

    def step(self, action):
        """
        执行一步交易

        Args:
            action: 0=BUY, 1=HOLD, 2=SELL, 3=SHORT, 4=COVER
        Returns:
            (next_state, reward, done, info)
        """
        current_price = self.df.iloc[self.current_step]["close"]
        position_before = self.position
        actual_action = action  # 记录实际执行的动作

        # ---- 执行交易 ----
        transaction_cost = 0.0

        if action == BUY and self.position == 0:
            # 空仓 -> 多头：全仓买入
            cost = self.balance * self.cost_rate
            available = self.balance - cost
            self.holdings = available / current_price
            self.balance = 0.0
            self.position = 1
            transaction_cost = cost

        elif action == SELL and self.position == 1:
            # 多头 -> 空仓：全部卖出
            sell_value = self.holdings * current_price
            cost = sell_value * self.cost_rate
            self.balance = sell_value - cost
            self.holdings = 0.0
            self.position = 0
            transaction_cost = cost

        elif action == SHORT and self.position == 0:
            # 空仓 -> 空头：做空
            cost = self.balance * self.cost_rate
            available = self.balance - cost
            self.short_holdings = available / current_price
            self.short_entry_price = current_price
            self.balance = 0.0
            self.position = -1
            transaction_cost = cost

        elif action == COVER and self.position == -1:
            # 空头 -> 空仓：平空
            # 计算当前空头的价值
            pnl = self.short_holdings * (self.short_entry_price - current_price)
            gross_value = self.short_holdings * self.short_entry_price + pnl
            cost = max(gross_value, 0) * self.cost_rate
            self.balance = max(gross_value - cost, 0)
            self.short_holdings = 0.0
            self.short_entry_price = 0.0
            self.position = 0
            transaction_cost = cost

        else:
            # HOLD，或非法动作 -> 映射为 HOLD
            actual_action = HOLD

        # ---- 推进时间 ----
        self.current_step += 1
        next_price = self.df.iloc[self.current_step]["close"]

        # ---- 更新 portfolio value ----
        self.prev_portfolio_value = self.portfolio_value
        if self.position == 1:
            # 多头：跟随价格
            self.portfolio_value = self.holdings * next_price
        elif self.position == -1:
            # 空头：反向跟随价格
            pnl = self.short_holdings * (self.short_entry_price - next_price)
            self.portfolio_value = self.short_holdings * self.short_entry_price + pnl
        else:
            # 空仓：保持现金
            self.portfolio_value = self.balance

        # 更新最大 portfolio value（回撤计算用）
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)

        # ---- 计算奖励 ----
        reward = self._compute_reward(
            position_before, transaction_cost, current_price, next_price
        )

        # ---- 检查是否结束 ----
        done = self.current_step >= self.end_step

        # ---- 构建 info ----
        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "action_taken": actual_action,
            "transaction_cost": transaction_cost,
            "current_price": next_price,
            "balance": self.balance,
            "holdings": self.holdings,
            "short_holdings": self.short_holdings,
        }

        return self._get_state(), reward, done, info

    def _get_state(self):
        """
        获取当前状态表示

        Returns:
            discrete mode: tuple of ints（可做 dict key）
            continuous mode: np.ndarray shape (7,)
        """
        row = self.df.iloc[self.current_step]
        features = {
            "price_change_24h": row["price_change_24h"],
            "price_change_4h": row["price_change_4h"],
            "volatility_24h": row["volatility_24h"],
            "rsi_14": row["rsi_14"],
            "hour_of_day": row["hour_of_day"],
            "position": self.position,
            "volume_ratio": row["volume_ratio"],
        }

        if self.mode == "discrete":
            return discretize_state(features)
        else:
            return normalize_state(features)

    def _compute_reward(self, position_before, transaction_cost, price_before, price_after):
        """
        根据 reward_mode 计算奖励

        position_before in {-1, 0, 1}:
        - position=1 (多头): portfolio_return = +hourly_return
        - position=0 (空仓): portfolio_return = 0
        - position=-1 (空头): portfolio_return = -hourly_return

        Args:
            position_before: 交易前的持仓状态
            transaction_cost: 本步交易成本
            price_before: 当前步价格
            price_after: 下一步价格
        Returns:
            float: 奖励值
        """
        # 核心项：持仓收益率（position 自然处理做空情况）
        if price_before > 0:
            hourly_return = (price_after - price_before) / price_before
        else:
            hourly_return = 0.0

        portfolio_return = position_before * hourly_return

        # 交易成本惩罚（归一化到与收益率同数量级）
        position_change = abs(self.position - position_before)
        cost_penalty = self.cost_rate * position_change

        if self.reward_mode == "simple":
            return portfolio_return - cost_penalty

        elif self.reward_mode == "sharpe":
            # Differential Sharpe Ratio
            eta = self.reward_params.get("sharpe_eta", 0.01)
            delta_A = portfolio_return - self._sharpe_A
            delta_B = portfolio_return ** 2 - self._sharpe_B

            # 更新 EMA
            self._sharpe_A += eta * delta_A
            self._sharpe_B += eta * delta_B

            # Differential Sharpe Ratio
            denom = (self._sharpe_B - self._sharpe_A ** 2)
            if denom > 1e-6:
                dsr = (self._sharpe_B * delta_A - 0.5 * self._sharpe_A * delta_B) / (
                    denom ** 1.5
                )
                dsr = float(np.clip(dsr, -10.0, 10.0))  # 防止极端值
            else:
                dsr = 0.0

            sharpe_bonus = self.reward_params.get("lambda_sharpe", 1.0) * dsr

            # 回撤惩罚
            risk_penalty = 0.0
            if self.max_portfolio_value > 0:
                drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
                threshold = self.reward_params.get("drawdown_threshold", 0.05)
                lambda_risk = self.reward_params.get("lambda_risk", 0.1)
                risk_penalty = lambda_risk * max(0.0, drawdown - threshold)

            return portfolio_return - cost_penalty - risk_penalty + sharpe_bonus

        elif self.reward_mode == "arbitrageur":
            # 理性套利者：均值回归 + 风险调整 + Sharpe
            # Differential Sharpe Ratio（复用 sharpe 模式逻辑）
            eta = self.reward_params.get("sharpe_eta", 0.01)
            delta_A = portfolio_return - self._sharpe_A
            delta_B = portfolio_return ** 2 - self._sharpe_B
            self._sharpe_A += eta * delta_A
            self._sharpe_B += eta * delta_B
            denom = (self._sharpe_B - self._sharpe_A ** 2)
            if denom > 1e-6:
                dsr = (self._sharpe_B * delta_A - 0.5 * self._sharpe_A * delta_B) / (
                    denom ** 1.5
                )
                dsr = float(np.clip(dsr, -10.0, 10.0))
            else:
                dsr = 0.0
            sharpe_bonus = self.reward_params.get("lambda_sharpe", 1.5) * dsr

            # 回撤惩罚
            risk_penalty = 0.0
            if self.max_portfolio_value > 0:
                drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
                threshold = self.reward_params.get("drawdown_threshold", 0.03)
                lambda_risk = self.reward_params.get("lambda_risk", 0.2)
                risk_penalty = lambda_risk * max(0.0, drawdown - threshold)

            # 均值回归奖励：RSI 极端时反向操作
            row = self.df.iloc[self.current_step]
            rsi = row.get("rsi_14", 0.5)
            mr_weight = self.reward_params.get("mean_reversion_bonus", 0.5)
            mean_reversion = 0.0
            if rsi < 0.3 and position_before == 1:       # 超卖时做多
                mean_reversion = mr_weight * (0.3 - rsi)
            elif rsi > 0.7 and position_before == -1:     # 超买时做空
                mean_reversion = mr_weight * (rsi - 0.7)

            return portfolio_return - cost_penalty - risk_penalty + sharpe_bonus + mean_reversion

        elif self.reward_mode == "manipulator":
            # 操纵者：动量追逐 + 成交量利用
            row = self.df.iloc[self.current_step]
            pc4h = row.get("price_change_4h", 0.0)
            vol_ratio = row.get("volume_ratio", 1.0)

            # 动量奖励：短期动量方向与仓位一致时给予正奖励
            momentum_weight = self.reward_params.get("momentum_bonus", 1.0)
            momentum = 0.0
            if position_before != 0 and pc4h != 0:
                # 仓位方向与动量方向一致为正
                alignment = position_before * np.sign(pc4h)
                momentum = momentum_weight * alignment * abs(pc4h)

            # 成交量异常奖励：高成交量时交易给予奖励
            vol_weight = self.reward_params.get("volume_bonus", 0.5)
            vol_threshold = self.reward_params.get("volume_threshold", 2.0)
            volume_bonus = 0.0
            if vol_ratio > vol_threshold and position_change > 0:
                volume_bonus = vol_weight * (vol_ratio - vol_threshold) / vol_threshold

            return portfolio_return - cost_penalty + momentum + volume_bonus

        elif self.reward_mode == "retail":
            # 从众散户：趋势追随 + 恐慌卖出 + 高风险厌恶
            row = self.df.iloc[self.current_step]
            pc24h = row.get("price_change_24h", 0.0)
            volatility = row.get("volatility_24h", 0.0)

            # 趋势追随奖励：24h 趋势方向与仓位一致
            trend_weight = self.reward_params.get("trend_bonus", 0.8)
            trend = 0.0
            if position_before != 0 and pc24h != 0:
                alignment = position_before * np.sign(pc24h)
                trend = trend_weight * alignment * abs(pc24h)

            # 恐慌卖出奖励：高波动率时离场给予奖励
            panic_weight = self.reward_params.get("panic_bonus", 0.3)
            vol_panic = self.reward_params.get("volatility_panic_threshold", 0.015)
            panic = 0.0
            if volatility > vol_panic and position_change > 0 and self.position == 0:
                # 从持仓变为空仓（卖出/平空），在高波动时视为"正确"恐慌
                panic = panic_weight * (volatility - vol_panic) / vol_panic

            # 高回撤惩罚
            risk_penalty = 0.0
            if self.max_portfolio_value > 0:
                drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
                threshold = self.reward_params.get("drawdown_threshold", 0.02)
                lambda_risk = self.reward_params.get("lambda_risk", 0.5)
                risk_penalty = lambda_risk * max(0.0, drawdown - threshold)

            return portfolio_return - cost_penalty - risk_penalty + trend + panic

        else:
            # 默认 fallback 到 simple
            return portfolio_return - cost_penalty

    def get_current_price(self):
        """获取当前步的收盘价"""
        return self.df.iloc[self.current_step]["close"]

    def get_current_regime(self):
        """获取当前步的 regime 标签"""
        if "regime" in self.df.columns:
            return self.df.iloc[self.current_step]["regime"]
        return "unknown"
