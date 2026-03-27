# Assignment 3: REINFORCE, Multi-Trader SHAP Analysis, and Regulatory Implications for TON Trading

## 1. Introduction and Data

We extend our TON (Toncoin) RL trading framework from Assignment 2 by implementing REINFORCE with Baseline, modeling three trader archetypes with differentiated reward functions, and analyzing regulatory implications through crisis frequency, stablecoin stability, and market efficiency metrics.

**Data**: TON hourly OHLCV data spanning March 2024 to February 2026 (17,243 rows), with 8 auxiliary assets (BTC, ETH, SOL, Gold, Silver, S&P 500, US Bond, SPY). We use an 80/20 chronological train/test split (13,736/3,435 hours). The test period (Aug 2025 -- Feb 2026) coincides with a significant TON price decline, making it a challenging out-of-sample environment where Buy & Hold yields -52%.

**State representation**: 8-dimensional continuous vector (price_change_24h, price_change_4h, volatility_24h, RSI-14, hour_sin, hour_cos, position, volume_ratio). **Action space**: {BUY, HOLD, SELL, SHORT, COVER} with full-position sizing.

## 2. REINFORCE with Baseline: Architecture and Iterative Development

### 2.1 Architecture

We implement REINFORCE with a learned baseline using two independent networks:
- **Policy Network**: Linear(8->128)->ReLU->Linear(128->128)->ReLU->Linear(128->5), outputting logits (no Softmax layer; Categorical distribution handles log-softmax internally for numerical stability).
- **Value Network**: Linear(8->128)->ReLU->Linear(128->128)->ReLU->Linear(128->1), estimating state value V(s).

The policy gradient update follows: L_policy = -E[log pi(a|s) * A(s,a)] - beta * H(pi), where A(s,a) = G_t - V(s_t) is the advantage and H is entropy regularization.

### 2.2 Iterative Optimization (5 Runs)

The REINFORCE agent required 5 rounds of diagnosis-and-fix iterations to achieve meaningful learning signals. This process reveals fundamental challenges of policy gradient methods in financial environments:

| Run | Changes | Entropy (final) | Key Observation |
|-----|---------|-----------------|-----------------|
| 0 (baseline) | Original config | 1.6094 (=ln5) | Policy loss stuck at -0.0161; zero learning |
| 1 | Remove Softmax; remove advantage normalization; lr 3e-4->1e-3; entropy_coeff 0.01->0.001 | 1.60 | Gradients now flowing; policy loss varies |
| 2 | + Reward scaling x100; gamma 0.99->0.95 | 1.59 | Rewards amplified but entropy drops too slowly |
| 3 | + lr->0.003; episode_length 720->200 | 0.63 (ep400) then 1.54 (ep500) | Policy converges then **collapses** |
| 4 | lr->0.001; episode_length=500; 1000 episodes | Oscillates 1.37-1.58 | Confirmed: learn-forget cycle |

**Root causes identified:**
1. **Softmax gradient suppression** (Run 0): Softmax as the final network layer creates near-zero gradients when the policy is close to uniform -- exactly when learning needs to begin.
2. **Advantage over-normalization** (Run 0): Per-episode z-score normalization of advantages zeroes out weak signals.
3. **Reward magnitude** (Runs 1-2): Hourly returns (~0.001) produce advantages too small to drive policy updates.
4. **High variance of vanilla REINFORCE** (Runs 3-4): Even with fixes, the policy oscillates between convergence and collapse -- a well-known limitation of Monte Carlo policy gradients in high-noise environments.

### 2.3 Policy Gradient vs. DQN Comparison

| Strategy | Total Return | Sharpe | Max Drawdown | Win Rate | Trades |
|----------|-------------|--------|--------------|----------|--------|
| **DQN (Dueling+PER)** | **+77.93%** | **2.44** | 26.90% | 37.71% | 302 |
| REINFORCE | +0.00% | 0.00 | 0.00% | -- | 0 |
| Buy & Hold | -52.05% | -1.84 | 57.35% | 50.06% | 1 |
| RSI Rule | -35.20% | -0.93 | 48.67% | 45.84% | 37 |
| Random | -69.12% | -4.57 | 71.77% | 33.11% | 929 |

**Why DQN succeeds where REINFORCE fails:**
- **Experience replay** breaks temporal correlations, providing stable gradient estimates
- **Target network** prevents Q-value oscillation (the exact problem REINFORCE suffers from)
- **Epsilon-greedy exploration** provides structured exploration with gradual transition to exploitation
- **Per-step TD updates** are far more sample-efficient than Monte Carlo returns over 720-step episodes

This comparison demonstrates that value-based methods (DQN) are fundamentally better suited for financial trading environments where rewards are sparse, noisy, and delayed.

## 3. Multi-Trader Analysis and Shapley Values

We model three trader archetypes using differentiated reward functions, all trained with REINFORCE (lr=0.003, gamma=0.95, episode_length=200, reward_scale=100):

| Trader | Reward Design | Return | Sharpe | Top 3 Features (SHAP) |
|--------|--------------|--------|--------|----------------------|
| **Arbitrageur** | High cost (0.2%), DSR + mean reversion bonus | -52.10% | -1.85 | price_change_24h (0.14), rsi_14 (0.10), volatility_24h (0.07) |
| **Manipulator** | Low cost (0.02%), momentum + volume bonus | **+51.98%** | **2.40** | price_change_4h (0.26), price_change_24h (0.17), rsi_14 (0.09) |
| **Retail** | Trend-following + panic bonus + high drawdown penalty | +51.85% | 2.39 | rsi_14 (0.21), volatility_24h (0.17), price_change_24h (0.14) |

### 3.1 SHAP Feature Interpretation

**Arbitrageur** relies heavily on price_change_24h (mean reversion signal) and RSI (overbought/oversold identification). The dominance of longer-term price changes reflects the arbitrageur's strategy of identifying mispricings and waiting for correction.

**Manipulator** is most sensitive to price_change_4h -- short-term momentum that front-runners exploit. The high SHAP value (0.26) for 4-hour price changes, combined with low transaction costs, enables rapid position cycling that captures momentum before other participants.

**Retail** traders are primarily driven by RSI (0.21), reflecting their tendency to follow popular technical indicators. High sensitivity to volatility (0.17) matches the panic-selling behavior encoded in their reward function.

### 3.2 Regulatory Implications

The differentiated feature sensitivities reveal how strategic behavior manifests:
- **Manipulators exploit information asymmetry**: Their reliance on short-term momentum suggests front-running or wash trading, where they act on price movements before other market participants.
- **Retail traders are predictable**: Their RSI-driven behavior creates exploitable patterns that more sophisticated traders can anticipate.
- **Regulators must monitor**: (1) Unusually high short-term trading volumes (manipulator signature), (2) Clustered retail panic selling during high-volatility periods, (3) The time gap between manipulator position changes and subsequent retail reactions.

## 4. Regulatory Analysis

### 4.1 Crisis Frequency
Monthly decline regime proportion shows TON experiences crisis periods (72h return < -5%) approximately 19% of the time, with clustering during mid-2024 and late-2025 market stress events. Regulators should implement circuit breakers triggered by sustained decline regime persistence.

### 4.2 Stablecoin Stability
We measure 1/(1 + rolling_std(TON-BTC correlation)) as a proxy for cross-asset relationship stability. Periods of low stability correspond to regime transitions where correlations break down -- precisely when manipulative strategies become most profitable and retail traders most vulnerable.

### 4.3 Market Efficiency
Rolling lag-1 autocorrelation of hourly returns fluctuates between -0.15 and +0.10. Persistent positive autocorrelation periods indicate momentum effects exploitable by manipulators; negative autocorrelation suggests mean-reversion opportunities for arbitrageurs. The market is not consistently efficient at hourly frequencies, justifying regulatory attention to high-frequency trading strategies.

## 5. Conclusion

Our experiments demonstrate that DQN with experience replay and target networks significantly outperforms vanilla REINFORCE in cryptocurrency trading. The high variance of Monte Carlo policy gradients, combined with the sparse and noisy nature of financial rewards, makes REINFORCE poorly suited for this domain without advanced variance reduction techniques (PPO, GAE, or actor-critic methods).

The multi-trader SHAP analysis reveals that different market participants rely on fundamentally different features: manipulators exploit short-term momentum while retail traders follow longer-term technical signals. This feature-level decomposition provides regulators with actionable intelligence for detecting and mitigating strategic market behavior.
