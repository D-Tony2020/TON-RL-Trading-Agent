# Reinforcement Learning for Cryptocurrency Trading: A TON Case Study

> ORIE 5570 Biweekly Report

---

## Introduction

The idea behind this project is straightforward: use reinforcement learning to trade a cryptocurrency at the hourly level and see whether an RL agent can actually learn something smarter than buy-and-hold. I picked TON (Toncoin) as the asset — a mid-cap altcoin that's significantly more volatile than BTC, but with enough data to work with.

### Dataset

The data consists of hourly OHLCV (open, high, low, close, volume) bars, spanning from March 2024 to February 2026 — roughly two years. The raw CSV contains 17,243 rows (after stripping the multi-level header), and after truncating the warmup period needed for feature engineering, 17,171 rows remain usable.

Summary statistics for the closing price:

| Statistic | Full | Train | Test |
|-----------|------|-------|------|
| Count | 17,171 | 13,736 | 3,435 |
| Mean | $4.17 | $4.77 | $1.78 |
| Std | $1.82 | $1.53 | $0.38 |
| Min | $1.22 | $2.39 | $1.22 |
| 25% | $2.88 | $3.22 | $1.50 |
| Median | $3.64 | $4.98 | $1.64 |
| 75% | $5.62 | $5.93 | $2.06 |
| Max | $8.20 | $8.20 | $2.87 |

On the hourly returns side, the mean is approximately -0.0005% (essentially zero), with a standard deviation of 0.87%. The distribution is noticeably left-skewed (skewness = -1.19) with extreme kurtosis (45.68) — classic crypto behavior: quiet most of the time, with occasional sharp crashes. The 1st and 99th percentiles are -2.50% and +2.38%, roughly symmetric, but the minimum of -24.10% is far more extreme than the maximum of +10.80%, reflecting the "crashes harder than it rallies" asymmetry.

One data quality issue worth mentioning: on September 3, 2024 at 07:00 UTC, there was a flash crash where the price dropped from $5.24 to $0.31 (-94%), then bounced back to $5.19 the next hour. Leaving this in would wreck the feature calculations, so I replaced it with the average of the adjacent time points. Additionally, about 51.1% of hourly volume entries are zero — TON's liquidity is genuinely not great. This required special handling in the feature design (setting volume_ratio to a neutral 1.0 when volume is zero to avoid division issues or outliers).

### Train/Test Split

I split the data 80/20 in chronological order. This is critical — random splitting would leak future information into training (look-ahead bias). The result:

- **Training set**: 2024-03-09 to 2025-10-02 (572 days), price roughly flat ($2.80 → $2.82, +0.82%)
- **Test set**: 2025-10-02 to 2026-02-23 (144 days), price from $2.80 down to $1.34 (**-52.01%**)

The test set is a brutal, sustained bear market. I didn't choose this on purpose — it's just what falls out of a chronological split. But in a way, it's a good stress test: if the agent can still make money in this kind of market, it's clearly learned something real.

### Market Regime Labels

I classified market states into three categories using a 72-hour rolling return: above +5% is labeled *rise*, below -5% is *decline*, everything else is *steady*. In the training set, steady accounts for 64.2%, decline 18.1%, rise 17.7% — fairly balanced. But in the test set, decline climbs to 24.8% while rise drops to 16.0%, consistent with the price trajectory.

### Feature Engineering

To give the agent awareness of market conditions, I selected 6 features:

1. **price_change_24h**: 24-hour return, capturing the medium-term trend
2. **price_change_4h**: 4-hour return, capturing short-term momentum
3. **volatility_24h**: standard deviation of hourly returns over the past 24 hours
4. **rsi_14**: 14-period RSI, normalized to [0,1], a classic overbought/oversold signal
5. **hour_of_day**: current hour (0-23), to capture intraday effects
6. **volume_ratio**: volume relative to its 72-hour rolling average — unusual spikes may signal incoming moves

DQN also receives volume_ratio as a continuous input (Q-Learning's state space was already large enough without adding more dimensions).

### Cross-Asset Data

Beyond TON itself, I collected hourly data for 8 auxiliary assets for correlation analysis: BTC, ETH, SOL (crypto peers), gold, silver (precious metals), S&P 500 futures, U.S. Treasury bond futures, and SPY. The crypto assets have ~17K rows perfectly aligned with TON; traditional assets have ~11K rows (data only during trading hours).

---

## Key Research Questions

The core question this project tries to answer is really just one: **How well can an RL agent perform in a cryptocurrency market that's been crashing for months?**

Around that central question, several more specific ones emerged:

- How much difference does introducing short-selling make? In theory, shorting lets you profit from a decline, but can the agent actually learn to do it?
- How do tabular Q-Learning and a deep Q-Network compare? Where does each excel?
- How much do seemingly minor engineering details (epsilon decay rate, PER priority calculation, target network update method) actually affect the final outcome?
- How does TON's correlation with other assets (especially BTC and ETH) shift across different market regimes?

---

## Method

### MDP Formulation

I framed the trading problem as an MDP. Each episode randomly samples 720 consecutive hourly steps (~30 days) from the training set. At each step, the agent picks an action, and the environment returns a reward and the next state.

**Action space** (final version): 5 actions — BUY (flat → long), HOLD (do nothing), SELL (long → flat), SHORT (flat → short), COVER (short → flat). All-in execution, position ∈ {-1, 0, 1}. Invalid actions (e.g., trying to BUY when already long) are mapped to HOLD.

**Reward function**: simple mode, `reward = position × hourly_return - cost_rate × |position_change|`. Portfolio return minus transaction cost — nothing fancy. I did try a more complex Differential Sharpe Ratio reward; I'll explain later why it was abandoned.

**State representation**:
- Q-Learning: 6 discrete features binned into 6,720 possible states
- DQN: 8 continuous features (4 price features linearly normalized + hour encoded as sin/cos + position + volume_ratio)

### The Two Agents

**Q-Learning**: standard tabular method, Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]. 400 episodes, ε-greedy exploration, α=0.1, γ=0.97.

**Dueling Double DQN + PER**: neural network approach. The Dueling architecture decomposes Q-values into state value V(s) and action advantage A(s,a); Double DQN uses the online network to select actions and the target network to evaluate them, preventing Q-value overestimation; PER (Prioritized Experience Replay) samples high-value experiences based on TD error. 500 episodes, lr=3e-4 (Adam), γ=0.99, batch_size=32, target network hard-updated every 1000 steps.

### Baseline Strategies

- **Buy & Hold**: buy at the first step, hold forever. In a -52% test set, this just loses 52%.
- **Random**: uniformly random action at each step among all 5 actions.
- **RSI Rule**: classic technical indicator rules — RSI < 30 triggers buy/cover, RSI > 70 triggers sell/short.

---

## Results & Discussion

### From Zero to Profit: The Story of 8 Iterations

The experiment wasn't a single run. It went through 8 rounds of iteration, each one responding to problems discovered in the previous round. The process itself turned out to be more interesting than the final numbers.

#### Round 1: The First Results Were Ugly

The first version used ε_decay = 0.9995 (per episode), thinking a slow decay would be stable. Then I checked the math — Q-Learning trained for 400 episodes ended with ε = 0.82, and DQN trained for 200 episodes ended with ε = 0.90. In other words, at the end of training, the agents were still picking random actions 80-90% of the time. That's not learning a strategy; that's guessing.

| Strategy | Total Return | Sharpe | Max Drawdown | Trades |
|----------|-------------|--------|-------------|--------|
| Q-Learning | -65.68% | -3.03 | 69.02% | 304 |
| DQN | -60.07% | -2.51 | 61.73% | 67 |
| Buy & Hold | -52.05% | -1.84 | 57.35% | 1 |

Both RL agents underperformed Buy & Hold. Losing to Buy & Hold in a market that dropped 52% means the agents learned essentially nothing useful.

#### Round 2: Fixing the Epsilon Decay Rate

The problem was clear: ε decayed too slowly. I recalculated — to have ε reach exactly 0.05 by the end of training, you need decay = ε_min^(1/n_episodes). So Q-Learning was changed to 0.9925 (0.9925^400 ≈ 0.05), and DQN to 0.985 (0.985^200 ≈ 0.05).

The improvement was immediate. DQN went from -60% to -25% — only losing 25% in a market that fell 52%, representing +27 percentage points of alpha. Max drawdown dropped from 62% to 36%. Q-Learning improved less dramatically (-66% → -47%), still losing money.

At this stage, the action space was only 3 (BUY/HOLD/SELL), with position limited to {0, 1}. In a sustained bear market, the optimal strategy is simply "do nothing" — staying flat is the best you can do, capping returns at 0%. DQN figured this out: it HOLDs 94-99% of the time, only occasionally trading, which is why it only lost 25%.

#### Round 3: Introducing Short-Selling — Game Changer

I realized that with 3 actions, the ceiling was too low. In a -52% market, the best possible outcome is 0% (stay flat). So I expanded the action space to 5, adding SHORT and COVER. Position went from {0,1} to {-1,0,1}. Q-Learning's state space grew from 4,480 to 6,720.

The results were transformative:

| Strategy | Total Return | Sharpe | Max Drawdown | Trades |
|----------|-------------|--------|-------------|--------|
| **Q-Learning** | **+61.07%** | **+1.93** | 35.96% | 353 |
| DQN | -20.87% | -0.45 | 44.33% | 94 |
| Buy & Hold | -52.05% | -1.84 | 57.35% | 1 |

Q-Learning flipped from losing 47% to gaining 61%! In a market that crashed 52%, that's +113 percentage points of excess return. Looking at its trading behavior in the decline regime: SHORT accounts for 5.5% of actions, COVER 5.3% — it learned to profit from declines through short-selling, and the shorts and covers come in matched pairs, not one-sided bets.

DQN also improved (-25% → -21%), but far less than Q-Learning. This is actually an interesting phenomenon — why does the simpler tabular method outperform the neural network? My interpretation: Q-Learning gives exact Q-values for states it has visited, while DQN needs to generalize, and the network's approximation may be inaccurate for some states, especially for the new SHORT action where it tends to be overly conservative. DQN only made 94 trades (vs. Q-Learning's 353), with SHORT at just 2.5% in decline (vs. Q-Learning's 5.5%) — it wasn't fully exploiting short-selling opportunities.

> Note: This round also uncovered a bug in Q-Learning's greedy action selection — when multiple actions had the same Q-value, `np.random.choice` randomly broke ties, making backtests non-reproducible (the same Q-table produced returns ranging from +24% to +81%). Switching to `np.argmax` (always pick the lowest index) fixed this.

#### Rounds 4-8: Five Rounds of Closed-Loop DQN Optimization

Q-Learning was already performing well after Run #3, but DQN had a lot of room for improvement. The next 5 rounds focused on optimizing DQN.

**Run #4: Fixing a PER Bug**

While reviewing the code, I found a bug in PER's `push()` — new transitions were being pushed with `max_priority ** alpha` as their initial priority, but `update_priorities()` already applies alpha when computing priorities. This meant initial priorities were effectively raised to alpha² = 0.36 instead of alpha = 0.6, systematically underweighting new experiences.

After the fix, DQN improved from -20.87% to -9.42%, and max drawdown dropped sharply to 15.50% (from 44.33%). A single priority calculation bug caused an 11 percentage-point gap — PER implementation details really do matter.

I also fixed a numerical stability issue in the Differential Sharpe Ratio calculation (denominator threshold from 1e-12 to 1e-6, output clipped to [-10, 10]).

**Run #5: Trying Polyak Soft Updates — Failed**

I extended DQN from 200 to 500 episodes and switched the target network from hard updates to Polyak soft updates (τ=0.005). The idea was that soft updates should be smoother.

Instead, DQN degenerated to -32.23% with a single trade — it just sat in cash the entire time. Checking the training logs revealed Q-values had exploded to 22.6 (normal range is around 0.1). The cause: during early training, ε is still around 0.9, meaning the agent is exploring randomly 90% of the time. Soft updates continuously mix the online network's parameters into the target network, but the early online network is full of noise. That noise keeps propagating into the target, which is used to compute TD targets, creating a self-reinforcing inflation loop.

Lesson learned: **Polyak soft updates are harmful during the high-exploration phase.** Unlike DDPG, which starts with a reasonably good policy, DQN's early policy is essentially random — the target network needs to stay far from the online network at that stage.

**Run #6: Slower Epsilon Decay Doesn't Help Either**

I reverted soft updates, kept 500 episodes, but changed ε_decay from 0.985 to 0.994 (slower decay for "more thorough exploration"). Also added cosine learning rate annealing.

Result: -39.24%, about 30 percentage points worse than Run #4. The problem is that ε_decay=0.994 means at episode 100, ε is still 0.548 (55% random). A massive amount of low-quality random experience floods the replay buffer, diluting the useful experiences accumulated earlier. DQN ended up with BUY=2.3% > SHORT=0.7% in decline — the direction is backwards.

Lesson learned: **More episodes ≠ better.** If the exploration rate decay doesn't match the training length, longer training actually hurts.

**Run #7: Sharpe Reward — Catastrophic Failure**

I switched DQN to use Differential Sharpe Ratio as the reward function, and also changed hour_of_day from linear normalization to sin/cos cyclical encoding (under linear encoding, 23:00 and 00:00 are maximally distant, but they're actually adjacent).

DQN cratered to -64.04% with 975 trades — approaching random strategy levels. The reason is straightforward: DSR magnitudes are around ±10, while simple rewards are around ±0.01. That's a 1000× difference. Q-values inflated to 4-8, the agent became wildly overactive. In decline, BUY=9.0% > SHORT=4.0% — direction completely reversed.

Lesson learned: **Differential Sharpe Ratio is unsuitable as a DQN training reward.** Its high variance and non-stationary nature are incompatible with function approximation. It's better suited as a policy evaluation metric, not a training signal.

The sin/cos encoding itself was actually fine and was retained in Run #8.

**Run #8: Merging the Best Configuration — Breakthrough to +70%**

After three consecutive failures, I decided to go back to the most stable configuration and add incrementally. I merged Run #4's core setup (fixed PER + hard updates + simple reward) with two validated improvements (sin/cos encoding + 500 episodes), crucially using ε_decay=0.985 again.

Why does ε_decay=0.985 with 500 episodes work? Because 0.985^200 ≈ 0.05, meaning ε hits its minimum at episode 200. **The remaining 300 episodes are pure exploitation** — the agent picks greedy actions 95% of the time, continuously refining the strategy it already learned. Run #4 only had 200 episodes, so training ended right when ε reached its minimum, with no exploitation period to polish the policy.

| Strategy | Total Return | Sharpe | Max Drawdown | Win Rate | Final Value | Trades |
|----------|-------------|--------|-------------|----------|-------------|--------|
| **DQN** | **+70.05%** | **+2.16** | **31.69%** | 49.39% | **$17,005** | 55 |
| Q-Learning | +45.83% | +1.64 | 26.39% | 47.09% | $14,583 | 217 |
| RSI Rule | -35.20% | -0.93 | 48.67% | 45.84% | $6,480 | 37 |
| Buy & Hold | -52.05% | -1.84 | 57.35% | 50.06% | $4,795 | 1 |
| Random | -69.12% | -4.57 | 71.77% | 33.11% | $3,088 | 929 |

DQN finally surpassed Q-Learning, and by a wide margin — +70.05% vs. +45.83%. In a market that dropped 52%, it turned $10,000 into $17,005.

Looking at DQN's trading behavior: just 55 trades (1.6% trade rate), each contributing an average of +1.27% return. In decline, SHORT=1.5% and COVER=1.4% (paired), with almost no trading in rise or steady. Extreme selectivity — only acting when it's most confident.

### Summary Across Eight Rounds

| Run | Key Change | Q-Learning | DQN | Best |
|-----|-----------|------------|-----|------|
| #1 | Baseline (ε decay too slow) | -65.68% | -60.07% | RSI Rule |
| #2 | Fixed ε decay rate | -46.65% | -25.28% | DQN |
| #3 | Short-selling (5 actions) | **+61.07%** | -20.87% | Q-Learning |
| #4 | Fixed PER bug | +28.93% | -9.42% | Q-Learning |
| #5 | Soft update ❌ | +45.83% | -32.23% | Q-Learning |
| #6 | Slow ε decay ❌ | +45.83% | -39.24% | Q-Learning |
| #7 | Sharpe reward ❌ | +45.83% | -64.04% | Q-Learning |
| #8 | Best config merged | +45.83% | **+70.05%** | **DQN** |

Q-Learning stabilized at +45.83% from Run #5 onward because a global random seed (seed=42) was introduced, and Q-Learning's code and hyperparameters hadn't changed since Run #4. The earlier +61.07% in Run #3 was a "lucky" random training outcome (no seed), while Run #4's +28.93% was "unlucky." With the seed fixed, the result settled at +45.83%, right between the two.

DQN's trajectory is more dramatic: from -60% all the way to +70%, with three failed attempts along the way. In retrospect, only four changes actually worked — fixing the ε decay rate, introducing short-selling, fixing the PER bug, and extending the pure exploitation period. Everything else (soft updates, LR scheduling, Sharpe reward) had negative effects.

### Cross-Asset Correlation Analysis

Beyond the RL trading experiments, I also analyzed TON's correlation with other assets. A few findings stood out:

**Correlations spike during stress periods.** TON's overall correlation with BTC is 0.522, but broken down by regime, it's only 0.376 during rise and jumps to 0.592 during decline. ETH and SOL show the same pattern. This is textbook "crisis contagion" — assets go their own way when things are good, but crash together when things go bad.

| Asset | Rise | Steady | Decline | Change |
|-------|------|--------|---------|--------|
| BTC | 0.376 | 0.547 | 0.592 | +0.216 |
| ETH | 0.391 | 0.595 | 0.635 | +0.244 |
| SOL | 0.390 | 0.573 | 0.625 | +0.235 |
| S&P 500 | 0.132 | 0.242 | 0.322 | +0.190 |
| Gold | 0.023 | 0.038 | 0.113 | +0.090 |

**TON is not "digital gold."** TON's correlation with gold is extremely low across all regimes (0.02–0.11), and with U.S. Treasuries it's even slightly negative (-0.082 during decline). Using gold to hedge TON risk would be essentially useless.

**Intraday effects.** The Asian session (00-08 UTC) has the highest average return (+0.006%) and lowest volatility (0.847%); the U.S. session (16-24 UTC) has the lowest return (-0.009%) and highest volatility (0.893%). This might relate to U.S. traders' risk appetite — TON tends to sell off during their trading hours.

---

## Conclusions

Back to the original question: how well can an RL agent do in a crypto market that crashed 52%?

The answer: **DQN achieved +70%, Q-Learning achieved +46%.** Both agents significantly outperformed Buy & Hold (-52%), the RSI rule strategy (-35%), and the random strategy (-69%). DQN ultimately turned $10,000 into $17,005 with a Sharpe Ratio of 2.16.

A few key takeaways:

1. **Short-selling is the game changer in a bear market.** With 3 actions, the best you can do is 0% (stay flat). Adding short-selling lifts the theoretical ceiling dramatically. This isn't a profound insight, but the experiment demonstrated it vividly — Q-Learning flipped from -47% to +61% after short-selling was introduced.

2. **Engineering details matter far more than expected.** A wrong ε decay rate meant the agent couldn't learn at all (Run #1). An inconspicuous PER priority bug was worth 11 percentage points (Run #4). The difference in exploitation period length was the gap between -9% and +70% (Run #4 vs. Run #8). None of these are "algorithmic innovations" — they're purely implementation-level issues.

3. **More complex doesn't mean better.** Soft updates, LR scheduling, Sharpe reward — these more sophisticated techniques all failed. The final winning configuration was the simplest one: hard updates, fixed learning rate, simple reward. The only "advanced" modification that survived was sin/cos time encoding, which is really just a natural way to handle cyclical features.

4. **Closed-loop iteration beats one-shot design.** If I'd stacked soft updates + Sharpe reward + slow decay all at once from the start, the result would have been catastrophic. It was precisely by changing one variable per round, observing the result, then deciding the next step, that ineffective approaches got eliminated and effective improvements got preserved.

**Limitations:**
- Only one seed was tested (seed=42); robustness across multiple seeds is unverified
- The test set is only 144 days of a single bear market; generalization to other market conditions is unknown
- All-in position sizing with no risk management — fairly crude
- Transaction cost was set at a fixed 0.1%; in practice, slippage and liquidity impact could be larger

---

## Appendix

### A. Full Hyperparameter Table

**Q-Learning:**

| Parameter | Value |
|-----------|-------|
| n_actions | 5 |
| alpha (learning rate) | 0.1 |
| gamma (discount factor) | 0.97 |
| epsilon (initial) | 1.0 |
| epsilon_min | 0.05 |
| epsilon_decay | 0.9925 per episode |
| n_episodes | 400 |
| episode_length | 720 (~30 days) |
| State space | 6,720 (discrete) |

**DQN (final config, Run #8):**

| Parameter | Value |
|-----------|-------|
| state_dim | 8 |
| n_actions | 5 |
| lr | 0.0003 (Adam) |
| gamma | 0.99 |
| batch_size | 32 |
| buffer_size | 100,000 |
| target_update | 1000 steps (hard update) |
| epsilon → epsilon_min | 1.0 → 0.05 |
| epsilon_decay | 0.985 per episode |
| n_episodes | 500 |
| episode_length | 720 |
| gradient_clip | 1.0 |
| hidden_dim | 128 |
| v_stream / a_stream | 64 / 64 |
| PER alpha | 0.6 |
| PER beta | 0.4 → 1.0 |
| Loss | Huber (SmoothL1) |

**Environment:**

| Parameter | Value |
|-----------|-------|
| initial_balance | $10,000 |
| cost_rate | 0.001 (0.1%) |
| reward_mode | simple |

### B. Core Code

#### B.1 Trading Environment (environment.py)

```python
class CryptoTradingEnv:
    """
    OpenAI Gym-style trading environment
    Actions: {BUY(0), HOLD(1), SELL(2), SHORT(3), COVER(4)}
    Position: position in {-1(short), 0(flat), 1(long)}
    """
    def __init__(self, df, mode="discrete", reward_mode="simple",
                 initial_balance=None, cost_rate=None, **reward_kwargs):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.reward_mode = reward_mode
        self.initial_balance = initial_balance or ENV_CONFIG["initial_balance"]
        self.reward_params = REWARD_CONFIG.get(reward_mode, REWARD_CONFIG["simple"]).copy()
        self.cost_rate = self.reward_params["cost_rate"]
        # Differential Sharpe Ratio EMA state
        self._sharpe_A = 0.0
        self._sharpe_B = 0.0

    def reset(self, start_idx=None, episode_length=720):
        max_start = self.data_length - episode_length - 1
        if start_idx is None:
            self.current_step = np.random.randint(0, max(1, max_start))
        else:
            self.current_step = min(start_idx, max_start)
        self.end_step = self.current_step + episode_length
        self.position = 0
        self.balance = self.initial_balance
        return self._get_state()

    def step(self, action):
        current_price = self.df.iloc[self.current_step]["close"]
        position_before = self.position

        if action == BUY and self.position == 0:
            cost = self.balance * self.cost_rate
            self.holdings = (self.balance - cost) / current_price
            self.balance = 0.0
            self.position = 1
        elif action == SELL and self.position == 1:
            sell_value = self.holdings * current_price
            self.balance = sell_value * (1 - self.cost_rate)
            self.holdings = 0.0
            self.position = 0
        elif action == SHORT and self.position == 0:
            cost = self.balance * self.cost_rate
            self.short_holdings = (self.balance - cost) / current_price
            self.short_entry_price = current_price
            self.balance = 0.0
            self.position = -1
        elif action == COVER and self.position == -1:
            pnl = self.short_holdings * (self.short_entry_price - current_price)
            gross_value = self.short_holdings * self.short_entry_price + pnl
            self.balance = max(gross_value * (1 - self.cost_rate), 0)
            self.position = 0

        self.current_step += 1
        reward = self._compute_reward(position_before, ...)
        done = self.current_step >= self.end_step
        return self._get_state(), reward, done, info

    def _compute_reward(self, position_before, ...):
        hourly_return = (price_after - price_before) / price_before
        portfolio_return = position_before * hourly_return
        cost_penalty = self.cost_rate * abs(self.position - position_before)
        return portfolio_return - cost_penalty  # simple mode
```

#### B.2 Q-Learning Agent (q_learning.py)

```python
class QLearningAgent:
    def __init__(self, n_actions=5, alpha=0.1, gamma=0.97,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9925):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        max_actions = np.where(q_values == max_q)[0]
        return np.random.choice(max_actions)

    def select_greedy_action(self, state):
        """Deterministic greedy (for backtesting)"""
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        td_error = target - current_q
        self.q_table[state][action] = current_q + self.alpha * td_error
        return td_error
```

#### B.3 Dueling Double DQN + PER (dqn.py)

```python
class DuelingDQN(nn.Module):
    """Q(s,a) = V(s) + A(s,a) - mean(A)"""
    def __init__(self, state_dim=8, n_actions=5, hidden_dim=128, v_dim=64, a_dim=64):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.v_stream = nn.Sequential(nn.Linear(hidden_dim, v_dim), nn.ReLU(), nn.Linear(v_dim, 1))
        self.a_stream = nn.Sequential(nn.Linear(hidden_dim, a_dim), nn.ReLU(), nn.Linear(a_dim, n_actions))

    def forward(self, x):
        features = self.feature(x)
        v = self.v_stream(features)
        a = self.a_stream(features)
        return v + (a - a.mean(dim=1, keepdim=True))


class PrioritizedReplayBuffer:
    """SumTree-based PER with importance sampling"""
    def push(self, state, action, reward, next_state, done):
        priority = self.max_priority  # Initial priority = current max priority
        self.tree.add(priority, (state, action, reward, next_state, done))

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)


class DQNAgent:
    def train_step(self):
        batch, indices, weights = self.buffer.sample(self.batch_size)
        # Double DQN: online network selects actions, target network evaluates
        best_actions = self.online_net(next_states_t).argmax(dim=1)
        next_q = self.target_net(next_states_t).gather(1, best_actions.unsqueeze(1)).squeeze(1)
        target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        td_errors = (current_q - target_q).detach().cpu().numpy()
        loss = (self.loss_fn(current_q, target_q) * weights_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.buffer.update_priorities(indices, td_errors)

        # Hard update target network
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
```

#### B.4 Feature Engineering (data_pipeline.py)

```python
def compute_features(df):
    df["price_change_24h"] = df["close"].pct_change(24)
    df["price_change_4h"] = df["close"].pct_change(4)
    df["volatility_24h"] = df["close"].pct_change().rolling(24).std()
    df["rsi_14"] = compute_rsi(df["close"], period=14)
    df["hour_of_day"] = df.index.hour
    # Volume ratio (set to 1.0 when volume is zero)
    vol_rolling = df["volume"].replace(0, np.nan).rolling(72, min_periods=1).mean()
    df["volume_ratio"] = np.where(df["volume"] == 0, 1.0, df["volume"] / vol_rolling)
    return df

def normalize_state(features):
    """DQN state: 4 linearly normalized features + hour sin/cos + position + volume_ratio -> 8D"""
    state = []
    for name in ["price_change_24h", "price_change_4h", "volatility_24h", "rsi_14"]:
        state.append(clip_and_normalize(features[name], RANGES[name]))
    # hour: sin/cos cyclical encoding
    hour = features["hour_of_day"]
    state.append(np.sin(2 * np.pi * hour / 24.0))
    state.append(np.cos(2 * np.pi * hour / 24.0))
    for name in ["position", "volume_ratio"]:
        state.append(clip_and_normalize(features[name], RANGES[name]))
    return np.array(state, dtype=np.float32)  # shape (8,)
```

#### B.5 Backtesting Engine (backtest.py)

```python
def backtest(env, agent, agent_type="qlearning"):
    """Run full greedy backtest on test set"""
    state = env.reset(start_idx=0, episode_length=len(env.df) - 1)
    while not done:
        action = agent.select_greedy_action(state)
        state, reward, done, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
    return {"portfolio_values": ..., "actions": ..., "positions": ..., "regimes": ...}

def compute_metrics(portfolio_values):
    total_return = (pv[-1] - pv[0]) / pv[0]
    annualized_sharpe = (mean(returns) / std(returns)) * sqrt(8760)
    max_drawdown = max((cummax - pv) / cummax)
    win_rate = mean(returns > 0)
    return {total_return, annualized_sharpe, max_drawdown, win_rate, final_value}

def run_all_backtests(test_df, q_agent, dqn_agent):
    """One-stop run: Q-Learning, DQN, Buy & Hold, Random, RSI Rule"""
    ...
```

#### B.6 Main Entry Point (main.py)

```python
def set_global_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def main():
    set_global_seed(args.seed)
    full_df, train_df, test_df = load_and_prepare_ton()

    # Phase 2: Training
    env_q = CryptoTradingEnv(train_df, mode="discrete", reward_mode="simple")
    agent_q = QLearningAgent()
    train_qlearning(env_q, agent_q)  # 400 eps x 720 steps

    env_dqn = CryptoTradingEnv(train_df, mode="continuous", reward_mode="simple")
    agent_dqn = DQNAgent()
    train_dqn(env_dqn, agent_dqn)  # 500 eps x 720 steps

    # Phase 3: Backtesting
    results = run_all_backtests(test_df, agent_q, agent_dqn)

    # Phase 4: Correlation analysis
    run_correlation_analysis(full_df)
```
