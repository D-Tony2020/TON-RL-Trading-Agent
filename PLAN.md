# Assignment 3 实现计划 — REINFORCE + 多交易者 SHAP + 监管分析

## Context

1. 实现 REINFORCE with Baseline，与已有 DQN 对比
2. 建模 3 种交易者类型（套利者/操纵者/散户），用 SHAP 解释特征重要性
3. 绘制危机频率、稳定币稳定性、市场效率三张监管分析图
4. 撰写 2-3 页报告

项目路径：`D:\Desktop\RL\project 1\`

---

## Phase 1: config.py 扩展

**文件：** `src/config.py`

新增：
- `REINFORCE_PARAMS`: state_dim=8, n_actions=5, lr_policy=3e-4, lr_value=1e-3, gamma=0.99, hidden_dim=128, n_episodes=500, episode_length=720, gradient_clip=1.0, entropy_coeff=0.01, checkpoint_interval=20
- `FEATURE_NAMES`: 8 个特征名（对应 normalize_state 的 8D 输出）
- `SHAP_CONFIG`: n_background_samples=100, n_explain_samples=500
- `REWARD_CONFIG` 新增 3 个 trader 奖励模式：
  - `"arbitrageur"`: 高 cost_rate(0.002) + Sharpe 奖励 + 均值回归 bonus
  - `"manipulator"`: 极低 cost_rate(0.0002) + 动量 bonus + 成交量 bonus
  - `"retail"`: 趋势追随 bonus + 恐慌卖出 bonus + 高回撤惩罚

---

## Phase 2: environment.py — 3 种 trader 奖励

**文件：** `src/environment.py` -> `_compute_reward()` (L246-312)

在 `simple`/`sharpe` 后新增 3 个 `elif`，从 `self.df.iloc[self.current_step]` 读取特征：

- **arbitrageur**: `portfolio_return - cost_penalty - risk_penalty + sharpe_bonus + mean_reversion_bonus`
  - mean_reversion_bonus: RSI<0.3 做多或 RSI>0.7 做空时正奖励
- **manipulator**: `portfolio_return - cost_penalty + momentum_bonus + volume_bonus`
  - momentum_bonus: price_change_4h 方向与仓位一致；volume_bonus: volume_ratio > 阈值时交易
- **retail**: `portfolio_return - cost_penalty - risk_penalty + trend_bonus + panic_bonus`
  - trend_bonus: price_change_24h 方向与仓位一致；panic_bonus: 高波动率时卖出

---

## Phase 3: REINFORCE Agent

**新文件：** `src/agents/reinforce.py`

### 架构：双独立网络
- **PolicyNetwork**: Linear(8->128)->ReLU->Linear(128->128)->ReLU->Linear(128->5)->Softmax
- **ValueNetwork**: Linear(8->128)->ReLU->Linear(128->128)->ReLU->Linear(128->1)

### REINFORCEAgent 接口（与 DQNAgent 对齐）
- `select_action(state)` -> Categorical 采样，存 log_prob + value 到 episode buffer
- `select_greedy_action(state)` -> argmax(probs)
- `finish_episode()` -> 从后向前算 G_t，advantage = G_t - V(s_t)（标准化），policy_loss = -sum(log_prob * adv) - entropy_coeff * entropy，value_loss = MSE(V, G_t)
- `save_checkpoint()` / `load_checkpoint()`
- `decay_epsilon()` -> no-op（保留接口兼容）

### train_reinforce()
- 签名同 train_dqn()，每 episode 结束调 finish_episode()
- 返回 history: episode_rewards, portfolio_values, policy_losses, value_losses, entropy_history
- **GPU 支持**：PolicyNetwork/ValueNetwork 必须 `torch.device('cuda' if available else 'cpu')`，训练开始打印设备信息
- **进度打印 + ETA**：每 10 episode 打印 (reward, policy_loss, value_loss, entropy, elapsed, ETA)
- **Checkpoint**：每 20 episode 保存 policy_net + value_net + 两个 optimizer + episode 计数到 `output/checkpoints/reinforce_epN.pt`
- **--resume 支持**：`load_checkpoint()` 可恢复训练状态，main.py 中支持 `--resume` 从最近 checkpoint 继续
- 测试不通过不进入下一 Phase

---

## Phase 4: 测试

**新文件：** `tests/test_reinforce.py`

- PolicyNetwork/ValueNetwork forward shape 验证
- select_action 范围、select_greedy_action 确定性
- checkpoint 保存/加载一致性
- 5 episode 冒烟测试（无 NaN/崩溃）

---

## Phase 5: 多交易者 + SHAP

**新文件：** `src/traders.py`

| 交易者 | 奖励模式 | 预期行为 |
|--------|----------|----------|
| Arbitrageur | `arbitrageur` | 均值回归、低频、风险调整 |
| Manipulator | `manipulator` | 动量追逐、高频、量价利用 |
| Retail | `retail` | 趋势追随、恐慌卖出、高风险厌恶 |

三种均用 REINFORCE agent，仅奖励不同。

核心函数：
- `train_all_traders(train_df)` -> {type: (agent, history)}
- `compute_shap_values(agent, env)` -> 用 `shap.KernelExplainer`，封装 policy_net 为 numpy callable
- `explain_top_features(shap_values)` -> 每种 trader 的 top 3 特征
- `plot_shap_comparison(all_shap)` -> 3 行 bar chart 对比图

---

## Phase 6: 监管分析图表

**新文件：** `src/regulatory.py`

1. **Crisis Frequency**: 按周/月统计 `regime=='decline'` 占比，折线图
2. **Stablecoin Stability**: `1 / (1 + rolling_std(TON-BTC correlation))`，双 Y 轴图（复用 `correlation.py` 的 `rolling_correlation()`）
3. **Market Efficiency**: 滚动 lag-1 autocorrelation of hourly returns，带 +/-0.05 参考线

可视化加入 `src/visualization.py`：`plot_regulatory_dashboard()` 三合一子图

---

## Phase 7: main.py 集成

**文件：** `main.py`

新增 mode：
- `reinforce` -- 训练 REINFORCE + PG vs DQN 对比回测
- `traders` -- 训练 3 种 trader + SHAP 分析
- `regulatory` -- 3 张监管图
- `assignment3` -- 一键全部

不修改 `run_all_backtests()` 签名，在 main.py 层手动组装对比 results。

---

## Phase 8: 收尾

- 更新 `CLAUDE.md` 第 4 节：动作空间改为 5 actions {BUY,HOLD,SELL,SHORT,COVER}，position {-1,0,1}，DQN 8D 连续状态，新增 REINFORCE 算法
- `requirements.txt` 新增 `shap>=0.42`
- `pytest tests/ -v` 全量通过
- `python main.py --mode assignment3` 端到端验证

---

## 文件变更总结

| 操作 | 文件 |
|------|------|
| **新建** | `src/agents/reinforce.py` -- REINFORCE agent 核心 |
| **新建** | `src/traders.py` -- trader 管理 + SHAP |
| **新建** | `src/regulatory.py` -- 监管分析指标 |
| **新建** | `tests/test_reinforce.py` -- REINFORCE 测试 |
| **修改** | `src/config.py` -- REINFORCE_PARAMS, FEATURE_NAMES, SHAP_CONFIG, 3 种 REWARD |
| **修改** | `src/environment.py` -- _compute_reward() +3 个 elif |
| **修改** | `src/visualization.py` -- SHAP 图 + 监管仪表盘 |
| **修改** | `main.py` -- +4 种 mode |
| **修改** | `requirements.txt` -- +shap |

## 验证方案

1. `pytest tests/ -v` 全量通过
2. `python main.py --mode reinforce` -- 训练 REINFORCE，输出 PG vs DQN 对比表和图
3. `python main.py --mode traders` -- 训练 3 种 trader，输出 SHAP 分析图和 top 3 解释
4. `python main.py --mode regulatory` -- 输出 3 张监管分析图
5. `python main.py --mode assignment3` -- 端到端一键运行全部
