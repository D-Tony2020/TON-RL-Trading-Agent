"""Run 5: 重训 DQN(reward_scale=100) + 用 Run3 best REINFORCE + 重训交易者 + SHAP + 汇总"""
import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CHECKPOINT_DIR, RESULTS_DIR, OUTPUT_DIR, FIGURES_DIR
from src.data_pipeline import load_and_prepare_ton
from src.environment import CryptoTradingEnv
from src.agents.dqn import DQNAgent, train_dqn
from src.agents.reinforce import REINFORCEAgent, train_reinforce
from src.backtest import (
    backtest, compute_metrics, count_trades, format_results_table,
    run_all_backtests, backtest_buy_and_hold, backtest_random, backtest_rsi_rule,
)
from src.visualization import (
    plot_training_curves, plot_backtest_comparison, plot_agent_actions,
)
from src.traders import run_trader_analysis
from src.regulatory import plot_regulatory_dashboard

np.random.seed(42)
torch.manual_seed(42)

for d in [OUTPUT_DIR, CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

full_df, train_df, test_df = load_and_prepare_ton()
print(f"数据: train={len(train_df)}, test={len(test_df)}")

# ================================================================
# 1. 重训 DQN (reward_scale=100)
# ================================================================
print(f"\n{'='*60}")
print("Step 1: 重训 DQN (reward_scale=100)")
print(f"{'='*60}")

np.random.seed(42)
torch.manual_seed(42)
env_dqn = CryptoTradingEnv(train_df, mode="continuous", reward_mode="simple")
print(f"  Reward scale: {env_dqn.reward_scale}")
agent_dqn = DQNAgent()
t0 = time.time()
history_dqn = train_dqn(env_dqn, agent_dqn, n_episodes=500, verbose=True)
dqn_time = time.time() - t0

agent_dqn.save_checkpoint(str(CHECKPOINT_DIR / "dqn_run5_final.pt"))
plot_training_curves(history_dqn, agent_name="DQN-Run5")

# DQN 回测
env_test_dqn = CryptoTradingEnv(test_df, mode="continuous", reward_mode="simple")
bt_dqn = backtest(env_test_dqn, agent_dqn, agent_type="dqn")
metrics_dqn = compute_metrics(bt_dqn["portfolio_values"])
print(f"  DQN: Return={metrics_dqn['total_return']:+.2%}, Sharpe={metrics_dqn['annualized_sharpe']:.2f}, Trades={count_trades(bt_dqn['actions'])}")

# ================================================================
# 2. 训练 REINFORCE (用 Run3 最佳配置: lr=0.003, ep_length=200)
# ================================================================
print(f"\n{'='*60}")
print("Step 2: 训练 REINFORCE (Run3 配置)")
print(f"{'='*60}")

np.random.seed(42)
torch.manual_seed(42)
env_r = CryptoTradingEnv(train_df, mode="continuous", reward_mode="simple")
agent_r = REINFORCEAgent(lr_policy=0.003, gamma=0.95, entropy_coeff=0.001)
t0 = time.time()
history_r = train_reinforce(env_r, agent_r, n_episodes=500, episode_length=200, verbose=True)
r_time = time.time() - t0

agent_r.save_checkpoint(str(CHECKPOINT_DIR / "reinforce_run5_final.pt"))
plot_training_curves(history_r, agent_name="REINFORCE-Run5")

# REINFORCE 回测
env_test_r = CryptoTradingEnv(test_df, mode="continuous", reward_mode="simple")
bt_r = backtest(env_test_r, agent_r, agent_type="reinforce")
metrics_r = compute_metrics(bt_r["portfolio_values"])
print(f"  REINFORCE: Return={metrics_r['total_return']:+.2%}, Sharpe={metrics_r['annualized_sharpe']:.2f}, Trades={count_trades(bt_r['actions'])}")

# ================================================================
# 3. Baseline 策略
# ================================================================
print(f"\n{'='*60}")
print("Step 3: Baseline 策略回测")
print(f"{'='*60}")

# Buy & Hold
env_bh = CryptoTradingEnv(test_df, mode="discrete", reward_mode="simple")
bt_bh = backtest_buy_and_hold(env_bh)
metrics_bh = compute_metrics(bt_bh["portfolio_values"])

# Random
env_rand = CryptoTradingEnv(test_df, mode="discrete", reward_mode="simple")
bt_rand = backtest_random(env_rand)
metrics_rand = compute_metrics(bt_rand["portfolio_values"])

# RSI Rule
env_rsi = CryptoTradingEnv(test_df, mode="discrete", reward_mode="simple")
bt_rsi = backtest_rsi_rule(env_rsi)
metrics_rsi = compute_metrics(bt_rsi["portfolio_values"])

# 汇总对比
all_results = {
    "DQN": (bt_dqn, metrics_dqn),
    "REINFORCE": (bt_r, metrics_r),
    "Buy & Hold": (bt_bh, metrics_bh),
    "Random": (bt_rand, metrics_rand),
    "RSI Rule": (bt_rsi, metrics_rsi),
}
print("\n" + format_results_table(all_results))
plot_backtest_comparison(all_results)
plot_agent_actions(test_df, bt_dqn, agent_name="DQN", metrics=metrics_dqn)
plot_agent_actions(test_df, bt_r, agent_name="REINFORCE", metrics=metrics_r)

# ================================================================
# 4. 三种交易者 + SHAP
# ================================================================
print(f"\n{'='*60}")
print("Step 4: 三种交易者 + SHAP")
print(f"{'='*60}")

np.random.seed(42)
torch.manual_seed(42)
# 交易者用 lr=0.003, ep_length=200 的激进配置
import src.config as cfg
old_lr = cfg.REINFORCE_PARAMS["lr_policy"]
old_gamma = cfg.REINFORCE_PARAMS["gamma"]
old_ent = cfg.REINFORCE_PARAMS["entropy_coeff"]
cfg.REINFORCE_PARAMS["lr_policy"] = 0.003
cfg.REINFORCE_PARAMS["gamma"] = 0.95
cfg.REINFORCE_PARAMS["entropy_coeff"] = 0.001

trader_result = run_trader_analysis(
    train_df, n_episodes=300, episode_length=200, verbose=True,
)

# 恢复
cfg.REINFORCE_PARAMS["lr_policy"] = old_lr
cfg.REINFORCE_PARAMS["gamma"] = old_gamma
cfg.REINFORCE_PARAMS["entropy_coeff"] = old_ent

# 交易者回测
trader_bt = {}
for ttype, (agent, history) in trader_result["trader_results"].items():
    env_bt = CryptoTradingEnv(test_df, mode="continuous", reward_mode=ttype)
    bt = backtest(env_bt, agent, agent_type="reinforce")
    m = compute_metrics(bt["portfolio_values"])
    trader_bt[ttype] = (bt, m)
    print(f"  {ttype}: Return={m['total_return']:+.2%}, Sharpe={m['annualized_sharpe']:.2f}, Trades={count_trades(bt['actions'])}")

# ================================================================
# 5. 监管分析
# ================================================================
print(f"\n{'='*60}")
print("Step 5: 监管分析")
print(f"{'='*60}")

reg_path = plot_regulatory_dashboard(full_df)
print(f"  [OK] {reg_path}")

# ================================================================
# 6. 保存最终汇总
# ================================================================
summary = {
    "run": "5-final",
    "config_notes": "DQN: reward_scale=100; REINFORCE: lr=0.003, gamma=0.95, ep_length=200, reward_scale=100",
    "comparison": {
        "DQN": {"return": metrics_dqn["total_return"], "sharpe": metrics_dqn["annualized_sharpe"],
                "max_dd": metrics_dqn["max_drawdown"], "trades": int(count_trades(bt_dqn["actions"])),
                "training_time": dqn_time},
        "REINFORCE": {"return": metrics_r["total_return"], "sharpe": metrics_r["annualized_sharpe"],
                      "max_dd": metrics_r["max_drawdown"], "trades": int(count_trades(bt_r["actions"])),
                      "training_time": r_time},
        "Buy_Hold": {"return": metrics_bh["total_return"], "sharpe": metrics_bh["annualized_sharpe"]},
        "Random": {"return": metrics_rand["total_return"], "sharpe": metrics_rand["annualized_sharpe"]},
        "RSI_Rule": {"return": metrics_rsi["total_return"], "sharpe": metrics_rsi["annualized_sharpe"]},
    },
    "traders": {},
    "top_features": {},
}

for ttype, (bt, m) in trader_bt.items():
    summary["traders"][ttype] = {
        "return": m["total_return"], "sharpe": m["annualized_sharpe"],
        "max_dd": m["max_drawdown"], "trades": int(count_trades(bt["actions"])),
    }

for ttype, features in trader_result["top_features"].items():
    summary["top_features"][ttype] = {}
    for action, feats in features.items():
        summary["top_features"][ttype][action] = [(n, float(v)) for n, v in feats]

results_path = RESULTS_DIR / "run5_final_results.json"
with open(results_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n{'='*60}")
print("[OK] Run 5 全部完成！")
print(f"  结果: {results_path}")
print(f"{'='*60}")
