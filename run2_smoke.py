"""Run 2: Reward scaling ×100 + gamma 0.95, 100ep smoke test"""
import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CHECKPOINT_DIR, RESULTS_DIR, OUTPUT_DIR, FIGURES_DIR
from src.data_pipeline import load_and_prepare_ton
from src.environment import CryptoTradingEnv
from src.agents.reinforce import REINFORCEAgent, train_reinforce
from src.backtest import backtest, compute_metrics, count_trades

np.random.seed(42)
torch.manual_seed(42)

for d in [OUTPUT_DIR, CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

full_df, train_df, test_df = load_and_prepare_ton()
print(f"数据: train={len(train_df)}, test={len(test_df)}")

# 训练 100ep
env = CryptoTradingEnv(train_df, mode="continuous", reward_mode="simple")
print(f"Reward scale: {env.reward_scale}")
agent = REINFORCEAgent()
print(f"Gamma: {agent.gamma}, Entropy coeff: {agent.entropy_coeff}")

t0 = time.time()
history = train_reinforce(env, agent, n_episodes=100, verbose=True)
train_time = time.time() - t0

ckpt_path = str(CHECKPOINT_DIR / "reinforce_run2_ep100.pt")
agent.save_checkpoint(ckpt_path)

# 回测
env_test = CryptoTradingEnv(test_df, mode="continuous", reward_mode="simple")
bt = backtest(env_test, agent, agent_type="reinforce")
metrics = compute_metrics(bt["portfolio_values"])

result = {
    "run": 2,
    "episodes": 100,
    "changes": ["reward_scale=100", "gamma 0.99→0.95"],
    "training_time": train_time,
    "entropy_trend": {
        "ep10": history["entropy_history"][9],
        "ep50": history["entropy_history"][49],
        "ep100": history["entropy_history"][-1],
    },
    "policy_loss_trend": {
        "ep10": history["policy_losses"][9],
        "ep50": history["policy_losses"][49],
        "ep100": history["policy_losses"][-1],
    },
    "backtest": {
        "total_return": metrics["total_return"],
        "sharpe": metrics["annualized_sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "win_rate": metrics["win_rate"],
        "trades": int(count_trades(bt["actions"])),
    },
    "avg_reward_last10": float(np.mean(history["episode_rewards"][-10:])),
    "avg_pv_last10": float(np.mean(history["episode_portfolio_values"][-10:])),
}

results_path = RESULTS_DIR / "run2_metrics.json"
with open(results_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\n{'='*60}")
print(f"Run 2 诊断汇报")
print(f"{'='*60}")
print(f"  Entropy 趋势: ep10={result['entropy_trend']['ep10']:.4f} → ep50={result['entropy_trend']['ep50']:.4f} → ep100={result['entropy_trend']['ep100']:.4f}")
print(f"  Policy Loss: ep10={result['policy_loss_trend']['ep10']:.4f} → ep50={result['policy_loss_trend']['ep50']:.4f} → ep100={result['policy_loss_trend']['ep100']:.4f}")
print(f"  回测: Return={metrics['total_return']:+.2%}, Sharpe={metrics['annualized_sharpe']:.2f}, Trades={result['backtest']['trades']}")
print(f"  最近10ep 平均 PV: ${result['avg_pv_last10']:,.0f}")
print(f"  最近10ep 平均 Reward: {result['avg_reward_last10']:.2f}")

if result['entropy_trend']['ep100'] < 1.4:
    print(f"\n  ✅ Entropy 显著下降，策略在收敛")
elif result['entropy_trend']['ep100'] < 1.55:
    print(f"\n  ⚠️ Entropy 有下降但缓慢，可能需要更多 episodes")
else:
    print(f"\n  ❌ Entropy 仍高，需要进一步调整")
