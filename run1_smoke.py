"""Run 1: REINFORCE 核心修复后 100ep smoke test"""
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

# 加载数据
full_df, train_df, test_df = load_and_prepare_ton()
print(f"数据: train={len(train_df)}, test={len(test_df)}")

# 训练 100 episodes
env = CryptoTradingEnv(train_df, mode="continuous", reward_mode="simple")
agent = REINFORCEAgent()
t0 = time.time()
history = train_reinforce(env, agent, n_episodes=100, verbose=True)
train_time = time.time() - t0

# 保存 checkpoint
ckpt_path = str(CHECKPOINT_DIR / "reinforce_run1_ep100.pt")
agent.save_checkpoint(ckpt_path)
print(f"\nCheckpoint 保存: {ckpt_path}")

# 回测
env_test = CryptoTradingEnv(test_df, mode="continuous", reward_mode="simple")
bt = backtest(env_test, agent, agent_type="reinforce")
metrics = compute_metrics(bt["portfolio_values"])

# 汇总
result = {
    "run": 1,
    "episodes": 100,
    "changes": ["remove Softmax", "remove adv normalization", "lr_policy 0.0003→0.001", "entropy_coeff 0.01→0.001"],
    "training_time": train_time,
    "final_entropy": history["entropy_history"][-1],
    "entropy_trend": {
        "ep10": history["entropy_history"][9] if len(history["entropy_history"]) > 9 else None,
        "ep50": history["entropy_history"][49] if len(history["entropy_history"]) > 49 else None,
        "ep100": history["entropy_history"][-1],
    },
    "final_policy_loss": history["policy_losses"][-1],
    "policy_loss_trend": {
        "ep10": history["policy_losses"][9] if len(history["policy_losses"]) > 9 else None,
        "ep50": history["policy_losses"][49] if len(history["policy_losses"]) > 49 else None,
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

results_path = RESULTS_DIR / "run1_metrics.json"
with open(results_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n结果保存: {results_path}")

# 打印诊断
print(f"\n{'='*60}")
print(f"Run 1 诊断汇报")
print(f"{'='*60}")
print(f"  Entropy 趋势: ep10={result['entropy_trend']['ep10']:.4f} → ep50={result['entropy_trend']['ep50']:.4f} → ep100={result['entropy_trend']['ep100']:.4f}")
print(f"  Policy Loss 趋势: ep10={result['policy_loss_trend']['ep10']:.6f} → ep50={result['policy_loss_trend']['ep50']:.6f} → ep100={result['policy_loss_trend']['ep100']:.6f}")
print(f"  回测: Return={metrics['total_return']:+.2%}, Sharpe={metrics['annualized_sharpe']:.2f}, Trades={result['backtest']['trades']}")
print(f"  最近10ep 平均 PV: ${result['avg_pv_last10']:,.0f}")
print(f"  训练时间: {train_time:.1f}s")

# 收敛判断
if result['final_entropy'] < 1.5:
    print(f"\n  ✅ Entropy 下降（{result['final_entropy']:.4f} < 1.5），策略开始分化")
else:
    print(f"\n  ⚠️ Entropy 仍高（{result['final_entropy']:.4f}），可能需要 reward scaling")

if result['policy_loss_trend']['ep100'] != result['policy_loss_trend']['ep10']:
    print(f"  ✅ Policy loss 有变化，梯度在传播")
else:
    print(f"  ⚠️ Policy loss 无变化，梯度可能仍为零")
