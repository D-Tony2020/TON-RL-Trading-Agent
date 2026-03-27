"""Run 4: lr=0.001 稳定训练 + 更长 episode(500步) + 1000ep
目标: 避免策略崩塌，保持 entropy 稳步下降
"""
import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CHECKPOINT_DIR, RESULTS_DIR, OUTPUT_DIR, FIGURES_DIR
from src.data_pipeline import load_and_prepare_ton
from src.environment import CryptoTradingEnv
from src.agents.reinforce import REINFORCEAgent, train_reinforce
from src.backtest import backtest, compute_metrics, count_trades
from src.visualization import plot_training_curves
from src.agents.dqn import DQNAgent

np.random.seed(42)
torch.manual_seed(42)

for d in [OUTPUT_DIR, CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

full_df, train_df, test_df = load_and_prepare_ton()

# 稳定配置: lr 回到 0.001, 更长 episode(500步), 更多轮数
agent = REINFORCEAgent(lr_policy=0.001, gamma=0.95, entropy_coeff=0.001)
env = CryptoTradingEnv(train_df, mode="continuous", reward_mode="simple")
print(f"Run 4: lr=0.001, gamma=0.95, entropy=0.001, ep_length=500, n_ep=1000, reward_scale={env.reward_scale}")

t0 = time.time()
history = train_reinforce(env, agent, n_episodes=1000, episode_length=500,
                          checkpoint_interval=100, verbose=True)
train_time = time.time() - t0

ckpt_path = str(CHECKPOINT_DIR / "reinforce_run4_ep1000.pt")
agent.save_checkpoint(ckpt_path)

# 训练曲线
plot_training_curves(history, agent_name="REINFORCE-Run4")

# 回测
env_test = CryptoTradingEnv(test_df, mode="continuous", reward_mode="simple")
bt_r = backtest(env_test, agent, agent_type="reinforce")
metrics_r = compute_metrics(bt_r["portfolio_values"])

# DQN 对比
dqn_ckpt = CHECKPOINT_DIR / "dqn_final.pt"
metrics_dqn = None
if dqn_ckpt.exists():
    agent_dqn = DQNAgent()
    agent_dqn.load_checkpoint(str(dqn_ckpt))
    env_dqn = CryptoTradingEnv(test_df, mode="continuous", reward_mode="simple")
    bt_dqn = backtest(env_dqn, agent_dqn, agent_type="dqn")
    metrics_dqn = compute_metrics(bt_dqn["portfolio_values"])

# 保存结果
result = {
    "run": 4,
    "episodes": 1000,
    "episode_length": 500,
    "config": "lr=0.001, gamma=0.95, entropy=0.001, reward_scale=100",
    "training_time": train_time,
    "entropy_trend": {
        "ep10": history["entropy_history"][9],
        "ep100": history["entropy_history"][99],
        "ep250": history["entropy_history"][249],
        "ep500": history["entropy_history"][499],
        "ep750": history["entropy_history"][749],
        "ep1000": history["entropy_history"][-1],
    },
    "backtest": {
        "total_return": metrics_r["total_return"],
        "sharpe": metrics_r["annualized_sharpe"],
        "max_drawdown": metrics_r["max_drawdown"],
        "win_rate": metrics_r["win_rate"],
        "trades": int(count_trades(bt_r["actions"])),
    },
}
if metrics_dqn:
    result["backtest_dqn"] = {
        "total_return": metrics_dqn["total_return"],
        "sharpe": metrics_dqn["annualized_sharpe"],
        "trades": int(count_trades(bt_dqn["actions"])),
    }

results_path = RESULTS_DIR / "run4_metrics.json"
with open(results_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\n{'='*60}")
print(f"Run 4 诊断汇报")
print(f"{'='*60}")
for k, v in result["entropy_trend"].items():
    print(f"  Entropy {k}: {v:.4f}")
print(f"  REINFORCE: Return={metrics_r['total_return']:+.2%}, Sharpe={metrics_r['annualized_sharpe']:.2f}, Trades={result['backtest']['trades']}")
if metrics_dqn:
    print(f"  DQN(旧):   Return={metrics_dqn['total_return']:+.2%}, Sharpe={metrics_dqn['annualized_sharpe']:.2f}")
print(f"  训练时间: {train_time:.1f}s")
