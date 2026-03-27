"""Run 3: lr_policy=0.003, episode_length=200, 500ep
缩短 episode 让策略更新更频繁，信号更集中
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
from src.visualization import plot_training_curves, plot_backtest_comparison, plot_agent_actions
from src.agents.dqn import DQNAgent

np.random.seed(42)
torch.manual_seed(42)

for d in [OUTPUT_DIR, CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

full_df, train_df, test_df = load_and_prepare_ton()

# 自定义超参数覆盖 config
agent = REINFORCEAgent(lr_policy=0.003, gamma=0.95, entropy_coeff=0.001)
env = CryptoTradingEnv(train_df, mode="continuous", reward_mode="simple")
print(f"Run 3 配置: lr_policy=0.003, gamma=0.95, entropy_coeff=0.001, ep_length=200, reward_scale={env.reward_scale}")

t0 = time.time()
history = train_reinforce(env, agent, n_episodes=500, episode_length=200, verbose=True)
train_time = time.time() - t0

ckpt_path = str(CHECKPOINT_DIR / "reinforce_run3_ep500.pt")
agent.save_checkpoint(ckpt_path)

# 回测
env_test = CryptoTradingEnv(test_df, mode="continuous", reward_mode="simple")
bt_r = backtest(env_test, agent, agent_type="reinforce")
metrics_r = compute_metrics(bt_r["portfolio_values"])

# 与 DQN 对比
dqn_ckpt = CHECKPOINT_DIR / "dqn_final.pt"
if dqn_ckpt.exists():
    agent_dqn = DQNAgent()
    agent_dqn.load_checkpoint(str(dqn_ckpt))
    env_dqn = CryptoTradingEnv(test_df, mode="continuous", reward_mode="simple")
    bt_dqn = backtest(env_dqn, agent_dqn, agent_type="dqn")
    metrics_dqn = compute_metrics(bt_dqn["portfolio_values"])
else:
    metrics_dqn = None

# 训练曲线
plot_training_curves(history, agent_name="REINFORCE-Run3")

# 保存结果
result = {
    "run": 3,
    "episodes": 500,
    "episode_length": 200,
    "changes": ["lr_policy=0.003", "episode_length 720→200", "reward_scale=100", "gamma=0.95"],
    "training_time": train_time,
    "entropy_trend": {
        "ep10": history["entropy_history"][9],
        "ep100": history["entropy_history"][99],
        "ep250": history["entropy_history"][249],
        "ep500": history["entropy_history"][-1],
    },
    "policy_loss_trend": {
        "ep10": history["policy_losses"][9],
        "ep100": history["policy_losses"][99],
        "ep250": history["policy_losses"][249],
        "ep500": history["policy_losses"][-1],
    },
    "backtest_reinforce": {
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
        "max_drawdown": metrics_dqn["max_drawdown"],
        "trades": int(count_trades(bt_dqn["actions"])),
    }

results_path = RESULTS_DIR / "run3_metrics.json"
with open(results_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\n{'='*60}")
print(f"Run 3 诊断汇报")
print(f"{'='*60}")
print(f"  Entropy: ep10={result['entropy_trend']['ep10']:.4f} → ep100={result['entropy_trend']['ep100']:.4f} → ep250={result['entropy_trend']['ep250']:.4f} → ep500={result['entropy_trend']['ep500']:.4f}")
print(f"  P_Loss: ep10={result['policy_loss_trend']['ep10']:.4f} → ep500={result['policy_loss_trend']['ep500']:.4f}")
print(f"  REINFORCE: Return={metrics_r['total_return']:+.2%}, Sharpe={metrics_r['annualized_sharpe']:.2f}, MaxDD={metrics_r['max_drawdown']:.2%}, Trades={result['backtest_reinforce']['trades']}")
if metrics_dqn:
    print(f"  DQN(旧):   Return={metrics_dqn['total_return']:+.2%}, Sharpe={metrics_dqn['annualized_sharpe']:.2f}")
print(f"  训练时间: {train_time:.1f}s")
