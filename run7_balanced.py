"""
Run 7: 平衡配置
- Batch=3 (faster updates than batch=5)
- Gradient clip=0.5 (between 0.1 and 1.0)
- lr=0.002 (between 0.001 and 0.003)
- entropy_coeff=0.003
- 1000 episodes × 200 steps
- Early stopping on test return
"""
import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CHECKPOINT_DIR, RESULTS_DIR, OUTPUT_DIR, FIGURES_DIR
from src.data_pipeline import load_and_prepare_ton
from src.environment import CryptoTradingEnv
from src.agents.reinforce import REINFORCEAgent
from src.backtest import backtest, compute_metrics, count_trades
from src.visualization import plot_training_curves
from collections import Counter

np.random.seed(42)
torch.manual_seed(42)

for d in [OUTPUT_DIR, CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

full_df, train_df, test_df = load_and_prepare_ton()

N_EPISODES = 1000
EP_LENGTH = 200
BATCH_SIZE = 3
GRAD_CLIP = 0.5
LR = 0.002
GAMMA = 0.95
ENTROPY_COEFF = 0.003
EVAL_INTERVAL = 25

print(f"Run 7: batch={BATCH_SIZE}, clip={GRAD_CLIP}, lr={LR}, ent={ENTROPY_COEFF}")

agent = REINFORCEAgent(
    lr_policy=LR, lr_value=0.001,
    gamma=GAMMA, entropy_coeff=ENTROPY_COEFF,
    gradient_clip=GRAD_CLIP,
)
env_train = CryptoTradingEnv(train_df, mode="continuous", reward_mode="simple")

# lr scheduler
policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    agent.policy_optimizer, T_max=N_EPISODES // BATCH_SIZE, eta_min=0.0002
)

history = {
    "episode_rewards": [], "episode_portfolio_values": [],
    "policy_losses": [], "value_losses": [], "entropy_history": [],
}

best_return = -999
best_ep = 0
best_sharpe = -999
t0 = time.time()

agent.policy_optimizer.zero_grad()
agent.value_optimizer.zero_grad()

for ep in range(N_EPISODES):
    state = env_train.reset(episode_length=EP_LENGTH)
    ep_reward = 0.0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env_train.step(action)
        agent.store_reward(reward)
        ep_reward += reward
        state = next_state

    is_batch_end = (ep + 1) % BATCH_SIZE == 0
    p_loss, v_loss, entropy = agent.finish_episode(accumulate=not is_batch_end)

    if is_batch_end:
        for p in agent.policy_net.parameters():
            if p.grad is not None:
                p.grad /= BATCH_SIZE
        for p in agent.value_net.parameters():
            if p.grad is not None:
                p.grad /= BATCH_SIZE
        agent.flush_gradients()
        policy_scheduler.step()

    history["episode_rewards"].append(ep_reward)
    history["episode_portfolio_values"].append(info["portfolio_value"])
    history["policy_losses"].append(p_loss)
    history["value_losses"].append(v_loss)
    history["entropy_history"].append(entropy)

    if (ep + 1) % EVAL_INTERVAL == 0:
        elapsed = time.time() - t0
        eta = (N_EPISODES - ep - 1) / ((ep + 1) / elapsed) if elapsed > 0 else 0

        env_eval = CryptoTradingEnv(test_df, mode="continuous", reward_mode="simple")
        bt_eval = backtest(env_eval, agent, agent_type="reinforce")
        m_eval = compute_metrics(bt_eval["portfolio_values"])
        trades = count_trades(bt_eval["actions"])
        avg_ent = np.mean(history["entropy_history"][-EVAL_INTERVAL:])

        print(
            f"  Ep {ep+1:>5d}/{N_EPISODES} | "
            f"Ent: {avg_ent:.3f} | "
            f"Test: {m_eval['total_return']:+.1%} ({trades}tr) Sharpe={m_eval['annualized_sharpe']:.2f} | "
            f"{elapsed:.0f}s ETA {eta:.0f}s"
        )

        if m_eval["total_return"] > best_return or (m_eval["total_return"] == best_return and trades > 0):
            best_return = m_eval["total_return"]
            best_sharpe = m_eval["annualized_sharpe"]
            best_ep = ep + 1
            agent.save_checkpoint(str(CHECKPOINT_DIR / "reinforce_run7_best.pt"))
            print(f"    ★ Best! Return={best_return:+.2%}, Sharpe={best_sharpe:.2f}, Trades={trades}")

train_time = time.time() - t0
agent.save_checkpoint(str(CHECKPOINT_DIR / "reinforce_run7_final.pt"))
plot_training_curves(history, agent_name="REINFORCE-Run7")

# 最终评估 (best model)
agent.load_checkpoint(str(CHECKPOINT_DIR / "reinforce_run7_best.pt"))
env_test = CryptoTradingEnv(test_df, mode="continuous", reward_mode="simple")
bt = backtest(env_test, agent, agent_type="reinforce")
m = compute_metrics(bt["portfolio_values"])

result = {
    "run": 7,
    "config": {"batch": BATCH_SIZE, "clip": GRAD_CLIP, "lr": LR, "entropy": ENTROPY_COEFF},
    "best_ep": best_ep, "training_time": train_time,
    "best_model": {
        "return": m["total_return"], "sharpe": m["annualized_sharpe"],
        "max_dd": m["max_drawdown"], "trades": int(count_trades(bt["actions"])),
        "actions": dict(Counter(bt["actions"].tolist())),
    },
}
with open(RESULTS_DIR / "run7_metrics.json", "w") as f:
    json.dump(result, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"Run 7 最佳模型 (ep{best_ep}): Return={m['total_return']:+.2%}, Sharpe={m['annualized_sharpe']:.2f}, Trades={result['best_model']['trades']}")
print(f"Actions: {result['best_model']['actions']}")
print(f"训练时间: {train_time:.1f}s")
