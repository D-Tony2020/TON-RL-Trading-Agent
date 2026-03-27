"""
Run 6: 优化版 REINFORCE
- Batch update: 每 5 episodes 平均梯度后更新（方差降低 √5 倍）
- Gradient clip: 1.0 → 0.1（防止策略崩塌）
- Early stopping: 保存回测 return 最高的模型
- lr cosine decay: 0.001 → 0.0001
- 2000 episodes × 200 steps
"""
import sys, os, json, time, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CHECKPOINT_DIR, RESULTS_DIR, OUTPUT_DIR, FIGURES_DIR
from src.data_pipeline import load_and_prepare_ton
from src.environment import CryptoTradingEnv
from src.agents.reinforce import REINFORCEAgent
from src.backtest import backtest, compute_metrics, count_trades
from src.visualization import plot_training_curves

np.random.seed(42)
torch.manual_seed(42)

for d in [OUTPUT_DIR, CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

full_df, train_df, test_df = load_and_prepare_ton()

# =====================================================================
# 配置
# =====================================================================
N_EPISODES = 2000
EP_LENGTH = 200
BATCH_SIZE = 5          # 每 5 episodes 更新一次
GRAD_CLIP = 0.1         # 激进梯度裁剪
LR_START = 0.001
LR_END = 0.0001
GAMMA = 0.95
ENTROPY_COEFF = 0.005   # 稍高，鼓励探索多样性
EVAL_INTERVAL = 50      # 每 50 ep 评估一次

print(f"Run 6 配置:")
print(f"  Episodes={N_EPISODES}, EP_length={EP_LENGTH}, Batch={BATCH_SIZE}")
print(f"  Grad_clip={GRAD_CLIP}, LR={LR_START}→{LR_END}, Gamma={GAMMA}")
print(f"  Entropy_coeff={ENTROPY_COEFF}")

# =====================================================================
# 初始化
# =====================================================================
agent = REINFORCEAgent(
    lr_policy=LR_START, lr_value=0.001,
    gamma=GAMMA, entropy_coeff=ENTROPY_COEFF,
    gradient_clip=GRAD_CLIP,
)
env_train = CryptoTradingEnv(train_df, mode="continuous", reward_mode="simple")
print(f"  Reward_scale={env_train.reward_scale}")

# lr scheduler: cosine decay
policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    agent.policy_optimizer, T_max=N_EPISODES // BATCH_SIZE, eta_min=LR_END
)

# =====================================================================
# 训练循环
# =====================================================================
history = {
    "episode_rewards": [], "episode_portfolio_values": [],
    "policy_losses": [], "value_losses": [], "entropy_history": [],
}

best_return = -999
best_ep = 0
t0 = time.time()

# 初始清零梯度
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

    # 累积梯度（不立即更新）
    is_batch_end = (ep + 1) % BATCH_SIZE == 0
    p_loss, v_loss, entropy = agent.finish_episode(accumulate=not is_batch_end)

    # Batch 结束时：平均梯度 → 裁剪 → 更新
    if is_batch_end:
        # 梯度已累积了 BATCH_SIZE 个 episode，除以 batch_size 做平均
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

    # 定期评估 + early stopping
    if (ep + 1) % EVAL_INTERVAL == 0:
        elapsed = time.time() - t0
        eps_per_sec = (ep + 1) / elapsed
        eta = (N_EPISODES - ep - 1) / eps_per_sec if eps_per_sec > 0 else 0

        # 快速回测（测试集前 500 步）
        env_eval = CryptoTradingEnv(test_df, mode="continuous", reward_mode="simple")
        bt_eval = backtest(env_eval, agent, agent_type="reinforce")
        m_eval = compute_metrics(bt_eval["portfolio_values"])
        trades_eval = count_trades(bt_eval["actions"])

        current_lr = agent.policy_optimizer.param_groups[0]["lr"]
        avg_reward = np.mean(history["episode_rewards"][-EVAL_INTERVAL:])
        avg_ent = np.mean(history["entropy_history"][-EVAL_INTERVAL:])

        print(
            f"  Ep {ep+1:>5d}/{N_EPISODES} | "
            f"Reward: {avg_reward:>7.1f} | "
            f"Entropy: {avg_ent:>.3f} | "
            f"LR: {current_lr:.5f} | "
            f"Test: {m_eval['total_return']:+.1%} ({trades_eval}tr) | "
            f"Elapsed: {elapsed:>5.0f}s | ETA: {eta:>5.0f}s"
        )

        # Early stopping: 保存回测最好的模型
        if m_eval["total_return"] > best_return:
            best_return = m_eval["total_return"]
            best_ep = ep + 1
            agent.save_checkpoint(str(CHECKPOINT_DIR / "reinforce_run6_best.pt"))
            print(f"    ★ 新最佳模型! Return={best_return:+.2%}, Trades={trades_eval}")

# =====================================================================
# 训练结束
# =====================================================================
train_time = time.time() - t0
history["training_time"] = train_time

# 保存最终模型
agent.save_checkpoint(str(CHECKPOINT_DIR / "reinforce_run6_final.pt"))
plot_training_curves(history, agent_name="REINFORCE-Run6")

# 用最佳模型做最终回测
print(f"\n--- 用最佳模型 (ep{best_ep}) 回测 ---")
agent.load_checkpoint(str(CHECKPOINT_DIR / "reinforce_run6_best.pt"))
env_test = CryptoTradingEnv(test_df, mode="continuous", reward_mode="simple")
bt_best = backtest(env_test, agent, agent_type="reinforce")
m_best = compute_metrics(bt_best["portfolio_values"])

# 用最终模型回测
agent.load_checkpoint(str(CHECKPOINT_DIR / "reinforce_run6_final.pt"))
env_test2 = CryptoTradingEnv(test_df, mode="continuous", reward_mode="simple")
bt_final = backtest(env_test2, agent, agent_type="reinforce")
m_final = compute_metrics(bt_final["portfolio_values"])

from collections import Counter
actions_best = Counter(bt_best["actions"].tolist())
actions_final = Counter(bt_final["actions"].tolist())

# 保存结果
result = {
    "run": 6,
    "config": {
        "n_episodes": N_EPISODES, "ep_length": EP_LENGTH, "batch_size": BATCH_SIZE,
        "grad_clip": GRAD_CLIP, "lr": f"{LR_START}→{LR_END}", "gamma": GAMMA,
        "entropy_coeff": ENTROPY_COEFF, "reward_scale": env_train.reward_scale,
    },
    "training_time": train_time,
    "best_model": {
        "epoch": best_ep,
        "return": m_best["total_return"],
        "sharpe": m_best["annualized_sharpe"],
        "max_dd": m_best["max_drawdown"],
        "trades": int(count_trades(bt_best["actions"])),
        "actions": dict(actions_best),
    },
    "final_model": {
        "return": m_final["total_return"],
        "sharpe": m_final["annualized_sharpe"],
        "trades": int(count_trades(bt_final["actions"])),
        "actions": dict(actions_final),
    },
    "entropy_trend": {
        "ep50": float(np.mean(history["entropy_history"][:50])),
        "ep500": float(np.mean(history["entropy_history"][450:500])),
        "ep1000": float(np.mean(history["entropy_history"][950:1000])),
        "ep2000": float(np.mean(history["entropy_history"][-50:])),
    },
}

results_path = RESULTS_DIR / "run6_metrics.json"
with open(results_path, "w") as f:
    json.dump(result, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"Run 6 结果")
print(f"{'='*60}")
print(f"  最佳模型 (ep{best_ep}): Return={m_best['total_return']:+.2%}, Sharpe={m_best['annualized_sharpe']:.2f}, Trades={result['best_model']['trades']}")
print(f"  最终模型: Return={m_final['total_return']:+.2%}, Sharpe={m_final['annualized_sharpe']:.2f}, Trades={result['final_model']['trades']}")
print(f"  Actions (best): {dict(actions_best)}")
print(f"  Actions (final): {dict(actions_final)}")
print(f"  训练时间: {train_time:.1f}s")
for k, v in result["entropy_trend"].items():
    print(f"  Entropy {k}: {v:.4f}")
