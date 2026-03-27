"""
Assignment 3 一体化执行脚本
一次性完成：DQN训练 → REINFORCE训练 → 对比回测 → 3种交易者SHAP → 监管分析
"""
import sys, os, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR, OUTPUT_DIR,
    DQN_PARAMS, REINFORCE_PARAMS,
)
from src.data_pipeline import load_and_prepare_ton
from src.environment import CryptoTradingEnv
from src.agents.dqn import DQNAgent, train_dqn
from src.agents.reinforce import REINFORCEAgent, train_reinforce
from src.backtest import (
    backtest, compute_metrics, count_trades, format_results_table,
    run_all_backtests, analyze_by_regime,
)
from src.visualization import (
    plot_training_curves, plot_backtest_comparison, plot_agent_actions,
)


def set_global_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)


def ensure_dirs():
    for d in [OUTPUT_DIR, CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def main():
    set_global_seed(42)
    ensure_dirs()

    device_str = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    print(f">> Assignment 3 一体化执行")
    print(f"   设备: {device_str}\n")

    # ================================================================
    # 1. 加载数据
    # ================================================================
    print("=" * 60)
    print("Phase 1: 加载数据")
    print("=" * 60)
    full_df, train_df, test_df = load_and_prepare_ton()
    print(f"  训练集: {len(train_df)} 行, 测试集: {len(test_df)} 行")
    print(f"  Regime: {dict(full_df['regime'].value_counts())}")

    # 收集所有结果
    all_results = {}

    # ================================================================
    # 2. 训练 DQN（用于对比）
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 2: 训练 DQN (Dueling Double + PER)")
    print("=" * 60)

    dqn_episodes = 500
    env_dqn = CryptoTradingEnv(train_df, mode="continuous", reward_mode="simple")
    agent_dqn = DQNAgent()
    t0 = time.time()
    history_dqn = train_dqn(env_dqn, agent_dqn, n_episodes=dqn_episodes, verbose=True)
    dqn_time = time.time() - t0

    agent_dqn.save_checkpoint(str(CHECKPOINT_DIR / "dqn_final.pt"))
    plot_training_curves(history_dqn, agent_name="DQN")
    print(f"\n  DQN 训练完成: {dqn_time:.1f}s")

    # DQN 回测
    env_test_dqn = CryptoTradingEnv(test_df, mode="continuous", reward_mode="simple")
    bt_dqn = backtest(env_test_dqn, agent_dqn, agent_type="dqn")
    metrics_dqn = compute_metrics(bt_dqn["portfolio_values"])
    all_results["DQN"] = (bt_dqn, metrics_dqn)
    print(f"  DQN 回测: Return={metrics_dqn['total_return']:+.2%}, Sharpe={metrics_dqn['annualized_sharpe']:.2f}, MaxDD={metrics_dqn['max_drawdown']:.2%}")

    # ================================================================
    # 3. 训练 REINFORCE with Baseline
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 3: 训练 REINFORCE with Baseline")
    print("=" * 60)

    reinforce_episodes = 500
    set_global_seed(42)  # 重置种子保证公平对比
    env_r = CryptoTradingEnv(train_df, mode="continuous", reward_mode="simple")
    agent_r = REINFORCEAgent()
    t0 = time.time()
    history_r = train_reinforce(env_r, agent_r, n_episodes=reinforce_episodes, verbose=True)
    reinforce_time = time.time() - t0

    agent_r.save_checkpoint(str(CHECKPOINT_DIR / "reinforce_final.pt"))
    plot_training_curves(history_r, agent_name="REINFORCE")
    print(f"\n  REINFORCE 训练完成: {reinforce_time:.1f}s")

    # REINFORCE 回测
    env_test_r = CryptoTradingEnv(test_df, mode="continuous", reward_mode="simple")
    bt_r = backtest(env_test_r, agent_r, agent_type="reinforce")
    metrics_r = compute_metrics(bt_r["portfolio_values"])
    all_results["REINFORCE"] = (bt_r, metrics_r)
    print(f"  REINFORCE 回测: Return={metrics_r['total_return']:+.2%}, Sharpe={metrics_r['annualized_sharpe']:.2f}, MaxDD={metrics_r['max_drawdown']:.2%}")

    # ================================================================
    # 4. PG vs DQN 对比
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 4: REINFORCE vs DQN 对比")
    print("=" * 60)

    comparison = {"REINFORCE": all_results["REINFORCE"], "DQN": all_results["DQN"]}
    print("\n" + format_results_table(comparison))
    plot_backtest_comparison(comparison)
    plot_agent_actions(test_df, bt_r, agent_name="REINFORCE", metrics=metrics_r)
    plot_agent_actions(test_df, bt_dqn, agent_name="DQN", metrics=metrics_dqn)

    # ================================================================
    # 5. 三种交易者 + SHAP
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 5: 三种交易者 + SHAP 分析")
    print("=" * 60)

    from src.traders import run_trader_analysis

    set_global_seed(42)
    trader_result = run_trader_analysis(
        train_df,
        n_episodes=300,  # 交易者用 300 episodes（够用且节省时间）
        verbose=True,
    )

    # 收集交易者回测结果
    trader_backtest = {}
    for ttype, (agent, history) in trader_result["trader_results"].items():
        env_bt = CryptoTradingEnv(test_df, mode="continuous", reward_mode=ttype)
        bt = backtest(env_bt, agent, agent_type="reinforce")
        m = compute_metrics(bt["portfolio_values"])
        trader_backtest[ttype] = (bt, m)
        print(f"  {ttype}: Return={m['total_return']:+.2%}, Sharpe={m['annualized_sharpe']:.2f}, Trades={count_trades(bt['actions'])}")

    # ================================================================
    # 6. 监管分析图表
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 6: 监管分析")
    print("=" * 60)

    from src.regulatory import plot_regulatory_dashboard
    reg_path = plot_regulatory_dashboard(full_df)
    print(f"  监管仪表盘已保存: {reg_path}")

    # ================================================================
    # 7. 保存汇总结果
    # ================================================================
    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)

    summary = {
        "dqn": {
            "total_return": metrics_dqn["total_return"],
            "sharpe": metrics_dqn["annualized_sharpe"],
            "max_drawdown": metrics_dqn["max_drawdown"],
            "trades": int(count_trades(bt_dqn["actions"])),
            "training_time": dqn_time,
            "episodes": dqn_episodes,
        },
        "reinforce": {
            "total_return": metrics_r["total_return"],
            "sharpe": metrics_r["annualized_sharpe"],
            "max_drawdown": metrics_r["max_drawdown"],
            "trades": int(count_trades(bt_r["actions"])),
            "training_time": reinforce_time,
            "episodes": reinforce_episodes,
        },
        "traders": {},
        "top_features": trader_result["top_features"],
    }

    for ttype, (bt, m) in trader_backtest.items():
        summary["traders"][ttype] = {
            "total_return": m["total_return"],
            "sharpe": m["annualized_sharpe"],
            "max_drawdown": m["max_drawdown"],
            "trades": int(count_trades(bt["actions"])),
        }

    # 保存 JSON
    results_path = RESULTS_DIR / "assignment3_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  结果已保存: {results_path}")

    # 打印最终对比表
    print("\n--- REINFORCE vs DQN ---")
    print(f"  {'':>12s} | {'Return':>10s} | {'Sharpe':>8s} | {'MaxDD':>8s} | {'Trades':>7s} | {'Time':>8s}")
    print(f"  {'-'*12} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*7} | {'-'*8}")
    for name, info in [("DQN", summary["dqn"]), ("REINFORCE", summary["reinforce"])]:
        print(f"  {name:>12s} | {info['total_return']:>+9.2%} | {info['sharpe']:>8.2f} | {info['max_drawdown']:>7.2%} | {info['trades']:>7d} | {info['training_time']:>7.1f}s")

    print("\n--- 交易者回测 ---")
    for ttype, info in summary["traders"].items():
        print(f"  {ttype:>12s} | Return={info['total_return']:+.2%} | Sharpe={info['sharpe']:.2f} | Trades={info['trades']}")

    print("\n--- SHAP Top 3 特征 ---")
    for ttype, features in summary["top_features"].items():
        print(f"\n  {ttype.upper()}:")
        for action, feats in features.items():
            feat_str = ", ".join([f"{n}({v:.4f})" for n, v in feats])
            print(f"    {action}: {feat_str}")

    print("\n" + "=" * 60)
    print("[OK] Assignment 3 全部完成！")
    print(f"  图表: {FIGURES_DIR}")
    print(f"  结果: {RESULTS_DIR}")
    print(f"  模型: {CHECKPOINT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
