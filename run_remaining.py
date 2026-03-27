"""
续跑脚本：从已有 checkpoint 加载，完成 SHAP 图表 + 监管分析 + 结果汇总
"""
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR, OUTPUT_DIR,
    TRADER_TYPES, FEATURE_NAMES,
)
from src.data_pipeline import load_and_prepare_ton
from src.environment import CryptoTradingEnv
from src.agents.dqn import DQNAgent
from src.agents.reinforce import REINFORCEAgent
from src.backtest import backtest, compute_metrics, count_trades, format_results_table
from src.traders import train_all_traders, compute_shap_values, explain_top_features, print_top_features, plot_shap_comparison


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    for d in [OUTPUT_DIR, CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 加载数据
    full_df, train_df, test_df = load_and_prepare_ton()
    print(f"数据加载完成: train={len(train_df)}, test={len(test_df)}")

    # 加载 DQN
    agent_dqn = DQNAgent()
    agent_dqn.load_checkpoint(str(CHECKPOINT_DIR / "dqn_final.pt"))
    env_dqn = CryptoTradingEnv(test_df, mode="continuous", reward_mode="simple")
    bt_dqn = backtest(env_dqn, agent_dqn, agent_type="dqn")
    metrics_dqn = compute_metrics(bt_dqn["portfolio_values"])

    # 加载 REINFORCE
    agent_r = REINFORCEAgent()
    agent_r.load_checkpoint(str(CHECKPOINT_DIR / "reinforce_final.pt"))
    env_r = CryptoTradingEnv(test_df, mode="continuous", reward_mode="simple")
    bt_r = backtest(env_r, agent_r, agent_type="reinforce")
    metrics_r = compute_metrics(bt_r["portfolio_values"])

    print("\n--- REINFORCE vs DQN ---")
    comparison = {"REINFORCE": (bt_r, metrics_r), "DQN": (bt_dqn, metrics_dqn)}
    print(format_results_table(comparison))

    # 训练 3 种交易者 + SHAP（重新训练，因为 checkpoint 被覆盖了）
    print("\n--- 重新训练交易者 + SHAP ---")
    np.random.seed(42)
    torch.manual_seed(42)

    trader_results = train_all_traders(train_df, n_episodes=300, verbose=True)

    all_shap = {}
    all_explain = {}
    all_top_features = {}
    trader_backtest = {}

    for ttype, (agent, history) in trader_results.items():
        print(f"\n--- SHAP 分析: {ttype} ---")
        env = CryptoTradingEnv(train_df, mode="continuous", reward_mode=ttype)
        shap_values, explain_data = compute_shap_values(agent, env)

        # 调试 SHAP 返回格式
        if isinstance(shap_values, list):
            print(f"  shap_values: list of {len(shap_values)}, first shape: {np.array(shap_values[0]).shape}")
        else:
            print(f"  shap_values: {type(shap_values)}, shape: {np.array(shap_values).shape}")

        top_features = explain_top_features(shap_values)
        all_shap[ttype] = shap_values
        all_explain[ttype] = explain_data
        all_top_features[ttype] = top_features
        print_top_features(ttype, top_features)

        # 回测
        env_bt = CryptoTradingEnv(test_df, mode="continuous", reward_mode=ttype)
        bt = backtest(env_bt, agent, agent_type="reinforce")
        m = compute_metrics(bt["portfolio_values"])
        trader_backtest[ttype] = (bt, m)
        print(f"  {ttype}: Return={m['total_return']:+.2%}, Sharpe={m['annualized_sharpe']:.2f}")

    # SHAP 图
    print("\n--- 生成 SHAP 对比图 ---")
    fig_path = plot_shap_comparison(all_shap, all_explain)
    print(f"  [OK] {fig_path}")

    # 监管分析
    print("\n--- 监管分析 ---")
    from src.regulatory import plot_regulatory_dashboard
    reg_path = plot_regulatory_dashboard(full_df)
    print(f"  [OK] {reg_path}")

    # 保存结果 JSON
    summary = {
        "dqn": {
            "total_return": float(metrics_dqn["total_return"]),
            "sharpe": float(metrics_dqn["annualized_sharpe"]),
            "max_drawdown": float(metrics_dqn["max_drawdown"]),
            "win_rate": float(metrics_dqn["win_rate"]),
            "trades": int(count_trades(bt_dqn["actions"])),
        },
        "reinforce": {
            "total_return": float(metrics_r["total_return"]),
            "sharpe": float(metrics_r["annualized_sharpe"]),
            "max_drawdown": float(metrics_r["max_drawdown"]),
            "win_rate": float(metrics_r["win_rate"]),
            "trades": int(count_trades(bt_r["actions"])),
        },
        "traders": {},
        "top_features": {},
    }

    for ttype, (bt, m) in trader_backtest.items():
        summary["traders"][ttype] = {
            "total_return": float(m["total_return"]),
            "sharpe": float(m["annualized_sharpe"]),
            "max_drawdown": float(m["max_drawdown"]),
            "trades": int(count_trades(bt["actions"])),
        }

    for ttype, features in all_top_features.items():
        summary["top_features"][ttype] = {}
        for action, feats in features.items():
            summary["top_features"][ttype][action] = [(n, float(v)) for n, v in feats]

    results_path = RESULTS_DIR / "assignment3_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n结果已保存: {results_path}")

    print("\n[OK] 全部完成!")


if __name__ == "__main__":
    main()
