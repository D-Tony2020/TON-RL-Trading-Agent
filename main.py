"""
TON RL Trading Agent — 主入口
用法：
    python main.py --mode smoke_test    # 快速验证（20 episodes）
    python main.py --mode full_train    # 完整训练
    python main.py --mode backtest      # 回测（需先训练）
    python main.py --mode report        # 生成全部报告素材
    python main.py --mode all           # 全流程
"""
import argparse
import sys
import os
import time
import numpy as np
import torch
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR, OUTPUT_DIR,
    QLEARNING_PARAMS, DQN_PARAMS,
)
from src.data_pipeline import load_and_prepare_ton, load_auxiliary_data
from src.environment import CryptoTradingEnv
from src.agents.q_learning import QLearningAgent, train_qlearning
from src.agents.dqn import DQNAgent, train_dqn
from src.backtest import (
    backtest, run_all_backtests, format_results_table,
    analyze_by_regime, compute_metrics, count_trades,
)
from src.correlation import (
    compute_correlation_matrix, rolling_correlation,
    regime_correlation, intraday_analysis, session_analysis,
)
from src.visualization import (
    plot_price_with_regimes, plot_training_curves,
    plot_backtest_comparison, plot_regime_action_distribution,
    plot_correlation_heatmap, plot_rolling_correlation,
    plot_intraday_effects, plot_regime_correlation_comparison,
)


def ensure_dirs():
    """确保输出目录存在"""
    for d in [OUTPUT_DIR, CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_data():
    """加载并准备数据"""
    print("=" * 60)
    print("Phase 1: 加载数据")
    print("=" * 60)
    full_df, train_df, test_df = load_and_prepare_ton()
    print(f"  完整数据: {len(full_df)} 行")
    print(f"  训练集:   {len(train_df)} 行 ({train_df.index[0].strftime('%Y-%m-%d')} ~ {train_df.index[-1].strftime('%Y-%m-%d')})")
    print(f"  测试集:   {len(test_df)} 行 ({test_df.index[0].strftime('%Y-%m-%d')} ~ {test_df.index[-1].strftime('%Y-%m-%d')})")
    print(f"  Regime 分布: {dict(full_df['regime'].value_counts())}")
    return full_df, train_df, test_df


def run_smoke_test(train_df, episodes=20):
    """快速冒烟测试"""
    print("\n" + "=" * 60)
    print("冒烟测试 (Smoke Test)")
    print("=" * 60)

    # Q-Learning
    print("\n--- Q-Learning (20 episodes × 100 steps) ---")
    env_q = CryptoTradingEnv(train_df, mode="discrete", reward_mode="simple")
    agent_q = QLearningAgent()
    history_q = train_qlearning(env_q, agent_q, n_episodes=episodes, episode_length=100,
                                checkpoint_interval=0, verbose=True)
    print(f"  Q 表大小: {agent_q.q_table_size}")
    print(f"  最终 reward: {history_q['episode_rewards'][-1]:.4f}")

    # DQN
    print(f"\n--- DQN (20 episodes × 100 steps) ---")
    env_dqn = CryptoTradingEnv(train_df, mode="continuous", reward_mode="simple")
    agent_dqn = DQNAgent(min_buffer_size=200)
    history_dqn = train_dqn(env_dqn, agent_dqn, n_episodes=episodes, episode_length=100,
                            checkpoint_interval=0, verbose=True)
    print(f"  Buffer 大小: {len(agent_dqn.buffer)}")
    print(f"  最终 reward: {history_dqn['episode_rewards'][-1]:.4f}")

    # 检查数值稳定性
    all_rewards = history_q["episode_rewards"] + history_dqn["episode_rewards"]
    assert all(not np.isnan(r) for r in all_rewards), "存在 NaN reward！"
    print("\n[OK] 冒烟测试通过：无崩溃，无 NaN")
    return agent_q, agent_dqn, history_q, history_dqn


def run_full_train(train_df):
    """完整训练"""
    print("\n" + "=" * 60)
    print("Phase 2: 完整训练")
    print("=" * 60)

    # Q-Learning
    q_params = QLEARNING_PARAMS
    print(f"\n--- Q-Learning ({q_params['n_episodes']} episodes × {q_params['episode_length']} steps) ---")
    env_q = CryptoTradingEnv(train_df, mode="discrete", reward_mode="simple")
    agent_q = QLearningAgent()
    history_q = train_qlearning(env_q, agent_q, verbose=True)

    # 保存最终模型
    agent_q.save_checkpoint(CHECKPOINT_DIR / "qlearning_final.pkl")
    print(f"  [OK] Q-Learning 模型已保存")

    # DQN
    dqn_params = DQN_PARAMS
    print(f"\n--- Dueling Double DQN + PER ({dqn_params['n_episodes']} episodes × {dqn_params['episode_length']} steps) ---")
    env_dqn = CryptoTradingEnv(train_df, mode="continuous", reward_mode="simple")
    agent_dqn = DQNAgent()
    history_dqn = train_dqn(env_dqn, agent_dqn, verbose=True)

    # 保存最终模型
    agent_dqn.save_checkpoint(CHECKPOINT_DIR / "dqn_final.pt")
    print(f"  [OK] DQN 模型已保存")

    # 绘制训练曲线
    plot_training_curves(history_q, agent_name="Q-Learning")
    plot_training_curves(history_dqn, agent_name="DQN")
    print(f"  [OK] 训练曲线已保存到 {FIGURES_DIR}")

    return agent_q, agent_dqn, history_q, history_dqn


def run_backtest(test_df, agent_q, agent_dqn):
    """运行回测"""
    print("\n" + "=" * 60)
    print("Phase 3: 回测")
    print("=" * 60)

    results = run_all_backtests(test_df, agent_q, agent_dqn)

    # 打印结果表
    print("\n" + format_results_table(results))

    # Regime 分析
    print("\n--- Regime 行为分析 ---")
    for name in ["Q-Learning", "DQN"]:
        bt_result, _ = results[name]
        regime_analysis = analyze_by_regime(bt_result)
        print(f"\n{name}:")
        for regime, info in regime_analysis.items():
            dist = info["action_distribution"]
            print(f"  {regime:>8s}: BUY={dist['BUY']:.1%} HOLD={dist['HOLD']:.1%} SELL={dist['SELL']:.1%} "
                  f"SHORT={dist['SHORT']:.1%} COVER={dist['COVER']:.1%} (n={info['count']})")

    # 绘制对比图
    plot_backtest_comparison(results)

    # 绘制 Regime 动作分布
    for name in ["Q-Learning", "DQN"]:
        bt_result, _ = results[name]
        regime_analysis = analyze_by_regime(bt_result)
        plot_regime_action_distribution(regime_analysis, agent_name=name)

    print(f"\n  [OK] 回测图表已保存到 {FIGURES_DIR}")
    return results


def run_correlation_analysis(full_df):
    """运行相关性分析"""
    print("\n" + "=" * 60)
    print("Phase 4: 跨资产相关性分析")
    print("=" * 60)

    # 加载辅助数据
    aux_dfs = load_auxiliary_data()
    print(f"  已加载 {len(aux_dfs)} 个辅助资产: {list(aux_dfs.keys())}")

    # 整体相关性矩阵
    print("\n--- 整体相关性矩阵 ---")
    corr_matrix = compute_correlation_matrix(full_df, aux_dfs)
    print(corr_matrix.to_string())
    plot_correlation_heatmap(corr_matrix)

    # 滚动相关性
    print("\n--- 计算30天滚动相关性... ---")
    rolling_corr = rolling_correlation(full_df, aux_dfs)
    plot_rolling_correlation(rolling_corr)
    print(f"  滚动相关性 shape: {rolling_corr.shape}")

    # 分 Regime 相关性
    print("\n--- 分 Regime 相关性 ---")
    regime_corrs = regime_correlation(full_df, aux_dfs)
    for regime, corr in regime_corrs.items():
        if "TON" in corr.index:
            ton_row = corr.loc["TON"].drop("TON")
            print(f"  {regime:>8s}: {dict(ton_row.round(3))}")
    plot_regime_correlation_comparison(regime_corrs)

    # 日内效应
    print("\n--- 日内效应分析 ---")
    hourly_stats = intraday_analysis(full_df)
    plot_intraday_effects(hourly_stats)

    # 时段分析
    session_stats = session_analysis(full_df)
    print(session_stats.to_string())

    print(f"\n  [OK] 相关性分析图表已保存到 {FIGURES_DIR}")


def run_report(full_df):
    """生成价格走势图（独立于训练）"""
    print("\n--- 生成 TON 价格走势图 ---")
    plot_price_with_regimes(full_df)
    print(f"  [OK] 价格图已保存到 {FIGURES_DIR}")


def main():
    parser = argparse.ArgumentParser(description="TON RL Trading Agent")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["smoke_test", "full_train", "backtest", "correlation", "report", "all"],
                        help="运行模式")
    parser.add_argument("--episodes", type=int, default=None,
                        help="覆盖训练轮数")
    parser.add_argument("--resume", action="store_true",
                        help="从 checkpoint 恢复训练")
    args = parser.parse_args()

    ensure_dirs()

    print(">> TON RL Trading Agent")
    print(f"  设备: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"  模式: {args.mode}")
    print()

    # 加载数据
    full_df, train_df, test_df = load_data()

    if args.mode == "smoke_test":
        episodes = args.episodes or 20
        run_smoke_test(train_df, episodes=episodes)

    elif args.mode == "full_train":
        agent_q, agent_dqn, _, _ = run_full_train(train_df)
        run_backtest(test_df, agent_q, agent_dqn)

    elif args.mode == "backtest":
        # 从 checkpoint 加载
        agent_q = QLearningAgent()
        agent_q.load_checkpoint(CHECKPOINT_DIR / "qlearning_final.pkl")
        agent_dqn = DQNAgent()
        agent_dqn.load_checkpoint(CHECKPOINT_DIR / "dqn_final.pt")
        run_backtest(test_df, agent_q, agent_dqn)

    elif args.mode == "correlation":
        run_correlation_analysis(full_df)

    elif args.mode == "report":
        run_report(full_df)

    elif args.mode == "all":
        # 全流程
        run_report(full_df)
        agent_q, agent_dqn, _, _ = run_full_train(train_df)
        run_backtest(test_df, agent_q, agent_dqn)
        run_correlation_analysis(full_df)

    print("\n" + "=" * 60)
    print("[OK] 全部完成！")
    print(f"  图表输出: {FIGURES_DIR}")
    print(f"  模型输出: {CHECKPOINT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
