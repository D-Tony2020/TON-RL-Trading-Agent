"""
多交易者模块 — 训练 3 种交易者 + SHAP 特征重要性分析

3 种交易者类型均使用 REINFORCE agent，仅奖励函数不同：
- Arbitrageur: 均值回归、低频、风险调整
- Manipulator: 动量追逐、高频、量价利用
- Retail: 趋势追随、恐慌卖出、高风险厌恶
"""
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
from datetime import datetime

from src.config import (
    TRADER_TYPES, FEATURE_NAMES, SHAP_CONFIG,
    REINFORCE_PARAMS, FIGURES_DIR,
)
from src.environment import CryptoTradingEnv
from src.agents.reinforce import REINFORCEAgent, train_reinforce


def train_all_traders(train_df, n_episodes=None, episode_length=None, verbose=True):
    """
    训练 3 种交易者类型

    每种交易者使用相同的 REINFORCE 架构，仅 reward_mode 不同

    Args:
        train_df: 训练集 DataFrame
        n_episodes: 训练轮数（默认从 config 读取）
        episode_length: 每轮步数
        verbose: 是否打印进度

    Returns:
        dict: {trader_type: (agent, history)}
    """
    results = {}

    for trader_type in TRADER_TYPES:
        if verbose:
            print(f"\n{'='*60}")
            print(f"训练交易者: {trader_type.upper()}")
            print(f"{'='*60}")

        env = CryptoTradingEnv(
            train_df,
            mode="continuous",
            reward_mode=trader_type,
        )
        agent = REINFORCEAgent()

        history = train_reinforce(
            env, agent,
            n_episodes=n_episodes,
            episode_length=episode_length,
            verbose=verbose,
        )

        results[trader_type] = (agent, history)

        if verbose:
            final_pv = history["episode_portfolio_values"][-1]
            print(f"  {trader_type} 训练完成 | 最终 PV: ${final_pv:,.0f}")

    return results


def _policy_predict_fn(agent, states_np):
    """
    将 agent 的策略网络封装为 SHAP 可用的 numpy callable

    Args:
        agent: REINFORCEAgent
        states_np: np.ndarray, shape (n_samples, state_dim)

    Returns:
        np.ndarray: shape (n_samples, n_actions) 动作概率
    """
    state_t = torch.FloatTensor(states_np).to(agent.device)
    with torch.no_grad():
        probs = agent.policy_net(state_t)
    return probs.cpu().numpy()


def compute_shap_values(agent, env, n_background=None, n_explain=None):
    """
    计算 SHAP 值，解释策略网络的决策

    使用 KernelExplainer 对策略网络的输出概率做解释

    Args:
        agent: 训练好的 REINFORCEAgent
        env: CryptoTradingEnv（用于采样状态）
        n_background: 背景数据量
        n_explain: 待解释样本量

    Returns:
        shap_values: list of np.ndarray, 每个动作的 SHAP 值
        explain_data: np.ndarray, 解释用的状态数据
    """
    n_background = n_background or SHAP_CONFIG["n_background_samples"]
    n_explain = n_explain or SHAP_CONFIG["n_explain_samples"]

    # 采样状态数据
    total_samples = n_background + n_explain
    states = []
    state = env.reset(episode_length=min(total_samples + 100, len(env.df) - 1))
    for _ in range(total_samples):
        action = np.random.randint(0, 5)
        next_state, _, done, _ = env.step(action)
        states.append(next_state)
        if done:
            state = env.reset(episode_length=min(total_samples + 100, len(env.df) - 1))
        else:
            state = next_state

    states = np.array(states)
    background_data = states[:n_background]
    explain_data = states[n_background:n_background + n_explain]

    # 构建 SHAP 解释器
    predict_fn = lambda x: _policy_predict_fn(agent, x)
    explainer = shap.KernelExplainer(predict_fn, background_data)

    # 计算 SHAP 值
    shap_values = explainer.shap_values(explain_data)

    return shap_values, explain_data


def explain_top_features(shap_values, top_k=3):
    """
    从 SHAP 值中提取每个动作的 top-k 重要特征

    Args:
        shap_values: compute_shap_values() 返回的 SHAP 值
        top_k: 取前 k 个特征

    Returns:
        dict: {action_idx: [(feature_name, mean_abs_shap), ...]}
    """
    action_names = ["BUY", "HOLD", "SELL", "SHORT", "COVER"]
    results = {}

    for action_idx, action_name in enumerate(action_names):
        if action_idx < len(shap_values):
            sv = shap_values[action_idx]  # (n_explain, n_features)
            mean_abs = np.mean(np.abs(sv), axis=0)
            top_indices = np.argsort(mean_abs)[::-1][:top_k]
            results[action_name] = [
                (FEATURE_NAMES[i], float(mean_abs[i]))
                for i in top_indices
            ]

    return results


def print_top_features(trader_type, top_features):
    """打印 top 特征"""
    print(f"\n  {trader_type.upper()} — Top 特征:")
    for action, features in top_features.items():
        feature_str = ", ".join([f"{name}({val:.4f})" for name, val in features])
        print(f"    {action}: {feature_str}")


def plot_shap_comparison(all_shap_results, all_explain_data):
    """
    绘制 3 种交易者的 SHAP 对比图

    每种交易者一行，显示所有动作的平均绝对 SHAP 值 bar chart

    Args:
        all_shap_results: {trader_type: shap_values}
        all_explain_data: {trader_type: explain_data}

    Returns:
        保存的文件路径
    """
    n_traders = len(all_shap_results)
    fig, axes = plt.subplots(n_traders, 1, figsize=(12, 4 * n_traders))
    if n_traders == 1:
        axes = [axes]

    action_names = ["BUY", "HOLD", "SELL", "SHORT", "COVER"]

    for idx, (trader_type, shap_values) in enumerate(all_shap_results.items()):
        ax = axes[idx]

        # 对所有动作的 SHAP 值取绝对值平均
        # shap_values 可能是 list of (n_explain, n_features) 或其他格式
        all_abs_shap = np.zeros(len(FEATURE_NAMES))
        if isinstance(shap_values, list):
            for action_sv in shap_values:
                sv = np.array(action_sv)
                if sv.ndim == 2 and sv.shape[1] == len(FEATURE_NAMES):
                    all_abs_shap += np.mean(np.abs(sv), axis=0)
                elif sv.ndim == 2 and sv.shape[0] == len(FEATURE_NAMES):
                    # 转置情况：(n_features, n_explain)
                    all_abs_shap += np.mean(np.abs(sv), axis=1)
            all_abs_shap /= max(len(shap_values), 1)
        else:
            sv = np.array(shap_values)
            if sv.ndim == 3:
                # (n_outputs, n_explain, n_features) 或 (n_explain, n_features, n_outputs)
                if sv.shape[-1] == len(FEATURE_NAMES):
                    all_abs_shap = np.mean(np.abs(sv), axis=(0, 1)) if sv.ndim == 3 else np.mean(np.abs(sv), axis=0)
                elif sv.shape[1] == len(FEATURE_NAMES):
                    all_abs_shap = np.mean(np.abs(sv), axis=(0, 2))
                else:
                    all_abs_shap = np.mean(np.abs(sv.reshape(-1, len(FEATURE_NAMES))), axis=0)
            elif sv.ndim == 2:
                all_abs_shap = np.mean(np.abs(sv), axis=0)[:len(FEATURE_NAMES)]

        # 按重要性排序
        sorted_indices = np.argsort(all_abs_shap)[::-1]
        sorted_names = [FEATURE_NAMES[i] for i in sorted_indices]
        sorted_values = all_abs_shap[sorted_indices]

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_names)))
        ax.barh(range(len(sorted_names)), sorted_values, color=colors)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{trader_type.capitalize()} — Feature Importance (averaged across actions)")

    fig.suptitle("SHAP Feature Importance by Trader Type", fontsize=14, y=1.02)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = FIGURES_DIR / f"shap_comparison_{ts}.png"
    fig.savefig(filepath, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return filepath


def run_trader_analysis(train_df, n_episodes=None, episode_length=None, verbose=True):
    """
    一站式运行：训练 3 种交易者 + SHAP 分析 + 图表

    Args:
        train_df: 训练集 DataFrame
        n_episodes: 训练轮数
        episode_length: 每轮步数
        verbose: 是否打印进度

    Returns:
        dict: {
            "trader_results": {type: (agent, history)},
            "shap_results": {type: shap_values},
            "top_features": {type: top_features_dict},
        }
    """
    # 1. 训练所有交易者
    trader_results = train_all_traders(
        train_df, n_episodes=n_episodes,
        episode_length=episode_length, verbose=verbose,
    )

    # 2. SHAP 分析
    all_shap = {}
    all_explain = {}
    all_top_features = {}

    for trader_type, (agent, history) in trader_results.items():
        if verbose:
            print(f"\n--- SHAP 分析: {trader_type} ---")

        env = CryptoTradingEnv(
            train_df,
            mode="continuous",
            reward_mode=trader_type,
        )

        shap_values, explain_data = compute_shap_values(agent, env)
        top_features = explain_top_features(shap_values)

        all_shap[trader_type] = shap_values
        all_explain[trader_type] = explain_data
        all_top_features[trader_type] = top_features

        if verbose:
            print_top_features(trader_type, top_features)

    # 3. 绘制对比图
    if verbose:
        print("\n--- 生成 SHAP 对比图 ---")
    fig_path = plot_shap_comparison(all_shap, all_explain)
    if verbose:
        print(f"  [OK] SHAP 图已保存: {fig_path}")

    return {
        "trader_results": trader_results,
        "shap_results": all_shap,
        "top_features": all_top_features,
    }
