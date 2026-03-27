# MEMORY.md — ORIE5570 TON RL Trading Agent 项目记忆

> **Purpose**: 跨设备/跨 session 的完整项目上下文，使新 Claude 实例能零信息损失地继续开发。

---

## 1. 项目身份

- **课程**: ORIE 5570 Reinforcement Learning for Good (Cornell Tech, Prof. Irene Aldridge)
- **资产**: TON (Toncoin) 加密货币
- **项目路径**: `D:/Desktop/RL/project 1/`
- **远程仓库**: `https://github.com/D-Tony2020/TON-RL-Trading-Agent.git`
- **规范文件**: `CLAUDE.md`（架构规则、测试纪律、算力管理）

---

## 2. Assignment 2（已完成）

### 2.1 核心成果

| 策略 | Total Return | Sharpe | Max Drawdown |
|------|-------------|--------|--------------|
| DQN (Dueling Double + PER) | +70.05% | 2.16 | 19.8% |
| Q-Learning (tabular) | +45.83% | 1.44 | 23.1% |
| Buy & Hold | -52.31% | -1.54 | 68.1% |
| RSI Rule | -35.42% | -0.92 | 54.3% |
| Random | -69.33% | -2.08 | 77.2% |

### 2.2 8 轮迭代历程

| Run | 变更 | DQN 结果 | Q-Learning 结果 |
|-----|------|----------|----------------|
| #1 | 初始 3-action, ε_decay=0.9995 | -11.33% | -47.30% |
| #2 | 添加 SHORT/COVER → 5-action | -19.85% | +61.07% (无种子) |
| #3 | Soft update + LR schedule | -36.46% | +61.07% |
| #4 | 回退硬更新 + 修 PER priority bug | -9.16% | -42.66% |
| #5 | seed=42 固定 | +0.69% | +45.83% |
| #6 | Sharpe reward (失败: 数量级 1000×) | -86.18% | +45.83% |
| #7 | sin/cos hour + 回退 simple reward | -62.79% | +45.83% |
| #8 | ε_decay=0.985 (快衰减+长纯利用期) | **+70.05%** | +45.83% |

### 2.3 关键修复与教训

- **ε 衰减校准**: decay = ε_min^(1/n_episodes), 0.985^200≈0.05 → ep200 后纯利用
- **PER 双 alpha bug**: `push()` 不应再 `** alpha`，`update_priorities()` 已做了
- **Polyak 软更新**: 高探索期 Q 值震荡传导到 target net → 爆炸，改回硬更新
- **DSR 失败**: Sharpe reward 与 simple reward 数量级差 1000×，梯度混乱
- **sin/cos 编码**: hour_of_day 用 sin/cos 周期编码，状态 7D→8D

### 2.4 已交付物

- `Report_EN.tex` / `Report_EN.md` — 英文报告（含市场响应分析 section）
- `submission.zip` — 精简打包

---

## 3. Assignment 3（进行中）

### 3.1 任务要求

1. 实现 **REINFORCE with Baseline**（两层 NN），与 DQN 对比
2. **Crypto Track**: 添加 3 种交易者（rational arbitrageur / manipulator / herd-following retail），计算 Shapley 值，解释 top 3 特征
3. 监管分析：绘制 **crisis frequency / stablecoin stability / market efficiency** 三张图
4. 撰写 2-3 页方法论与结果报告

### 3.2 实施计划（8 Phase）

详见 `.claude/plans/synthetic-foraging-candy.md`

### 3.3 进度

| Phase | 内容 | 状态 | 文件 |
|-------|------|------|------|
| 1 | config.py 扩展 | ✅ 已完成 | `src/config.py` |
| 2 | environment.py 新增 3 种 reward_mode | ✅ 已完成 | `src/environment.py` |
| 3 | REINFORCE Agent | ✅ 已完成 + bug 修复 | `src/agents/reinforce.py` |
| 4 | 测试验证 | ✅ 10/10 通过 | `tests/test_reinforce.py` |
| 5 | 多交易者 + SHAP | ✅ 已完成 | `src/traders.py` |
| 6 | 监管分析图表 | ✅ 已完成 | `src/regulatory.py` |
| 7 | main.py 集成 | ✅ 已完成 (+4 mode) | `main.py` |
| 8 | 收尾 | ✅ 已完成 | `requirements.txt`, `visualization.py` |

### 3.4 已完成的具体工作

**Phase 1 — config.py**:
- `REINFORCE_PARAMS`: state_dim=8, n_actions=5, lr_policy=3e-4, lr_value=1e-3, gamma=0.99, hidden_dim=128, entropy_coeff=0.01
- `FEATURE_NAMES`: 8 维特征名列表
- `SHAP_CONFIG`: n_background=100, n_explain=500
- `REWARD_CONFIG` 新增: arbitrageur(高成本+Sharpe+均值回归) / manipulator(低成本+动量+成交量) / retail(趋势+恐慌+高风险厌恶)
- `TRADER_TYPES = ["arbitrageur", "manipulator", "retail"]`

**Phase 2 — environment.py**:
- `_compute_reward()` 新增 3 个 elif 分支
- arbitrageur: DSR + mean_reversion_bonus (RSI<0.3做多/RSI>0.7做空)
- manipulator: momentum_bonus (4h动量×仓位) + volume_bonus (vol_ratio>阈值)
- retail: trend_bonus (24h趋势×仓位) + panic_bonus (高波动率离场) + 高回撤惩罚

**Phase 3 — reinforce.py**:
- `PolicyNetwork`: 8→128→ReLU→128→ReLU→5→Softmax
- `ValueNetwork`: 8→128→ReLU→128→ReLU→1
- `REINFORCEAgent`: select_action(Categorical采样) / select_greedy_action(argmax) / store_reward / finish_episode(G_t计算→advantage标准化→policy_loss+entropy→value_loss→分别反向传播) / save/load_checkpoint
- `train_reinforce()`: GPU 支持 / 进度+ETA 每 10 ep / checkpoint 每 20 ep

**Phase 4 — test_reinforce.py**:
- 10/10 单元测试通过（macOS Python 3.9.6 + torch 2.8.0）
- 修复了 `finish_episode()` 中 value_net 梯度 bug：value 原在 `torch.no_grad()` 下计算，改为存储 state 并在 `finish_episode()` 中重新计算

**Phase 5 — src/traders.py**:
- `train_all_traders(train_df)`: 用 REINFORCE 分别训练 3 种 trader
- `compute_shap_values(agent, env)`: `shap.KernelExplainer` 封装 policy_net
- `explain_top_features(shap_values)`: 每种 trader 的 top 3 特征
- `plot_shap_comparison()`: 3 行 bar chart 对比（按重要性排序）
- `run_trader_analysis()`: 一站式入口

**Phase 6 — src/regulatory.py**:
- `compute_crisis_frequency()`: 按月统计 decline regime 占比
- `compute_stablecoin_stability()`: `1 / (1 + rolling_std(TON-BTC corr))`
- `compute_market_efficiency()`: 滚动 lag-1 autocorrelation
- `plot_regulatory_dashboard()`: 三合一仪表盘（含双 Y 轴）

**Phase 7 — main.py**:
- 新增 mode: `reinforce` / `traders` / `regulatory` / `assignment3`
- `run_reinforce()`: 训练 + 回测 + 与 DQN 对比
- `run_traders()`: 委托 `run_trader_analysis()`
- `run_regulatory_analysis()`: 委托 `plot_regulatory_dashboard()`
- 不修改 `run_all_backtests()` 签名

**Phase 8 — 收尾**:
- `requirements.txt` 新增 `shap>=0.42`
- `visualization.py`: `plot_training_curves()` 重构为动态 panel，兼容 REINFORCE 的 policy_loss/value_loss/entropy
- MEMORY.md 更新进度

---

## 4. 当前 MDP 设计（实际状态）

- **动作空间**: 5 actions {BUY(0), HOLD(1), SELL(2), SHORT(3), COVER(4)}
- **持仓**: position ∈ {-1(空头), 0(空仓), 1(多头)}
- **状态**: Q-Learning 6D 离散 (6720 states), DQN/REINFORCE 8D 连续 (含 sin/cos hour)
- **奖励**: 6 种模式 — simple / sharpe / arbitrageur / manipulator / retail (+ 默认 fallback)
- **数据**: TON hourly OHLCV (17243 行, 2024-03 ~ 2026-02) + 8 辅助资产, 80/20 时间顺序切分
- **种子**: seed=42 全局固定

---

## 5. 文件结构

```
D:/Desktop/RL/project 1/
├── main.py                          # 入口 (mode: smoke_test/full_train/backtest/correlation/report/all)
├── requirements.txt                 # numpy, pandas, torch, matplotlib, seaborn, scipy, pytest, tqdm
├── CLAUDE.md                        # 项目规范（第4节待更新）
├── MEMORY.md                        # 本文件
├── src/
│   ├── config.py                    # 全局超参数（含 REINFORCE_PARAMS, 3种 trader REWARD）
│   ├── data_pipeline.py             # 数据加载/清洗/特征/regime/切分
│   ├── environment.py               # CryptoTradingEnv (6种 reward_mode)
│   ├── backtest.py                  # 回测引擎 + 5种基线策略
│   ├── correlation.py               # 跨资产相关性分析
│   ├── visualization.py             # 8+类图表（含 REINFORCE 训练曲线支持）
│   ├── traders.py                   # 多交易者训练 + SHAP 分析（新建）
│   ├── regulatory.py                # 监管分析仪表盘（新建）
│   └── agents/
│       ├── q_learning.py            # 表格 Q-Learning
│       ├── dqn.py                   # Dueling Double DQN + PER
│       └── reinforce.py             # REINFORCE with Baseline（新建）
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_environment.py
│   ├── test_agents.py
│   ├── test_backtest.py
│   └── test_reinforce.py            # REINFORCE 测试（新建，待验证）
├── data/
│   ├── TON/TON11419-USD_DataHr.csv  # 主资产
│   └── futures/                     # BTC, ETH, SOL, Gold, Silver, SP500, USBond, SPY
├── output/
│   ├── checkpoints/                 # .pt, .pkl
│   ├── figures/                     # 带时间戳 PNG
│   └── results/
├── Report_EN.tex                    # Assignment 2 英文报告
├── Report_EN.md                     # Assignment 2 英文报告 (MD)
├── Report.md                        # Assignment 2 中文报告
├── Strategy.md                      # 8轮实验日志
└── submission.zip                   # Assignment 2 提交包
```

---

## 6. 已知问题

- **Python 3.14 + torch**: 原机 (LEGION) Python 3.14 环境导致 torch 导入极慢(>180s)。macOS Python 3.9.6 + torch 2.8.0 测试通过。
- **无数据目录**: 仓库不包含 `data/` 目录（.gitignore），依赖数据的测试（smoke_train, reward_mode）需要先放入数据文件。
- **CLAUDE.md 缺失**: 仓库中无 CLAUDE.md 文件，需要创建。

---

## 7. 用户偏好

- 简体中文回复，技术术语英文
- 代码注释中文，变量名 snake_case 英文
- 报告风格：叙事驱动，展现决策历程，不要正式论文结构
- 不假设数据来源（不写 yfinance 之类），只描述数据特征
- commit message 英文
