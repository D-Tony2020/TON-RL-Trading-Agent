# CLAUDE.md — TON RL Trading Agent 项目规范

## 1. 项目身份

ORIE5570 (Cornell Tech, Prof. Irene Aldridge) bi-weekly assignment。用 Q-Learning (TD) + Dueling Double DQN (PER) + REINFORCE with Baseline 建模理性投资者在 TON 加密货币三种市场场景（稳态/急跌/急涨）下的最优行为。Assignment 3 扩展：多交易者类型建模 + SHAP 特征解释 + 监管分析。

## 2. 语言与风格

- 回复使用简体中文，技术术语保留英文
- 代码注释使用中文
- 变量/函数命名 snake_case 英文
- commit message 英文

## 3. 架构规则

- 模块职责严格分离：`data_pipeline` / `environment` / `agents` / `backtest` / `correlation` / `visualization` / `traders` / `regulatory`
- **所有超参数集中在 `src/config.py`**，代码中绝不硬编码超参数值
- Environment 统一 Gym 风格接口：`reset()` → state, `step(action)` → (next_state, reward, done, info)
- 每个模块对外暴露清晰的函数/类接口，模块间通过接口通信

## 4. 已定稿的设计决策（不可随意修改）

- 状态空间：Q-Learning 6维离散(6720 states)，DQN/REINFORCE 8维连续（含 sin/cos hour 编码）
- 动作空间：{BUY(0), HOLD(1), SELL(2), SHORT(3), COVER(4)}，全仓，支持做空
- 持仓状态：position ∈ {-1(空头), 0(空仓), 1(多头)}
- 奖励函数：可配置结构（`reward_mode` = `'simple'` / `'sharpe'` / `'arbitrageur'` / `'manipulator'` / `'retail'`），系数可调
- 算法：Q-Learning + Dueling Double DQN + PER + REINFORCE with Baseline
- 非法动作处理：无效动作（如空仓SELL、满仓BUY、空仓COVER 等）→ 映射为 HOLD
- 全局种子：seed=42 保证可复现性

## 5. 数据处理红线

- `data/` 目录下所有原始 CSV **只读，绝不修改**
- 所有 CSV 文件统一 `pd.read_csv(path, skiprows=2)`，列名重命名为 `['datetime','close','high','low','open','volume']`
- 闪崩异常值（2024-09-03 07:00，$5.24→$0.31）**必须用前后均值插值处理**
- 数据切分：**时间顺序 80/20，严禁随机切分**（防止未来数据泄露）

## 6. 测试纪律

- 每个模块写完后，必须跑对应的 `pytest tests/test_<module>.py -v`
- 全流程改动后，跑 `pytest tests/ -v` 全量验证
- 测试不通过不进入下一个 Phase

## 7. 算力管理与长任务规范（关键！）

### 7.1 分批执行，不要一次性全跑
- 网格搜索、交叉验证等吃算力任务**必须分批执行**
- 例：超参数搜索拆成多轮小批量，每轮完成后汇报结果，再决定下一轮
- 禁止一次性提交全部组合然后等待数小时

### 7.2 中间进度实时监控
- 所有训练循环必须实现**进度条 + 中间指标打印**
- 最低要求：每 N 个 episode 打印一次 (episode_reward, loss, epsilon, elapsed_time)
- 长任务（>1分钟）必须显示预计剩余时间 (ETA)
- 使用 tqdm 或自定义 progress reporter，不允许"黑箱等待"

### 7.3 断点续训（Checkpoint）
- 所有训练过程必须支持**定期保存 checkpoint**
- Q-Learning：每 50 episodes 保存 Q-table（pickle）
- DQN：每 20 episodes 保存 model state_dict + optimizer state + replay buffer 状态 + epsilon + episode 计数
- REINFORCE：每 20 episodes 保存 policy_net + value_net + 两个 optimizer 的 state_dict
- Checkpoint 保存到 `output/checkpoints/`，文件名含 episode 编号
- 必须实现 `--resume` 参数，从最近的 checkpoint 恢复训练
- 目的：中途崩溃（内存/断电）不会丢失全部训练进度

### 7.4 GPU 使用
- 本地有 RTX 30 系 GPU，DQN/REINFORCE 训练时**必须检测并使用 GPU（如可用）**
- 使用 `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- 在训练开始时打印当前使用的设备信息
- Q-Learning 为纯 numpy，不需要 GPU

## 8. 输出约定

- 所有图表、模型、结果输出到 `output/` 目录
- 子目录结构：`output/checkpoints/`、`output/figures/`、`output/results/`
- 不自动覆盖已有输出，文件名带时间戳（格式：`YYYYMMDD_HHMMSS`）
