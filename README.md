# DQN Portfolio Management

A reinforcement-learning framework for dynamic portfolio allocation, based on
*"Reinforcement Learning in Portfolio Management with Sharpe Ratio Rewarding
Based Framework."* Two Deep Q-Network (DQN) agents — one rewarded on rolling
Sharpe ratio, one on raw daily return — are trained on historical OHLCV data
and benchmarked against Mean-Variance Optimization (MVO) and a naïve
equal-weight buy-and-hold strategy.

**Asset universe:** 10 European energy stocks — Shell (`SHEL`), BP (`BP`),
TotalEnergies (`TTE`), Enel (`ENEL.MI`), Iberdrola (`IBE.MC`), RWE (`RWE.DE`),
E.ON (`EOAN.DE`), Equinor (`EQNR`), Ørsted (`ORSTED.CO`), Repsol (`REP.MC`).

**Data window:** train Nov 2006 – Nov 2019, test Nov 2019 – Nov 2021.

---

## How to run

```bash
python main.py                 # train + evaluate both DQN agents
python main.py --skip-train    # evaluate only, using saved checkpoints
python main.py --reward sharpe # train/evaluate a single agent
```

Or run the individual scripts (`train.py`, `evaluate.py`) directly if you
only need one stage.

---

## File overview

### `config.py`
Central hyperparameter file — every other module reads its settings from
here. Nothing else needs to be edited to change tickers, dates, network
size, reward type, or the various optional features described below.

Key groups:
- **Data**: `TICKERS`, `TRAIN_START`/`TRAIN_END`/`TEST_START`/`TEST_END`, `FEATURES` (OHLCV), `WINDOW_SIZE` (10-day lookback).
- **Environment**: `INITIAL_CAPITAL`, `TRANSACTION_FEE`, `RISK_FREE_RATE_ANN`, `ALLOW_SHORT`.
- **DQN architecture**: `HIDDEN_LAYERS`, `ACTIVATION`, `DROPOUT_RATE`, `USE_DUELING`.
- **DQN training**: `LEARNING_RATE`, `GAMMA`, `BATCH_SIZE`, `REPLAY_CAPACITY`, `NUM_EPISODES`, `EPISODE_LENGTH`.
- **Exploration**: `EPS_START`/`EPS_END`/`EPS_DECAY` (ε-greedy schedule).
- **Agent selection**: `AGENT = "sharpe" | "return" | "both"` — the single switch that controls which agent(s) train/evaluate.
- **Reward shaping toggles** (all off by default, matching the paper): `REWARD_SHAPING`, `CONCENTRATION_PENALTY`, `REWARD_CLIP`, `USE_DIFFERENTIAL_SHARPE`.
- **MVO benchmark**: `MVO_REBALANCE_FREQ`.

### `data.py`
Downloads and preprocesses market data.
- `download_data()` — pulls OHLCV data via `yfinance` for all tickers over the full date range.
- `compute_pct_change()` — converts prices to daily percentage returns.
- `build_windows()` — reshapes returns into the paper's state tensor: `(n_stocks × n_features × window)` = 10 × 5 × 10 = 500 inputs per state. Clips price features to `±STATE_CLIP` and normalizes volume by a global training-set standard deviation.
- `load_split_data()` — orchestrates the above into aligned train/test state tensors and close-price arrays, ready for the environment.

### `environment.py`
`PortfolioEnv` — the Gym-style environment the DQN agents interact with.
- **State**: a `(10, 5, 10)` window of recent OHLCV percentage changes.
- **Action**: a portfolio weight vector over the 10 assets (normalized to sum to 1).
- **Reward**: either
  - **DQN-Return** — daily portfolio return scaled by `RETURN_REWARD_SCALE` (50×) so its gradient magnitude matches DQN-Sharpe's (raw daily returns are too small on their own and cause gradient collapse), or
  - **DQN-Sharpe** — a rolling-window Sharpe ratio (`SHARPE_WINDOW` days), clipped to `[-3, 3]` and scaled by `SHARPE_REWARD_SCALE`.
- **Transaction costs**: each step computes turnover (total absolute change in weights vs. the previous step) and subtracts `TRANSACTION_FEE × turnover` from the return before it affects both the portfolio value and the reward. This is disabled by setting `TRANSACTION_FEE = 0.0` in config.
- **Concentration penalty** (optional, off by default): penalizes reward based on how far the portfolio's Herfindahl-Hirschman Index sits above the equal-weight baseline — discourages the agent from piling into one or two assets. Applies to both reward types when enabled.
- Other optional, paper-disabled features: reward shaping vs. equal-weight benchmark, reward clipping, and a differential Sharpe ratio formulation (Moody & Saffell, 1998).

### `agent.py`
`DQNAgent` — the learning algorithm.
- **`QNetwork`**: a fully-connected network (`HIDDEN_LAYERS`, ReLU, dropout) with an optional dueling head (`USE_DUELING`, off by default — not in the paper).
- **Replay buffer**: standard uniform replay (paper-compliant) by default, with an optional prioritized variant (`SumTree` + `PrioritisedReplayBuffer`, enabled via `USE_PER`).
- **Action space**: since portfolio weights are continuous, `build_action_space()` discretizes them into a fixed set of candidate weight vectors — one-hot single-asset bets, the equal-weight vector (repeated to boost sampling frequency), and either a full grid (small universes) or randomly sampled Dirichlet weight vectors (`NUM_RANDOM_ACTIONS`, used here since there are 10 assets).
- **Learning**: Double-DQN target computation, Smooth L1 loss, gradient clipping, a cosine-annealed learning-rate schedule, and periodic target-network syncing (`TARGET_UPDATE`).
- **Checkpointing**: `save()`/`load()` persist the online/target networks, optimizer, and scheduler state. `load()` uses `weights_only=True` to avoid PyTorch's arbitrary-code-execution risk when loading checkpoints.

### `mvo.py`
Benchmark strategies to compare the DQN agents against.
- `compute_mvo_weights()` — solves for the maximum-Sharpe long-only portfolio (SLSQP) given a trailing return window.
- `simulate_mvo()` — rebalances at `MVO_REBALANCE_FREQ` (daily/weekly/monthly) using a trailing 252-day window, paying the same turnover-based `TRANSACTION_FEE` cost as the DQN environment on each rebalance.
- `simulate_naive()` — static equal-weight, buy-and-hold. No ongoing transaction cost applies, since weights are never rebalanced after the initial allocation.
- `_performance_metrics()` — shared metrics computation (annualized return, Sharpe ratio, max drawdown, daily return stats) used by both benchmarks and the DQN evaluation.

### `train.py`
Training loop, run once per agent (`sharpe` and/or `return`).
- Pre-fills the replay buffer with random transitions before training starts.
- Each episode samples a random contiguous `EPISODE_LENGTH`-day (504, ≈2 years) window from the training set, so the agent experiences multi-year dynamics (drawdowns, recoveries) during training rather than only at test time.
- Updates the network every `UPDATE_EVERY` steps, decays ε after each episode, and logs reward, portfolio return, gradient norm, and loss.
- Runs a final **sanity check**: a greedy (ε=0) pass over the last contiguous `EPISODE_LENGTH` days of the training set, to catch agents that memorized random sub-windows instead of learning something that holds up on a contiguous stretch.
- Saves the trained checkpoint and a JSON training history (used later by `evaluate.py`'s plots).

### `evaluate.py`
Runs trained agents and benchmarks on the held-out test set and produces all reporting artifacts.
- `evaluate_dqn()` — loads a checkpoint, runs the agent greedily (ε=0) over the test set, and computes performance metrics.
- `print_comparison_table()` — prints a side-by-side metrics table.
- `plot_portfolio_values()` — normalized portfolio value over the test period for every strategy → `portfolio_performance.png`.
- `plot_drawdowns()` — drawdown-from-peak over time for every strategy → `drawdowns.png`.
- `plot_training_history()` — per-agent and combined plots of cumulative reward, episode return, and gradient norm across training → `training_history_<reward>.png` / `training_history.png`.
- `save_results_txt()` — writes the full text report (per-strategy metrics + comparison table + training history summary) → `results_summary.txt`.

### `main.py`
Top-level entry point that wires everything together: train (unless
`--skip-train`), load the test set, run both benchmarks, evaluate the
selected DQN agent(s), and produce all comparison tables/plots.

---

## Known limitations / things to check before trusting results

- **Currency mismatch**: the 10 tickers span five currencies (GBp, EUR, USD, NOK, DKK). `yfinance` returns each in its local currency, and the pipeline currently treats price columns as directly comparable (e.g. in volume normalization and MVO's covariance estimate) without FX conversion. This will distort return and covariance estimates unless addressed.
- **Overfitting risk**: DQN-Sharpe's test-set performance (see `results_summary.txt`) is substantially worse than the naïve equal-weight benchmark, despite training reward climbing steadily. Training episode returns are extremely high-variance (mean 711%, std 554%, min/max 14%/2714%), which is more consistent with the agent exploiting specific historical sub-windows than learning a generalizable policy. The transaction-cost and concentration-penalty fixes in `config.py` / `environment.py` / `mvo.py` are a first attempt at addressing this — re-run training and compare the new sanity-check and test-set numbers before drawing conclusions.
- **Random episode sampling**: training episodes are drawn from random 504-day windows, which can over- or under-represent certain historical regimes (e.g. the 2008 crash and recovery). Consider systematic/rolling coverage of the training period if instability persists.
