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
Note: `SHEL`, `BP`, `TTE`, `EQNR` are USD-denominated US listings; `ENEL.MI`,
`IBE.MC`, `RWE.DE`, `EOAN.DE`, `REP.MC` are EUR-denominated; `ORSTED.CO` is
DKK-denominated. Three currencies are mixed into one return/covariance matrix
with no FX conversion applied anywhere in the pipeline — see **Known
limitations** below.

**Data window:**
| Split | Window | Purpose |
|---|---|---|
| Train | Nov 2006 – Nov 2017 | DQN training / replay buffer |
| Validation | Nov 2017 – Nov 2019 | Model checkpointing, MVO rebalance-frequency selection — **never used to report final numbers** |
| Test | Nov 2019 – Nov 2021 | Final, single-pass evaluation only |

The train/test dates match the paper's original split; the final two years of
the paper's training window are now carved out as a validation split (see
**Train / validation / test discipline** below for why).

---

## How to run

```
python main.py                 # train + evaluate both DQN agents
python main.py --skip-train    # evaluate only, using saved checkpoints
python main.py --reward sharpe # train/evaluate a single agent
```

Or run the individual scripts (`train.py`, `evaluate.py`) directly if you
only need one stage.

---

## Train / validation / test discipline

An earlier version of this codebase tuned several hyperparameters
(`TRANSACTION_FEE`, `EPISODE_LENGTH`, `CONCENTRATION_PENALTY`, the MVO
rebalance frequency) by inspecting results on the test set and then changing
config values in response — a data-snooping loop that quietly invalidates
the "held-out" test set the moment its results inform a design decision.

This version closes that loop:

- **`config.py`** now defines three date ranges: `TRAIN_START`/`TRAIN_END`,
  `VAL_START`/`VAL_END`, and `TEST_START`/`TEST_END`. `TEST_START`/`TEST_END`
  are unchanged from the paper.
- **`train.py`** runs a greedy evaluation pass over the validation split
  every `VALIDATION_INTERVAL` episodes and overwrites the saved checkpoint
  only when validation Sharpe improves. The end-of-training "sanity check"
  pass over the training set is a separate, purely diagnostic step and does
  not affect which checkpoint is kept.
- **`mvo.py`** exposes `select_best_rebalance_freq()`, which chooses the MVO
  rebalance frequency (daily/weekly/monthly) by comparing Sharpe ratios on
  the validation split, rather than hard-coding a single frequency with no
  documented rationale.
- **`main.py`** and **`evaluate.py`** run this validation-based selection
  first, then evaluate every strategy on the test split exactly once, for
  the numbers that get reported.

If you re-tune any hyperparameter, do it against `results/train_history_*.json`'s
`val_sharpe` entries (or a fresh validation-only run), not against
`results_summary.txt`'s test-set numbers.

---

## File overview

### `config.py`

Central hyperparameter file — every other module reads its settings from
here. Nothing else needs to be edited to change tickers, dates, network
size, reward type, or the various optional features described below.

Key groups:

- **Data**: `TICKERS`, `TRAIN_START`/`TRAIN_END`, `VAL_START`/`VAL_END`,
  `TEST_START`/`TEST_END`, `FEATURES` (OHLCV), `WINDOW_SIZE` (10-day
  lookback).
- **Environment**: `INITIAL_CAPITAL`, `TRANSACTION_FEE`, `RISK_FREE_RATE_ANN`,
  `ALLOW_SHORT`.
- **DQN architecture**: `HIDDEN_LAYERS`, `ACTIVATION`, `DROPOUT_RATE`,
  `USE_DUELING`.
- **DQN training**: `LEARNING_RATE`, `GAMMA`, `BATCH_SIZE`,
  `REPLAY_CAPACITY`, `NUM_EPISODES`, `EPISODE_LENGTH`,
  `VALIDATION_INTERVAL` (episodes between validation passes for
  checkpointing).
- **Exploration**: `EPS_START`/`EPS_END`/`EPS_DECAY` (ε-greedy schedule).
- **Agent selection**: `AGENT = "sharpe" | "return" | "both"` — the single
  switch that controls which agent(s) train/evaluate.
- **Reward shaping toggles** (all off by default, matching the paper):
  `REWARD_SHAPING`, `CONCENTRATION_PENALTY`, `REWARD_CLIP`,
  `USE_DIFFERENTIAL_SHARPE`.
- **MVO benchmark**: `MVO_REBALANCE_FREQ` — a fallback default only; the
  pipeline normally overrides this via `select_best_rebalance_freq()` on
  the validation split.

### `data.py`

Downloads and preprocesses market data.

- `download_data()` — pulls OHLCV data via `yfinance` for all tickers over
  the full date range. Explicitly reindexes the resulting columns to match
  the order of `TICKERS` — yfinance's own column ordering for multi-ticker
  downloads is not guaranteed to match the list you passed in (it is often
  alphabetical instead), and every downstream reshape assumes column *N*
  corresponds to `TICKERS[N]`.
- `compute_pct_change()` — converts prices to daily percentage returns.
- `get_aligned_pct()` — slices a pct-change `DataFrame` to `[start, end]`
  and drops the first `WINDOW_SIZE` rows, so the result lines up
  index-for-index with the corresponding `*_states`/`*_closes` arrays.
  Centralises an offset that was previously duplicated separately in
  `main.py` and `evaluate.py`.
- `build_windows()` — reshapes returns into the paper's state tensor:
  `(n_stocks × n_features × window)` = 10 × 5 × 10 = 500 inputs per state.
  Clips price features to `±STATE_CLIP` and normalizes volume by a global
  *training-set* standard deviation, reused (not recomputed) for the
  validation and test splits.
- `load_split_data()` — orchestrates the above into aligned train / **validation** /
  test state tensors and close-price arrays: returns
  `(train_states, val_states, test_states, train_closes, val_closes, test_closes)`.

### `environment.py`

`PortfolioEnv` — the Gym-style environment the DQN agents interact with.
Split-agnostic: it's instantiated identically on whichever states/closes
arrays are passed in (train, validation, or test).

- **State**: a `(10, 5, 10)` window of recent OHLCV percentage changes.
- **Action**: a portfolio weight vector over the 10 assets (normalized to
  sum to 1).
- **Reward**: either
  * **DQN-Return** — daily portfolio return scaled by `RETURN_REWARD_SCALE`
    (50×) so its gradient magnitude matches DQN-Sharpe's (raw daily returns
    are too small on their own and cause gradient collapse), or
  * **DQN-Sharpe** — a rolling-window Sharpe ratio (`SHARPE_WINDOW` days),
    clipped to `[-3, 3]` and scaled by `SHARPE_REWARD_SCALE`.
- **Transaction costs**: each step computes turnover (total absolute change
  in weights vs. the previous step) and subtracts
  `TRANSACTION_FEE × turnover` from the return before it affects both the
  portfolio value and the reward. Disabled by setting `TRANSACTION_FEE = 0.0`
  in config.
- **Concentration penalty** (optional, off by default): penalizes reward
  based on how far the portfolio's Herfindahl-Hirschman Index sits above
  the equal-weight baseline — discourages the agent from piling into one or
  two assets. Applies to both reward types when enabled.
- Other optional, paper-disabled features: reward shaping vs. equal-weight
  benchmark, reward clipping, and a differential Sharpe ratio formulation
  (Moody & Saffell, 1998).

### `agent.py`

`DQNAgent` — the learning algorithm.

- **`QNetwork`**: a fully-connected network (`HIDDEN_LAYERS`, ReLU, dropout)
  with an optional dueling head (`USE_DUELING`, off by default — not in the
  paper).
- **Replay buffer**: standard uniform replay (paper-compliant) by default,
  with an optional prioritized variant (`SumTree` + `PrioritisedReplayBuffer`,
  enabled via `USE_PER`).
- **Action space**: since portfolio weights are continuous,
  `build_action_space()` discretizes them into a fixed set of candidate
  weight vectors — one-hot single-asset bets, the equal-weight vector
  (repeated to boost sampling frequency), and either a full grid (small
  universes) or randomly sampled Dirichlet weight vectors
  (`NUM_RANDOM_ACTIONS`, used here since there are 10 assets). **Note:**
  this set is fixed once at construction (65 candidate portfolios total for
  10 assets) and reproduced identically across training and evaluation via
  a fixed seed — the agent selects among these fixed candidates rather than
  learning an arbitrary continuous weight vector.
- **Learning**: Double-DQN target computation, Smooth L1 loss, gradient
  clipping, a cosine-annealed learning-rate schedule, and periodic
  target-network syncing (`TARGET_UPDATE`).
- **Checkpointing**: `save()`/`load()` persist the online/target networks,
  optimizer, and scheduler state. `load()` uses `weights_only=True` to
  avoid PyTorch's arbitrary-code-execution risk when loading checkpoints.

### `mvo.py`

Benchmark strategies to compare the DQN agents against.

- `compute_mvo_weights()` — solves for the maximum-Sharpe long-only
  portfolio (SLSQP) given a trailing return window.
- `simulate_mvo()` — rebalances at a given frequency (daily/weekly/monthly)
  using a trailing 252-day window, paying the same turnover-based
  `TRANSACTION_FEE` cost as the DQN environment on each rebalance.
- `simulate_naive()` — static equal-weight, buy-and-hold. No ongoing
  transaction cost applies, since weights are never rebalanced after the
  initial allocation.
- `select_best_rebalance_freq()` — chooses the rebalance frequency by
  comparing Sharpe ratios **on the validation split only**, so this
  decision doesn't leak information from the test set.
- `_performance_metrics()` — shared metrics computation (annualized return,
  Sharpe ratio, max drawdown, daily return stats) used by both benchmarks
  and the DQN evaluation.

### `train.py`

Training loop, run once per agent (`sharpe` and/or `return`).

- Pre-fills the replay buffer with random transitions before training
  starts.
- Each episode samples a random contiguous `EPISODE_LENGTH`-day (504,
  ≈2 years) window from the training set, so the agent experiences
  multi-year dynamics (drawdowns, recoveries) during training rather than
  only at evaluation time.
- Updates the network every `UPDATE_EVERY` steps, decays ε after each
  episode, and logs reward, portfolio return, gradient norm, and loss.
- **Validation-based checkpointing**: every `VALIDATION_INTERVAL` episodes,
  runs a greedy (ε=0) pass over the validation split and overwrites the
  saved checkpoint only if validation Sharpe improves. This is the
  mechanism that replaces "look at test results, then retune" as a
  model-selection process.
- Runs a final, separate **sanity check**: a greedy pass over the last
  contiguous `EPISODE_LENGTH` days of the *training* set, to catch agents
  that memorized random sub-windows instead of learning something that
  holds up on a contiguous stretch. This is diagnostic only and does not
  affect checkpoint selection.
- Saves a JSON training history (including per-validation-pass Sharpe,
  used later by `evaluate.py`'s plots).

### `evaluate.py`

Runs trained agents and benchmarks on the held-out test set and produces
all reporting artifacts.

- `evaluate_dqn()` — loads a checkpoint, runs the agent greedily (ε=0) over
  whichever states/closes are passed in (the test split, in normal use),
  and computes performance metrics.
- `print_comparison_table()` — prints a side-by-side metrics table.
- `plot_portfolio_values()` — normalized portfolio value over the test
  period for every strategy → `portfolio_performance.png`.
- `plot_drawdowns()` — drawdown-from-peak over time for every strategy →
  `drawdowns.png`.
- `plot_training_history()` — per-agent and combined plots of cumulative
  reward, episode return, and gradient norm across training, with
  validation-Sharpe checkpoints overlaid where available →
  `training_history_<reward>.png` / `training_history.png`.
- `save_results_txt()` — writes the full text report (per-strategy metrics
  + comparison table + training history summary, including best validation
  Sharpe) → `results_summary.txt`.

### `main.py`

Top-level entry point that wires everything together: train (unless
`--skip-train`), select the MVO rebalance frequency on the validation set,
load the test set, run both benchmarks, evaluate the selected DQN agent(s)
on test, and produce all comparison tables/plots.

---

## Known limitations / things to check before trusting results

- **Currency mismatch**: the 10 tickers span three currencies (USD for
  `SHEL`/`BP`/`TTE`/`EQNR`, EUR for `ENEL.MI`/`IBE.MC`/`RWE.DE`/`EOAN.DE`/`REP.MC`,
  DKK for `ORSTED.CO`). `yfinance` returns each in its local/listing
  currency, and the pipeline currently treats price columns as directly
  comparable (e.g. in volume normalization and MVO's covariance estimate)
  without FX conversion. This will distort return and covariance estimates
  unless addressed.
- **Fixed, small discrete action space**: with 10 assets, the DQN agents
  choose among only 65 fixed candidate portfolios (10 one-hot + 5 duplicated
  equal-weight + 50 Dirichlet(0.5) draws sampled once at startup), not a
  continuous weight vector. `NUM_RANDOM_ACTIONS` in `config.py` can be
  raised for finer resolution, at the cost of a larger action space to
  learn over.
- **Validation-set size**: at ~2 years, the validation split is short
  relative to a full market cycle. Checkpoint selection and rebalance-
  frequency selection based on it may still be somewhat noisy; a rolling/
  expanding-window validation scheme would be a natural robustness
  extension.
- **Overfitting risk**: earlier versions of this pipeline showed DQN-Sharpe
  test-set performance substantially worse than the naïve equal-weight
  benchmark, with training episode returns that were extremely
  high-variance (mean 711%, std 554%, min/max 14%/2714%) — more consistent
  with the agent exploiting specific historical sub-windows than learning
  a generalizable policy. The transaction-cost, concentration-penalty, and
  now validation-based checkpointing fixes are intended to address this;
  re-run training and check `results/train_history_*.json`'s `val_sharpe`
  trend and the sanity-check output before drawing conclusions.
- **Random episode sampling**: training episodes are drawn from random
  504-day windows, which can over- or under-represent certain historical
  regimes (e.g. the 2008 crash and recovery). Consider systematic/rolling
  coverage of the training period if instability persists.
