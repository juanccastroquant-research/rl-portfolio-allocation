# DQN Portfolio Management

A reinforcement-learning approach to portfolio allocation. A Deep Q-Network (DQN) agent learns to allocate capital across a 10-asset European energy universe, trained on Sharpe-ratio and/or raw-return rewards, and benchmarked against Mean-Variance Optimization (MVO) and a naïve equal-weight buy-and-hold strategy.

## Results (Test Set: Nov 2019 – Nov 2021)

| Strategy | Annual Return | Total Return | Sharpe Ratio | Max Drawdown | Daily Ret Std |
|---|---|---|---|---|---|
| MVO | 15.79% | 31.76% | 0.363 | 42.77% | 2.54% |
| Naive (equal-weight, buy & hold) | 10.79% | 21.27% | 0.324 | 40.43% | 1.87% |
| **DQN-SHARPE** | **28.20%** | **59.55%** | **0.807** | **37.47%** | 2.11% |

The DQN agent trained with the Sharpe-ratio reward more than doubles the risk-adjusted return (Sharpe) of both benchmarks while also achieving the lowest maximum drawdown — see `results/portfolio_performance.png` and `results/drawdowns.png`.

Best validation Sharpe during training: **0.7255** (episode 320 of 800). See `results/training_history.png` for the full learning curve, epsilon decay, and gradient-norm diagnostics.

## Project Structure

```
.
├── config.py          # All hyperparameters, data splits, asset universe
├── data.py            # Download, preprocess, and split OHLCV data (Yahoo Finance)
├── environment.py      # Gym-style PortfolioEnv (state, action, reward)
├── agent.py            # DQN agent, Q-network, replay buffer (uniform + PER)
├── mvo.py              # Mean-Variance Optimization & naive benchmarks
├── train.py             # Training loop with validation-based checkpointing
├── evaluate.py          # Test-set evaluation, plots, results reporting
├── main.py              # End-to-end pipeline: train → validate → evaluate
├── checkpoints/          # Saved model weights (created at runtime)
└── results/              # Metrics, plots, and text reports (created at runtime)
```

## Methodology

### Asset Universe
10 European energy-sector equities: Shell (SHEL), BP, TotalEnergies (TTE), Enel (ENEL.MI), Iberdrola (IBE.MC), RWE (RWE.DE), E.ON (EOAN.DE), Equinor (EQNR), Ørsted (ORSTED.CO), and Repsol (REP.MC).

### Data Split
Because Ørsted only IPO'd in June 2016, the usable joint history across all 10 tickers starts there rather than the paper's original 2006 start date. The data is split chronologically into three non-overlapping windows:

| Split | Range | Purpose |
|---|---|---|
| Train | 2016-06-15 → 2018-11-06 | Agent training |
| Validation | 2018-11-07 → 2019-11-06 | Checkpoint selection & MVO rebalance-frequency selection |
| Test | 2019-11-07 → 2021-11-06 | Final, single-pass evaluation only |

The test set is never used for any decision (checkpointing, hyperparameters, rebalance frequency) prior to the final evaluation pass in `main.py` / `evaluate.py`.

### State & Action Space
- **State**: a `(10 stocks × 5 OHLCV features × 10-day window)` tensor, matching the paper's 500-input specification.
- **Action**: a discrete candidate set of long-only portfolio weight vectors (single-asset anchors, equal-weight anchors, and Dirichlet-sampled random portfolios), selected via ε-greedy / greedy argmax over Q-values.

### Reward
Two reward formulations are supported (`config.AGENT`):
- **DQN-SHARPE**: rolling 20-day Sharpe ratio of portfolio returns.
- **DQN-RETURN**: scaled daily portfolio return (scaled ×50 to match gradient magnitude with the Sharpe reward).

Both reward types are optionally combined with:
- A **transaction-cost penalty** (5 bps, turnover-based), applied identically to DQN, MVO, and Naive so all strategies compete on equal footing.
- A **concentration penalty** (Herfindahl-Hirschman Index-based), discouraging the agent from over-concentrating into a small number of assets.

### Agent
A standard (non-dueling) fully-connected Q-network (`[512, 256, 128]` hidden layers, ReLU, dropout), trained with Double-DQN targets, Adam + cosine-annealed learning rate, gradient clipping, and a uniform experience replay buffer (Prioritized Experience Replay is implemented but disabled by default, matching the paper).

### Model Selection
Every `VALIDATION_INTERVAL` episodes, a greedy pass is run over the validation split; the checkpoint with the best validation Sharpe ratio is kept. The MVO benchmark's rebalance frequency (daily / weekly / monthly) is likewise chosen on the validation split, via `mvo.select_best_rebalance_freq()`.

## Setup

```bash
pip install numpy pandas scipy torch matplotlib yfinance
```

## Usage

Run the full pipeline (download data → train → validate → evaluate → plot):

```bash
python main.py
```

Train and evaluate only one reward type:

```bash
python main.py --reward sharpe     # or --reward return
```

Skip training and evaluate existing checkpoints only:

```bash
python main.py --skip-train
```

Run training or evaluation independently:

```bash
python train.py         # trains whichever REWARD_TYPES are set in config.py
python evaluate.py       # evaluates saved checkpoints against MVO/Naive on the test set
python data.py           # downloads data and prints train/val/test shape diagnostics
```

## Configuration

All hyperparameters, data-split dates, and reward/environment options live in `config.py`, including:
- `AGENT`: `"sharpe"`, `"return"`, or `"both"` — which agent(s) to train/evaluate.
- `EPISODE_LENGTH`, `NUM_EPISODES`, `LEARNING_RATE`, `GAMMA`, etc. — standard DQN hyperparameters.
- `TRANSACTION_FEE`, `CONCENTRATION_PENALTY`, `REWARD_SHAPING`, `REWARD_CLIP` — optional environment/reward modifiers (see inline comments in `config.py` for the rationale behind each default).

## Output

After a full run, `results/` contains:
- `results_summary.txt` — full metrics for every strategy plus training diagnostics.
- `metrics.json` — machine-readable version of the same metrics.
- `portfolio_performance.png` — normalized portfolio value over the test period.
- `drawdowns.png` — drawdown curves for all strategies over the test period.
- `training_history.png` / `train_history_<reward_type>.json` — reward, return, epsilon, gradient-norm, and validation-Sharpe curves over training.

`checkpoints/` contains the best-validation-Sharpe model weights for each trained agent (`dqn_sharpe.pt`, `dqn_return.pt`).

## Notes & Caveats

- Results are reported on a single test window (2019–2021, which includes the COVID-19 crash) — a demanding but idiosyncratic stress test. Treat results as indicative rather than fully generalizable.
- Validation Sharpe is noisy across checkpoints; the best-performing checkpoint by validation Sharpe is not necessarily the one from the final training episode.
- The discrete action space is a fixed candidate set of portfolios (seeded for reproducibility), not a continuously learned weight vector — see `agent.py`'s `build_action_space()`.
