# ============================================================
#  mvo.py  –  Mean-Variance Optimization & Naïve benchmarks
# ============================================================
import numpy as np
from scipy.optimize import minimize
from config import RISK_FREE_RATE_ANN, MVO_REBALANCE_FREQ, TICKERS, TRANSACTION_FEE

N_STOCKS  = len(TICKERS)
DAILY_RFR = RISK_FREE_RATE_ANN / 252


# ──────────────────────────────────────────────────────────────────────
#  Shared print-and-write helper
# ──────────────────────────────────────────────────────────────────────
def _log(line: str = "", file=None):
    print(line)
    if file:
        file.write(line + "\n")


# ──────────────────────────────────────────────────────────────────────
#  Sharpe Ratio utilities
# ──────────────────────────────────────────────────────────────────────
def portfolio_sharpe(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """Annualised Sharpe ratio (negative for minimisation)."""
    ret = float(weights @ mu) * 252
    std = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(252)
    return -(ret - RISK_FREE_RATE_ANN) / (std + 1e-8)


def compute_mvo_weights(returns_window: np.ndarray) -> np.ndarray:
    """
    Solve MVO for maximum Sharpe ratio given a return history window.

    Parameters
    ----------
    returns_window : (T, n_stocks)  daily pct returns

    Returns
    -------
    weights : (n_stocks,)  long-only, sum-to-1
    """
    mu  = returns_window.mean(axis=0)
    cov = np.cov(returns_window.T) + np.eye(N_STOCKS) * 1e-8

    w0          = np.ones(N_STOCKS) / N_STOCKS
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds      = [(0, 1)] * N_STOCKS

    result = minimize(
        portfolio_sharpe, w0,
        args=(mu, cov),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9}
    )
    if result.success:
        w = np.clip(result.x, 0, 1)
        return w / w.sum()
    return w0


# ──────────────────────────────────────────────────────────────────────
#  Simulate strategies over the test set
# ──────────────────────────────────────────────────────────────────────
def _rebalance_steps(n_days: int, freq: str) -> set:
    if freq == "daily":
        return set(range(n_days))
    elif freq == "weekly":
        return set(range(0, n_days, 5))
    elif freq == "monthly":
        return set(range(0, n_days, 21))
    else:
        return {0}


def simulate_mvo(closes: np.ndarray,
                 pct_returns: np.ndarray,
                 initial_val: float = 1_000_000,
                 freq: str = MVO_REBALANCE_FREQ,
                 report_file=None) -> dict:
    """
    Simulate MVO strategy.

    Parameters
    ----------
    closes      : (T, n_stocks)
    pct_returns : (T, n_features*n_stocks) OR (T, n_stocks)
    report_file : open file handle to write results to (optional)
    """
    # Extract Close-only returns if full feature matrix was passed.
    # Robust to the expanded feature set (OHLCV + engineered features).
    n_cols = pct_returns.shape[1]
    if n_cols == N_STOCKS:
        pct_close = pct_returns
    elif n_cols % N_STOCKS == 0:
        # Assume Close is the 4th column (index 3) in the base FEATURES list
        n_feat_per_stock = n_cols // N_STOCKS
        close_idx        = 3   # "Close" position in FEATURES
        close_cols       = [i * n_feat_per_stock + close_idx
                            for i in range(N_STOCKS)]
        pct_close        = pct_returns[:, close_cols]
    else:
        raise ValueError(f"Unexpected pct_returns shape: {pct_returns.shape}")

    T                = closes.shape[0] - 1
    rebal_days       = _rebalance_steps(T, freq)
    portfolio_values = [initial_val]
    weights          = np.ones(N_STOCKS) / N_STOCKS
    daily_rets       = []

    for t in range(T):
        txn_cost = 0.0
        if t in rebal_days and t > 0:
            hist = pct_close[max(0, t - 252): t]
            if len(hist) >= 20:
                new_weights = compute_mvo_weights(hist)
                # FIX: turnover-based transaction cost, same model as
                # PortfolioEnv.step() in environment.py, so DQN and MVO
                # pay comparable friction and remain fairly comparable.
                turnover = float(np.sum(np.abs(new_weights - weights)))
                txn_cost = TRANSACTION_FEE * turnover
                weights  = new_weights

        stock_rets = (closes[t + 1] - closes[t]) / (closes[t] + 1e-8)
        daily_r    = float(weights @ stock_rets) - txn_cost
        daily_rets.append(daily_r)
        portfolio_values.append(portfolio_values[-1] * (1 + daily_r))

    return _performance_metrics(portfolio_values, daily_rets, "MVO",
                                report_file=report_file)


def simulate_naive(closes: np.ndarray,
                   initial_val: float = 1_000_000,
                   report_file=None) -> dict:
    """Naïve Portfolio Allocation: equal weight, buy-and-hold.

    No ongoing TRANSACTION_FEE cost applies here by design — weights
    are never rebalanced after the initial buy, so turnover (and thus
    cost) is zero for the remainder of the test period. This mirrors
    real buy-and-hold behavior and keeps Naive the cheapest strategy
    to run, consistent with its simplicity.
    """
    T       = closes.shape[0] - 1
    weights = np.ones(N_STOCKS) / N_STOCKS
    portfolio_values = [initial_val]
    daily_rets       = []

    for t in range(T):
        stock_rets = (closes[t + 1] - closes[t]) / (closes[t] + 1e-8)
        daily_r    = float(weights @ stock_rets)
        daily_rets.append(daily_r)
        portfolio_values.append(portfolio_values[-1] * (1 + daily_r))

    return _performance_metrics(portfolio_values, daily_rets, "Naive",
                                report_file=report_file)


# ──────────────────────────────────────────────────────────────────────
#  Performance metrics
# ──────────────────────────────────────────────────────────────────────
def _performance_metrics(portfolio_values: list, daily_rets: list,
                         name: str, report_file=None) -> dict:
    vals  = np.array(portfolio_values)
    rets  = np.array(daily_rets)

    total_return  = (vals[-1] - vals[0]) / vals[0]
    n_years       = len(rets) / 252
    annual_return = (1 + total_return) ** (1 / max(n_years, 1e-8)) - 1
    ann_std       = rets.std() * np.sqrt(252) + 1e-8
    sharpe        = (annual_return - RISK_FREE_RATE_ANN) / ann_std

    peak = np.maximum.accumulate(vals)
    mdd  = float(np.max((peak - vals) / (peak + 1e-8)))

    sep = "=" * 40
    _log(f"\n{sep}",                                              report_file)
    _log(f"  Strategy     : {name}",                             report_file)
    _log(f"  Annual Return: {annual_return:.4f}  "
         f"({annual_return*100:.2f}%)",                          report_file)
    _log(f"  Total Return : {total_return:.4f}  "
         f"({total_return*100:.2f}%)",                           report_file)
    _log(f"  Sharpe Ratio : {sharpe:.4f}",                       report_file)
    _log(f"  Max Drawdown : {mdd:.4f}",                          report_file)
    _log(f"  Daily Ret Std: {rets.std()*100:.4f}%",              report_file)
    _log(f"  Daily Ret Min: {rets.min()*100:.4f}%",              report_file)
    _log(f"  Daily Ret Max: {rets.max()*100:.4f}%",              report_file)
    _log(sep,                                                     report_file)

    return {
        "name":             name,
        "portfolio_values": vals.tolist(),
        "daily_returns":    rets.tolist(),
        "total_return":     total_return,
        "annual_return":    annual_return,
        "sharpe_ratio":     sharpe,
        "max_drawdown":     mdd,
    }
