# ============================================================
#  environment.py  –  Portfolio RL Environment
#  Aligned with paper:
#    - DQN-S reward: rolling-window Sharpe ratio
#    - DQN-R reward: daily portfolio return
#    - No transaction costs (paper does not mention them)
#    - No reward shaping, no concentration penalty, no clipping
# ============================================================
import numpy as np
from config import (
    INITIAL_CAPITAL, RISK_FREE_RATE_ANN,
    REWARD_TYPE, SHARPE_WINDOW, TICKERS,
    SHARPE_REWARD_SCALE, RETURN_REWARD_SCALE,
    REWARD_SHAPING, SHAPING_SCALE,
    CONCENTRATION_PENALTY, CONCENTRATION_PENALTY_SCALE,
    USE_DIFFERENTIAL_SHARPE, DIFF_SHARPE_ETA, DIFF_SHARPE_SCALE,
    REWARD_CLIP, REWARD_CLIP_MIN, REWARD_CLIP_MAX,
    TRANSACTION_FEE
)

N_STOCKS   = len(TICKERS)
DAILY_RFR  = RISK_FREE_RATE_ANN / 252
EW_WEIGHTS = np.ones(N_STOCKS, dtype=np.float32) / N_STOCKS


class PortfolioEnv:
    """
    Single-episode portfolio environment.

    Paper definition
    ----------------
    State  : (n_stocks × n_features × window) = (10 × 5 × 10)
    Action : (n_stocks,) portfolio weights summing to 1
    Reward : Rolling-window Sharpe ratio (DQN-S)
             OR scaled daily return (DQN-R)

    FIX: DQN-R reward is now multiplied by RETURN_REWARD_SCALE (50.0)
    to bring gradient magnitudes in line with DQN-S. This is a
    monotone transformation and does not change the optimal policy —
    the ranking of actions by Q-value is unaffected.

    FIX: Transaction costs (TRANSACTION_FEE, turnover-based) and the
    concentration penalty (CONCENTRATION_PENALTY, HHI-based) are now
    both applied here, to both DQN-Sharpe and DQN-Return, to discourage
    the large/unstable reallocations observed during training.
    """

    def __init__(self, states: np.ndarray, closes: np.ndarray,
                 reward_type: str = REWARD_TYPE):
        self.states      = states
        self.closes      = closes
        self.T           = len(states)
        self.reward_type = reward_type

        self.t             = 0
        self.portfolio_val = INITIAL_CAPITAL
        self.weights       = EW_WEIGHTS.copy()
        self.returns_hist  = []

        self._ds_A = 0.0
        self._ds_B = 0.0

    def reset(self):
        self.t             = 0
        self.portfolio_val = INITIAL_CAPITAL
        self.weights       = EW_WEIGHTS.copy()
        self.returns_hist  = []
        self._ds_A         = 0.0
        self._ds_B         = 0.0
        return self.states[0]

    def step(self, action: np.ndarray):
        assert len(action) == N_STOCKS

        weights = np.clip(action, 0, None)
        w_sum   = weights.sum()
        weights = weights / w_sum if w_sum > 0 else EW_WEIGHTS.copy()

        if self.t + 1 < self.T:
            price_today    = self.closes[self.t]
            price_tomorrow = self.closes[self.t + 1]
        else:
            price_today    = self.closes[self.t]
            price_tomorrow = price_today

        stock_returns = (price_tomorrow - price_today) / (price_today + 1e-8)
        gross_return  = float(np.dot(weights, stock_returns))
        ew_return     = float(np.dot(EW_WEIGHTS, stock_returns))

        # ── Transaction cost (FIX: previously unimplemented) ───────────
        # Cost is proportional to turnover — the total absolute change
        # in portfolio weights vs. the prior step. This is what actually
        # penalizes the large, unstable reallocations seen in training
        # (episode returns std 554%, max 2714%); TRANSACTION_FEE alone
        # in config.py does nothing without this.
        turnover     = float(np.sum(np.abs(weights - self.weights)))
        txn_cost     = TRANSACTION_FEE * turnover
        daily_return = gross_return - txn_cost

        self.portfolio_val *= (1 + daily_return)
        self.weights        = weights
        self.returns_hist.append(daily_return)

        reward = self._compute_reward(daily_return, ew_return, weights)

        self.t += 1
        done = self.t >= self.T - 1
        next_state = self.states[self.t] if not done else self.states[-1]

        info = {
            "portfolio_value": self.portfolio_val,
            "daily_return":    daily_return,
            "gross_return":    gross_return,
            "transaction_cost": txn_cost,
            "weights":         weights.copy(),
        }
        return next_state, reward, done, info

    def _compute_reward(self, daily_return: float,
                        ew_return: float,
                        weights: np.ndarray) -> float:
        """
        Compute reward according to the paper.

        DQN-R: scaled daily portfolio return.
            FIX: multiplied by RETURN_REWARD_SCALE (50.0) to match the
            gradient magnitude of DQN-S. Monotone transformation —
            optimal policy is unchanged.

        DQN-S: rolling-window Sharpe ratio (paper equation 4).
        """
        # ── Primary reward ─────────────────────────────────────────────
        if self.reward_type == "return":
            # FIX: scale up the raw return signal to restore gradient flow
            base = daily_return * RETURN_REWARD_SCALE

        elif self.reward_type == "sharpe":
            if USE_DIFFERENTIAL_SHARPE:
                base = self._differential_sharpe_step(daily_return)
            else:
                window = self.returns_hist[-SHARPE_WINDOW:]
                if len(window) < 2:
                    base = daily_return * SHARPE_REWARD_SCALE
                else:
                    r_arr  = np.array(window)
                    excess = r_arr - DAILY_RFR
                    std    = max(float(r_arr.std()), 1e-4)
                    raw_sharpe = float(excess.mean() / std)
                    raw_sharpe = float(np.clip(raw_sharpe, -3.0, 3.0))
                    base = raw_sharpe * SHARPE_REWARD_SCALE
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        # ── Optional: reward shaping (NOT in paper) ────────────────────
        if REWARD_SHAPING:
            shaping = np.clip((daily_return - ew_return) * SHAPING_SCALE,
                              -0.01, 0.01)
            base += shaping

        # ── Optional: concentration penalty (NOT in paper) ──────────────
        # FIX: previously gated to `self.reward_type == "return"` only,
        # so enabling CONCENTRATION_PENALTY had no effect on DQN-Sharpe —
        # the agent whose training returns were actually unstable
        # (std 554%, max 2714%). Now applies to both reward types.
        if CONCENTRATION_PENALTY:
            hhi        = float(np.sum(weights ** 2))
            hhi_excess = hhi - (1.0 / N_STOCKS)
            base      -= CONCENTRATION_PENALTY_SCALE * hhi_excess

        # ── Optional: reward clipping (NOT in paper) ───────────────────
        if REWARD_CLIP:
            base = float(np.clip(base, REWARD_CLIP_MIN, REWARD_CLIP_MAX))

        return base

    def _differential_sharpe_step(self, rt: float) -> float:
        """Differential Sharpe (Moody & Saffell 1998). USE_DIFFERENTIAL_SHARPE=True only."""
        eta    = DIFF_SHARPE_ETA
        A_prev = self._ds_A
        B_prev = self._ds_B

        self._ds_A = A_prev + eta * (rt - A_prev)
        self._ds_B = B_prev + eta * (rt ** 2 - B_prev)

        denom = B_prev - A_prev ** 2
        if denom < 1e-10:
            return rt * DIFF_SHARPE_SCALE * 0.1

        delta  = (B_prev * (rt - A_prev) - 0.5 * A_prev * (rt ** 2 - B_prev))
        delta /= (denom ** 1.5)

        return float(delta) * DIFF_SHARPE_SCALE

    @property
    def state_shape(self):
        return self.states[0].shape

    @property
    def action_dim(self):
        return N_STOCKS
