# ============================================================
#  data.py  –  Download & preprocess stock data
#  Aligned with paper: 5 OHLCV features only, window = 10
#
#  FIX SUMMARY (vs previous version):
#    1. Explicit column reindexing after download: yfinance does not
#       guarantee the ticker order within its MultiIndex columns
#       matches the order of the `tickers` list passed in (it is
#       commonly alphabetical instead). build_windows() assumes the
#       stock axis is ordered exactly as TICKERS, so that order is
#       now pinned explicitly here rather than assumed.
#    2. load_split_data() now returns THREE splits (train / validation
#       / test) instead of two. The validation split exists so that
#       hyperparameter and model-selection decisions can be made
#       without looking at the test set (see train.py / mvo.py).
#    3. New get_aligned_pct() helper centralises the "slice pct-change
#       DataFrame to [start, end], then drop the first WINDOW_SIZE
#       rows" logic that was previously duplicated (and easy to get
#       subtly wrong) across main.py and evaluate.py.
# ============================================================
import os
import numpy as np
import pandas as pd
import yfinance as yf

import config
from config import (
    TICKERS, TRAIN_START, TRAIN_END, VAL_START, VAL_END,
    TEST_START, TEST_END, FEATURES, WINDOW_SIZE, RESULTS_DIR,
    STATE_CLIP
)


def download_data(tickers=TICKERS, start=TRAIN_START, end=TEST_END) -> pd.DataFrame:
    print(f"Downloading data for {tickers} …")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    raw = raw[FEATURES]

    # FIX 1: pin the ticker order explicitly. yfinance's MultiIndex
    # column order for the ticker sub-level is not guaranteed to match
    # `tickers` (commonly alphabetical instead). Everything downstream
    # (build_windows' reshape, get_close_prices, MVO's covariance
    # matrix) assumes column N corresponds to TICKERS[N], so that must
    # be pinned here rather than assumed.
    raw = raw.reindex(columns=pd.MultiIndex.from_product([FEATURES, tickers]))

    raw.dropna(inplace=True)
    return raw


def compute_pct_change(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change().dropna()


def get_aligned_pct(pct_all: pd.DataFrame, start: str, end: str) -> np.ndarray:
    """
    Slice a pct-change DataFrame to [start, end] and drop the first
    WINDOW_SIZE rows, so the result lines up index-for-index with the
    *_states / *_closes arrays returned by load_split_data() for the
    same [start, end] window.

    Centralised here so callers (main.py, evaluate.py, mvo.py) don't
    each re-implement this offset separately and risk a subtle
    off-by-WINDOW_SIZE misalignment.
    """
    mask = (pct_all.index >= start) & (pct_all.index <= end)
    return pct_all[mask].values[WINDOW_SIZE:]


def build_windows(pct_df: pd.DataFrame, window: int = WINDOW_SIZE,
                  volume_std: float = None):
    """
    Build sliding-window state tensors.

    Paper: "each environment state S is a multidimensional matrix,
    which contains 500 data input (10 × 5 × 10)."
    → 10 stocks × 5 features × 10 days = 500 inputs per state.

    Parameters
    ----------
    pct_df      : DataFrame of pct-change OHLCV values
    window      : lookback window (paper: 10)
    volume_std  : global std of volume pct-changes from the TRAINING
                  set. If None, it is computed from pct_df and stored
                  (only ever done for the training-split call) — the
                  validation and test splits must reuse this same
                  value rather than leaking their own scale statistics
                  back into normalisation.
    """
    n_features = len(FEATURES)      # 5: Open, High, Low, Close, Volume
    n_stocks   = len(TICKERS)       # 10
    T          = len(pct_df)

    arr = pct_df.values.reshape(T, n_features, n_stocks)
    arr = arr.transpose(0, 2, 1)    # (T, n_stocks, n_features)

    volume_idx  = FEATURES.index("Volume")
    price_mask  = np.ones(n_features, dtype=bool)
    price_mask[volume_idx] = False

    # Clip price pct-change features
    arr[:, :, price_mask] = np.clip(arr[:, :, price_mask],
                                    -STATE_CLIP, STATE_CLIP)

    # Volume normalisation with global (training-set) std
    arr[:, :, volume_idx] = np.clip(arr[:, :, volume_idx], -10.0, 10.0)

    if volume_std is None:
        volume_std = float(arr[:, :, volume_idx].std()) + 1e-8
        config.VOLUME_STD_GLOBAL = volume_std

    arr[:, :, volume_idx] = arr[:, :, volume_idx] / (volume_std * 10.0)
    arr[:, :, volume_idx] = np.clip(arr[:, :, volume_idx], -1.0, 1.0)

    # Build sliding windows → (n_stocks, n_features, window)
    states = []
    for t in range(window, T):
        window_data = arr[t - window: t]
        states.append(window_data.transpose(1, 2, 0))  # (n_stocks, n_features, window)

    return np.array(states, dtype=np.float32)


def get_close_prices(raw_df: pd.DataFrame) -> np.ndarray:
    return raw_df["Close"].values.astype(np.float32)


def load_split_data():
    """
    Returns
    -------
    train_states, val_states, test_states, train_closes, val_closes, test_closes

    FIX 2: three-way split (train / validation / test) instead of two.
    TRAIN_END was pulled back two years (see config.py) so that
    [VAL_START, VAL_END] can serve as a genuine held-out set for
    hyperparameter and model-selection decisions, leaving
    [TEST_START, TEST_END] untouched until the single, final
    evaluation.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    raw = download_data()
    config.ALL_FEATURES = list(FEATURES)

    pct = compute_pct_change(raw)

    train_mask = (pct.index >= TRAIN_START) & (pct.index <= TRAIN_END)
    val_mask   = (pct.index >= VAL_START)   & (pct.index <= VAL_END)
    test_mask  = (pct.index >= TEST_START)  & (pct.index <= TEST_END)

    train_pct = pct[train_mask]
    val_pct   = pct[val_mask]
    test_pct  = pct[test_mask]

    # Training windows first — this is the ONLY call that computes
    # (and stores, via config.VOLUME_STD_GLOBAL) the global volume std.
    train_states = build_windows(train_pct, volume_std=None)
    val_states   = build_windows(val_pct,  volume_std=config.VOLUME_STD_GLOBAL)
    test_states  = build_windows(test_pct, volume_std=config.VOLUME_STD_GLOBAL)

    raw_close = get_close_prices(raw)
    raw_dates = raw.index
    train_close_mask = (raw_dates >= TRAIN_START) & (raw_dates <= TRAIN_END)
    val_close_mask   = (raw_dates >= VAL_START)   & (raw_dates <= VAL_END)
    test_close_mask  = (raw_dates >= TEST_START)  & (raw_dates <= TEST_END)

    train_closes = raw_close[train_close_mask][WINDOW_SIZE:]
    val_closes   = raw_close[val_close_mask][WINDOW_SIZE:]
    test_closes  = raw_close[test_close_mask][WINDOW_SIZE:]

    print(f"Train states: {train_states.shape}, "
          f"Val states: {val_states.shape}, "
          f"Test states: {test_states.shape}")
    print(f"Features ({len(FEATURES)}): {list(FEATURES)}")
    print(f"State size per step: {train_states.shape[1:]} "
          f"= {train_states[0].size} inputs  "
          f"(paper: 10×5×10 = 500)")
    print(f"Global volume std: {config.VOLUME_STD_GLOBAL:.4f}")
    return train_states, val_states, test_states, train_closes, val_closes, test_closes


if __name__ == "__main__":
    tr_s, va_s, te_s, tr_c, va_c, te_c = load_split_data()
    print("Train states:", tr_s.shape)
    print("Val   states:", va_s.shape)
    print("Test  states:", te_s.shape)
