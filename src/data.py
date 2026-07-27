# ============================================================
#  data.py  –  Download & preprocess stock data
#  Aligned with paper: 5 OHLCV features only, window = 10
# ============================================================
import os
import numpy as np
import pandas as pd
import yfinance as yf

import config
from config import (
    TICKERS, TRAIN_START, TRAIN_END,
    TEST_START, TEST_END, FEATURES, WINDOW_SIZE, RESULTS_DIR,
    STATE_CLIP
)


def download_data(tickers=TICKERS, start=TRAIN_START, end=TEST_END) -> pd.DataFrame:
    print(f"Downloading data for {tickers} …")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    raw = raw[FEATURES]
    raw.dropna(inplace=True)
    return raw


def compute_pct_change(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change().dropna()


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
    volume_std  : global std of volume pct-changes from training set.
                  If None, computed from pct_df (training-set call).
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

    # Volume normalisation with global std
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
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Paper uses raw OHLCV only — no engineered features
    raw = download_data()
    config.ALL_FEATURES = list(FEATURES)

    pct = compute_pct_change(raw)

    train_mask = (pct.index >= TRAIN_START) & (pct.index <= TRAIN_END)
    test_mask  = (pct.index >= TEST_START)  & (pct.index <= TEST_END)

    train_pct = pct[train_mask]
    test_pct  = pct[test_mask]

    # Training windows first (computes + stores global volume std)
    train_states = build_windows(train_pct, volume_std=None)
    test_states  = build_windows(test_pct,  volume_std=config.VOLUME_STD_GLOBAL)

    # Close prices aligned to the windowed states
    raw_close        = get_close_prices(raw)
    raw_dates        = raw.index
    train_close_mask = (raw_dates >= TRAIN_START) & (raw_dates <= TRAIN_END)
    test_close_mask  = (raw_dates >= TEST_START)  & (raw_dates <= TEST_END)

    train_closes = raw_close[train_close_mask][WINDOW_SIZE:]
    test_closes  = raw_close[test_close_mask][WINDOW_SIZE:]

    print(f"Train states: {train_states.shape}, Test states: {test_states.shape}")
    print(f"Features ({len(FEATURES)}): {list(FEATURES)}")
    print(f"State size per step: {train_states.shape[1:]} "
          f"= {train_states[0].size} inputs  "
          f"(paper: 10×5×10 = 500)")
    print(f"Global volume std: {config.VOLUME_STD_GLOBAL:.4f}")
    return train_states, test_states, train_closes, test_closes


if __name__ == "__main__":
    tr_s, te_s, tr_c, te_c = load_split_data()
    print("Train states:", tr_s.shape)
    print("Test  states:", te_s.shape)
