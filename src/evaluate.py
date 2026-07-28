# ============================================================
#  evaluate.py  –  Evaluate all strategies & produce plots
#
#  FIX (vs previous version): __main__ now unpacks the three-way
#  train/val/test split from data.load_split_data(), selects the MVO
#  rebalance frequency on the validation split only (mirrors the
#  validation-based checkpointing now done in train.py), and evaluates
#  every strategy on the test split exactly once.
# ============================================================
import os
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from config     import CHECKPOINT_DIR, RESULTS_DIR, INITIAL_CAPITAL, TICKERS
from data       import load_split_data, download_data, compute_pct_change, get_aligned_pct
from environment import PortfolioEnv
from agent      import DQNAgent
from mvo        import simulate_mvo, simulate_naive, _performance_metrics, select_best_rebalance_freq

import pandas as pd

N_STOCKS    = len(TICKERS)
REPORT_PATH = os.path.join(RESULTS_DIR, "results_report.txt")


# ──────────────────────────────────────────────────────────────────────
#  Shared print-and-write helper
# ──────────────────────────────────────────────────────────────────────
def _log(line: str = "", file=None):
    print(line)
    if file:
        file.write(line + "\n")


# ──────────────────────────────────────────────────────────────────────
#  Evaluate one DQN agent on the test set
# ──────────────────────────────────────────────────────────────────────
def evaluate_dqn(reward_type: str,
                 test_states: np.ndarray,
                 test_closes: np.ndarray,
                 report_file=None) -> dict:
    """
    Evaluate a trained DQN agent, greedily (epsilon=0), on whichever
    states/closes are passed in. Callers should pass the TEST split
    here — this function itself does not enforce that, so double-check
    the caller when reusing it (e.g. main.py always passes test data).
    """
    env   = PortfolioEnv(test_states, test_closes, reward_type=reward_type)
    agent = DQNAgent(env.state_shape, env.action_dim, reward_type=reward_type)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"dqn_{reward_type}.pt")
    if not os.path.exists(ckpt_path):
        msg = f"  [WARNING] No checkpoint found at {ckpt_path} – using random weights."
        _log(msg, report_file)
    else:
        agent.load(ckpt_path)
    agent.epsilon = 0.0   # greedy evaluation

    state = env.reset()
    portfolio_values = []   # env.portfolio_val == INITIAL_CAPITAL after reset;
                            # we collect it on the first step via info so there
                            # is no duplicate entry.
    daily_rets       = []
    done             = False
    first_step       = True

    while not done:
        action_weights, _ = agent.select_action(state)
        state, _, done, info = env.step(action_weights)
        if first_step:
            # Reconstruct the starting value from the first daily return so
            # the series has exactly T+1 points (including t=0).
            start_val = info["portfolio_value"] / (1 + info["daily_return"])
            portfolio_values.append(start_val)
            first_step = False
        portfolio_values.append(info["portfolio_value"])
        daily_rets.append(info["daily_return"])

    name = f"DQN-{reward_type.upper()}"
    return _performance_metrics(portfolio_values, daily_rets, name,
                                report_file=report_file)


# ──────────────────────────────────────────────────────────────────────
#  Comparison table
# ──────────────────────────────────────────────────────────────────────
def print_comparison_table(results: list[dict], report_file=None):
    sep  = "=" * 60
    dash = "-" * 60
    _log("", report_file)
    _log(sep, report_file)
    _log(f"{'Strategy':<15} {'Annual Return':>14} {'Sharpe':>10} {'MDD':>10}",
         report_file)
    _log(dash, report_file)
    for r in results:
        _log(f"{r['name']:<15} "
             f"{r['annual_return']*100:>13.2f}%  "
             f"{r['sharpe_ratio']:>10.4f}  "
             f"{r['max_drawdown']:>10.4f}",
             report_file)
    _log(sep, report_file)


# ──────────────────────────────────────────────────────────────────────
#  Plots
# ──────────────────────────────────────────────────────────────────────
def plot_portfolio_values(results: list[dict], save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors  = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

    for i, r in enumerate(results):
        vals  = np.array(r["portfolio_values"]) / INITIAL_CAPITAL
        ax.plot(vals, label=r["name"], color=colors[i % len(colors)], linewidth=1.8)

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, label="Initial")
    ax.set_title("Portfolio Performance – Test Set (2019–2021)", fontsize=14)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Normalised Portfolio Value")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = save_path or os.path.join(RESULTS_DIR, "portfolio_performance.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved → {path}")


def _smooth(values: list, window: int = 10) -> np.ndarray:
    """Simple rolling mean to expose the learning trend under episode noise."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel   = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode='valid')
    pad      = np.full(window - 1, np.nan)
    return np.concatenate([pad, smoothed])


def plot_training_history(save_path: str = None):
    """
    Save three plot files per agent plus a combined comparison.

    Each individual plot has three panels:
        Left   : cumulative reward (raw faint + smoothed bold) + ε twin axis
        Middle : episode portfolio return (raw faint + smoothed bold)
        Right  : gradient norm per episode

    The gradient norm panel lets you diagnose:
        • Exploding gradients  (sudden spikes → lower LR or clip threshold)
        • Vanishing gradients  (norms near 0  → network not learning)
        • Convergence plateau  (norms stabilise → learning has flattened)

    If the history also contains "val_episodes"/"val_sharpe" (written by
    train.py's validation-based checkpointing), those points are overlaid
    on the middle panel as a secondary axis so it's visible where the
    saved checkpoint actually came from.
    """
    colors = {"sharpe": "#2196F3", "return": "#FF9800"}
    SMOOTH_W = 10

    histories = {}
    for rt in ["sharpe", "return"]:
        hist_path = os.path.join(RESULTS_DIR, f"train_history_{rt}.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                histories[rt] = json.load(f)

    # ── Individual plots ─────────────────────────────────────────────
    for rt, h in histories.items():
        has_gnorms = "grad_norms" in h and len(h["grad_norms"]) > 0
        ncols  = 3 if has_gnorms else 2
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
        color = colors[rt]
        label = f"DQN-{rt.upper()}"
        eps   = list(range(1, len(h["episode_rewards"]) + 1))

        # Left: cumulative reward
        raw_r    = h["episode_rewards"]
        smooth_r = _smooth(raw_r, SMOOTH_W)
        axes[0].plot(eps, raw_r, color=color, alpha=0.25, linewidth=1.0)
        axes[0].plot(eps, smooth_r, color=color, alpha=0.95, linewidth=2.0,
                     label=f"Smoothed (w={SMOOTH_W})")
        axes[0].set_title(f"{label} – Cumulative Reward")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Cumulative Reward")
        axes[0].legend(loc="upper left")
        axes[0].grid(alpha=0.3)

        if "epsilon" in h:
            ax_eps = axes[0].twinx()
            ax_eps.plot(eps, h["epsilon"], color="grey", linestyle="--",
                        linewidth=1.0, alpha=0.6, label="ε")
            ax_eps.set_ylabel("ε (epsilon)", color="grey")
            ax_eps.tick_params(axis="y", labelcolor="grey")
            ax_eps.set_ylim(0, 1.05)
            ax_eps.legend(loc="upper right")

        # Middle: episode return, with validation Sharpe overlaid
        raw_ret    = [r * 100 for r in h["episode_returns"]]
        smooth_ret = _smooth(raw_ret, SMOOTH_W)
        axes[1].plot(eps, raw_ret, color=color, alpha=0.25, linewidth=1.0)
        axes[1].plot(eps, smooth_ret, color=color, alpha=0.95, linewidth=2.0,
                     label=f"Smoothed (w={SMOOTH_W})")
        axes[1].set_title(f"{label} – Episode Return (%)")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Return (%)")
        axes[1].legend(loc="upper left")
        axes[1].grid(alpha=0.3)

        if h.get("val_episodes") and h.get("val_sharpe"):
            ax_val = axes[1].twinx()
            ax_val.scatter(h["val_episodes"], h["val_sharpe"],
                          color="black", marker="x", s=40,
                          label="Validation Sharpe")
            ax_val.set_ylabel("Validation Sharpe", color="black")
            ax_val.legend(loc="lower right")

        # Right: gradient norms
        if has_gnorms:
            gnorms        = h["grad_norms"]
            smooth_gnorms = _smooth(gnorms, SMOOTH_W)
            axes[2].plot(eps, gnorms, color=color, alpha=0.25, linewidth=1.0)
            axes[2].plot(eps, smooth_gnorms, color=color, alpha=0.95, linewidth=2.0,
                        label=f"Smoothed (w={SMOOTH_W})")
            axes[2].set_title(f"{label} – Gradient Norm")
            axes[2].set_xlabel("Episode")
            axes[2].set_ylabel("Grad Norm (L2)")
            axes[2].legend(loc="upper left")
            axes[2].grid(alpha=0.3)

        plt.tight_layout()
        ind_path = os.path.join(RESULTS_DIR, f"training_history_{rt}.png")
        plt.savefig(ind_path, dpi=150)
        plt.close()
        print(f"  Plot saved → {ind_path}")

    # ── Combined plot ────────────────────────────────────────────────
    if histories:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for rt, h in histories.items():
            color = colors[rt]
            label = f"DQN-{rt.upper()}"
            eps   = list(range(1, len(h["episode_rewards"]) + 1))

            raw_r    = h["episode_rewards"]
            smooth_r = _smooth(raw_r, SMOOTH_W)
            axes[0].plot(eps, raw_r,    color=color, alpha=0.20, linewidth=1.0)
            axes[0].plot(eps, smooth_r, color=color, alpha=0.95, linewidth=2.0,
                         label=label)

            raw_ret    = [r * 100 for r in h["episode_returns"]]
            smooth_ret = _smooth(raw_ret, SMOOTH_W)
            axes[1].plot(eps, raw_ret,    color=color, alpha=0.20, linewidth=1.0)
            axes[1].plot(eps, smooth_ret, color=color, alpha=0.95, linewidth=2.0,
                         label=label)

        axes[0].set_title("Cumulative Reward (smoothed)")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Cumulative Reward")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].set_title("Episode Portfolio Return (smoothed, %)")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Return (%)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        combined_path = save_path or os.path.join(RESULTS_DIR, "training_history.png")
        plt.savefig(combined_path, dpi=150)
        plt.close()
        print(f"  Plot saved → {combined_path}")


def plot_drawdowns(results: list[dict], save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

    for i, r in enumerate(results):
        vals = np.array(r["portfolio_values"])
        peak = np.maximum.accumulate(vals)
        dd   = (peak - vals) / (peak + 1e-8)
        ax.fill_between(range(len(dd)), -dd, alpha=0.4,
                        color=colors[i % len(colors)], label=r["name"])

    ax.set_title("Drawdown – Test Set", fontsize=14)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = save_path or os.path.join(RESULTS_DIR, "drawdowns.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved → {path}")


# ──────────────────────────────────────────────────────────────────────
#  Full results text file
# ──────────────────────────────────────────────────────────────────────
def save_results_txt(results: list[dict], save_path: str = None):
    path = save_path or os.path.join(RESULTS_DIR, "results_summary.txt")
    sep  = "=" * 60
    dash = "-" * 60

    lines = []
    lines.append(sep)
    lines.append("  DQN PORTFOLIO MANAGEMENT – FULL RESULTS SUMMARY")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)
    lines.append("")
    lines.append("[ INDIVIDUAL STRATEGY RESULTS ]")
    lines.append("")
    for r in results:
        lines.append(sep)
        lines.append(f"  Strategy     : {r['name']}")
        lines.append(f"  Annual Return: {r['annual_return']:.4f}  ({r['annual_return']*100:.2f}%)")
        lines.append(f"  Total Return : {r['total_return']:.4f}  ({r['total_return']*100:.2f}%)")
        lines.append(f"  Sharpe Ratio : {r['sharpe_ratio']:.4f}")
        lines.append(f"  Max Drawdown : {r['max_drawdown']:.4f}  ({r['max_drawdown']*100:.2f}%)")
        daily_rets = np.array(r['daily_returns'])
        lines.append(f"  Daily Ret Std: {daily_rets.std()*100:.4f}%")
        lines.append(f"  Daily Ret Min: {daily_rets.min()*100:.4f}%")
        lines.append(f"  Daily Ret Max: {daily_rets.max()*100:.4f}%")
        lines.append(sep)

    lines.append("")
    lines.append("[ COMPARISON TABLE ]")
    lines.append("")
    lines.append(f"{'Strategy':<15} {'Annual Ret':>12} {'Total Ret':>12} "
                 f"{'Sharpe':>10} {'MDD':>10} {'DailyStd':>10}")
    lines.append(dash)
    for r in results:
        daily_std = np.array(r['daily_returns']).std() * 100
        lines.append(
            f"{r['name']:<15} "
            f"{r['annual_return']*100:>11.2f}%  "
            f"{r['total_return']*100:>11.2f}%  "
            f"{r['sharpe_ratio']:>10.4f}  "
            f"{r['max_drawdown']:>10.4f}  "
            f"{daily_std:>9.4f}%"
        )
    lines.append(dash)

    lines.append("")
    lines.append("[ TRAINING HISTORY SUMMARY ]")
    lines.append("")
    for rt in ["sharpe", "return"]:
        hist_path = os.path.join(RESULTS_DIR, f"train_history_{rt}.json")
        if not os.path.exists(hist_path):
            continue
        with open(hist_path) as f:
            h = json.load(f)
        rewards  = h["episode_rewards"]
        ep_rets  = [r * 100 for r in h["episode_returns"]]
        epsilons = h.get("epsilon", [])
        gnorms   = h.get("grad_norms", [])
        val_sharpe = h.get("val_sharpe", [])
        lines.append(f"  DQN-{rt.upper()}")
        lines.append(f"    Episodes trained   : {len(rewards)}")
        lines.append(f"    Reward  – mean     : {np.mean(rewards):.4f}")
        lines.append(f"    Reward  – std      : {np.std(rewards):.4f}")
        lines.append(f"    Reward  – min/max  : {min(rewards):.4f} / {max(rewards):.4f}")
        lines.append(f"    Ep Ret  – mean     : {np.mean(ep_rets):.2f}%")
        lines.append(f"    Ep Ret  – std      : {np.std(ep_rets):.2f}%")
        lines.append(f"    Ep Ret  – min/max  : {min(ep_rets):.2f}% / {max(ep_rets):.2f}%")
        if epsilons:
            lines.append(f"    Final ε            : {epsilons[-1]:.4f}")
        if gnorms:
            lines.append(f"    GradNorm – mean    : {np.mean(gnorms):.4f}")
            lines.append(f"    GradNorm – final10 : {np.mean(gnorms[-10:]):.4f}")
        if val_sharpe:
            lines.append(f"    Best Val Sharpe    : {max(val_sharpe):.4f} "
                        f"(episode {h['val_episodes'][int(np.argmax(val_sharpe))]})")
        lines.append("")

    lines.append(sep)
    lines.append("  END OF REPORT")
    lines.append(sep)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Results summary saved → {path}")


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    from config import TEST_START, TEST_END, VAL_START, VAL_END

    _, val_states, test_states, _, val_closes, test_closes = load_split_data()

    raw     = download_data()
    pct_all = compute_pct_change(raw)
    val_pct  = get_aligned_pct(pct_all, VAL_START, VAL_END)
    test_pct = get_aligned_pct(pct_all, TEST_START, TEST_END)

    with open(REPORT_PATH, "w") as rf:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log("=" * 60, rf)
        _log("  DQN PORTFOLIO MANAGEMENT – RESULTS REPORT", rf)
        _log(f"  Generated: {timestamp}", rf)
        _log("=" * 60, rf)

        _log("\n[ VALIDATION-BASED MODEL SELECTION ]\n", rf)
        best_freq = select_best_rebalance_freq(val_closes, val_pct)
        _log(f"  MVO rebalance frequency selected on validation: {best_freq}", rf)

        _log("\n[ BENCHMARK STRATEGIES (test set) ]\n", rf)
        mvo_res   = simulate_mvo(test_closes, test_pct, freq=best_freq, report_file=rf)
        naive_res = simulate_naive(test_closes, report_file=rf)

        _log("\n[ DQN STRATEGIES (test set) ]\n", rf)
        dqn_s_res = evaluate_dqn("sharpe", test_states, test_closes, report_file=rf)
        dqn_r_res = evaluate_dqn("return", test_states, test_closes, report_file=rf)

        all_results = [mvo_res, naive_res, dqn_s_res, dqn_r_res]

        _log("\n[ COMPARISON TABLE ]\n", rf)
        print_comparison_table(all_results, report_file=rf)
        _log(f"\n  Full report saved → {REPORT_PATH}", rf)

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(
            [{k: v for k, v in r.items()
              if k not in ("portfolio_values", "daily_returns")}
             for r in all_results],
            f, indent=2
        )

    plot_portfolio_values(all_results)
    plot_drawdowns(all_results)
    plot_training_history()
    save_results_txt(all_results)
