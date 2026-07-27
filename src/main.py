# ============================================================
#  main.py  –  Full pipeline: train → evaluate
# ============================================================
"""
Run this script to execute the full paper pipeline:
  1. Download European energy sector stock data from Yahoo Finance
  2. Train DQN-S (Sharpe reward) and DQN-R (Return reward)
  3. Run MVO and Naïve benchmarks
  4. Evaluate all strategies on the test set
  5. Print performance table and save plots

Usage
-----
    python main.py                     # train + evaluate both agents
    python main.py --skip-train        # evaluate only (requires saved checkpoints)
    python main.py --reward sharpe     # train/evaluate only DQN-S
"""
import argparse
import os

from config import RESULTS_DIR, CHECKPOINT_DIR, REWARD_TYPE, REWARD_TYPES


def main():
    parser = argparse.ArgumentParser(description="DQN Portfolio Management")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training and use existing checkpoints.")
    parser.add_argument("--reward", choices=["sharpe", "return", "both"],
                        default="both",
                        help="Which reward type to train/evaluate.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR,     exist_ok=True)
    os.makedirs(CHECKPOINT_DIR,  exist_ok=True)

    reward_types = (REWARD_TYPES if args.reward == "both"
                    else [args.reward])

    # ── Training ────────────────────────────────────────────────────
    if not args.skip_train:
        from train import train_agent
        for rt in reward_types:
            print(f"\n{'#'*55}")
            print(f"  TRAINING  DQN-{rt.upper()}")
            print(f"{'#'*55}\n")
            train_agent(reward_type=rt)

    # ── Evaluation ──────────────────────────────────────────────────
    print(f"\n{'#'*55}")
    print(f"  EVALUATION")
    print(f"{'#'*55}\n")

    from data       import load_split_data, download_data, compute_pct_change
    from mvo        import simulate_mvo, simulate_naive
    from evaluate   import (evaluate_dqn, print_comparison_table,
                            plot_portfolio_values, plot_drawdowns,
                            plot_training_history, save_results_txt)
    from config     import TEST_START, TEST_END, WINDOW_SIZE

    _, test_states, _, test_closes = load_split_data()

    raw     = download_data()
    pct_all = compute_pct_change(raw)
    test_mask = (pct_all.index >= TEST_START) & (pct_all.index <= TEST_END)
    test_pct  = pct_all[test_mask].values[WINDOW_SIZE:]

    mvo_res   = simulate_mvo(test_closes, test_pct)
    naive_res = simulate_naive(test_closes)

    all_results = [mvo_res, naive_res]
    for rt in reward_types:
        all_results.append(evaluate_dqn(rt, test_states, test_closes))

    print_comparison_table(all_results)
    plot_portfolio_values(all_results)
    plot_drawdowns(all_results)
    plot_training_history()
    save_results_txt(all_results)


if __name__ == "__main__":
    main()
