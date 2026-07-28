# ============================================================
#  main.py  –  Full pipeline: train → validate → evaluate
#
#  FIX (vs previous version): the pipeline now runs three phases in
#  strict order —
#    1. TRAIN on [TRAIN_START, TRAIN_END], with model checkpointing
#       decided by periodic passes over [VAL_START, VAL_END] (train.py)
#    2. Select the MVO rebalance frequency, also on the validation
#       split only (mvo.select_best_rebalance_freq)
#    3. Evaluate every strategy — DQN agents and both benchmarks — on
#       [TEST_START, TEST_END] exactly once, for the final reported
#       numbers.
#  The test split is never used for any decision prior to step 3.
# ============================================================
"""
Run this script to execute the full pipeline:
  1. Download European energy sector stock data from Yahoo Finance
  2. Train DQN-S (Sharpe reward) and DQN-R (Return reward), selecting
     checkpoints by validation-set Sharpe
  3. Select the MVO rebalance frequency on the validation set
  4. Evaluate all strategies on the held-out test set
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

    # ── Training (checkpoints selected via validation, see train.py) ──
    if not args.skip_train:
        from train import train_agent
        for rt in reward_types:
            print(f"\n{'#'*55}")
            print(f"  TRAINING  DQN-{rt.upper()}")
            print(f"{'#'*55}\n")
            train_agent(reward_type=rt)

    # ── Evaluation (test set touched exactly once, here) ──────────────
    print(f"\n{'#'*55}")
    print(f"  EVALUATION")
    print(f"{'#'*55}\n")

    from data       import (load_split_data, download_data, compute_pct_change,
                            get_aligned_pct)
    from mvo        import simulate_mvo, simulate_naive, select_best_rebalance_freq
    from evaluate   import (evaluate_dqn, print_comparison_table,
                            plot_portfolio_values, plot_drawdowns,
                            plot_training_history, save_results_txt)
    from config     import TEST_START, TEST_END, VAL_START, VAL_END

    (train_states, val_states, test_states,
     train_closes, val_closes, test_closes) = load_split_data()

    raw     = download_data()
    pct_all = compute_pct_change(raw)
    val_pct  = get_aligned_pct(pct_all, VAL_START, VAL_END)
    test_pct = get_aligned_pct(pct_all, TEST_START, TEST_END)

    # Model-selection step on validation only — never touches test_*.
    best_freq = select_best_rebalance_freq(val_closes, val_pct)

    # Final, single-pass evaluation on the held-out test set.
    mvo_res   = simulate_mvo(test_closes, test_pct, freq=best_freq)
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
