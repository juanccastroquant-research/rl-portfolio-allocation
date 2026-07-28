# ============================================================
#  config.py  –  Hyperparameter file
#  Paper: "Reinforcement Learning in Portfolio Management
#          with Sharpe Ratio Rewarding Based Framework"
#
#  FIX SUMMARY (vs previous version):
#    1. EPISODE_LENGTH: 252 → 504  (agents now see 2-year windows;
#       train–test regime mismatch was the #1 cause of failure)
#    2. RETURN_REWARD_SCALE: new param = 50.0  (fixes gradient
#       collapse in DQN-Return; grad norms were ~130× smaller than
#       DQN-Sharpe because raw daily returns ~0.1% are tiny)
#    3. NUM_EPISODES: 1200 → 800  (longer episodes ⟹ fewer fit in
#       same wall-clock time; 800 × 504 steps ≈ same total steps)
#    4. LR_T_MAX adjusted to match new total gradient steps
#    5. TRANSACTION_FEE: 0.0 → 0.0005  (re-enabled; test-set results
#       showed DQN-Sharpe making large, unstable reallocations with
#       zero friction, likely encouraging overconfident concentration
#       that overfits to specific training windows. Cost logic also
#       added to simulate_mvo()/simulate_naive() in mvo.py so all
#       strategies remain comparable on equal footing.)
#    6. CONCENTRATION_PENALTY: False → True  (training returns were
#       wildly unstable — mean 711%, std 554%, max 2714% — consistent
#       with the agent piling into a few assets rather than learning
#       a robust allocation. Penalty discourages weight concentration
#       away from equal-weight.)
#    7. TRAIN_START: 2006-11-07 → 2016-06-15  (Ørsted/ORSTED.CO only
#       has price history from its June 2016 IPO onward; the joint
#       dropna() across all 10 tickers in data.py was silently
#       truncating the whole dataset to mid-2016 regardless of the
#       configured TRAIN_START, so this makes the real window explicit)
#    8. EPISODE_LENGTH: 504 → 126 (re-derived; only ~3.3 years of
#       training data remain after fix 7, nowhere near enough to
#       sample 504-day episodes from). LR_T_MAX recomputed to match.
#    9. VAL_START/VAL_END added: 2018-11-07 → 2019-11-06. Matches the
#       train/val/test split now implemented in data.py — validation
#       is used for model-selection/checkpointing during training so
#       the test set stays untouched until final evaluation.
#       TRAIN_END pulled back to 2018-11-06 to make room for it.
#   10. VALIDATION_INTERVAL: new param = 20. Required by train.py's
#       validation-based checkpointing (previously caused an
#       ImportError — the param didn't exist in config.py yet).
# ============================================================

# -----------------------------------------------------------
# DATA
# -----------------------------------------------------------
# ASSET UNIVERSE: 10 European Energy assets
#   Shell (SHEL), BP, TotalEnergies (TTE), Enel (ENEL.MI),
#   Iberdrola (IBE.MC), RWE (RWE.DE), E.ON (EOAN.DE),
#   Equinor (EQNR), Ørsted (ORSTED.CO), Repsol (REP.MC)
TICKERS = [
    "SHEL", "BP", "TTE", "ENEL.MI",
    "IBE.MC", "RWE.DE", "EOAN.DE",
    "EQNR", "ORSTED.CO", "REP.MC"
]

# Paper dates (slide 16 of BenchmarkPaper.pptx):
#   Training: Nov 7 2006 – Nov 6 2019
#   Testing:  Nov 7 2019 – Nov 6 2021
#
# FIX: TRAIN_START moved from 2006-11-07 → 2016-06-15.
# Root cause: Ørsted (ORSTED.CO, formerly DONG Energy) only IPO'd on
# Nasdaq Copenhagen on 9 June 2016 — Yahoo Finance has no price data
# for it before that date. download_data() in data.py does a joint
# `raw.dropna()` across all 10 tickers, so any row missing Ørsted gets
# dropped for every ticker — collapsing the entire usable date range
# to mid-2016 onward regardless of what TRAIN_START says. Setting
# TRAIN_START explicitly to just after the IPO (with a ~1-week buffer
# past the 9 Jun completion date for the market to settle) makes the
# actual training window honest instead of silently truncated.
# TEST_START/TEST_END are untouched — both already fall after 2016,
# so they were never affected by this.
#
# FIX (this pass): added VAL_START/VAL_END. data.py now does a 3-way
# train/val/test split (validation used for model-selection/checkpoint
# decisions, keeping the test set untouched until final evaluation —
# see train.py's _validation_pass()). With TRAIN_START now at 2016,
# the total usable range is only ~5.4 years, so validation is carved
# out as the one year immediately before TEST_START rather than the
# paper's original 2-year pullback (that assumed a ~13-year training
# set; here it would leave almost nothing for training). This still
# gives ~2.4 years for training (comfortably more than EPISODE_LENGTH
# requires) and a genuine 1-year held-out validation window.
TRAIN_START = "2016-06-15"
TRAIN_END   = "2018-11-06"
VAL_START   = "2018-11-07"
VAL_END     = "2019-11-06"
TEST_START  = "2019-11-07"
TEST_END    = "2021-11-06"

# Paper: "5 parameters: Open, Close, High, Low price and the volume"
# Paper: "10 sequential historical date is utilized as environment"
FEATURES    = ["Open", "High", "Low", "Close", "Volume"]
WINDOW_SIZE = 10        # Paper: "10 sequential historical dates"

# -----------------------------------------------------------
# ENVIRONMENT
# -----------------------------------------------------------
INITIAL_CAPITAL    = 1_000_000
# FIX 5: re-enabled. Paper itself doesn't model transaction costs, but
# a frictionless environment let DQN-Sharpe reallocate aggressively at
# zero cost, which likely fed the unstable/overfit training returns
# seen in results_summary.txt. 0.0005 (5 bps) is a light, standard
# retail/institutional-equity assumption. Cost logic now also lives in
# simulate_mvo()/simulate_naive() in mvo.py so the benchmarks pay the
# same friction and the comparison stays apples-to-apples.
TRANSACTION_FEE    = 0.0005
RISK_FREE_RATE_ANN = 0.0117     # Paper: "average RFR will be 1.17%"
ALLOW_SHORT        = False      # Paper: "short strategy is not considered"

# -----------------------------------------------------------
# DQN ARCHITECTURE
# -----------------------------------------------------------
HIDDEN_LAYERS = [512, 256, 128]
ACTIVATION    = "relu"
DROPOUT_RATE  = 0.05

USE_DUELING   = False           # Not described in paper

# -----------------------------------------------------------
# DQN TRAINING
# -----------------------------------------------------------
LEARNING_RATE   = 3e-4
GAMMA           = 0.95
BATCH_SIZE      = 128
REPLAY_CAPACITY = 100_000
TARGET_UPDATE   = 100

# FIX 3: Reduced from 1200 → 800.
# EPISODE_LENGTH doubled to 504, so total gradient steps are roughly
# preserved: 800 × (504/4) = 100 800 vs 1200 × (252/4) = 75 600.
# More steps per episode also means a fuller replay buffer earlier.
NUM_EPISODES    = 800

MAX_STEPS       = None

# FIX 1 (original): doubled from 252 → 504 (2 trading years) to match
# the test horizon.
#
# FIX 7 (this pass): 504 → 126 (~half a trading year).
# TRAIN_START was moved to 2016-06-15 (see DATA section) because the
# joint dataset was being silently truncated to Ørsted's 2016 IPO date
# regardless of what TRAIN_START said, and TRAIN_END was pulled back
# further to 2018-11-06 to make room for a 1-year VAL split. That
# leaves only ~2.4 years of real training data (vs. the ~13 years the
# paper's 2006 start assumed) — nowhere near enough to sample random,
# non-overlapping 504-day (2-year) episodes from. 126 leaves a
# reasonable number of distinct start points for episode diversity
# while still giving the agent several months of contiguous market
# behavior per episode.
#
# IMPORTANT: run `python data.py` first and check the printed
# "Train states" count. If it's still <= EPISODE_LENGTH, lower this
# further (e.g. to 63, one trading quarter) until
# Train states > EPISODE_LENGTH + 1.
EPISODE_LENGTH  = 126

UPDATE_EVERY    = 4

# -----------------------------------------------------------
# PRIORITISED EXPERIENCE REPLAY (PER)
# -----------------------------------------------------------
USE_PER      = False            # Not described in paper
PER_ALPHA    = 0.6
PER_BETA_0   = 0.4
PER_BETA_END = 1.0
PER_EPS      = 1e-6

# -----------------------------------------------------------
# LR SCHEDULER
# -----------------------------------------------------------
# FIX 4 (original): adjusted for EPISODE_LENGTH=504.
# FIX 7 (this pass): recomputed for EPISODE_LENGTH=126.
# 800 episodes × (126 steps / 4) = 25 200 gradient steps
LR_T_MAX   = 25_200
LR_ETA_MIN = 5e-5

# -----------------------------------------------------------
# EXPLORATION (ε-greedy)
# -----------------------------------------------------------
# EPS_DECAY kept at 0.9930 — with 800 episodes ε still reaches 0.05
# by episode ~418, giving ~382 greedy episodes to consolidate.
# Formula: 1.0 × 0.9930^418 ≈ 0.05 ✓
EPS_START = 1.0
EPS_END   = 0.05
EPS_DECAY = 0.9930

# -----------------------------------------------------------
# ACTION SPACE
# -----------------------------------------------------------
NUM_WEIGHT_LEVELS  = 3
NUM_RANDOM_ACTIONS = 50

# -----------------------------------------------------------
# AGENT SELECTION  ←  CHOOSE YOUR AGENT(S) HERE
# -----------------------------------------------------------
#
#   Set AGENT to one of three options:
#
#     "sharpe"  →  train/evaluate DQN-S only  (Sharpe ratio reward)
#     "return"  →  train/evaluate DQN-R only  (daily return reward)
#     "both"    →  train/evaluate DQN-S and DQN-R
#
AGENT = "sharpe" #"both"   # ← change this line only

# Derived — do not edit below this line
if AGENT == "sharpe":
    REWARD_TYPES = ["sharpe"]
elif AGENT == "return":
    REWARD_TYPES = ["return"]
elif AGENT == "both":
    REWARD_TYPES = ["sharpe", "return"]
else:
    raise ValueError(f"AGENT must be 'sharpe', 'return', or 'both'. Got: '{AGENT}'")

REWARD_TYPE = REWARD_TYPES[0]   # fallback used by single-agent entry-points

USE_DIFFERENTIAL_SHARPE = False
DIFF_SHARPE_ETA         = 0.01
DIFF_SHARPE_SCALE       = 5.0

# Rolling-window Sharpe reward (DQN-S):
SHARPE_WINDOW       = 20
SHARPE_REWARD_SCALE = 1.0

# FIX 2: New parameter — scale factor for DQN-Return reward.
# Root cause of gradient collapse: raw daily returns average ~0.001
# (0.1%), producing Q-value targets so small that gradients vanish
# through 3 hidden layers (grad norms ~0.003 vs Sharpe's ~0.39,
# ~130× smaller). Scaling by 50 brings return rewards to the same
# magnitude as the Sharpe reward signal (~0.05–0.15 per step) and
# restores meaningful gradient flow without changing the policy's
# preference ordering (monotone transformation).
RETURN_REWARD_SCALE = 50.0

# -----------------------------------------------------------
# REWARD SHAPING  (NOT in paper — disabled)
# -----------------------------------------------------------
REWARD_SHAPING = False
SHAPING_SCALE  = 0.3

# -----------------------------------------------------------
# CONCENTRATION PENALTY  (NOT in paper — enabled as a fix)
# -----------------------------------------------------------
# FIX 6: enabled. Training returns were extremely unstable (std 554%,
# max single-episode return 2714%), consistent with the agent piling
# into a small number of assets rather than diversifying. This penalty
# subtracts a scaled Herfindahl-Hirschman Index excess (concentration
# above equal-weight) from the reward, discouraging that behavior.
CONCENTRATION_PENALTY       = True
CONCENTRATION_PENALTY_SCALE = 0.05

# -----------------------------------------------------------
# REWARD CLIPPING  (NOT in paper — disabled)
# -----------------------------------------------------------
REWARD_CLIP     = False
REWARD_CLIP_MIN = -0.10
REWARD_CLIP_MAX =  0.10

# -----------------------------------------------------------
# STATE NORMALISATION
# -----------------------------------------------------------
STATE_CLIP        = 0.10
VOLUME_SCALE      = 1e-6
VOLUME_STD_GLOBAL = None          # set by load_split_data() at runtime

# Engineered features (NOT used — paper has only 5 OHLCV features)
MOM_WINDOWS  = []
VOL_WINDOW   = 20

# -----------------------------------------------------------
# MVO BENCHMARK
# -----------------------------------------------------------
MVO_REBALANCE_FREQ = "monthly"

# -----------------------------------------------------------
# LOGGING & CHECKPOINTS
# -----------------------------------------------------------
SEED           = 42
LOG_INTERVAL   = 10
# NEW: required by train.py's _validation_pass() / validation-based
# checkpointing (added there to select models on VAL performance
# instead of peeking at the test set). Every VALIDATION_INTERVAL
# episodes, a greedy pass runs over the VAL split and the checkpoint
# is kept only if it beats the best validation Sharpe seen so far.
# 20 gives 40 validation passes across NUM_EPISODES=800 — frequent
# enough to track progress without adding significant overhead (each
# pass is a single greedy sweep over ~250 VAL days, far cheaper than
# an episode of training with its gradient updates).
VALIDATION_INTERVAL = 20
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"
