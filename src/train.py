# ============================================================
#  train.py  –  Train DQN-S and DQN-R agents
#
#  FIX SUMMARY (vs previous version):
#    1. Prefill buffer uses full EPISODE_LENGTH window (was capped
#       at 252 regardless of EPISODE_LENGTH setting).
#    2. Episode sub-window sampling accounts for the longer
#       EPISODE_LENGTH=504 so we never overflow train_states.
#    3. Added a contiguous end-of-training evaluation pass over
#       the last 504 days of the training set to verify the agent
#       hasn't just memorised random sub-windows.
# ============================================================
import os
import json
import numpy as np

from config import (
    NUM_EPISODES, LOG_INTERVAL, CHECKPOINT_DIR,
    RESULTS_DIR, REWARD_TYPE, REWARD_TYPES, SEED, EPISODE_LENGTH,
    UPDATE_EVERY, BATCH_SIZE
)
from data        import load_split_data
from environment import PortfolioEnv
from agent       import DQNAgent

np.random.seed(SEED)
rng = np.random.default_rng(SEED)


def _grad_norm(agent: DQNAgent) -> float:
    """Compute total L2 norm of online-network gradients for diagnostics."""
    total = 0.0
    for p in agent.online.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def train_agent(reward_type: str = REWARD_TYPE):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_states, _, train_closes, _ = load_split_data()
    T = len(train_states)

    if T < EPISODE_LENGTH + 1:
        raise ValueError(
            f"Training set ({T} steps) shorter than EPISODE_LENGTH ({EPISODE_LENGTH}). "
            f"Reduce EPISODE_LENGTH or extend TRAIN_END."
        )

    env   = PortfolioEnv(train_states, train_closes, reward_type=reward_type)
    agent = DQNAgent(env.state_shape, env.action_dim, reward_type=reward_type)

    history = {
        "episode_rewards": [],
        "episode_returns": [],
        "epsilon":         [],
        "grad_norms":      [],
    }

    # ── Pre-fill replay buffer ───────────────────────────────────────
    # FIX: use the correct EPISODE_LENGTH (was hard-coded to 252).
    # Pre-fill with 4× BATCH_SIZE random transitions from a rolling
    # window so the buffer is diverse from the first gradient update.
    prefill_len   = min(EPISODE_LENGTH, T)
    prefill_steps = min(BATCH_SIZE * 4, prefill_len - 1)
    print(f"  Pre-filling replay buffer ({prefill_steps} random steps) …")

    prefill_env = PortfolioEnv(
        train_states[:prefill_len],
        train_closes[:prefill_len],
        reward_type=reward_type
    )
    state = prefill_env.reset()
    import random as _random
    for _ in range(prefill_steps):
        action_idx      = _random.randrange(agent.n_actions)
        action_w        = agent.actions[action_idx]
        ns, r, done, _  = prefill_env.step(action_w)
        agent.store(state, action_idx, r, ns, done)
        state = ns if not done else prefill_env.reset()
    print(f"  Buffer size after pre-fill: {len(agent.buffer)}")

    # ── Training loop ────────────────────────────────────────────────
    for ep in range(1, NUM_EPISODES + 1):

        # FIX: ensure end_idx never overflows train_states.
        # With EPISODE_LENGTH=504, max_start must leave at least 504 steps.
        max_start = max(0, T - EPISODE_LENGTH - 1)
        start_idx = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        end_idx   = start_idx + EPISODE_LENGTH

        ep_states  = train_states[start_idx:end_idx]
        ep_closes  = train_closes[start_idx:end_idx]

        env_ep      = PortfolioEnv(ep_states, ep_closes, reward_type=reward_type)
        state       = env_ep.reset()
        ep_reward   = 0.0
        ep_pf_start = env_ep.portfolio_val
        done        = False
        step_in_ep  = 0
        ep_losses   = []

        while not done:
            action_weights, action_idx = agent.select_action(state)
            next_state, reward, done, info = env_ep.step(action_weights)

            agent.store(state, action_idx, reward, next_state, done)

            step_in_ep += 1
            if step_in_ep % UPDATE_EVERY == 0:
                loss = agent.update()
                if loss is not None:
                    ep_losses.append(loss)

            state      = next_state
            ep_reward += reward

        # One final update at episode end
        loss = agent.update()
        if loss is not None:
            ep_losses.append(loss)

        agent.decay_epsilon()

        ep_return = (env_ep.portfolio_val - ep_pf_start) / ep_pf_start
        ep_gnorm  = _grad_norm(agent)

        history["episode_rewards"].append(ep_reward)
        history["episode_returns"].append(ep_return)
        history["epsilon"].append(agent.epsilon)
        history["grad_norms"].append(ep_gnorm)

        if ep % LOG_INTERVAL == 0 or ep == 1:
            recent_rets  = history["episode_returns"][-10:]
            smooth_ret   = np.mean(recent_rets) * 100
            avg_loss_str = f"{np.mean(ep_losses):.4f}" if ep_losses else "N/A"
            print(f"Episode {ep:4d}/{NUM_EPISODES} | "
                  f"Reward: {ep_reward:8.4f} | "
                  f"Ep Return: {ep_return*100:6.2f}% | "
                  f"Smooth(10): {smooth_ret:6.2f}% | "
                  f"Loss: {avg_loss_str} | "
                  f"GradNorm: {ep_gnorm:.4f} | "
                  f"ε: {agent.epsilon:.4f}")

    # ── End-of-training sanity check ─────────────────────────────────
    # FIX: Run one greedy pass over the final EPISODE_LENGTH days of
    # training data (contiguous, not random). This is a quick sanity
    # check: if the agent scores negatively here it has overfit badly
    # to specific sub-windows and should be re-trained.
    print("\n  [Sanity] Greedy pass over last training window …")
    sanity_states = train_states[T - EPISODE_LENGTH:]
    sanity_closes = train_closes[T - EPISODE_LENGTH:]
    sanity_env    = PortfolioEnv(sanity_states, sanity_closes, reward_type=reward_type)
    saved_eps     = agent.epsilon
    agent.epsilon = 0.0
    s_state       = sanity_env.reset()
    s_done        = False
    while not s_done:
        aw, _ = agent.select_action(s_state)
        s_state, _, s_done, _ = sanity_env.step(aw)
    sanity_return = (sanity_env.portfolio_val - INITIAL_CAPITAL) / INITIAL_CAPITAL
    agent.epsilon = saved_eps
    print(f"  [Sanity] Return over last {EPISODE_LENGTH} training days: "
          f"{sanity_return*100:.2f}%")

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"dqn_{reward_type}.pt")
    agent.save(ckpt_path)

    hist_path = os.path.join(RESULTS_DIR, f"train_history_{reward_type}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f)
    print(f"\nTraining history saved → {hist_path}")

    return agent, history


# Import for sanity check
from config import INITIAL_CAPITAL

if __name__ == "__main__":
    for rt in REWARD_TYPES:
        print(f"\n{'#'*50}")
        print(f"  Training DQN-{rt.upper()}")
        print(f"{'#'*50}")
        train_agent(reward_type=rt)
