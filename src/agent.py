# ============================================================
#  agent.py  –  Deep Q-Network Agent
#  Aligned with paper:
#    - Standard (non-dueling) Q-network
#    - Uniform experience replay (no PER)
#    - Double-DQN target (standard DQN improvement, kept)
#
#  FIX: torch.load now passes weights_only=True to suppress the
#  FutureWarning in PyTorch ≥2.0 and avoid arbitrary code execution
#  risk when loading checkpoints from untrusted sources.
# ============================================================
import os
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    HIDDEN_LAYERS, LEARNING_RATE, GAMMA,
    BATCH_SIZE, REPLAY_CAPACITY, TARGET_UPDATE,
    EPS_START, EPS_END, EPS_DECAY,
    NUM_WEIGHT_LEVELS, NUM_RANDOM_ACTIONS, SEED, CHECKPOINT_DIR,
    DROPOUT_RATE, LR_T_MAX, LR_ETA_MIN,
    USE_DUELING, USE_PER,
    PER_ALPHA, PER_BETA_0, PER_BETA_END, PER_EPS,
    NUM_EPISODES
)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ──────────────────────────────────────────────────────────────────────
#  Q-Network
# ──────────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """
    Standard fully-connected Q-network (paper-compliant when USE_DUELING=False).

    Optional dueling architecture (Wang et al. 2016) available via
    USE_DUELING=True in config:
        Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

        trunk_layers = []
        in_dim = state_dim
        for i, h in enumerate(HIDDEN_LAYERS):
            trunk_layers.append(nn.Linear(in_dim, h))
            if i == 0:
                trunk_layers.append(nn.LayerNorm(h))
            trunk_layers.append(nn.ReLU())
            if DROPOUT_RATE > 0:
                trunk_layers.append(nn.Dropout(p=DROPOUT_RATE))
            in_dim = h
        self.trunk = nn.Sequential(*trunk_layers)

        if USE_DUELING:
            self.value_head = nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 1)
            )
            self.adv_head = nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, action_dim)
            )
            nn.init.uniform_(self.value_head[-1].weight, -3e-3, 3e-3)
            nn.init.zeros_(self.value_head[-1].bias)
            nn.init.uniform_(self.adv_head[-1].weight, -3e-3, 3e-3)
            nn.init.zeros_(self.adv_head[-1].bias)
        else:
            self.q_head = nn.Linear(in_dim, action_dim)
            nn.init.uniform_(self.q_head.weight, -3e-3, 3e-3)
            nn.init.zeros_(self.q_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        if USE_DUELING:
            v   = self.value_head(h)
            adv = self.adv_head(h)
            return v + adv - adv.mean(dim=1, keepdim=True)
        else:
            return self.q_head(h)


# ──────────────────────────────────────────────────────────────────────
#  Prioritised Experience Replay (optional, not used by paper)
# ──────────────────────────────────────────────────────────────────────
class SumTree:
    """Binary sum-tree for O(log N) priority sampling (USE_PER=True only)."""

    def __init__(self, capacity: int):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data      = np.empty(capacity, dtype=object)
        self.write     = 0
        self.n_entries = 0

    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if self.tree[left] == 0 and self.tree[right] == 0:
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write     = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def get(self, s: float):
        idx      = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritisedReplayBuffer:
    """Replay buffer backed by a SumTree (USE_PER=True only)."""

    def __init__(self, capacity: int = REPLAY_CAPACITY):
        self.tree     = SumTree(capacity)
        self.capacity = capacity
        self.alpha    = PER_ALPHA
        self._max_p   = 1.0

    def push(self, state, action_idx, reward, next_state, done):
        priority = self._max_p ** self.alpha
        self.tree.add(priority, (state, action_idx, reward, next_state, done))

    def sample(self, batch_size: int, beta: float):
        total   = self.tree.total
        segment = total / batch_size
        indices, weights, samples = [], [], []

        min_prob = (self.tree.tree[self.tree.capacity - 1 + 0] + PER_EPS) / (total + 1e-8)
        min_prob = max(min_prob, 1e-8)

        for i in range(batch_size):
            s   = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            if data is None:
                data = self.tree.data[0]
            prob = (p + PER_EPS) / (total + 1e-8)
            w    = (prob / min_prob) ** (-beta)
            indices.append(idx)
            weights.append(w)
            samples.append(data)

        max_w = max(weights) + 1e-8
        weights = np.array(weights, dtype=np.float32) / max_w

        s, a, r, ns, d = zip(*samples)
        return (
            (np.array(s,  dtype=np.float32),
             np.array(a,  dtype=np.int64),
             np.array(r,  dtype=np.float32),
             np.array(ns, dtype=np.float32),
             np.array(d,  dtype=np.float32)),
            weights, indices
        )

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            p = (abs(float(err)) + PER_EPS) ** self.alpha
            self.tree.update(idx, p)
            if p > self._max_p:
                self._max_p = p

    def __len__(self):
        return self.tree.n_entries


# ──────────────────────────────────────────────────────────────────────
#  Uniform Replay Buffer  (paper-compliant)
# ──────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    """Standard uniform experience replay buffer (paper-compliant)."""

    def __init__(self, capacity: int = REPLAY_CAPACITY):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buf.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.int64),
                np.array(r,  dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d,  dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# ──────────────────────────────────────────────────────────────────────
#  Discrete Action Space
# ──────────────────────────────────────────────────────────────────────
def build_action_space(n_stocks: int, n_levels: int = NUM_WEIGHT_LEVELS):
    rng = np.random.default_rng(SEED)

    ew = np.ones(n_stocks, dtype=np.float32) / n_stocks

    # Anchor: 100% in one stock
    single_stock = [np.eye(n_stocks, dtype=np.float32)[i] for i in range(n_stocks)]

    # Equal-weight anchor repeated 5× to boost exploration frequency
    ew_anchors = [ew.copy() for _ in range(5)]

    if n_stocks <= 5:
        import itertools
        levels  = np.linspace(0, 1, n_levels)
        combos  = list(itertools.product(levels, repeat=n_stocks))
        actions = []
        for c in combos:
            w = np.array(c, dtype=np.float32)
            s = w.sum()
            if s > 0:
                actions.append(w / s)
    else:
        random_actions = rng.dirichlet(
            np.full(n_stocks, 0.5), size=NUM_RANDOM_ACTIONS
        ).astype(np.float32)
        actions = list(random_actions)

    return single_stock + ew_anchors + actions


# ──────────────────────────────────────────────────────────────────────
#  DQN Agent
# ──────────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_shape: tuple, n_stocks: int, reward_type: str = "sharpe"):
        self.n_stocks    = n_stocks
        self.reward_type = reward_type
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actions   = build_action_space(n_stocks)
        self.n_actions = len(self.actions)

        state_dim = int(np.prod(state_shape))

        self.online = QNetwork(state_dim, self.n_actions).to(self.device)
        self.target = QNetwork(state_dim, self.n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(),
                                    lr=LEARNING_RATE, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=LR_T_MAX, eta_min=LR_ETA_MIN
        )
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        if USE_PER:
            self.buffer    = PrioritisedReplayBuffer()
            self._beta_inc = (PER_BETA_END - PER_BETA_0) / max(NUM_EPISODES, 1)
            self._beta     = PER_BETA_0
        else:
            self.buffer = ReplayBuffer()

        self.epsilon = EPS_START
        self.steps   = 0

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        arch_tag = "dueling" if USE_DUELING else "standard"
        buf_tag  = "PER"     if USE_PER     else "uniform"
        print(f"DQN-{reward_type.upper()} | actions={self.n_actions} | "
              f"state_dim={state_dim} | arch={arch_tag} | replay={buf_tag} | "
              f"device={self.device}")

    def select_action(self, state: np.ndarray) -> tuple:
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            idx = random.randrange(self.n_actions)
        else:
            self.online.eval()
            with torch.no_grad():
                s   = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
                idx = int(self.online(s).argmax(dim=1).item())
            self.online.train()
        return self.actions[idx], idx

    def store(self, state, action_idx, reward, next_state, done):
        self.buffer.push(
            state.flatten(), action_idx, reward,
            next_state.flatten(), float(done)
        )

    def update(self) -> float | None:
        if len(self.buffer) < BATCH_SIZE:
            return None

        if USE_PER:
            (s, a, r, ns, d), is_weights, tree_indices = self.buffer.sample(
                BATCH_SIZE, beta=self._beta
            )
            is_w = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)
        else:
            s, a, r, ns, d = self.buffer.sample(BATCH_SIZE)
            is_w           = None
            tree_indices   = None

        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r  = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d  = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        q_vals = self.online(s).gather(1, a)

        # Double-DQN: online net selects action, target net evaluates it
        with torch.no_grad():
            best_next_actions = self.online(ns).argmax(dim=1, keepdim=True)
            max_next_q        = self.target(ns).gather(1, best_next_actions)
            q_targets         = r + GAMMA * max_next_q * (1 - d)

        per_sample_loss = self.loss_fn(q_vals, q_targets)

        if USE_PER:
            loss = (is_w * per_sample_loss).mean()
            with torch.no_grad():
                td_errors = (q_targets - q_vals).abs().cpu().numpy().flatten()
            self.buffer.update_priorities(tree_indices, td_errors)
        else:
            loss = per_sample_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Epsilon decay after each episode."""
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)
        if USE_PER:
            self._beta = min(PER_BETA_END, self._beta + self._beta_inc)

    def save(self, path: str):
        ckpt = {
            "online":    self.online.state_dict(),
            "target":    self.target.state_dict(),
            "epsilon":   self.epsilon,
            "steps":     self.steps,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        if USE_PER:
            ckpt["per_beta"] = self._beta
        torch.save(ckpt, path)
        print(f"  Checkpoint saved → {path}")

    def load(self, path: str):
        # FIX: weights_only=True suppresses FutureWarning in PyTorch ≥2.0
        # and prevents arbitrary code execution from untrusted checkpoints.
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.epsilon = ckpt["epsilon"]
        self.steps   = ckpt["steps"]
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if USE_PER and "per_beta" in ckpt:
            self._beta = ckpt["per_beta"]
        print(f"  Checkpoint loaded ← {path}")
