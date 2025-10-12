
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    raise ImportError("PyTorch is required for DQNAgent. Install torch first.") from e


def _infer_n_actions(action_space: Any) -> int:
    # Works with gymnasium spaces or dict of spaces
    if hasattr(action_space, "n"):
        return int(action_space.n)
    if isinstance(action_space, dict):
        # pick the first space, assume homogeneous action spaces
        first = next(iter(action_space.values()))
        if hasattr(first, "n"):
            return int(first.n)
    raise ValueError("Cannot infer number of actions from action_space")


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100_000
    start_learning_after: int = 1_000
    train_freq: int = 1
    target_update_freq: int = 1_000
    tau: float = 1.0  # 1.0 -> hard update
    grad_clip_norm: Optional[float] = 10.0
    hidden_sizes: tuple = (256, 256)
    device: str = "cuda" if (torch.cuda.is_available()) else "cpu"
    seed: int = 0


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_sizes=(256, 256)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, n_actions)

        # Init (Kaiming for ReLU)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.buf = deque(maxlen=capacity)
        random.seed(seed)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.stack(s),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    """
    CUDA-enabled DQN agent with the same public API used in main.py:
      - __init__(starting_state, state_space, action_space, alpha, gamma, exploration_strategy)
      - act() -> int
      - learn(next_state, reward) -> None
      - attribute: state (current state, updated externally on reset)
    Notes:
      * We ignore `alpha` (kept for signature compatibility). Use optimizer lr instead.
      * `exploration_strategy` is expected to be EpsilonGreedy-like. We fall back to
        typical exponential decay if methods are missing.
    """
    def __init__(
        self,
        starting_state: Any,
        state_space: Any,
        action_space: Any,
        alpha: float = 0.1,   # kept for compatibility, not used directly
        gamma: float = 0.99,
        exploration_strategy: Optional[Any] = None,
        config: Optional[DQNConfig] = None,
    ):
        self.state = self._to_state_vec(starting_state)
        self.state_dim = int(self.state.shape[-1])
        self.n_actions = _infer_n_actions(action_space)

        self.cfg = config or DQNConfig(gamma=gamma)
        self.device = torch.device(self.cfg.device)

        # RNG
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)

        # Networks
        self.online = QNetwork(self.state_dim, self.n_actions, hidden_sizes=self.cfg.hidden_sizes).to(self.device)
        self.target = QNetwork(self.state_dim, self.n_actions, hidden_sizes=self.cfg.hidden_sizes).to(self.device)
        self._hard_update()

        self.opt = torch.optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        self.replay = ReplayBuffer(self.cfg.buffer_size, seed=self.cfg.seed)

        # exploration
        self.exploration_strategy = exploration_strategy
        self._eps0 = getattr(exploration_strategy, "initial_epsilon", 1.0) if exploration_strategy else 1.0
        self._eps_min = getattr(exploration_strategy, "min_epsilon", 0.05) if exploration_strategy else 0.05
        self._decay = getattr(exploration_strategy, "decay", 1.0) if exploration_strategy else 0.9995

        self.step_count = 0
        self.last_action = None

    def _to_state_vec(self, state: Any) -> np.ndarray:
        if isinstance(state, (list, tuple)):
            arr = np.asarray(state, dtype=np.float32)
        elif isinstance(state, np.ndarray):
            arr = state.astype(np.float32, copy=False)
        elif isinstance(state, dict):
            # Flatten deterministic order
            arr = np.asarray([state[k] for k in sorted(state.keys())], dtype=np.float32).reshape(-1)
        else:
            arr = np.asarray(state, dtype=np.float32).reshape(-1)
        return arr.reshape(1, -1)  # 2D

    def _epsilon(self) -> float:
        # Try duck-typing: if strategy has callable epsilon(state) or value(t) etc.
        if self.exploration_strategy is not None:
            eps = max(self._eps_min, self._eps0 * (self._decay ** self.step_count))
            return float(eps)
        return max(self._eps_min, self._eps0 * (self._decay ** self.step_count))

    def _hard_update(self):
        self.target.load_state_dict(self.online.state_dict())

    @torch.no_grad()
    def _q_values(self, state_np: np.ndarray):
        s = torch.from_numpy(state_np).to(self.device)
        return self.online(s)

    def act(self) -> int:
        self.step_count += 1
        eps = self._epsilon()
        if random.random() < eps:
            a = random.randrange(self.n_actions)
        else:
            q = self._q_values(self.state)
            a = int(torch.argmax(q, dim=1).item())
        self.last_action = a
        return a

    def learn(self, next_state: Any, reward: float) -> None:
        # Store transition
        ns = self._to_state_vec(next_state)
        r = float(reward)
        d = 0.0  # Without terminal flag in main.py, treat as continuing task
        a = self.last_action if self.last_action is not None else random.randrange(self.n_actions)
        self.replay.push(self.state.squeeze(0), a, r, ns.squeeze(0), d)

        # Move to next
        self.state = ns

        # Train
        if len(self.replay) < self.cfg.start_learning_after:
            return
        if self.step_count % self.cfg.train_freq != 0:
            return

        s, a, r, s2, d = self.replay.sample(self.cfg.batch_size)

        s_t = torch.from_numpy(s).to(self.device)
        a_t = torch.from_numpy(a).to(self.device).long()
        r_t = torch.from_numpy(r).to(self.device)
        s2_t = torch.from_numpy(s2).to(self.device)
        d_t = torch.from_numpy(d).to(self.device)

        # Q(s,a)
        q = self.online(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)

        # target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            q_next = self.target(s2_t).max(1)[0]
            target = r_t + self.cfg.gamma * q_next * (1.0 - d_t)

        loss = F.smooth_l1_loss(q, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip_norm)
        self.opt.step()

        # target update
        if self.step_count % self.cfg.target_update_freq == 0:
            if self.cfg.tau >= 1.0:
                self._hard_update()
            else:
                # Polyak
                with torch.no_grad():
                    for p_targ, p in zip(self.target.parameters(), self.online.parameters()):
                        p_targ.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
