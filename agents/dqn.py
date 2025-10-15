"""
DQN hoàn chỉnh (sumo-rl compatible + CUDA-ready)

Tính năng:
- Cấu hình trung tâm (DQNConfig)
- ReplayBuffer (seeded)
- QNetwork linh hoạt (hidden_dims)
- EpsilonGreedy exploration helper
- DQNAgent tương thích với sumo-rl (act/learn/save/load)
- Huber loss (Smooth L1), gradient clipping và Polyak (tau) target update
- Thiết bị tự động: dùng CUDA nếu có

Sử dụng: copy file này vào project và import DQNAgent.
Mỗi nút (node) có thể khởi tạo một DQNAgent riêng để mở rộng sang multi-agent.
"""

from dataclasses import dataclass
from collections import deque
from typing import Any, Optional, Sequence, Tuple, Dict
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
    hidden_dims: Tuple[int, ...] = (256, 256)
    device: Optional[str] = None  # if None, auto-detect
    seed: int = 0


class ReplayBuffer:
    """Replay buffer storing transitions and returning minibatches.

    Stored element: (state, action, reward, next_state, done)
    States are stored as raw numpy arrays (not tensors) to keep memory lean.
    """

    def __init__(self, capacity: int = 100_000, seed: int = 0):
        self.capacity = int(capacity)
        self.buf = deque(maxlen=self.capacity)
        random.seed(seed)

    def push(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        self.buf.append((np.array(state, dtype=np.float32), int(action), float(reward), np.array(next_state, dtype=np.float32), bool(done)))

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


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int] = (256, 256)):
        super().__init__()
        layers = []
        in_dim = int(state_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.ReLU())
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, int(action_dim)))
        self.net = nn.Sequential(*layers)

        # init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EpsilonGreedy:
    """Simple epsilon schedule helper.

    Use .get_epsilon(step) to query epsilon (default exponential decay).
    """

    def __init__(self, eps_start=1.0, eps_end=0.05, eps_decay=1e-4):
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay = float(eps_decay)

    def get_epsilon(self, step: int) -> float:
        # exponential: eps = eps_end + (eps_start - eps_end) * exp(-decay * step)
        return float(self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.eps_decay * step))


class DQNAgent:
    """DQN agent compatible with sumo-rl style interface.

    Key methods:
        - act() -> action (int)
        - learn(next_state, reward, done=False) -> perform a learning step
        - save(path) / load(path)

    Notes:
        - The agent keeps an internal .state (current observation). Set .state externally on reset if needed.
        - This implementation assumes continuous observation vectors (numpy arrays or lists). For dict/state spaces, user
          should pre-process to flattened vectors before passing to DQNAgent.
    """

    def __init__(
        self,
        starting_state: Any,
        state_space: Any,
        action_space: Any,
        config: Optional[DQNConfig] = None,
        exploration: Optional[EpsilonGreedy] = None,
    ):
        # config
        self.cfg = config or DQNConfig()
        if self.cfg.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)

        # seeds
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)

        # state / action dims
        self.state = self._to_state_np(starting_state)
        self.state_dim = int(np.prod(self.state.shape))
        # action space assumed discrete
        if hasattr(action_space, 'n'):
            self.n_actions = int(action_space.n)
        elif isinstance(action_space, int):
            self.n_actions = int(action_space)
        else:
            raise ValueError('action_space must have .n or be an int')

        # networks
        self.q_net = QNetwork(self.state_dim, self.n_actions, hidden_dims=self.cfg.hidden_dims).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.n_actions, hidden_dims=self.cfg.hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # optimizer & loss
        self.opt = optim.Adam(self.q_net.parameters(), lr=self.cfg.lr)
        # we'll use functional smooth_l1

        # replay
        self.replay = ReplayBuffer(self.cfg.buffer_size, seed=self.cfg.seed)

        # exploration
        self.exploration = exploration or EpsilonGreedy()

        # bookkeeping
        self.step_count = 0
        self.learn_steps = 0
        self.last_action: Optional[int] = None

    # -------------------- helper --------------------
    def _to_state_np(self, state: Any) -> np.ndarray:
        if isinstance(state, np.ndarray):
            arr = state.astype(np.float32, copy=False)
        elif isinstance(state, (list, tuple)):
            arr = np.asarray(state, dtype=np.float32).reshape(-1)
        elif isinstance(state, (int, float)):
            arr = np.array([float(state)], dtype=np.float32)
        elif state is None:
            arr = np.zeros((self.state_dim,), dtype=np.float32) if hasattr(self, 'state_dim') else np.zeros((1,), dtype=np.float32)
        else:
            # try convert
            arr = np.asarray(state, dtype=np.float32).reshape(-1)
        return arr

    def _to_tensor(self, state_np: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(state_np.astype(np.float32)).view(1, -1).to(self.device)
        return t

    # -------------------- public API --------------------
    def act(self) -> int:
        """Chọn action (epsilon-greedy)."""
        self.step_count += 1
        eps = self.exploration.get_epsilon(self.step_count)
        if random.random() < eps:
            a = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                s_t = self._to_tensor(self.state)
                q = self.q_net(s_t)
                a = int(torch.argmax(q, dim=1).item())
        self.last_action = a
        return a

    def store_transition(self, state, action, reward, next_state, done: bool):
        self.replay.push(state, action, reward, next_state, done)

    def learn(self, next_state: Any, reward: float, done: bool = False):
        """Lưu transition và thực hiện 1 step huấn luyện (nếu đủ dữ liệu).

        Ghi chú: giữ API giống sumo-rl: learn(next_state, reward, done)
        """
        # push transition (use last_action; if None, choose random)
        a = self.last_action if self.last_action is not None else random.randrange(self.n_actions)
        self.replay.push(self.state, a, reward, next_state, done)

        # advance state
        self.state = self._to_state_np(next_state)

        # training condition
        if len(self.replay) < self.cfg.start_learning_after:
            return
        if self.step_count % self.cfg.train_freq != 0:
            return

        # sample
        s, a_batch, r, s2, d = self.replay.sample(self.cfg.batch_size)

        # to tensors
        s_t = torch.from_numpy(s).float().to(self.device)
        a_t = torch.from_numpy(a_batch).long().to(self.device)
        r_t = torch.from_numpy(r).float().to(self.device)
        s2_t = torch.from_numpy(s2).float().to(self.device)
        d_t = torch.from_numpy(d).float().to(self.device)

        # current Q for taken actions
        q_values = self.q_net(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)

        # compute targets using target network
        with torch.no_grad():
            q_next = self.target_net(s2_t).max(1)[0]
            target = r_t + self.cfg.gamma * q_next * (1.0 - d_t)

        # loss (smooth l1)
        loss = F.smooth_l1_loss(q_values, target)

        # optimize
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.grad_clip_norm)
        self.opt.step()

        # target update
        self.learn_steps += 1
        if self.learn_steps % self.cfg.target_update_freq == 0:
            if self.cfg.tau >= 1.0:
                # hard update
                self.target_net.load_state_dict(self.q_net.state_dict())
            else:
                # soft update (Polyak)
                with torch.no_grad():
                    for p_t, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                        p_t.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

    def save(self, path: str):
        ckpt = {
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'opt': self.opt.state_dict(),
            'cfg': self.cfg,
            'step_count': self.step_count,
            'learn_steps': self.learn_steps,
        }
        torch.save(ckpt, path)

    def load(self, path: str, map_location: Optional[str] = None):
        map_loc = map_location or (self.device)
        ckpt = torch.load(path, map_location=map_loc)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_net.load_state_dict(ckpt.get('target_net', ckpt['q_net']))
        if 'opt' in ckpt:
            try:
                self.opt.load_state_dict(ckpt['opt'])
            except Exception:
                # optimizer state may be incompatible across devices/different net shapes
                pass
        # restore counters if present
        self.step_count = ckpt.get('step_count', self.step_count)
        self.learn_steps = ckpt.get('learn_steps', self.learn_steps)

    # helper to set state externally (useful on env.reset())
    def set_state(self, state: Any):
        self.state = self._to_state_np(state)


# If run as script, provide a tiny smoke test (CPU-only) to ensure no syntax/runtime errors
if __name__ == '__main__':
    class DummySpace:
        def __init__(self, n=None, shape=None):
            self.n = n
            self.shape = shape

    # tiny smoke test
    state0 = np.zeros(4)
    action_space = DummySpace(n=2)
    agent = DQNAgent(starting_state=state0, state_space=None, action_space=action_space)
    for i in range(10):
        a = agent.act()
        next_s = state0 + np.random.randn(4) * 0.01
        agent.learn(next_s, reward=0.0, done=False)
    print('smoke test done')
