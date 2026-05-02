"""Contextual-bandit Q-network with GPU-resident replay and decoupled CPU inference.

Why this design (vs. the previous one):

- Previous code did `predict()` on GPU per dispatch and `train_step()` on GPU per reward.
  Each call paid CPU↔GPU transfer cost; for a small MLP that overhead dominates.
- Here we split the work:
    * Inference (`act`) runs on a CPU-resident copy of the network. Small MLP on CPU is
      faster than transferring tensors to GPU for a tiny forward pass.
    * Training (`train_step`) runs on the GPU model with a *preallocated GPU tensor pool*
      as replay buffer — no python list, no per-sample numpy→torch conversion.
    * The CPU inference network is synced from the GPU training network every
      `inference_sync_every` train steps.

The replay only stores `(features, scalar_reward)` because the simulator-side reward signal
is contextual-bandit-style (no Bellman bootstrap). This is intentional: the simulator's
asynchronous tool-level decisions don't have a clean "next state" for a value function,
and adding a target network would not improve learning here.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import random
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for RL DQN training. Install project dependencies with torch enabled."
    ) from exc


@dataclass(frozen=True)
class DQNConfig:
    input_dim: int
    hidden_dim: int = 64
    hidden_layers: int = 2
    learning_rate: float = 3e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9995
    replay_capacity: int = 100_000
    batch_size: int = 1024
    grad_clip_norm: float = 5.0
    seed: int = 42
    device: str = "auto"
    inference_device: str = "cpu"     # small MLP runs faster on CPU
    inference_sync_every: int = 200   # train-steps between weight syncs to CPU model


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(width, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DQNAgent:
    """Candidate-scoring contextual bandit. Action = argmax over candidate features."""

    def __init__(self, config: DQNConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.inference_device = self._resolve_device(config.inference_device)
        self.py_rng = random.Random(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Training network (GPU if available)
        self.network = QNetwork(config.input_dim, config.hidden_dim, config.hidden_layers).to(self.device)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        # Inference network (CPU copy). Small MLP → CPU inference avoids per-call transfer.
        self.inference_network = copy.deepcopy(self.network).to(self.inference_device)
        self.inference_network.eval()
        for p in self.inference_network.parameters():
            p.requires_grad_(False)

        # GPU-resident ring-buffer replay
        self._replay_features = torch.zeros(
            (config.replay_capacity, config.input_dim), dtype=torch.float32, device=self.device,
        )
        self._replay_rewards = torch.zeros(
            (config.replay_capacity,), dtype=torch.float32, device=self.device,
        )
        self._replay_pos = 0
        self._replay_full = False

        self.epsilon = config.epsilon_start
        self.training_steps = 0

    # ----- replay -----
    def remember(self, features: np.ndarray, reward: float) -> None:
        # features: (input_dim,) numpy. Convert once, write into preallocated GPU buffer.
        idx = self._replay_pos
        self._replay_features[idx].copy_(torch.from_numpy(features.astype(np.float32, copy=False)))
        self._replay_rewards[idx] = float(reward)
        self._replay_pos = (idx + 1) % self.config.replay_capacity
        if self._replay_pos == 0:
            self._replay_full = True

    def replay_size(self) -> int:
        return self.config.replay_capacity if self._replay_full else self._replay_pos

    # ----- act -----
    def act(self, candidate_features: np.ndarray, explore: bool = True) -> int:
        n = len(candidate_features)
        if n == 0:
            raise ValueError("DQNAgent.act requires at least one candidate")
        if explore and self.py_rng.random() < self.epsilon:
            return self.py_rng.randrange(n)
        with torch.no_grad():
            x = torch.from_numpy(candidate_features.astype(np.float32, copy=False))
            if x.device != self.inference_device:
                x = x.to(self.inference_device, non_blocking=True)
            q = self.inference_network(x)
            return int(torch.argmax(q).item())

    # ----- train -----
    def train_step(self) -> float | None:
        size = self.replay_size()
        if size < self.config.batch_size:
            return None
        idx = torch.randint(0, size, (self.config.batch_size,), device=self.device)
        x = self._replay_features.index_select(0, idx)
        target = self._replay_rewards.index_select(0, idx)

        self.network.train()
        prediction = self.network(x)
        loss = self.loss_fn(prediction, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.config.inference_sync_every == 0:
            self._sync_inference_network()
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        return float(loss.detach().item())

    def _sync_inference_network(self) -> None:
        # CPU copy of training weights for inference
        cpu_state = {k: v.detach().to(self.inference_device) for k, v in self.network.state_dict().items()}
        self.inference_network.load_state_dict(cpu_state)
        self.inference_network.eval()

    # ----- save / load -----
    def save(self, path: str) -> None:
        payload: dict[str, Any] = {
            "config": asdict(self.config),
            "model_state": {k: v.cpu() for k, v in self.network.state_dict().items()},
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "DQNAgent":
        payload = torch.load(path, map_location="cpu")
        config_dict = dict(payload["config"])
        if device is not None:
            config_dict["device"] = device
        config = DQNConfig(**config_dict)
        agent = cls(config)
        agent.network.load_state_dict({k: v.to(agent.device) for k, v in payload["model_state"].items()})
        agent._sync_inference_network()
        agent.epsilon = payload["epsilon"]
        agent.training_steps = payload["training_steps"]
        return agent

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # back-compat alias for older callers
    @property
    def replay(self) -> list:
        return [None] * self.replay_size()
