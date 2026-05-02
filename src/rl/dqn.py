from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import random
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing.
    raise ImportError(
        "PyTorch is required for RL DQN training. Install project dependencies with torch enabled."
    ) from exc


@dataclass(frozen=True)
class DQNConfig:
    input_dim: int
    hidden_dim: int = 128
    hidden_layers: int = 2
    learning_rate: float = 3e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9995
    replay_capacity: int = 100_000
    batch_size: int = 256
    seed: int = 42
    device: str = "auto"
    grad_clip_norm: float = 5.0


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
    """PyTorch candidate-scoring DQN.

    Each waiting lot is one discrete action. The network receives candidate features
    and predicts a scalar action value; argmax selects the lot.
    """

    def __init__(self, config: DQNConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.py_rng = random.Random(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.network = QNetwork(config.input_dim, config.hidden_dim, config.hidden_layers).to(self.device)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.epsilon = config.epsilon_start
        self.replay: deque[tuple[np.ndarray, float]] = deque(maxlen=config.replay_capacity)
        self.training_steps = 0

    def act(self, candidate_features: np.ndarray, explore: bool = True) -> int:
        if len(candidate_features) == 0:
            raise ValueError("DQNAgent.act requires at least one candidate")
        if explore and self.py_rng.random() < self.epsilon:
            return self.py_rng.randrange(len(candidate_features))
        q_values = self.predict(candidate_features)
        return int(np.argmax(q_values))

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            values = self.network(tensor)
        return values.detach().cpu().numpy()

    def remember(self, features: np.ndarray, reward: float) -> None:
        self.replay.append((features.astype(np.float32), float(reward)))

    def train_step(self) -> float | None:
        if len(self.replay) < self.config.batch_size:
            return None
        batch = self.py_rng.sample(list(self.replay), self.config.batch_size)
        x = torch.as_tensor(np.asarray([item[0] for item in batch]), dtype=torch.float32, device=self.device)
        target = torch.as_tensor([item[1] for item in batch], dtype=torch.float32, device=self.device)

        self.network.train()
        prediction = self.network(x)
        loss = self.loss_fn(prediction, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        self.training_steps += 1
        return float(loss.detach().cpu().item())

    def save(self, path: str) -> None:
        payload: dict[str, Any] = {
            "config": asdict(self.config),
            "model_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "DQNAgent":
        load_device = cls._resolve_device(device or "auto")
        payload = torch.load(path, map_location=load_device)
        config = DQNConfig(**payload["config"])
        if device is not None:
            config = DQNConfig(**{**asdict(config), "device": device})
        agent = cls(config)
        agent.network.load_state_dict(payload["model_state"])
        agent.optimizer.load_state_dict(payload["optimizer_state"])
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
