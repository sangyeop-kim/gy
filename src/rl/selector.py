"""Connect a DQN agent to Simulator dispatch decisions.

Reward & training schedule are deliberately simple:
- Reward is computed once per decision, when the lot finishes its current step
  (`on_lot_arrival`) or completes the route (`on_lot_complete`). The simulator's
  asynchronous tool-level decisions don't have a clean "next state", so we treat
  this as a contextual bandit.
- We do *not* call train_step on every reward (the previous version did, which
  saturated GPU sync overhead). Instead we count rewards and call train_step every
  `train_every` rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.rl.dqn import DQNAgent
from src.rl.features import DispatchFeatureEncoder
from src.simulator.engine import MINUTES_PER_DAY
from src.simulator.model import Lot


@dataclass
class PendingDecision:
    features: np.ndarray
    decision_time: float
    due_time: float | None
    remaining_before: float
    projected_tardiness_before: float
    cqt_violations_before: int


class RLDispatchSelector:
    """Simple, fast bridge between DQN agent and Simulator."""

    def __init__(
        self,
        agent: DQNAgent,
        encoder: DispatchFeatureEncoder | None = None,
        *,
        explore: bool = True,
        train_online: bool = True,
        train_every: int = 64,
        fallback_probability: float = 0.0,
    ) -> None:
        self.agent = agent
        self.encoder = encoder or DispatchFeatureEncoder()
        self.explore = explore
        self.train_online = train_online
        self.train_every = max(1, int(train_every))
        self.fallback_probability = fallback_probability
        self.pending: dict[int, PendingDecision] = {}
        self.decisions = 0
        self.rewards = 0
        self._train_counter = 0
        self.losses: list[float] = []
        self.reward_components: list[dict[str, float]] = []

    def select_lot(self, *, simulator, toolgroup, candidates: tuple[Lot, ...], fallback_policy, tool=None) -> Lot:
        if not candidates:
            raise ValueError("RLDispatchSelector received no candidates")

        if self.fallback_probability > 0.0 and self.agent.py_rng.random() < self.fallback_probability:
            selected = fallback_policy.select_lot(candidates, simulator.now, tool)
            selected_index = candidates.index(selected)
            features = self.encoder.encode_candidates(simulator, toolgroup, candidates, tool)
            chosen_features = features[selected_index]
        else:
            features = self.encoder.encode_candidates(simulator, toolgroup, candidates, tool)
            selected_index = self.agent.act(features, explore=self.explore)
            selected = candidates[selected_index]
            chosen_features = features[selected_index]

        remaining = selected.route.remaining_nominal_minutes(selected.step_index)
        self.pending[selected.id] = PendingDecision(
            features=chosen_features,
            decision_time=simulator.now,
            due_time=selected.due_time,
            remaining_before=remaining,
            projected_tardiness_before=self._projected_tardiness(simulator.now, remaining, selected.due_time),
            cqt_violations_before=int(simulator.cqt_counts_by_lot[selected.id].get("violations", 0)),
        )
        self.decisions += 1
        return selected

    # ---- reward hooks (called by Simulator) ----
    def on_lot_arrival(self, *, simulator, lot: Lot, now: float) -> None:
        decision = self.pending.pop(lot.id, None)
        if decision is not None:
            self._commit_reward(simulator, lot, decision, now, terminal=False)

    def on_lot_complete(self, *, simulator, lot: Lot, now: float) -> None:
        decision = self.pending.pop(lot.id, None)
        if decision is not None:
            self._commit_reward(simulator, lot, decision, now, terminal=True)

    # ---- internals ----
    def _commit_reward(self, simulator, lot: Lot, decision: PendingDecision, now: float, terminal: bool) -> None:
        reward, components = self._reward(simulator, lot, decision, now, terminal)
        self.agent.remember(decision.features, reward)
        self.reward_components.append(components)
        self.rewards += 1
        if self.train_online:
            self._train_counter += 1
            if self._train_counter >= self.train_every:
                self._train_counter = 0
                loss = self.agent.train_step()
                if loss is not None:
                    self.losses.append(loss)

    def _reward(self, simulator, lot: Lot, decision: PendingDecision, now: float, terminal: bool) -> tuple[float, dict[str, float]]:
        """Single scalar reward; cheap to compute (no O(n^2) candidate comparison)."""
        remaining_after = 0.0 if terminal else lot.route.remaining_nominal_minutes(lot.step_index)
        proj_after = self._projected_tardiness(now, remaining_after, decision.due_time)

        # Tardiness improvement, in days. Positive when we reduced projected lateness.
        tardiness_improvement_days = (decision.projected_tardiness_before - proj_after) / MINUTES_PER_DAY

        # New CQT violation since this decision.
        cqt_violations_after = int(simulator.cqt_counts_by_lot[lot.id].get("violations", 0))
        cqt_new_violations = max(0, cqt_violations_after - decision.cqt_violations_before)

        # Priority weight: super-hot 2x, hot 1.5x, regular 1x
        priority_weight = 1.0
        if lot.priority >= 30 or lot.super_hot_lot:
            priority_weight = 2.0
        elif lot.priority >= 20:
            priority_weight = 1.5

        terminal_bonus = 0.0
        terminal_late_penalty = 0.0
        if terminal and decision.due_time is not None:
            tardy_days = max(0.0, (now - decision.due_time) / MINUTES_PER_DAY)
            if tardy_days <= 0.0:
                terminal_bonus = 1.0 * priority_weight
            else:
                terminal_late_penalty = priority_weight * min(5.0, tardy_days)

        reward = (
            priority_weight * tardiness_improvement_days
            - 3.0 * cqt_new_violations
            + terminal_bonus
            - terminal_late_penalty
        )
        components = {
            "reward": float(reward),
            "tardiness_improvement_days": float(tardiness_improvement_days),
            "cqt_new_violations": float(cqt_new_violations),
            "terminal_bonus": float(terminal_bonus),
            "terminal_late_penalty": float(terminal_late_penalty),
        }
        return float(reward), components

    @staticmethod
    def _projected_tardiness(now: float, remaining: float, due_time: float | None) -> float:
        if due_time is None:
            return 0.0
        return max(0.0, now + remaining - due_time)
