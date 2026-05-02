from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.rl.dqn import DQNAgent
from src.rl.features import DispatchFeatureEncoder
from src.simulator.engine import MINUTES_PER_DAY
from src.simulator.model import Lot


@dataclass
class PendingDecision:
    features: np.ndarray
    decision_time: float
    wait_age: float
    due_time: float | None
    step_index: int
    remaining_before: float
    projected_tardiness_before: float
    cqt_remaining_before: float | None
    cqt_violations_before: int
    cqt_passes_before: int


class RLDispatchSelector:
    """Connect a DQN agent to Simulator dispatch decisions."""

    def __init__(
        self,
        agent: DQNAgent,
        encoder: DispatchFeatureEncoder | None = None,
        *,
        explore: bool = True,
        train_online: bool = True,
        fallback_probability: float = 0.0,
    ) -> None:
        self.agent = agent
        self.encoder = encoder or DispatchFeatureEncoder()
        self.explore = explore
        self.train_online = train_online
        self.fallback_probability = fallback_probability
        self.pending: dict[int, PendingDecision] = {}
        self.decisions = 0
        self.rewards = 0
        self.losses: list[float] = []
        self.reward_components: list[dict[str, float]] = []

    def select_lot(self, *, simulator, toolgroup, candidates: tuple[Lot, ...], fallback_policy, tool=None) -> Lot:
        if not candidates:
            raise ValueError("RLDispatchSelector received no candidates")
        if self.fallback_probability > 0.0 and self.agent.py_rng.random() < self.fallback_probability:
            selected = fallback_policy.select_lot(candidates, simulator.now, tool)
            selected_index = candidates.index(selected)
            features = self.encoder.encode_candidates(simulator, toolgroup, candidates, tool)[selected_index]
        else:
            matrix = self.encoder.encode_candidates(simulator, toolgroup, candidates, tool)
            selected_index = self.agent.act(matrix, explore=self.explore)
            selected = candidates[selected_index]
            features = matrix[selected_index]
        wait_since = selected.waiting_since if selected.waiting_since is not None else selected.release_time
        self.pending[selected.id] = PendingDecision(
            features=features,
            decision_time=simulator.now,
            wait_age=max(0.0, simulator.now - wait_since),
            due_time=selected.due_time,
            step_index=selected.step_index,
            remaining_before=selected.route.remaining_nominal_minutes(selected.step_index),
            projected_tardiness_before=self._projected_tardiness(
                simulator.now,
                selected.route.remaining_nominal_minutes(selected.step_index),
                selected.due_time,
            ),
            cqt_remaining_before=self._active_cqt_remaining(simulator, selected),
            cqt_violations_before=int(simulator.cqt_counts_by_lot[selected.id].get("violations", 0)),
            cqt_passes_before=int(simulator.cqt_counts_by_lot[selected.id].get("passes", 0)),
        )
        self._commit_immediate_reward(simulator, toolgroup, selected, candidates, tool)
        self.decisions += 1
        return selected

    def _commit_immediate_reward(self, simulator, toolgroup, selected: Lot, candidates: tuple[Lot, ...], tool) -> None:
        decision = self.pending[selected.id]
        reward, components = self._immediate_reward(simulator, toolgroup, selected, candidates, tool)
        self.agent.remember(decision.features, reward)
        self.reward_components.append(components)
        self.rewards += 1
        if self.train_online:
            loss = self.agent.train_step()
            if loss is not None:
                self.losses.append(loss)

    def on_lot_arrival(self, *, simulator, lot: Lot, now: float) -> None:
        decision = self.pending.pop(lot.id, None)
        if decision is not None:
            self._commit_reward(simulator, lot, decision, now, terminal=False)

    def on_lot_complete(self, *, simulator, lot: Lot, now: float) -> None:
        decision = self.pending.pop(lot.id, None)
        if decision is not None:
            self._commit_reward(simulator, lot, decision, now, terminal=True)

    def _commit_reward(self, simulator, lot: Lot, decision: PendingDecision, now: float, terminal: bool) -> None:
        reward, components = self._reward(simulator, lot, decision, now, terminal)
        self.agent.remember(decision.features, reward)
        self.reward_components.append(components)
        self.rewards += 1
        if self.train_online:
            loss = self.agent.train_step()
            if loss is not None:
                self.losses.append(loss)

    def _immediate_reward(self, simulator, toolgroup, selected: Lot, candidates: tuple[Lot, ...], tool) -> tuple[float, dict[str, float]]:
        candidate_scores = [
            self._candidate_dense_score(simulator, toolgroup, lot, candidates, tool)
            for lot in candidates
        ]
        selected_score = self._candidate_dense_score(simulator, toolgroup, selected, candidates, tool)
        best_score = max(candidate_scores) if candidate_scores else selected_score
        mean_score = sum(candidate_scores) / len(candidate_scores) if candidate_scores else selected_score
        relative_quality = selected_score - mean_score
        regret = best_score - selected_score
        urgency = self._urgency_score(simulator.now, selected)
        setup_penalty = self._setup_penalty(simulator, toolgroup, selected, tool)
        batch_reward, batch_penalty = self._batch_reward_penalty(simulator, toolgroup, selected, candidates)
        cqt_reward, cqt_penalty = self._cqt_reward_penalty(simulator, selected)
        hot_priority_bonus = 0.05 * max(0.0, selected.priority - 10.0) / 20.0
        if selected.super_hot_lot:
            hot_priority_bonus += 0.15
        reward = (
            0.35 * relative_quality
            - 0.25 * regret
            + 0.25 * urgency
            + batch_reward
            + cqt_reward
            + hot_priority_bonus
            - setup_penalty
            - batch_penalty
            - cqt_penalty
        )
        components = {
            "reward": float(reward),
            "immediate_relative_quality": float(relative_quality),
            "immediate_regret": float(regret),
            "immediate_urgency": float(urgency),
            "immediate_setup_penalty": float(setup_penalty),
            "immediate_batch_reward": float(batch_reward),
            "immediate_batch_penalty": float(batch_penalty),
            "immediate_cqt_reward": float(cqt_reward),
            "immediate_cqt_penalty": float(cqt_penalty),
            "immediate_hot_priority_bonus": float(hot_priority_bonus),
        }
        return float(reward), components

    def _candidate_dense_score(self, simulator, toolgroup, lot: Lot, candidates: tuple[Lot, ...], tool) -> float:
        urgency = self._urgency_score(simulator.now, lot)
        setup_penalty = self._setup_penalty(simulator, toolgroup, lot, tool)
        batch_reward, batch_penalty = self._batch_reward_penalty(simulator, toolgroup, lot, candidates)
        cqt_reward, cqt_penalty = self._cqt_reward_penalty(simulator, lot)
        priority_bonus = 0.05 * max(0.0, lot.priority - 10.0) / 20.0
        if lot.super_hot_lot:
            priority_bonus += 0.15
        return urgency + batch_reward + cqt_reward + priority_bonus - setup_penalty - batch_penalty - cqt_penalty

    def _urgency_score(self, now: float, lot: Lot) -> float:
        remaining = max(lot.route.remaining_nominal_minutes(lot.step_index), 1.0)
        if lot.due_time is None:
            slack_score = 0.0
            cr_score = 0.0
        else:
            slack_days = (lot.due_time - now - remaining) / MINUTES_PER_DAY
            critical_ratio = (lot.due_time - now) / remaining
            slack_score = max(-1.0, min(1.0, -slack_days / 7.0))
            cr_score = max(0.0, min(1.0, (1.5 - critical_ratio) / 1.5))
        wait_since = lot.waiting_since if lot.waiting_since is not None else lot.release_time
        wait_score = max(0.0, min(1.0, (now - wait_since) / MINUTES_PER_DAY))
        return 0.45 * slack_score + 0.35 * cr_score + 0.20 * wait_score

    def _setup_penalty(self, simulator, toolgroup, lot: Lot, tool) -> float:
        if tool is not None:
            setup = simulator._setup_duration_preview(tool, lot.current_step)
        elif toolgroup.idle_tools:
            setup = min(simulator._setup_duration_preview(item, lot.current_step) for item in toolgroup.idle_tools)
        else:
            setup = 0.0
        return 0.20 * max(0.0, min(1.0, setup / MINUTES_PER_DAY))

    def _batch_reward_penalty(self, simulator, toolgroup, lot: Lot, candidates: tuple[Lot, ...]) -> tuple[float, float]:
        if not simulator._is_batch_step(toolgroup, lot.current_step):
            return 0.0, 0.0
        minimum = lot.current_step.batch_minimum or 0.0
        maximum = lot.current_step.batch_maximum or max(minimum, 1.0)
        compatible = [
            candidate for candidate in candidates
            if simulator._batch_compatible(toolgroup, lot, candidate)
        ]
        quantity = simulator._batch_quantity(toolgroup, compatible)
        fill_ratio = quantity / max(maximum, 1.0)
        if quantity >= minimum:
            return 0.20 * min(1.0, fill_ratio), 0.0
        shortage = (minimum - quantity) / max(minimum, 1.0)
        return 0.0, 0.30 * min(1.0, shortage)

    def _cqt_reward_penalty(self, simulator, lot: Lot) -> tuple[float, float]:
        remaining = self._active_cqt_remaining(simulator, lot)
        if remaining is None:
            return 0.0, 0.0
        if remaining < 0:
            return 0.0, 0.75
        remaining_days = remaining / MINUTES_PER_DAY
        if remaining_days <= 0.25:
            return 0.35, 0.0
        if remaining_days <= 1.0:
            return 0.20, 0.0
        return 0.05, 0.0

    def _reward(self, simulator, lot: Lot, decision: PendingDecision, now: float, terminal: bool) -> tuple[float, dict[str, float]]:
        remaining_after = 0.0 if terminal else lot.route.remaining_nominal_minutes(lot.step_index)
        projected_tardiness_after = self._projected_tardiness(now, remaining_after, decision.due_time)
        tardiness_improvement = (decision.projected_tardiness_before - projected_tardiness_after) / MINUTES_PER_DAY
        tardiness_penalty = projected_tardiness_after / MINUTES_PER_DAY
        cycle_penalty = max(0.0, now - decision.decision_time) / MINUTES_PER_DAY
        wait_penalty = decision.wait_age / MINUTES_PER_DAY
        remaining_reduction = max(0.0, decision.remaining_before - remaining_after) / MINUTES_PER_DAY
        priority_weight = 1.0 + max(0, lot.priority - 10) / 20.0 + (1.0 if lot.super_hot_lot else 0.0)
        due_bonus = 0.0
        due_late_penalty = 0.0
        if terminal and decision.due_time is not None:
            if now <= decision.due_time:
                due_bonus = 2.0 * priority_weight
            else:
                due_late_penalty = priority_weight * min(5.0, (now - decision.due_time) / MINUTES_PER_DAY)
        cqt_bonus = 0.0
        cqt_penalty = 0.0
        cqt_violations_after = int(simulator.cqt_counts_by_lot[lot.id].get("violations", 0))
        cqt_passes_after = int(simulator.cqt_counts_by_lot[lot.id].get("passes", 0))
        if cqt_violations_after > decision.cqt_violations_before:
            cqt_penalty += 3.0 * (cqt_violations_after - decision.cqt_violations_before)
        if cqt_passes_after > decision.cqt_passes_before:
            cqt_bonus += 0.75 * (cqt_passes_after - decision.cqt_passes_before)
        cqt_remaining_after = self._active_cqt_remaining(simulator, lot)
        if decision.cqt_remaining_before is not None and cqt_remaining_after is not None:
            consumed = max(0.0, decision.cqt_remaining_before - cqt_remaining_after) / MINUTES_PER_DAY
            cqt_penalty += 0.25 * consumed
            if cqt_remaining_after < 0:
                cqt_penalty += min(3.0, abs(cqt_remaining_after) / MINUTES_PER_DAY)
        completion_bonus = 1.0 if terminal else 0.0
        reward = (
            0.4 * remaining_reduction
            + 1.5 * priority_weight * tardiness_improvement
            + due_bonus
            + cqt_bonus
            + completion_bonus
            - 1.2 * priority_weight * tardiness_penalty
            - due_late_penalty
            - cqt_penalty
            - 0.08 * cycle_penalty
            - 0.03 * wait_penalty
        )
        components = {
            "reward": float(reward),
            "remaining_reduction": float(remaining_reduction),
            "tardiness_improvement": float(tardiness_improvement),
            "tardiness_penalty": float(tardiness_penalty),
            "due_bonus": float(due_bonus),
            "due_late_penalty": float(due_late_penalty),
            "cqt_bonus": float(cqt_bonus),
            "cqt_penalty": float(cqt_penalty),
            "cycle_penalty": float(cycle_penalty),
            "wait_penalty": float(wait_penalty),
            "completion_bonus": float(completion_bonus),
        }
        return float(reward), components

    def _projected_tardiness(self, now: float, remaining: float, due_time: float | None) -> float:
        if due_time is None:
            return 0.0
        return max(0.0, now + remaining - due_time)

    def _active_cqt_remaining(self, simulator, lot: Lot) -> float | None:
        open_windows = [
            start + limit - simulator.now
            for (lot_id, _step_number), (start, limit) in simulator._cqt_open.items()
            if lot_id == lot.id
        ]
        if not open_windows:
            return None
        return min(open_windows)
