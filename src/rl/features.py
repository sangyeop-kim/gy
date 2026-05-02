"""Vectorized candidate feature encoder.

Design goals:
- No simulator-internal calls in the hot path (no batch_compatible / setup preview lookups).
- All-numpy vectorized construction (no per-candidate list comprehension).
- Fixed 12-dim feature, all clipped to [-1, 1] for stable gradients.

This intentionally drops the heavier features (setup-preview, compatible-batch-qty)
that the previous encoder computed per candidate. Those touch simulator internals and
turn dispatch into an O(n^2) call, which dominated training time. If we want to bring
that information back, it should be precomputed once per dispatch and broadcast — not
recomputed per candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.simulator.engine import MINUTES_PER_DAY
from src.simulator.model import Lot, Tool, ToolGroup


FEATURE_NAMES = (
    "is_super_hot",        # 0/1
    "priority_norm",       # (priority - 10) / 20  in [-0.5, 1.0]
    "due_in_norm",         # (due - now) / 30d, clip [-1, 1]
    "slack_norm",          # (due - now - remaining) / 30d, clip [-1, 1]
    "cr_norm",             # CR / 3, clip [0, 1]; <1 means urgent
    "current_pt_norm",     # current step nominal pt / 1d, clip [0, 1]
    "remaining_norm",      # remaining / 30d, clip [0, 1]
    "route_progress",      # step_index / len(route)
    "wait_age_norm",       # wait_age / 1d, clip [0, 1]
    "cqt_active",          # 0/1, lot has any open cqt window
    "cqt_remaining_norm",  # cqt_remaining_min / 1d, clip [-1, 1]
    "queue_len_norm",      # queue_len / 100, clip [0, 1]   (context, broadcast)
)


@dataclass(frozen=True)
class DispatchFeatureEncoder:
    """Vectorized fixed-size candidate feature encoder."""

    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    @property
    def size(self) -> int:
        return len(FEATURE_NAMES)

    def encode_candidates(
        self,
        simulator,
        toolgroup: ToolGroup,
        candidates: tuple[Lot, ...],
        tool: Tool | None = None,
    ) -> np.ndarray:
        n = len(candidates)
        if n == 0:
            return np.zeros((0, self.size), dtype=np.float32)

        now = float(simulator.now)
        cqt_open = simulator._cqt_open  # dict[(lot_id, step_no)] -> (start, limit)
        queue_len = float(len(toolgroup.waiting_lot_ids))

        # ---- per-candidate raw arrays (single python loop, no inner method calls) ----
        priority = np.empty(n, dtype=np.float32)
        super_hot = np.empty(n, dtype=np.float32)
        due_in = np.empty(n, dtype=np.float32)
        remaining = np.empty(n, dtype=np.float32)
        current_pt = np.empty(n, dtype=np.float32)
        wait_age = np.empty(n, dtype=np.float32)
        progress = np.empty(n, dtype=np.float32)
        cqt_active = np.zeros(n, dtype=np.float32)
        cqt_remaining = np.full(n, MINUTES_PER_DAY, dtype=np.float32)  # default = +1 day buffer
        has_due = np.empty(n, dtype=bool)

        for i, lot in enumerate(candidates):
            step = lot.current_step
            base = step.process_time.mean
            if step.processing_unit.lower() == "wafer":
                base *= lot.wafers_per_lot
            current_pt[i] = max(base, 0.0)
            remaining[i] = max(lot.route.remaining_nominal_minutes(lot.step_index), 0.0)
            priority[i] = float(lot.priority)
            super_hot[i] = 1.0 if lot.super_hot_lot else 0.0
            ws = lot.waiting_since if lot.waiting_since is not None else lot.release_time
            wait_age[i] = max(0.0, now - float(ws))
            progress[i] = lot.step_index / max(1, len(lot.route.steps))
            if lot.due_time is None:
                has_due[i] = False
                due_in[i] = 30.0 * MINUTES_PER_DAY
            else:
                has_due[i] = True
                due_in[i] = float(lot.due_time) - now

            # cqt: only check this lot's id; iterate cqt_open keys is O(open_windows),
            # but we expect that to be small (at most ~1 per lot at a time).
            best = math.inf
            for (lot_id, _step_no), (start, limit) in cqt_open.items():
                if lot_id == lot.id:
                    rem = (start + limit) - now
                    if rem < best:
                        best = rem
            if math.isfinite(best):
                cqt_active[i] = 1.0
                cqt_remaining[i] = best

        slack = due_in - remaining
        cr = np.where(remaining > 1.0, due_in / np.maximum(remaining, 1.0), 10.0)

        # ---- normalize, clip, fill output ----
        out = np.empty((n, self.size), dtype=np.float32)
        out[:, 0] = super_hot
        out[:, 1] = np.clip((priority - 10.0) / 20.0, -1.0, 1.0)
        out[:, 2] = np.clip(due_in / (30.0 * MINUTES_PER_DAY), -1.0, 1.0)
        out[:, 3] = np.clip(slack / (30.0 * MINUTES_PER_DAY), -1.0, 1.0)
        out[:, 4] = np.clip(cr / 3.0, 0.0, 1.0)
        out[:, 5] = np.clip(current_pt / MINUTES_PER_DAY, 0.0, 1.0)
        out[:, 6] = np.clip(remaining / (30.0 * MINUTES_PER_DAY), 0.0, 1.0)
        out[:, 7] = np.clip(progress, 0.0, 1.0)
        out[:, 8] = np.clip(wait_age / MINUTES_PER_DAY, 0.0, 1.0)
        out[:, 9] = cqt_active
        out[:, 10] = np.clip(cqt_remaining / MINUTES_PER_DAY, -1.0, 1.0)
        out[:, 11] = np.clip(queue_len / 100.0, 0.0, 1.0)

        # missing-due → due_in / slack go to neutral 0
        if not has_due.all():
            mask = ~has_due
            out[mask, 2] = 0.0
            out[mask, 3] = 0.0
            out[mask, 4] = 1.0  # CR: not urgent

        return out
