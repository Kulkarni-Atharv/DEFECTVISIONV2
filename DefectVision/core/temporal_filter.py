from __future__ import annotations
from collections import deque
import numpy as np
from config import TEMPORAL_WINDOW, TEMPORAL_DEFECT_RATIO


class TemporalFilter:
    """
    Rolling-window consistency filter.

    A defect is "confirmed" only when at least TEMPORAL_DEFECT_RATIO
    of the last TEMPORAL_WINDOW frames independently flagged a defect.
    This eliminates single-frame spikes caused by sensor noise, partial
    occlusions, or transient lighting glitches.

    The smoothed score (mean over the window) is also returned so the
    UI can show a stable numeric readout.
    """

    def __init__(self) -> None:
        self._scores: deque[float] = deque(maxlen=TEMPORAL_WINDOW)
        self._flags: deque[int]  = deque(maxlen=TEMPORAL_WINDOW)

    def update(self, defect_score: float, is_raw_defect: bool) -> tuple[float, bool]:
        """
        Push a new frame result and return (smoothed_score, confirmed_defect).
        confirmed_defect is False until the window is at least half full.
        """
        self._scores.append(defect_score)
        self._flags.append(1 if is_raw_defect else 0)

        smoothed = float(np.mean(self._scores))

        # Need at least half the window filled before issuing a verdict.
        if len(self._flags) < max(2, TEMPORAL_WINDOW // 2):
            return smoothed, False

        ratio = sum(self._flags) / len(self._flags)
        confirmed = ratio >= TEMPORAL_DEFECT_RATIO
        return smoothed, confirmed

    def reset(self) -> None:
        """Call after a new reference is captured to avoid stale history."""
        self._scores.clear()
        self._flags.clear()

    @property
    def window_full(self) -> bool:
        return len(self._flags) == TEMPORAL_WINDOW
