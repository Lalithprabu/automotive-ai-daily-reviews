"""
CurriculumScheduler: cosine-decaying pseudo-label confidence threshold.

Implements the paper's curriculum learning strategy for self-training on
collaborator pseudo-labels: only trust very confident pseudo-labels early
in training (when the teacher's shared features are new / the student is
weak), and progressively relax the trust threshold as the student adapts.
"""

import math


class CurriculumScheduler:
    def __init__(self, start: float = 0.90, end: float = 0.50, total_steps: int = 150):
        assert 0.0 <= end <= start <= 1.0, "expected end <= start, both in [0, 1]"
        self.start = start
        self.end = end
        self.total_steps = max(1, total_steps)

    def get_threshold(self, step: int) -> float:
        """Cosine decay from `start` at step 0 to `end` at step total_steps,
        then held flat at `end` for any step beyond total_steps."""
        t = min(max(step, 0), self.total_steps)
        cos_term = 0.5 * (1.0 + math.cos(math.pi * t / self.total_steps))
        return self.end + (self.start - self.end) * cos_term

    def __call__(self, step: int) -> float:
        return self.get_threshold(step)
