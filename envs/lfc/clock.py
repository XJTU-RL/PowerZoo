"""Opt-in LFC decision clock. Legacy/paper timing is not routed through it."""

from dataclasses import dataclass
import math


@dataclass
class CouplingClock:
    horizon: int
    previous_target: float | None = None
    previous_pause: float | None = None
    previous_clock_sample: float | None = None

    @property
    def stop_time(self) -> int:
        # The last pause target is horizon+1; allow solver callbacks past it.
        return self.horizon + 2

    def target(self, step: int | None = None) -> int:
        if step is None:
            return 1
        if not 0 <= step < self.horizon:
            raise ValueError("LFC control step outside episode")
        return 2 + step

    def reset(self) -> None:
        self.previous_target = self.previous_pause = self.previous_clock_sample = None

    def record(self, target: float, pause: float, sample: float) -> dict:
        if not all(math.isfinite(x) for x in (target, pause, sample)):
            raise RuntimeError("LFC clock received a non-finite timestamp")
        tol = 1e-8
        # R2025b IEEE9 pause callbacks observed +2/60 s; allow three 60-Hz
        # ticks of callback delay. out_t is separately logged every 0.2 s.
        pause_slack, sample_slack = 3 / 60, 0.2
        if not (
            target - tol <= pause <= target + pause_slack + tol
            and target - sample_slack - tol <= sample <= pause + tol
        ):
            raise RuntimeError(
                f"LFC timestamp inconsistent: target={target}, pause={pause}, sample={sample}"
            )
        expected_target = 1 if self.previous_target is None else self.previous_target + 1
        if abs(target - expected_target) > tol:
            raise RuntimeError(f"LFC expected target {expected_target}, got {target}")
        if self.previous_pause is not None and (
            pause <= self.previous_pause + tol or sample <= self.previous_clock_sample + tol
        ):
            raise RuntimeError(
                f"LFC stale clock trace: pause={pause}, sample={sample}, "
                f"previous pause={self.previous_pause}, sample={self.previous_clock_sample}"
            )
        dt = None if self.previous_pause is None else pause - self.previous_pause
        self.previous_target = target
        self.previous_pause, self.previous_clock_sample = pause, sample
        return dict(target_s=target, pause_s=pause, clock_sample_s=sample, dt_s=dt)


@dataclass
class AlignedEpisodeGate:
    """An attempt may mutate DSS/MATLAB; never retry it without a good reset."""

    ready: bool = False

    def reset(self, operation):
        self.ready = False
        result = operation()
        self.ready = True
        return result

    def step(self, operation):
        if not self.ready:
            raise RuntimeError("LFC aligned episode requires a successful reset before stepping")
        self.ready = False
        result = operation()
        self.ready = not bool(result[2])
        return result
