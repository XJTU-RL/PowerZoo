"""Pure draft adapter for the calibrated IEEE34 one-way Bus8 probe."""
from dataclasses import dataclass
import math

STATIC_COMMIT = "fdfb8680a15e76fd87b4801c1cb5f0c85172a3ac"
ANCHOR_PROBE_COMMIT = "24dddb130f6eaae546d81bf373fabe73f6592c83"
BUS8_ANCHOR = (7851.739821876062, 12942.530488711882)  # kW, kvar
PUBLISHED_IMPORT = (1486.417, -138.427)  # rounded reference
MAX_DELTA_KW_KVAR = (50.0, 40.0)  # draft experiment envelope, not field limits
EPS = 1e-8


@dataclass(frozen=True)
class Snapshot:
    """Measurements by a future DSS runner, never inferred from action commands."""
    time_s: float
    source_terminal_kw: float  # into Vsource.Source terminal 1
    source_terminal_kvar: float
    remaining_source_terminal_kva: float
    voltages_pu: tuple[float, ...]
    winding_kva_ratios: tuple[float, ...]  # both SubXF, both XFM1
    phase_current_ratios: tuple[float, ...]  # max ABC/rating on each of those terminals
    taps: tuple[int, ...]
    residual_kw: float
    residual_kvar: float
    enabled_storage_count: int
    converged: bool
    controls_settled: bool

    def checked_import(self) -> tuple[float, float]:
        vals = (self.time_s, self.source_terminal_kw, self.source_terminal_kvar,
                self.remaining_source_terminal_kva, self.residual_kw, self.residual_kvar,
                *self.voltages_pu, *self.winding_kva_ratios, *self.phase_current_ratios)
        if not all(math.isfinite(x) for x in vals):
            raise ValueError("non-finite measurement")
        if self.converged is not True or self.controls_settled is not True:
            raise ValueError("solve or STATIC controls incomplete")
        if self.time_s < 0 or not 0 <= self.remaining_source_terminal_kva <= 1e-6:
            raise ValueError("invalid source measurement")
        if len(self.voltages_pu) != 95 or not all(.95 <= x <= 1.05 for x in self.voltages_pu):
            raise ValueError("phase voltage gate")
        for ratios in (self.winding_kva_ratios, self.phase_current_ratios):
            if len(ratios) != 4 or not all(0 <= x <= 1 for x in ratios):
                raise ValueError("SubXF/XFM1 terminal rating gate")
        if len(self.taps) != 6 or any(type(x) is not int for x in self.taps):
            raise ValueError("expected six regulator tap positions")
        if type(self.enabled_storage_count) is not int or self.enabled_storage_count != 0:
            raise ValueError("original storage must be disabled")
        tol = max(.01, 1e-6 * max(abs(self.source_terminal_kw), abs(self.source_terminal_kvar)))
        if max(abs(self.residual_kw), abs(self.residual_kvar)) > tol:
            raise ValueError("global complex-power accounting gate")
        return -self.source_terminal_kw, -self.source_terminal_kvar


@dataclass(frozen=True)
class Command:
    feeder_time_s: float
    write_target_s: float
    observed_pause_s: float
    feeder_import_kw_kvar: tuple[float, float]
    feeder_delta_kw_kvar: tuple[float, float]
    bus8_load_w_var: tuple[float, float]


class AnchoredBus8:
    def __init__(self, reference: Snapshot, *, initial_write_target_s: float):
        self.reference = reference.checked_import()
        if any(abs(a - b) > .002 for a, b in zip(self.reference, PUBLISHED_IMPORT)):
            raise ValueError("reference does not match published static point")
        if not math.isfinite(initial_write_target_s) or initial_write_target_s < 0:
            raise ValueError("invalid Bus8 start time")
        self.start = initial_write_target_s
        self.feeder_start = reference.time_s
        self.taps = reference.taps

    def project(self, feeder: Snapshot, *, index: int, observed_pause_s: float) -> Command:
        # One second is a declared exchange schedule, not static DSS elapsed integration.
        if type(index) is not int or index < 0 or not math.isfinite(observed_pause_s):
            raise ValueError("invalid exchange clock")
        target = self.start + index
        power = feeder.checked_import()
        if (abs(feeder.time_s - (self.feeder_start + index)) > EPS
                or not target - EPS <= observed_pause_s <= target + 2 / 60 + EPS):
            raise ValueError("IEEE34/Bus8 exchange clocks do not match")
        if feeder.taps != self.taps:
            raise ValueError("tap change confounds matched perturbation")
        delta = tuple(x - y for x, y in zip(power, self.reference))
        if index == 0 and max(map(abs, delta)) > EPS:
            raise ValueError("zero-delta reference replay required")
        if any(abs(x) > cap + EPS for x, cap in zip(delta, MAX_DELTA_KW_KVAR)):
            raise ValueError("measured perturbation outside Bus8 envelope")
        load = tuple(1000 * (x + d) for x, d in zip(BUS8_ANCHOR, delta))
        return Command(feeder.time_s, target, observed_pause_s, power, delta, load)
