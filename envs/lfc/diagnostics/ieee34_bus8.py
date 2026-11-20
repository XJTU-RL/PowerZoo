"""Pure draft contract for calibrated IEEE34 → Bus8. No simulator I/O."""
from dataclasses import dataclass
import hashlib
import json
import math
import re

STATIC_COMMIT = "fdfb8680a15e76fd87b4801c1cb5f0c85172a3ac"
DECK_TREE = "8fbc1a9a3b7a2a5da636224908298a256b4c2b43"
ANCHOR_PROBE_COMMIT = "24dddb130f6eaae546d81bf373fabe73f6592c83"
BUS8_ANCHOR = (7851.739821876062, 12942.530488711882)  # kW, kvar
PUBLISHED_IMPORT = (1486.417, -138.427)  # rounded; exact is in accepted report
MAX_DELTA_KW_KVAR = (50.0, 40.0)  # draft envelope, not field limits
EPS = 1e-8
REGULATORS = ("reg1a", "reg1b", "reg1c", "reg2a", "reg2b", "reg2c")
RATINGS = {
    "subxf:1": ("sourcebus", 25000., 69.),
    "subxf:2": ("800", 25000., 24.9),
    "xfm1:1": ("832", 500., 24.9),
    "xfm1:2": ("888", 500., 4.16),
}


def _keyed(rows, label):
    out = {}
    for key, value in rows:
        if not isinstance(key, str) or not key or key in out:
            raise ValueError(f"empty or duplicate {label} ID")
        out[key] = value
    return out


@dataclass(frozen=True)
class Binding:
    source_commit: str
    deck_tree: str
    source_pu: float
    load_multiplier: float
    shunt: tuple[str, str, float, float]  # bus, connection, line kV, nameplate kvar
    solver_tolerance: float
    control_mode: str
    original_disabled_der: tuple[str, ...]
    active_der: tuple[str, ...]
    active_profile_bindings: tuple[str, ...]


PINNED_BINDING = Binding(
    STATIC_COMMIT, DECK_TREE, 1.045, .75, ("890.1.2.3", "delta", 4.16, 75.), 1e-9,
    "STATIC", ("storage.batt1", "storage.batt2"), (), (),
)


@dataclass(frozen=True)
class Winding:
    terminal_id: str
    bus: str
    rating_kva: float
    rating_kv_ll: float
    measured_kva: float
    node_current_amp: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class ObservedReference:
    """The audited manifest, not an arbitrary array from the caller."""
    sha256: str
    phase_ids: tuple[str, ...]
    taps: tuple[tuple[str, float], ...]
    feeder_import_kw_kvar: tuple[float, float]

    @classmethod
    def read_report(cls, report_bytes: bytes, *, expected_sha256: str):
        """Require explicit independent approval of the exact external report digest."""
        digest = hashlib.sha256(report_bytes).hexdigest()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or digest != expected_sha256:
            raise ValueError("unapproved observed static report digest")
        data = json.loads(report_bytes, object_pairs_hook=lambda pairs: _keyed(pairs, "JSON"))
        if (data["mode"], data["source_pu"], data["load_multiplier"]) != (
            "optimized-headroom-pair", 1.045, .75
        ):
            raise ValueError("wrong static scenario")
        accepted = [x for x in data["scenarios"] if x["study_model_feasible_static_candidate"] is True]
        if len(accepted) != 1:
            raise ValueError("expected exactly one admitted reference row")
        row = accepted[0]
        ev = row["shunt_study_evidence"]
        if (row["source_commit"], row["input_tree"], row["source_pu_command"],
            row["native_load_multiplier"], ev["synthetic_shunt_nameplate_kvar"],
            row["native_voltage_convergence_tolerance"], row["control_mode"],
            tuple(row["disabled_der"]), row["native_control_actions_done"]) != (
                STATIC_COMMIT, DECK_TREE, 1.045, .75, 75., 1e-9, "STATIC",
                PINNED_BINDING.original_disabled_der, True):
            raise ValueError("wrong accepted report source/deck/settings")
        r = row["stages"]["standard_snapshot"]
        if r["converged"] is not True or r["network_power_balance_ok"] is not True:
            raise ValueError("reference physical gates failed")
        phases = r["voltage_phase_pu"]
        if len(phases) != 95 or not all(re.fullmatch(r"[a-zA-Z0-9_]+\.[123]", x) for x in phases):
            raise ValueError("unexpected reference phase IDs")
        if not all(.95 <= x["v_pu"] <= 1.05 for x in phases.values()):
            raise ValueError("reference phase voltage gate failed")
        taps = row["regulator_secondary_taps"]
        if set(taps) != set(REGULATORS) or not all(math.isfinite(x) and x > 0 for x in taps.values()):
            raise ValueError("missing or invalid observed regulator taps")
        power = r["pcc_import_kw"], r["pcc_import_kvar"]
        if not all(math.isfinite(a) and abs(a-b) <= .002 for a, b in zip(power, PUBLISHED_IMPORT)):
            raise ValueError("reference P/Q differs from published static point")
        if any(n.lower().startswith(("pvsystem.", "storage.", "generator."))
               for n in r["enabled_electrical_elements_in_power_accounting"]):
            raise ValueError("reference contains active DER")
        screens = row["model_winding_screens"]
        if set(screens) != {"SubXF", "XFM1"}:
            raise ValueError("reference transformer IDs differ")
        for xf, windings in screens.items():
            if len(windings) != 2:
                raise ValueError("reference winding count differs")
            for i, w in enumerate(windings, 1):
                expect = RATINGS[f"{xf.lower()}:{i}"]
                if (w["terminal_bus"].split(".")[0].lower(), w["rating_kva"],
                    w["rating_kv_ll"]) != expect or w["passed_model_nameplate_screen"] is not True:
                    raise ValueError("reference model winding evidence differs")
                abc, neutral = w["phase_current_amp_abc"], w["neutral_current_amp"]
                if len(abc) != 3 or not all(math.isfinite(v) and v >= 0 for v in (*abc, neutral)):
                    raise ValueError("reference winding lacks ABC or neutral current")
        return cls(digest, tuple(sorted(phases)), tuple(sorted(taps.items())), power)


@dataclass(frozen=True)
class Snapshot:
    """Per-exchange native observations from a future, independently audited runner."""
    binding: Binding
    time_s: float  # declared exchange slot, not static DSS elapsed integration
    source_terminal_kw: float  # into Vsource.Source terminal one
    source_terminal_kvar: float
    remaining_source_terminal_kva: float
    phases_pu: tuple[tuple[str, float], ...]
    windings: tuple[Winding, ...]
    taps: tuple[tuple[str, float], ...]
    residual_kw: float
    residual_kvar: float
    converged: bool
    controls_settled: bool

    def checked_import(self, reference: ObservedReference) -> tuple[float, float]:
        vals = (self.time_s, self.source_terminal_kw, self.source_terminal_kvar,
                self.remaining_source_terminal_kva, self.residual_kw, self.residual_kvar)
        if self.binding != PINNED_BINDING:
            raise ValueError("source/deck/settings or active DER/profile binding changed")
        if not all(math.isfinite(x) for x in vals):
            raise ValueError("non-finite measurement")
        if self.converged is not True or self.controls_settled is not True:
            raise ValueError("solve or STATIC controls incomplete")
        if self.time_s < 0 or not 0 <= self.remaining_source_terminal_kva <= 1e-6:
            raise ValueError("invalid source measurement")
        phase = _keyed(self.phases_pu, "phase")
        if set(phase) != set(reference.phase_ids) or not all(.95 <= x <= 1.05 for x in phase.values()):
            raise ValueError("phase ID or voltage gate")
        windings = _keyed(((x.terminal_id, x) for x in self.windings), "winding")
        if set(windings) != set(RATINGS):
            raise ValueError("missing or unexpected winding ID")
        for name, w in windings.items():
            bus, kva, kv = RATINGS[name]
            if (w.bus, w.rating_kva, w.rating_kv_ll) != (bus, kva, kv):
                raise ValueError("winding terminal bus or model rating changed")
            amps = dict(w.node_current_amp)
            if (len(w.node_current_amp) != 4 or set(amps) != {0, 1, 2, 3}
                or any(type(n) is not int for n in amps)
                or not all(math.isfinite(x) and x >= 0 for x in amps.values())):
                raise ValueError("winding ABC/neutral inventory gate")
            if not 0 <= w.measured_kva <= kva or max(amps[n] for n in (1, 2, 3)) > kva/(math.sqrt(3)*kv):
                raise ValueError("winding ABC/model nameplate gate")
        if _keyed(self.taps, "tap") != dict(reference.taps):
            raise ValueError("observed regulator tap IDs or positions changed")
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


class AnchorSequence:
    """Single-use, strictly ordered admission; the engine runner remains responsible for reset."""
    def __init__(self, report_bytes: bytes, *, approved_report_sha256: str):
        self._reference = ObservedReference.read_report(
            report_bytes, expected_sha256=approved_report_sha256)
        self._next = None

    def reset(self, snapshot: Snapshot, *, initial_write_target_s: float, observed_pause_s: float):
        self._next = None
        if not math.isfinite(initial_write_target_s) or initial_write_target_s < 0:
            raise ValueError("invalid Bus8 start time")
        self._start = initial_write_target_s
        self._feeder_start = snapshot.time_s
        self._next = 0
        return self.project(snapshot, index=0, observed_pause_s=observed_pause_s)

    def invalidate(self):
        """Call after an actual engine failure (including a partial P/Q write)."""
        self._next = None

    def project(self, feeder: Snapshot, *, index: int, observed_pause_s: float) -> Command:
        expected = self._next
        self.invalidate()
        if expected is None or type(index) is not int or index != expected:
            raise ValueError("invalid/replayed exchange; reset required")
        target = self._start + index
        power = feeder.checked_import(self._reference)
        if (abs(feeder.time_s - (self._feeder_start + index)) > EPS
                or not target - EPS <= observed_pause_s <= target + 2/60 + EPS):
            raise ValueError("IEEE34/Bus8 exchange clocks do not match")
        delta = tuple(x - y for x, y in zip(power, self._reference.feeder_import_kw_kvar))
        if index == 0 and max(map(abs, delta)) > EPS:
            raise ValueError("zero-delta reference replay required")
        if any(abs(x) > cap + EPS for x, cap in zip(delta, MAX_DELTA_KW_KVAR)):
            raise ValueError("measured perturbation outside Bus8 envelope")
        load = tuple(1000 * (x + d) for x, d in zip(BUS8_ANCHOR, delta))
        self._next = index + 1
        return Command(feeder.time_s, target, observed_pause_s, power, delta, load)
