"""Opt-in IEEE123 distribution-side LFC probe; no MATLAB or policy registration.

The unmodified PowerZoo native 123 feeder is solved with its native controls
once at reset and held at those taps thereafter. There is no transmission
feedback, time-series profile playback or assertion of paper reproduction.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCENARIO_PATH = ROOT / "scenarios/lfc/ieee123-native/probe_v1.json"


def stock_probe_spec():
    """Load checked-in v1 scenario and return its source-byte digest."""
    raw = SCENARIO_PATH.read_bytes()
    spec = json.loads(raw)
    if spec["schema_version"] != 1 or spec["step_seconds"] != 1:
        raise ValueError("unsupported IEEE123 probe schema or interval")
    return spec, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class PQ:
    """kW and kvar; positive represents import or injection as named."""

    kw: float
    kvar: float

    def __post_init__(self):
        if not all(map(math.isfinite, (self.kw, self.kvar))):
            raise ValueError("P/Q must be finite")


@dataclass(frozen=True)
class GSPCalibration:
    """No scaling: background + measured feeder demand; units remain kW/kvar.

    The target must come from an independently validated transmission
    equilibrium before attempting any Simulink integration. This driver
    does not select or validate the transmission operating point.
    """

    target_t0: PQ
    feeder_t0: PQ

    @property
    def background(self) -> PQ:
        return PQ(
            self.target_t0.kw - self.feeder_t0.kw,
            self.target_t0.kvar - self.feeder_t0.kvar,
        )

    def project(self, feeder_import: PQ) -> PQ:
        return PQ(
            self.target_t0.kw + (feeder_import.kw - self.feeder_t0.kw),
            self.target_t0.kvar + (feeder_import.kvar - self.feeder_t0.kvar),
        )


@dataclass(frozen=True)
class StorageReading:
    device: str
    requested_kw: float
    applied_kw: float
    measured_injection: PQ
    soc_before: float
    soc_after: float


@dataclass(frozen=True)
class IntervalReading:
    start_s: float
    end_s: float
    feeder_import: PQ
    equivalent_transmission_load: PQ
    load_consumption: PQ
    feeder_loss: PQ
    active_balance_error_kw: float
    storage: tuple[StorageReading, ...]


def _requests(by_device, ids):
    if set(by_device) != set(ids):
        raise ValueError("one explicit request is required for each stock Storage ID")
    out = {}
    for device in ids:
        kw = by_device[device]
        if (
            isinstance(kw, bool)
            or not isinstance(kw, (int, float))
            or not math.isfinite(kw)
        ):
            raise ValueError(f"{device} needs finite kW")
        out[device] = {"kw": float(kw)}
    return out


def _verify_stock_binding(session, devices):
    if set(session.batteries) != {row["id"].split(".")[1] for row in devices}:
        raise ValueError("native case must have exactly the four stock Storage devices")
    if len(devices) != 4:
        raise ValueError("v1 probe requires four storage mappings")
    for row in devices:
        device, bus, phases, kv, conn = (
            row["id"],
            row["bus"],
            tuple(row["phases"]),
            row["kv"],
            row["connection"],
        )
        wiring = session.connection(device)
        actual_bus = wiring["buses"][0].split(".", 1)[0]
        actual_phases = tuple(n for n in wiring["nodes"][: wiring["conductors"]] if n)
        dev = session.batteries[device.split(".")[1]]
        if (
            not wiring["enabled"]
            or wiring["terminals"] != 1
            or actual_bus != bus
            or actual_phases != phases
            or session.read_property(device, "conn").lower() != conn
            or not math.isclose(float(session.read_property(device, "kV")), kv)
            or not math.isclose(dev.max_kw, row["kw"])
            or not math.isclose(dev.max_kwh, row["kwh"])
        ):
            raise ValueError(f"unexpected stock binding for {device}: {wiring}")


class Native123Feeder:
    """Run stock feeder with explicit four-agent kW inputs and a one-second clock.

    Positive requested, applied and measured *injection* is discharge. Source
    import uses the opposite sign of OpenDSS's consumed-positive source
    terminal; load consumption has the same sign as its load terminals.
    """

    def __init__(self, equilibrium_target: PQ, *, _session_factory=None):
        if not isinstance(equilibrium_target, PQ):
            raise TypeError("supply the explicit t0 transmission demand as PQ")
        if _session_factory is None:
            from envs.powergrid.session import DSSSession

            _session_factory = DSSSession
        spec, self.scenario_sha256 = stock_probe_spec()
        self._source_id = spec["source_device"]
        self.storage_ids = tuple(row["id"] for row in spec["storage_devices"])
        self.session = _session_factory(
            ROOT / spec["feeder_directory"],
            spec["feeder_master"],
            step_seconds=spec["step_seconds"],
        )
        try:
            _verify_stock_binding(self.session, spec["storage_devices"])
            self.reset(equilibrium_target)
        except Exception:
            self.close()
            raise

    def _source_import(self):
        p, q = self.session.powers(self._source_id)
        return PQ(-float(p), -float(q))

    def reset(self, equilibrium_target: PQ) -> GSPCalibration:
        if not isinstance(equilibrium_target, PQ):
            raise TypeError("supply the explicit t0 transmission demand as PQ")
        self.session.reset(seconds=0)
        self.session.apply(
            _requests(dict.fromkeys(self.storage_ids, 0), self.storage_ids)
        )
        self.session.solve(native_controls=True)
        if not math.isclose(self.session.time_s, 0, rel_tol=0, abs_tol=1e-6):
            raise RuntimeError(
                "t0 regulator settlement unexpectedly advanced the clock"
            )
        self.calibration = GSPCalibration(equilibrium_target, self._source_import())
        return self.calibration

    def step(self, by_device) -> IntervalReading:
        requests = _requests(
            by_device, self.storage_ids
        )  # refuse bad frames before changing the engine
        s = self.session
        start = s.time_s
        soc = {name: b.soc for name, b in s.batteries.items()}
        s.begin_interval(start)
        trace = s.apply(requests)
        s.solve(native_controls=False)
        source = self._source_import()
        loads = PQ(*map(float, s.total_load_power()))
        losses = PQ(*map(float, s.total_loss()))
        powers = {}
        for device in self.storage_ids:
            p, q = s.powers(device)
            powers[device] = PQ(-float(p), -float(q))
        s.finish_interval()  # sole owner of native energy/time advancement
        end = s.time_s
        residual = math.fsum(
            [source.kw, -loads.kw, -losses.kw] + [row.kw for row in powers.values()]
        )
        readings = tuple(
            StorageReading(
                device=device,
                requested_kw=requests[device]["kw"],
                applied_kw=trace[device]["applied"]["kw"],
                measured_injection=powers[device],
                soc_before=soc[device.split(".")[1]],
                soc_after=s.batteries[device.split(".")[1]].soc,
            )
            for device in self.storage_ids
        )
        return IntervalReading(
            start,
            end,
            source,
            self.calibration.project(source),
            loads,
            losses,
            residual,
            readings,
        )

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
