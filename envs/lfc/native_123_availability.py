"""Opt-in, static-scenario 123 device envelope and scratch feasibility guard.

No action reaches the live feeder here. This module does not certify observed
network line ampacity or infer future per-device power limits from one hold.
"""

import cmath
from dataclasses import dataclass
import json
import math
from pathlib import Path

from envs.lfc.receiver_123 import load_protocol
from envs.lfc.native_feeder_123 import Native123Feeder, PQ, stock_probe_spec

SPEC = Path(__file__).resolve().parents[2] / "scenarios/lfc/ieee123-native/damped_envelope_v1.json"


class DeviceDispatchMismatch(RuntimeError):
    """Direct terminal P/Q delivery mismatch; never a cleanup/parity wrapper."""


@dataclass(frozen=True)
class Conductor:
    element: str
    terminal: int
    node: int
    volts: complex
    amps: float
    opened: bool


@dataclass(frozen=True)
class OperatingPoint:
    clock_s: float
    state: tuple
    soc: tuple[float, ...]
    storage_pq: tuple[PQ, ...]
    source_pq: PQ
    conductors: tuple[Conductor, ...]


def envelope():
    return json.loads(SPEC.read_bytes())


def _near(a, b, tol):
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


def _sample_element(s, element, terminals):
    geometry = s.connection(element)
    obj = s.native.ActiveCktElement
    if not geometry["enabled"]:
        raise RuntimeError(f"{element}: disabled element")
    if list(obj.NodeOrder) != geometry["nodes"]:
        raise RuntimeError(f"{element}: runtime NodeOrder differs from pinned connection")
    n = geometry["conductors"]
    nt = geometry["terminals"]
    volt, curr = tuple(obj.VoltagesMagAng), tuple(obj.CurrentsMagAng)
    if (
        n <= 0 or len(geometry["nodes"]) != n * nt
        or len(volt) != len(curr) or len(volt) != 2 * n * nt
        or not all(math.isfinite(x) for x in volt + curr)
        or any(x < 0 for x in volt[::2] + curr[::2])
    ):
        raise RuntimeError(f"{element}: malformed native conductor phasors")
    result = []
    for terminal in terminals:
        if not 1 <= terminal <= nt:
            raise RuntimeError(f"{element}: invalid terminal {terminal}")
        start = (terminal - 1) * n
        nodes = geometry["nodes"][start:start + n]
        if len(set(nodes)) != len(nodes):
            raise RuntimeError(f"{element}: ambiguous conductor IDs")
        for i, node in enumerate(nodes, start):
            result.append(Conductor(
                element, terminal, node,
                cmath.rect(volt[2*i], math.radians(volt[2*i+1])),
                curr[2*i], geometry["open"][terminal - 1][i - start],
            ))
    return result


def _state(session):
    """Static fingerprints; ineligible mutable components are not synchronized."""
    native = session.native
    if session.overlays or session.pvs or session.svgs:
        raise RuntimeError("dynamic or overlaid feeder is out of scope")
    topology = []
    loads = []
    sources = []
    taps = []
    transformers = session.native.Transformers
    for element in sorted(native.AllElementNames):
        kind, _, name = element.lower().partition(".")
        if kind in ("generator", "invcontrol", "storagecontroller"):
            raise RuntimeError(f"unsupported external state owner {element}")
        if kind in ("line", "transformer", "storage", "load", "capacitor", "vsource"):
            c = session.connection(element)
            topology.append((element, tuple(c["buses"]), tuple(c["nodes"]), c["enabled"],
                             tuple(tuple(x) for x in c["open"])))
        if kind == "load":
            loads.append((element, tuple(session.read_property(element, key)
                                        for key in ("kW", "kvar", "model", "daily", "duty", "yearly"))))
        if kind == "vsource":
            sources.append((element, tuple(session.read_property(element, key)
                                          for key in ("basekV", "pu", "Angle", "frequency"))))
        if kind == "transformer":
            transformers.Name = name
            winding_taps = []
            for wdg in range(1, transformers.NumWindings + 1):
                transformers.Wdg = wdg
                winding_taps.append(float(transformers.Tap))
            taps.append((element, tuple(winding_taps)))
    caps = tuple((name, session.capacitor_state(name)) for name in sorted(session.capacitors))
    return (tuple(topology), tuple(loads), tuple(sources), tuple(taps), caps,
            float(native.Solution.LoadMult), float(native.Solution.Frequency))


def collect(feeder, spec):
    """Read one solved native operating point; no commands/time/energy writes."""
    s = feeder.session
    if s.model_sha256 != spec["feeder_sha256"]:
        raise RuntimeError("native feeder hash differs from screened model")
    if not s._measurement_valid or not s.native.Solution.Converged:
        raise RuntimeError("no converged native measurement")
    at = s.time_s
    scope, _ = stock_probe_spec()
    devices = scope["storage_devices"]
    if tuple(d["id"] for d in devices) != tuple(feeder.storage_ids):
        raise RuntimeError("unexpected native storage identity/order")
    conductors = []
    powers = []
    soc = []
    state = _state(s)
    for d in devices:
        element = d["id"]
        c = s.connection(element)
        if c["terminals"] != 1 or c["buses"][0].split(".")[0] != d["bus"]:
            raise RuntimeError(f"{element}: stock terminal binding mismatch")
        conductors.extend(_sample_element(s, element, (1,)))
        p, q = s.powers(element, terminal=1)
        powers.append(PQ(-float(p), -float(q)))
        current_soc = float(s.read_property(element, "%stored")) / 100
        if not math.isfinite(current_soc):
            raise RuntimeError(f"{element}: invalid native SOC")
        soc.append(current_soc)
        if not all(
            _near(float(s.read_property(element, prop)), d[key], 1e-9)
            for prop, key in (("kV", "kv"), ("kWRated", "kw"), ("kWhRated", "kwh"),
                              ("kVA", "kw"))
        ) or s.read_property(element, "conn").lower() != d["connection"]:
            raise RuntimeError(f"{element}: native storage model rating/connection mismatch")
    for element, terminals in spec["diagnostic_branch_terminals"].items():
        conductors.extend(_sample_element(s, element, terminals))
    p, q = s.powers(scope["source_device"], terminal=1)
    result = OperatingPoint(at, state, tuple(soc), tuple(powers), PQ(-float(p), -float(q)),
                            tuple(conductors))
    if not _near(s.time_s, at, 0):
        raise RuntimeError("collection changed native clock")
    return result


def _device_samples(point, device):
    samples = tuple(x for x in point.conductors if x.element.lower() == device.lower())
    if not samples or any(x.terminal != 1 or x.opened for x in samples):
        raise RuntimeError(f"{device}: missing/open storage conductor")
    by_node = {x.node: x for x in samples}
    if len(by_node) != len(samples):
        raise RuntimeError(f"{device}: duplicate storage conductor")
    return by_node


def _device_voltages(point, device):
    nodes = _device_samples(point, device["id"])
    phases = tuple(device["phases"])
    if device["connection"] == "wye" and len(phases) == 1 and set(nodes) == {0, *phases}:
        pairs = ((phases[0], 0),)
    elif device["connection"] == "delta" and phases == (1, 2, 3) and set(nodes) == set(phases):
        pairs = ((1, 2), (2, 3), (3, 1))
    else:
        raise RuntimeError(f"{device['id']}: native phase/neutral mapping mismatch")
    return tuple(abs(nodes[a].volts - nodes[b].volts) / (1000 * device["kv"])
                 for a, b in pairs)


def verify_device_envelope(point, vector, spec, controller, fixed):
    devices, _ = stock_probe_spec()
    ids = tuple(d["id"] for d in devices["storage_devices"])
    if set(vector) != set(ids) or len(point.soc) != len(ids) or len(point.storage_pq) != len(ids):
        raise RuntimeError("prospective vector must cover the exact four device identities")
    params, limits = controller["storage"], controller["controller"]
    gates = fixed["gates"]
    low, high = spec["terminal_voltage_pu"]
    for device, soc, pq in zip(devices["storage_devices"], point.soc, point.storage_pq, strict=True):
        id_ = device["id"]
        requested = vector[id_]
        samples = _device_samples(point, id_)
        if not all(low <= pu <= high for pu in _device_voltages(point, device)):
            raise RuntimeError(f"{id_}: terminal voltage outside [{low}, {high}] engineering screen")
        # One-phase line and neutral; three-phase delta line conductors,
        # not the branch current inside the delta.
        derived_a = device["kw"] / (device["kv"] * (math.sqrt(3) if len(device["phases"]) == 3 else 1))
        if max(p.amps for p in samples.values()) > derived_a or math.hypot(pq.kw, pq.kvar) > device["kw"]:
            raise RuntimeError(f"{id_}: native current or apparent power exceeds model-device screen")
        if (
            not 0 <= requested <= limits["per_device_limit_kw"]
            or not _near(pq.kw, requested, gates["storage_p_kw"])
            or not _near(pq.kvar, 0, gates["storage_q_kvar"])
        ):
            raise DeviceDispatchMismatch(f"{id_}: signed terminal P/Q delivery outside screen")
        remaining = soc * params["capacity_kwh"] - (
            max(0, pq.kw) * limits["exchange_s"] / (3600 * params["discharge_efficiency"])
        )
        if not (params["reserve_soc"] <= soc <= 1
                and remaining >= params["reserve_soc"] * params["capacity_kwh"]):
            raise RuntimeError(f"{id_}: insufficient measured discharge energy above reserve")


def _assert_parity(a, b, spec):
    tol = spec["parity_abs"]
    pa = (*a.storage_pq, a.source_pq)
    pb = (*b.storage_pq, b.source_pq)
    if (
        a.state != b.state or not _near(a.clock_s, b.clock_s, tol["seconds"])
        or len(a.soc) != len(b.soc) or len(pa) != len(pb)
        or len(a.conductors) != len(b.conductors)
        or not all(_near(x, y, tol["soc"]) for x, y in zip(a.soc, b.soc))
        or not all(_near(p.kw, q.kw, tol["power_kw_or_kvar"])
                   and _near(p.kvar, q.kvar, tol["power_kw_or_kvar"]) for p, q in zip(pa, pb))
        or any(
            (x.element, x.terminal, x.node, x.opened) != (y.element, y.terminal, y.node, y.opened)
            or abs(x.volts - y.volts) > tol["volts"]
            or not _near(x.amps, y.amps, tol["amps"])
            for x, y in zip(a.conductors, b.conductors)
        )
    ):
        raise RuntimeError("scratch baseline differs from live native operating point")


def _probe_same_state(point, base, tol):
    if (
        point.state != base.state or not _near(point.clock_s, base.clock_s, tol["seconds"])
        or point.soc != base.soc
    ):
        raise RuntimeError("scratch solve changed clock, native SOC or static feeder state")


def prospective(live, last_vector, proposed_vector, spec=None, *, _factory=Native123Feeder):
    """Check one complete constrained vector. Never writes a live dispatch.

    Only the exact pinned static model is eligible. If reconstruction or
    pre-action baseline parity fails, there is no usable authority.
    """
    spec = envelope() if spec is None else spec
    fixed = load_protocol()
    controller = fixed["device_limits"]
    current = collect(live, spec)
    ids = tuple(live.storage_ids)
    verify_device_envelope(current, last_vector, spec, controller, fixed)
    if set(proposed_vector) != set(ids) or not all(
        isinstance(u, (float, int)) and not isinstance(u, bool) and math.isfinite(u)
        and 0 <= u <= controller["controller"]["per_device_limit_kw"] for u in proposed_vector.values()
    ):
        raise RuntimeError("invalid complete prospective dispatch vector")
    original = None
    try:
        # Check object, solver context and filesystem identities before taking
        # ownership. Exiting an alias could close the live native context or
        # delete its model copy. An injected factory must return a bare feeder.
        scratch = _factory(live.calibration.target_t0)
        s = scratch.session
        if scratch is live or s is live.session or s.dss is live.session.dss:
            raise RuntimeError("independent DSS context required")
        scratch_path, live_path = s.folder.resolve(), live.session.folder.resolve()
        if scratch_path.is_relative_to(live_path) or live_path.is_relative_to(scratch_path):
            raise RuntimeError("independent DSS temporary model copy required")
        scratch_error = None
        try:
            if s.source != live.session.source:
                raise RuntimeError("scratch deck identity mismatch")
            pristine = collect(scratch, spec)
            if pristine.state != current.state:
                raise RuntimeError("static live state cannot be reconstructed from pinned reset")
            for device, soc in zip(ids, current.soc, strict=True):
                s.command(f"Edit {device} %stored={100 * soc:.17g}")
                short = device.split(".", 1)[1]
                s.batteries[short].soc = float(s.read_property(device, "%stored")) / 100
            restored_soc = tuple(s.batteries[device.split(".", 1)[1]].soc for device in ids)
            s.apply({device: {"kw": last_vector[device]} for device in ids})
            s.solve(seconds=current.clock_s, native_controls=False)
            restored = collect(scratch, spec)
            if restored.soc != restored_soc:
                raise RuntimeError("scratch baseline solve advanced native SOC")
            _assert_parity(current, restored, spec)
            s.apply({device: {"kw": proposed_vector[device]} for device in ids})
            s.solve(native_controls=False)
            projected = collect(scratch, spec)
            _probe_same_state(projected, restored, spec["parity_abs"])
            verify_device_envelope(projected, proposed_vector, spec, controller, fixed)
        except Exception as exc:
            scratch_error = exc
            raise
        finally:
            try:
                scratch.close()
            except Exception as close_error:
                if scratch_error is not None:
                    raise RuntimeError(
                        f"scratch cleanup failed ({close_error}) after {scratch_error}"
                    ) from scratch_error
                raise
    except Exception as exc:
        original = exc
        raise
    finally:
        # Verify live electrical and energy state on success AND rejection.
        # Keep the original scratch failure in the exception chain if both fail.
        try:
            _assert_parity(current, collect(live, spec), spec)
        except Exception as parity:
            if original is not None:
                raise RuntimeError(
                    f"LIVE feeder parity verification failed ({parity}) after scratch "
                    f"rejection: {original}"
                ) from original
            raise RuntimeError(f"LIVE feeder parity verification failed: {parity}") from parity
    return projected
