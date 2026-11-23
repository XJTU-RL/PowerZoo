"""Replay measured feeder injections through the pinned native 60 Hz IEEE9 model."""
import argparse
from hashlib import sha256
from contextlib import contextmanager
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree
from zipfile import ZipFile

from scripts.check_lfc_123_coupling import ROOT, SCENARIO, _copy_model, _poll
from scripts.lfc_bus8_receiver import _advance, _instrument, _native_frequency
from research.lfc_dual_pcc.stage0 import CONFIG, write_ports
from research.lfc_dual_pcc.topology import check


START = 20
RELEASE = 4
SITES = ("A", "B", "C")


def port_specs(source, wiring):
    """Bus5's third dynamic load is on its own branch, parallel to fixed Load A."""
    path = ROOT / source["model"]
    check(path, wiring)
    with ZipFile(path) as archive:
        tree = ElementTree.fromstring(archive.read("simulink/systems/system_root.xml"))
    blocks = {b.get("SID"): b.get("Name") for b in tree.findall("Block")}
    edges = [{p.text for p in line.iter("P") if p.get("Name") in {"Src", "Dst"}}
             for line in tree.findall("Line")]
    for a, b in (("1891#rconn:2", "3686#lconn:1"), ("3689#out:1", "3687#in:1"),
                 ("3690#out:1", "3688#in:1"), ("3687#rconn:1", "3686#lconn:2"),
                 ("3688#rconn:1", "3686#lconn:3")):
        if edges.count({a, b}) != 1:
            raise ValueError("Bus5 dynamic load or control path changed")
    if [blocks[k] for k in ("3689", "3690")] != ["loadbus5p1", "loadbus5q1"]:
        raise ValueError("unexpected Bus5 control blocks")
    # Positive common operating point for bidirectional incremental P and Q.
    return dict(wiring["ports"], C=dict(bus=5, blocks=["loadbus5p1", "loadbus5q1"],
                                         anchor_kw_kvar=[20., 20.]))


def injections(episode, specs):
    count = episode["n_sites"]
    if count not in (2, 3) or not episode["steps"]:
        raise ValueError("expected a two- or three-feeder episode")
    rows = []
    for step in episode["steps"]:
        p, q, target = (step[k] for k in ("delivered_kw", "q_support_kvar", "target_kw"))
        if any(len(v) != count or not all(map(math.isfinite, v)) for v in (p, q, target)):
            raise ValueError("invalid per-feeder delivery or target")
        command = {s: list(spec["anchor_kw_kvar"]) for s, spec in specs.items()}
        demand = step["request_kw"] * step["activation"]
        if not math.isfinite(demand) or demand < 0:
            raise ValueError("invalid external service call")
        command["A"][0] += demand
        for s, dp, dq in zip(SITES[:count], p, q, strict=True):
            command[s][0] -= dp
            command[s][1] -= dq
        rows.append(command)
    return rows


def simulate(eng, model, clock, specs, commands):
    neutral = {s: spec["anchor_kw_kvar"] for s, spec in specs.items()}
    timeline = [*commands, *([neutral] * RELEASE)]
    eng.set_param(model, "StopTime", str(START + len(timeline) + 1), nargout=0)
    eng.set_param(f"{model}/pause_time", "Value", str(START), nargout=0)
    write_ports(eng, model, specs, neutral)
    eng.set_param(model, "SimulationCommand", "start", nargout=0)
    _poll(eng, model, "paused", timeout_s=160)
    now = float(eng.get_param(model, "SimulationTime"))
    if not START-1e-7 <= now <= START+clock["max_pause_lag_s"]+1e-7:
        raise ValueError("unexpected initial time")
    writes = []
    for i, values in enumerate(timeline):
        write_ports(eng, model, specs, values)
        end = _advance(eng, model, now, START+i+1, clock)
        writes.append([now, end])
        now = end
    eng.set_param(model, "SimulationCommand", "stop", nargout=0)
    _poll(eng, model, "stopped")
    samples = [[f.time_s, f.hz] for f in _native_frequency(eng)]
    if (not samples or abs(samples[0][0]) > 1e-7 or abs(samples[0][1]-60) > .01
            or samples[-1][0] < START+len(timeline)-.21
            or any(not math.isfinite(f) for _, f in samples)
            or any(abs(b[0]-a[0]-.2) > 1e-7 for a, b in zip(samples, samples[1:]))):
        raise ValueError("invalid native Bus8 frequency record")
    return dict(writes_s=writes, bus8_time_hz=samples)


def measure(result, reference):
    f, f0 = result["bus8_time_hz"], reference["bus8_time_hz"]
    if [t for t, _ in f] != [t for t, _ in f0] or result["writes_s"] != reference["writes_s"]:
        raise ValueError("unmatched time grid or physical holds")
    start = result["writes_s"][0][0]
    if any(abs(x-y) > 1e-7 for (t, x), (_, y) in zip(f, f0) if t <= start):
        raise ValueError("initial native frequency mismatch")
    diff = [(t, x-y) for (t, x), (_, y) in zip(f, f0) if t >= start]
    integral = sum((b[0]-a[0])*(abs(a[1])+abs(b[1]))/2 for a, b in zip(diff, diff[1:]))
    return dict(iae_from_neutral_hz_s=integral, peak_from_neutral_hz=max(abs(x) for _, x in diff),
                nadir_hz=min(x for t, x in f if t >= start))


@contextmanager
def engine():
    import matlab.engine
    session = matlab.engine.start_matlab()
    try:
        yield session
    finally:
        session.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics", type=Path, help="native per-feeder trajectories; omit for neutral and Bus5 probe")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.physics) != bool(args.calibration) or args.steps < 1 or args.out.exists():
        parser.error("supply physics and calibration together; use a new output file")
    wire = json.loads(CONFIG.read_bytes())
    source = json.loads(SCENARIO.read_bytes())
    specs = port_specs(source, wire)
    neutral = {s: spec["anchor_kw_kvar"] for s, spec in specs.items()}
    result = dict(model_sha256=source["model_sha256"], ports=specs, steps=args.steps, release_s=RELEASE,
                  scope="Open-loop replay: measured feeder P/Q and exogenous activation, no frequency feedback")
    with TemporaryDirectory() as tmp, engine() as eng:
        name = _copy_model(source, Path(tmp))
        eng.cd(tmp, nargout=0)
        eng.load_system(name, nargout=0)
        _instrument(eng, name, source["sensor"])
        if abs(float(eng.eval("Ts", nargout=1))-1/60) > 1e-7:
            raise ValueError("requires native 60 Hz model")
        if args.physics is None:
            common = [neutral] * args.steps
            ref = simulate(eng, name, wire["clock"], specs, common)
            probe = dict(neutral, C=[neutral["C"][0]+30., neutral["C"][1]])
            calibration = simulate(eng, name, wire["clock"], specs, [probe] * args.steps)
            diagnostic = measure(calibration, ref)
            if diagnostic["peak_from_neutral_hz"] < wire["gates"]["frequency_detection_hz"]:
                raise ValueError("Bus5 independent intervention undetected")
            result.update(neutral=ref, bus5_plus_30_kw=calibration, bus5_effect=diagnostic)
        else:
            calibration = json.loads(args.calibration.read_bytes())
            if any(calibration[k] != result[k] for k in ("model_sha256", "ports", "steps", "release_s")):
                raise ValueError("mismatched calibration")
            result.update(physics_sha256=sha256(args.physics.read_bytes()).hexdigest(),
                          calibration_sha256=sha256(args.calibration.read_bytes()).hexdigest(), episodes=[])
            for episode in json.loads(args.physics.read_bytes())["episodes"]:
                rows = injections(episode, specs)
                if len(rows) != args.steps:
                    raise ValueError("mismatched episode duration")
                run = simulate(eng, name, wire["clock"], specs, rows)
                identity = {k: episode[k] for k in ("n_sites", "arm", "train_seed", "event_seed")}
                result["episodes"].append(dict(**identity, **run, **measure(run, calibration["neutral"])))
                print(identity, result["episodes"][-1]["iae_from_neutral_hz_s"], flush=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
