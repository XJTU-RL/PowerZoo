"""Two independent IEEE34 solves and two measured Simscape load branches on one clock."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference
from envs.lfc.native_feeder_123 import PQ
from envs.lfc.receiver_123 import read_causal
from scripts.check_lfc34_flexible_load import DESIGN, FlexibleLoad, ROOT, native
from scripts.check_lfc_123_coupling import SCENARIO, _check_measurement, _copy_model, _poll
from scripts.lfc_bus8_receiver import _advance, _instrument, _native_frequency, _physical_sensor, _verify_prefixes
from .topology import check


CONFIG = Path(__file__).with_name("stage0.json")


def demands(cfg, case, slot):
    result = {port: (0.0, 0.0) for port in cfg["ports"]}
    for event in cfg["events"]:
        if event["from_s"] <= slot < event["until_s"]:
            for port in result:
                if case in ("joint", port.lower() + "_only"):
                    result[port] = tuple(event[port])
    return result


def write_ports(engine, model, specs, values):
    for port, spec in specs.items():
        for block, v in zip(spec["blocks"], values[port], strict=True):
            if not math.isfinite(v):
                raise ValueError("nonfinite transmission target")
            engine.set_param(f"{model}/{block}", "Value", f"{1000*v:.17g}", nargout=0)
    for port, spec in specs.items():
        for block, v in zip(spec["blocks"], values[port], strict=True):
            actual = float(engine.get_param(f"{model}/{block}", "Value"))
            if not math.isclose(actual, 1000*v, rel_tol=0, abs_tol=1e-6):
                raise RuntimeError(f"{port}/{block}: target readback mismatch")


def acquire(case, reference, emit):
    import dss
    import matlab.engine

    cfg = json.loads(CONFIG.read_bytes())
    if case not in cfg["cases"]:
        raise ValueError("unknown predeclared case")
    source = json.loads(SCENARIO.read_bytes())
    design = json.loads(DESIGN.read_bytes())
    wiring = check(ROOT / source["model"], cfg)
    ref = ObservedReference.read_report(reference.read_bytes())
    clock, ports = cfg["clock"], cfg["ports"]
    first, stop = clock["first_hold_s"], clock["stop_s"]
    eps = clock["timestamp_epsilon_s"]
    anchors = {p: spec["anchor_kw_kvar"] for p, spec in ports.items()}
    emit(dict(phase="manifest", case=case, git=subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(), topology=wiring,
        reference_sha256=ref.sha256, deck_tree=native.DECK_TREE, dss_version=dss.__version__))
    with ExitStack() as stack:
        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        deck = native.archived_deck(ROOT, tmp)
        plants = {}
        for port in ports:
            ctx = dss.DSS.NewContext()
            ctx.AllowChangeDir = False
            stack.callback(ctx.ClearAll)
            plants[port] = FlexibleLoad(ctx, deck, ref, design)
        if plants["A"].ctx is plants["B"].ctx:
            raise RuntimeError("two independent feeder contexts required")
        initial = {p: list(plant.operating_point) for p, plant in plants.items()}
        if max(abs(x-y) for x,y in zip(initial["A"], initial["B"])) > 1e-6:
            raise RuntimeError("unmatched initial feeder imports")
        emit(dict(phase="baseline", feeder=initial, anchor=anchors))
        model = _copy_model(source, tmp)
        eng = matlab.engine.start_matlab()
        stack.callback(eng.quit)
        eng.cd(str(tmp), nargout=0)
        eng.load_system(model, nargout=0)
        _instrument(eng, model, source["sensor"])
        if abs(float(eng.eval("Ts", nargout=1)) - source["clock"]["simulink_base_step_s"]) > eps:
            raise RuntimeError("base step changed")
        eng.set_param(model, "StopTime", str(stop+1), nargout=0)
        eng.set_param(f"{model}/pause_time", "Value", str(first), nargout=0)
        write_ports(eng, model, ports, anchors)
        eng.set_param(model, "SimulationCommand", "start", nargout=0)
        _poll(eng, model, "paused", timeout_s=160)
        start = float(eng.get_param(model, "SimulationTime"))
        if not first-eps <= start <= first+clock["max_pause_lag_s"]+eps:
            raise RuntimeError("invalid first physical pause")
        reports, periods, prefixes = [], [], []
        previous = start-1
        for slot in range(first, stop):
            samples = _native_frequency(eng)
            hz = read_causal(lambda: samples, previous, start, clock)
            prefixes.append(samples)
            eligible = [r for r in reports if r["arrival_s"] < start]
            prior = eligible[-1] if eligible else None
            values = {p: [x+(y-z) for x,y,z in zip(anchors[p],
                       prior["pcc"][p] if prior else initial[p], initial[p], strict=True)] for p in ports}
            local = demands(cfg, case, slot)
            write_ports(eng, model, ports, values)
            if eng.get_param(model, "SimulationStatus") != "paused" or float(eng.get_param(model,"SimulationTime")) != start:
                raise RuntimeError("master advanced during two-port transaction")
            end = _advance(eng, model, start, slot+1, clock)
            periods.append((slot,start,end,values))
            # Fixed feeder source voltage allows the two decoupled local intervals
            # to be solved after the master hold. Only then can their reports exist.
            rows = {}
            for p, plant in plants.items():
                a, b = local[p]
                row = plant.step(design["neutral_kw"]+a, b)
                if (abs(row["start_s"]-(slot-first)) > eps
                        or abs(row["end_s"]-row["start_s"]-1) > eps):
                    raise RuntimeError("DSS interval mismatch")
                rows[p] = row
            report = {"slot": slot, "generated_s": end, "arrival_s": end+clock["report_latency_s"],
                      "pcc": {p: rows[p]["pcc"] for p in ports}}
            reports.append(report)
            emit(dict(phase="hold",slot=slot,write_s=start,end_s=end,frequency=asdict(hz),
                      used_report=prior["slot"] if prior else None,request=local,target=values,
                      report=report,feeders=rows))
            previous,start = start,end
        eng.set_param(model, "SimulationCommand", "stop", nargout=0)
        _poll(eng, model, "stopped")
        frequency = _native_frequency(eng)
        _verify_prefixes(prefixes, frequency)
        trace = [(s.time_s,s.hz) for s in frequency]
        if (not trace or trace[0][0] > first or trace[-1][0] < stop-.2-eps or
                any(abs(b[0]-a[0]-.2)>eps for a,b in zip(trace,trace[1:]))):
            raise RuntimeError("native frequency coverage or period mismatch")
        emit(dict(phase="frequency", trace=trace))
        for port, spec in ports.items():
            sensor = dict(source["sensor"], block=spec["sensor"])
            signals = _physical_sensor(eng,model,sensor)
            emit(dict(phase="physical_trace",port=port,**signals))
            signals["frequency_hz"] = trace
            policy = source["warmup"].copy()
            for key,v in zip(("p_abs_tol_kw","q_abs_tol_kvar"),anchors[port],strict=True):
                policy[key] = max(cfg["gates"]["absolute_sensor_floor_kw_kvar"],
                                  cfg["gates"]["absolute_sensor_fraction"]*abs(v))
            gaps = dict(power=source["clock"]["simulink_base_step_s"]+eps,frequency=.2+eps)
            warm = _check_measurement(PQ(*anchors[port]),signals,policy,first,3,
                                      dict(gaps,power=2*source["clock"]["simulink_base_step_s"]+eps),settled=True)
            emit(dict(phase="warm",port=port,measurement=warm))
            for slot,start,end,values in periods:
                actual=_check_measurement(PQ(*values[port]),signals,policy,end,.2,gaps,
                                          settled=False,hold=(start,end))
                emit(dict(phase="sensor",port=port,slot=slot,write_s=start,end_s=end,
                          pq=[actual["mean_p_kw"],actual["mean_q_kvar"]]))
        emit(dict(phase="complete",holds=len(periods)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case",choices=json.loads(CONFIG.read_bytes())["cases"])
    parser.add_argument("reference",type=Path)
    parser.add_argument("report",type=Path)
    args=parser.parse_args()
    os.umask(0o077)
    with args.report.open("x") as f:
        def emit(row):
            f.write(json.dumps(row,allow_nan=False)+"\n")
            f.flush()
        try:
            acquire(args.case,args.reference,emit)
        except Exception as exc:
            emit(dict(phase="failed",reason=repr(exc)))
            raise


if __name__=="__main__":
    main()
