"""Native frequency feedback, one-second delayed feeder support, matched disturbance."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference
from envs.lfc.receiver_123 import read_causal
from scripts.check_lfc34_flexible_load import ROOT, native
from scripts.check_lfc_123_coupling import SCENARIO, _copy_model, _poll
from scripts.lfc_bus8_receiver import _advance, _instrument, _native_frequency, _verify_prefixes
from research.lfc_dual_pcc.capacity_feeders import NativePolicyControl, digest, feeder_config
from research.lfc_dual_pcc.case_a_plant import feeder
from research.lfc_dual_pcc.replay_capacity_transmission import (
    RELEASE, SITES, START, engine, measure, port_specs, simulate,
)
from research.lfc_dual_pcc.stage0 import CONFIG, write_ports


# Fixed duty cycle and droop scale. The switch times index the one-second holds.
DISTURBANCE_KW = (30., 30., 30., 0., 0., 0., 20., 20., 20., 0., 0., 0.)
FULL_ACTIVATION_HZ = .00020


def grid_input(specs, load, response):
    ports = {site: spec["anchor_kw_kvar"][:] for site, spec in specs.items()}
    ports["A"][0] += load
    for site, support in response.items():
        ports[site] = [v - dv for v, dv in zip(ports[site], support, strict=True)]
    return ports


def feedback(reference_hz, measured_hz):
    return min(1., max(0., (reference_hz - measured_hz) / FULL_ACTIVATION_HZ))


def run_episode(eng, model, clock, specs, deck, reference, weights, episode):
    count = episode["n_sites"]
    cfg, labels = feeder_config(episode["energy_kws"], episode["ramp_up_kw_s"])
    ctrl = NativePolicyControl(episode, weights)
    active = len(DISTURBANCE_KW)
    if len(episode["steps"]) != active or len(labels) != count:
        raise ValueError("incompatible feeder population or capacity schedule")
    end_slot = START + active + RELEASE
    eng.set_param(model, "StopTime", str(end_slot + 1), nargout=0)
    eng.set_param(f"{model}/pause_time", "Value", str(START), nargout=0)
    write_ports(eng, model, specs, grid_input(specs, 0., {}))
    eng.set_param(model, "SimulationCommand", "start", nargout=0)
    _poll(eng, model, "paused", timeout_s=160)
    now = float(eng.get_param(model, "SimulationTime"))
    if not START - 1e-7 <= now <= START + clock["max_pause_lag_s"] + 1e-7:
        raise ValueError("invalid native initial pause")
    previous = now - 1
    prefixes, holds = [], []
    pending = {}
    with ExitStack() as stack:
        plants = [stack.enter_context(feeder(deck, reference, cfg, name)) for name in labels]
        for i in range(active + RELEASE):
            samples = _native_frequency(eng)
            prefixes.append(samples)
            observed = read_causal(lambda: samples, previous, now, clock)
            if i == 0:
                quiet = [f.hz for f in samples if START - 2 <= f.time_s < START]
                if len(quiet) != 10:
                    raise ValueError("incomplete two-second pre-event baseline")
                reference_hz = fmean(quiet)
            request = episode["steps"][i]["request_kw"] if i < active else 0.
            activation = feedback(reference_hz, observed.hz) if i < active else 0.
            load = DISTURBANCE_KW[i] if i < active else 0.
            values = grid_input(specs, load, pending)
            applied = pending
            pending = {}
            actions, capacity, target = [], [], []
            if i < active:
                actions, capacity, target = ctrl.dispatch(plants, i, request, activation)
            write_ports(eng, model, specs, values)
            if float(eng.get_param(model, "SimulationTime")) != now:
                raise ValueError("clock advanced inside the port write transaction")
            end = _advance(eng, model, now, START+i+1, clock)
            if i < active:
                for site, plant, action in zip(SITES[:count], plants, actions, strict=True):
                    row = plant.execute(action)
                    pending[site] = [row["delivered_kw"], plant.base[1] - row["pcc"][1]]
            holds.append(dict(start_s=now, end_s=end, freq_sample_s=observed.time_s, freq_hz=observed.hz,
                              disturbance_kw=load, request_kw=request, activation=activation,
                              call_kw=request*activation, capacity_kw=capacity, targets_kw=target, actions=actions,
                              measured_support_kw_kvar=pending, applied_support_kw_kvar=applied,
                              port_setpoints_kw_kvar=values))
            previous, now = now, end
    eng.set_param(model, "SimulationCommand", "stop", nargout=0)
    _poll(eng, model, "stopped")
    stopped = _native_frequency(eng)
    _verify_prefixes(prefixes, stopped)
    samples = [[f.time_s, f.hz] for f in stopped]
    errors = [h["call_kw"] - sum(v[0] for v in h["measured_support_kw_kvar"].values())
              for h in holds[:active]]
    # As seen by the grid during this hold: includes one initial hold of transport lag
    # and the last outstanding response, which is flushed during release.
    grid_errors = [h["call_kw"] - sum(v[0] for v in h["applied_support_kw_kvar"].values()) for h in holds]
    return dict(n_sites=count, arm=episode["arm"], train_seed=episode["train_seed"], event_seed=episode["event_seed"],
                reference_hz=reference_hz, writes_s=[[h["start_s"], h["end_s"]] for h in holds],
                bus8_time_hz=samples, holds=holds, call_unmet_wh=sum(max(0., x) for x in errors)/3.6,
                call_excess_wh=sum(max(0., -x) for x in errors)/3.6,
                grid_call_unmet_wh=sum(max(0., x) for x in grid_errors)/3.6,
                grid_call_excess_wh=sum(max(0., -x) for x in grid_errors)/3.6)


def compare(result):
    """Audit feedback causality and contrast the two arms against their no-support trace."""
    arms = {a["arm"]: a for a in result["episodes"]}
    rule, fixed = (arms[k] for k in ("rule", "fixed_lower"))
    if rule["writes_s"] != fixed["writes_s"] or rule["reference_hz"] != fixed["reference_hz"]:
        raise ValueError("mismatched initial state or event clock")
    alpha = {}
    for name, arm in arms.items():
        last = {}
        active = arm["holds"][:len(DISTURBANCE_KW)]
        for i, hold in enumerate(arm["holds"]):
            expected = DISTURBANCE_KW[i] if i < len(active) else 0.
            if hold["disturbance_kw"] != expected or hold["applied_support_kw_kvar"] != last:
                raise ValueError("invalid pulse or causal feeder hold")
            if not hold["freq_sample_s"] < hold["start_s"] < hold["end_s"]:
                raise ValueError("invalid frequency sampling order")
            expected_alpha = feedback(arm["reference_hz"], hold["freq_hz"]) if i < len(active) else 0.
            if hold["activation"] != expected_alpha:
                raise ValueError("feedback differs from observed-frequency activation")
            last = hold["measured_support_kw_kvar"]
        alpha[name] = [min(h["activation"] for h in active), max(h["activation"] for h in active)]
    pairs = list(zip(rule["holds"][:len(DISTURBANCE_KW)], fixed["holds"][:len(DISTURBANCE_KW)], strict=True))
    cap_gaps = [max(abs(a-b) for a,b in zip(r["capacity_kw"],f["capacity_kw"],strict=True))
                for r,f in pairs[::2]]
    iae0 = result["no_support"]["iae_from_neutral_hz_s"]
    return dict(changed_capacity_windows=sum(d > 1e-6 for d in cap_gaps),
                max_capacity_gap_kw=max(cap_gaps),
                changed_fast_holds=sum(r["actions"] != f["actions"] for r,f in pairs),
                changed_station_actions=sum(a != b for r,f in pairs
                                            for a,b in zip(r["actions"], f["actions"], strict=True)),
                max_frequency_gap_hz=max(abs(a[1]-b[1]) for a,b in
                                         zip(rule["bus8_time_hz"], fixed["bus8_time_hz"], strict=True)),
                alpha_range=alpha,
                iae_change_from_no_support_hz_s={k:v["iae_from_neutral_hz_s"]-iae0 for k,v in arms.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("reference", "weights", "episodes", "calibration", "out"):
        p.add_argument("--"+key, required=True, type=Path)
    p.add_argument("--n-sites", required=True, type=int, choices=(2,3))
    args = p.parse_args()
    if args.out.exists():
        p.error("supply a new output path")
    wire = json.loads(CONFIG.read_bytes())
    source = json.loads(SCENARIO.read_bytes())
    specs = port_specs(source, wire)
    calibration = json.loads(args.calibration.read_bytes())
    if (calibration["model_sha256"] != source["model_sha256"] or calibration["ports"] != specs
            or calibration["steps"] != len(DISTURBANCE_KW) or calibration["release_s"] != RELEASE):
        raise ValueError("incompatible earlier neutral reference")
    clock = wire["clock"]
    ref = ObservedReference.read_report(args.reference.read_bytes())
    weights = json.loads(args.weights.read_bytes())
    templates = [e for e in json.loads(args.episodes.read_bytes())["episodes"]
                 if e["n_sites"] == args.n_sites and e["arm"] in ("rule", "fixed_lower")]
    if len(templates) != 2 or len({e["arm"] for e in templates}) != 2:
        raise ValueError("expected a rule and an upper-learning arm")
    a,b = templates
    if any(a[k] != b[k] for k in ("n_sites", "train_seed", "event_seed", "energy_kws", "ramp_up_kw_s")
           ) or [s["request_kw"] for s in a["steps"]] != [s["request_kw"] for s in b["steps"]]:
        raise ValueError("mismatched initial state or exogenous capacity schedule")
    result = dict(model_sha256=source["model_sha256"], ports=specs, full_activation_hz=FULL_ACTIVATION_HZ,
                  disturbance_kw=DISTURBANCE_KW, feeder_transport_delay_s=1, calibration_sha256=digest(args.calibration),
                  weights_sha256=digest(args.weights), reference_sha256=digest(args.reference),
                  episodes_sha256=digest(args.episodes), episodes=[])
    with TemporaryDirectory() as tmp, engine() as eng:
        scratch = Path(tmp)
        deck = native.archived_deck(ROOT, scratch)
        model = _copy_model(source, scratch)
        eng.cd(str(scratch), nargout=0)
        eng.load_system(model, nargout=0)
        _instrument(eng, model, source["sensor"])
        if not math.isclose(float(eng.eval("Ts", nargout=1)), 1/60, abs_tol=1e-9):
            raise ValueError("expected the pinned 60 Hz model")
        neutral = calibration["neutral"]
        no_support = simulate(eng, model, clock, specs, [grid_input(specs, kw, {}) for kw in DISTURBANCE_KW])
        result["no_support"] = dict(**no_support, **measure(no_support, neutral))
        for template in templates:
            run = run_episode(eng, model, clock, specs, deck, ref, weights, template)
            run.update(measure(run, neutral))
            result["episodes"].append(run)
            print(template["arm"], run["iae_from_neutral_hz_s"], flush=True)
    result["comparison"] = compare(result)
    with args.out.open("x") as f:
        json.dump(result, f, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
