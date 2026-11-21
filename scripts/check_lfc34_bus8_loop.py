"""IEEE34 flexible load and Bus8 measured frequency, one serial engine per arm."""
import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

from scripts.check_lfc34_flexible_load import DESIGN, FlexibleLoad, ROOT, native
from envs.lfc.diagnostics.ieee34_bus8 import BUS8_ANCHOR, ObservedReference

MANIFEST = DESIGN.with_name("ieee34-bus8-loop-v1.json")


def import_receiver(repo, version):
    # Pinned release projection of the receiver used by the archived protocol.
    # These byte identities work in a clean checkout without private Git objects.
    if version != "5e95976da768d92cefec31f35cbad2078f3bd856" or repo.resolve() != ROOT.resolve():
        raise ValueError("unexpected engineering receiver revision or checkout")
    repo = repo.resolve()
    files = {
        "scripts/lfc_bus8_exchange.py":
            "1e42436eb9d162c6525e669927d38abe2b2f31f9679c2bf92baf38b91bbd460b",
        "scripts/lfc_bus8_receiver.py":
            "94b2c407604c289dda622cf60de086b86cda6e4062f1a301ecbea284a22e7867",
        "scripts/check_lfc_123_coupling.py":
            "f52049811a699a38633f7b40b598c8077df7cb0619b1a2c892bfdd8d03307092",
        "envs/lfc/native_feeder_123.py":
            "a8c3a7c93366288e775a1706ba92749088cf78f5ccd93f998a1d58312a79ad0a",
        "envs/lfc/receiver_123.py":
            "76b3d807759503f5fa23e7a5d01139e4bfd7fe9b402ec67fae3c1d13d2d50636",
        "scenarios/lfc/ieee123-native/coupling_v1.json":
            "80ee971b37bc7ed90771942a36c1673c3e346f64715b13948eee9a09adba9329",
    }
    for rel, expected in files.items():
        if hashlib.sha256((repo / rel).read_bytes()).hexdigest() != expected:
            raise ValueError(f"receiver source mismatch {rel}")
    from scripts.lfc_bus8_exchange import exchange
    for rel in files:
        if rel.endswith(".py"):
            module = sys.modules[rel[:-3].replace("/", ".")]
            if Path(module.__file__).resolve() != repo/rel:
                raise ValueError(f"unexpected import {rel}")
    return exchange


def support(hz, reference, previous, policy):
    error = reference - hz
    demand = math.copysign(max(0, abs(error) - policy["deadband_hz"]), error)
    limit, ramp = policy["capacity_kw"], policy["ramp_kw_s"]
    target = max(-limit, min(limit, policy["gain_kw_per_hz"] * demand))
    return max(previous-ramp, min(previous+ramp, target))


def run(spec, design, reference, exchange, emit):
    import dss

    c, g = spec["clock"], spec["gates"]
    groups = {}
    with tempfile.TemporaryDirectory() as tmp:
        deck = native.archived_deck(ROOT, Path(tmp))
        for name, pulse in spec["events_kw_kvar"].items():
            ctx = dss.DSS.NewContext()
            ctx.AllowChangeDir = False
            controlled = name in ("feedback_cut_return", "closed")
            history = []
            emit({"case": name, "phase": "started"})
            try:
                plant = FlexibleLoad(ctx, deck, reference, design)
                def dispatch(slot, pause, measured):
                    active = c["event_start_s"] <= slot < c["event_stop_s"]
                    prior = history[-1]["support_kw"] if history else 0.
                    request = 0.
                    if controlled and slot >= c["event_start_s"]:
                        request = support(measured.hz, f_ref, prior, spec["controller"])
                    result = plant.step(design["neutral_kw"] - request)
                    if abs(result["start_s"] - (slot - c["first_hold_s"])) > 1e-7:
                        raise RuntimeError("DSS frame does not match receiver boundary")
                    delta = result["pcc_delta"]
                    applied = tuple(a + (p if active else 0) +
                                    (0 if name == "feedback_cut_return" else d)
                                    for a,p,d in zip(BUS8_ANCHOR, pulse, delta))
                    history.append({"slot":slot, "support_kw":request, "pcc_delta":delta,
                                    "measured":measured, "target":applied})
                    emit({"case":name, "phase":"feeder", "write_boundary":pause, **history[-1],
                          "dss": result})
                    return applied
                hz, rows = exchange(c, dispatch, lambda r: emit({"case": name, **r}))
            finally:
                ctx.ClearAll()
            groups[name] = hz, rows, history
            if name == "zero_a":
                continue
            zero, base_rows, base_history = groups["zero_a"]
            difference = []
            for a,b in zip(hz, zero, strict=True):
                if abs(a.time_s-b.time_s) > 1e-7:
                    raise ValueError("unaligned native time")
                difference.append((a.time_s, a.hz-b.hz))
            for row, base, cmd, other in zip(rows, base_rows, history, base_history, strict=True):
                if abs(row["write_s"]-base["write_s"]) > 1e-7 or abs(row["end_s"]-base["end_s"]) > 1e-7:
                    raise ValueError("unpaired actual hold")
                tol = g["zero_power"] if name == "zero_b" else g["sensor_delta_error"]
                for m, b, target, ref in zip(row["sensor"], base["sensor"], cmd["target"], other["target"]):
                    if abs((m-b) - (target-ref)) > tol:
                        raise ValueError(f"{name} paired receiving P/Q mismatch at {row['slot']}")
            write = next(r["write_s"] for r in rows if r["slot"] == c["event_start_s"])
            release = next(r["write_s"] for r in rows if r["slot"] == c["event_stop_s"])
            if name == "zero_b":
                noise = max(abs(v) for t,v in difference if t >= rows[0]["write_s"])
                if noise > g["zero_hz"]:
                    raise ValueError("independent zero repeat unresolved")
                band = max(g["frequency_floor_hz"], 3*noise)
            if any(abs(v) > band for t,v in difference if t < write):
                raise ValueError("unpaired pre-event frequency")
            on = [(t,v) for t,v in difference if write < t < release]
            if name == "plus":
                f_ref = statistics.mean(s.hz for s in hz if write-2 < s.time_s < write)
            sign = -1 if pulse[0] > 0 else 1
            seen = [sign*v > band if pulse[0] else abs(v) > band for _,v in on]
            onset = next((t for ((t,_),a,b) in zip(on, seen, seen[1:]) if a and b), None)
            event = [h for h in history if c["event_start_s"] <= h["slot"] < c["event_stop_s"]]
            entry = dict(phase="assessment",case=name,band_hz=band,onset_s=onset,
                         event_min_hz=min(v for _,v in on),event_max_hz=max(v for _,v in on),
                         pcc_event_kwh=sum(-h["pcc_delta"][0] for h in event)/3600)
            emit(entry)
            if name in ("plus","minus") and onset is None:
                raise RuntimeError(f"{name} signed response unresolved; stop before feedback")
            if name == "closed":
                cut = groups["feedback_cut_return"]
                good = []
                for r, h, off, b in zip(rows, history, cut[1], groups["plus"][2], strict=True):
                    expected = h["pcc_delta"][0] - b["pcc_delta"][0]
                    delivered = r["sensor"][0] - off["sensor"][0]
                    good.append(abs(expected) > g["delivery_min_kw"] and expected*delivered > 0
                                and abs(delivered-expected) <= g["sensor_delta_error"])
                count = g["consecutive_deliveries"]
                if not any(all(good[i:i+count]) for i in range(len(good)-count+1)):
                    raise RuntimeError("insufficient consecutive measured return-path deliveries")
    emit({"phase":"complete"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runtime",type=Path)
    p.add_argument("baseline",type=Path)
    p.add_argument("report",type=Path)
    args = p.parse_args()
    spec, design = (json.loads(f.read_bytes()) for f in (MANIFEST, DESIGN))
    receiver = import_receiver(args.runtime, spec["runtime"])
    baseline = ObservedReference.read_report(args.baseline.read_bytes())
    os.umask(0o077)
    with args.report.open("x") as out:
        def encode(value):
            if is_dataclass(value):
                return asdict(value)
            raise TypeError(type(value).__name__)
        def emit(record):
            out.write(json.dumps(record, default=encode, allow_nan=False) + "\n")
            out.flush()
            if record["phase"] in ("assessment","failed","complete"):
                print(json.dumps(record), flush=True)
        emit({"phase":"design","config":spec,"dss":design,"git":native.git(ROOT,"rev-parse","HEAD"),
              "reference":baseline.sha256,"manifest":hashlib.sha256(MANIFEST.read_bytes()).hexdigest()})
        try:
            run(spec, design, baseline, receiver, emit)
        except Exception as error:
            emit({"phase":"failed","reason":repr(error)})
            raise


if __name__ == "__main__":
    main()
