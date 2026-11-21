"""Seconds-scale measured IEEE34 flexible demand at 890 in the held study."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

from scripts import check_lfc34_coupling_inputs as native
from scripts import check_lfc34_tap_hold as held
from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "scenarios/lfc/ieee34-flexible-load-v1.json"


class FlexibleLoad:
    def __init__(self, context, deck, baseline, design):
        self.ctx = context
        self.reference = baseline
        self.design = design
        self.circuit = native._configure(context, deck, (0., 0.))
        circuit = self.circuit
        neutral, _ = native._capture(context, circuit, (0., 0.), 0)
        original = neutral.checked_import(baseline)
        if max(abs(a-b) for a,b in zip(original, baseline.feeder_import_kw_kvar)) > 1e-6:
            raise RuntimeError("original working point differs")
        self.taps = held.read_taps(circuit)
        context.Text.Command = "Set ControlMode=OFF"
        circuit.Solution.StepSize = 1.0
        if circuit.Solution.StepSize != 1 or self.time() != 0:
            raise RuntimeError("DSS initialization clock")
        self.elapsed = 0
        self.operating_point = self.solve(design["neutral_kw"], 0)["pcc"]
        self.previous_kw = design["neutral_kw"]

    def time(self):
        return held.read_dss_clock(self.circuit.Solution)["absolute_seconds"]

    def solve(self, kw, kvar):
        circuit = self.circuit
        if not (0 <= kw <= self.design["maximum_kw"] and abs(kvar) <= self.design["max_q_kvar"]):
            raise ValueError("flexible demand envelope")
        held.apply_arm(self.ctx, circuit, (kw, kvar))
        circuit.Solution.SolveNoControl()
        if held.read_taps(circuit) != self.taps or abs(self.time() - self.elapsed) > 1e-7:
            raise RuntimeError("mechanical or DSS clock mismatch")
        request = (kw, kvar) if kw or kvar else None
        snapshot, details = native._capture(self.ctx, circuit, request, self.elapsed,
                                            control_mode="OFF", actual_clock_s=self.time())
        pcc = snapshot.checked_import(self.reference, expected_binding=held.HELD_BINDING)
        if not held.all_finite(details):
            raise RuntimeError("invalid native electrical evidence")
        return dict(start_s=self.time(), command_kw_kvar=(kw, kvar), pcc=pcc,
                    snapshot=asdict(snapshot), **details)

    def step(self, kw, kvar=0.):
        if abs(kw - self.previous_kw) > self.design["ramp_kw_per_s"]:
            raise ValueError("one-second ramp gate")
        record = self.solve(kw, kvar)
        self.circuit.Solution.FinishTimeStep()
        self.elapsed += 1
        end = self.time()
        if abs(end - self.elapsed) > 1e-7 or held.read_taps(self.circuit) != self.taps:
            raise RuntimeError("DSS did not finish exactly one held second")
        self.previous_kw = kw
        record["end_s"] = end
        record["pcc_delta"] = tuple(a-b for a,b in zip(record["pcc"], self.operating_point))
        record["pcc_support_kwh"] = -record["pcc_delta"][0] * (end - record["start_s"]) / 3600
        return record


def run(reference, output):
    import dss

    design = json.loads(DESIGN.read_bytes())
    baseline = ObservedReference.read_report(reference.read_bytes())
    native.verify_checkout(ROOT, Path.cwd())
    with tempfile.TemporaryDirectory() as temp, output.open("x") as stream:
        deck = native.archived_deck(ROOT, Path(temp))
        for name, pulse in design["pulse_kw_kvar"].items():
            ctx = dss.DSS.NewContext()
            ctx.AllowChangeDir = False
            try:
                driver = FlexibleLoad(ctx, deck, baseline, design)
                for slot in range(design["seconds"]):
                    active = design["start_s"] <= slot < design["release_s"]
                    kw = design["neutral_kw"] + (pulse[0] if active else 0)
                    row = driver.step(kw, pulse[1] if active else 0)
                    stream.write(json.dumps(dict(case=name, **row), allow_nan=False) + "\n")
                    stream.flush()
                    for observed, requested in zip(row["pcc_delta"], pulse):
                        if active and requested and requested * observed <= 0:
                            raise RuntimeError(f"{name}: wrong measured PCC direction")
                    if not active and max(map(abs, row["pcc_delta"])) > design["neutral_tol_kw_kvar"]:
                        raise RuntimeError(f"{name}: neutral did not recover")
            finally:
                ctx.ClearAll()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.reference, args.output)
