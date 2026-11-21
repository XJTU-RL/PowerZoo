"""End-to-end wiring check on the assembled engineering tree without a study controller."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

from envs.lfc.native_123_availability import collect, envelope, _assert_parity
from envs.lfc.native_feeder_123 import Native123Feeder, PQ
from scripts.check_lfc123_storage_absorption import PROTOCOL as STORAGE, inspect_hold, rating
from scripts.check_lfc34_flexible_load import DESIGN, FlexibleLoad, native
from scripts.check_lfc34_bus8_loop import ObservedReference, import_receiver, MANIFEST
from scripts.check_lfc_123_coupling import SCENARIO
from scripts.lfc_bus8_receiver import _json_default

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/"scenarios/lfc/platform/witness_v1.json"


@contextmanager
def power_port(feeder, anchor, reference):
    if feeder == "123":
        spec, bounds = envelope(), json.loads(STORAGE.read_bytes())
        with Native123Feeder(anchor) as live, Native123Feeder(anchor) as predictor:
            if live.session.dss is predictor.session.dss or live.session.folder == predictor.session.folder:
                raise ValueError("distinct native predictor required")
            ratings = {name:rating(live.session,name) for name in live.storage_ids}
            if ratings != {name:rating(predictor.session,name) for name in predictor.storage_ids}:
                raise ValueError("predictor nameplates differ")
            if any(abs(v["%stored"]/100-bounds["initial_soc"])>1e-9 for v in ratings.values()):
                raise ValueError("initial SOC differs")
            init = live.calibration.feeder_t0
            def advance(request):
                per_device = request/len(live.storage_ids)
                if abs(per_device) > 4:
                    raise ValueError("outside physical absorption screen")
                _assert_parity(collect(predictor,spec),collect(live,spec),spec)
                preview = inspect_hold(predictor,per_device,bounds,ratings)
                if preview["veto"]:
                    raise ValueError(f"predictor veto {preview['veto']}")
                actual = inspect_hold(live,per_device,bounds,ratings)
                if actual["veto"]:
                    raise ValueError(f"live veto {actual['veto']}")
                _assert_parity(collect(predictor,spec),collect(live,spec),spec)
                pcc = [actual["pcc"][k] for k in ("kw","kvar")]
                return pcc, actual["start_s"],actual["end_s"],actual
            yield (init.kw,init.kvar), advance
    else:
        import dss
        design = json.loads(DESIGN.read_bytes())
        baseline = ObservedReference.read_report(reference.read_bytes())
        ctx = dss.DSS.NewContext()
        ctx.AllowChangeDir=False
        try:
            with tempfile.TemporaryDirectory() as scratch:
                deck = native.archived_deck(ROOT,Path(scratch))
                live = FlexibleLoad(ctx,deck,baseline,design)
                def advance(request):
                    observed = live.step(design["neutral_kw"]-request)
                    return observed["pcc"],observed["start_s"],observed["end_s"],observed
                yield live.operating_point,advance
        finally:
            ctx.ClearAll()


def verify(report):
    from scripts.verify_lfc_platform_evidence import reject_constant
    with report.open() as stream:
        rows = [json.loads(line,parse_constant=reject_constant) for line in stream]
    cfg = json.loads(CONFIG.read_bytes())
    if (not rows or rows[-1]["phase"] != "complete" or rows[0]["protocol"] != hashlib.sha256(CONFIG.read_bytes()).hexdigest()
            or any(r["phase"]=="failed" for r in rows)):
        raise ValueError("incomplete or different engineering check")
    feeder=rows[0]["feeder"]
    if feeder not in ("123","34"):
        raise ValueError("unrecognized native feeder")
    h = [r for r in rows if r["phase"]=="port"]
    sensors = [r for r in rows if r["phase"]=="sensor_hold"]
    writes = [r for r in rows if r["phase"]=="write"]
    signals, = [r["signals"] for r in rows if r["phase"]=="native_logs"]
    reference, = [r["hz"] for r in rows if r["phase"]=="reference"]
    anchor = json.loads(SCENARIO.read_bytes())["source_anchor"]
    clock, rule, limits = cfg["clock"],cfg["actuation"],cfg["criteria"]
    points = dict(signals["frequency_hz"])
    start, = [w["pause_s"] for w in writes if w["slot"]==clock["event_start_s"]]
    quiet = [w["native"]["hz"] for w in writes if start-rule["reference_window_s"]<w["native"]["time_s"]<start]
    if len(quiet)!=round(rule["reference_window_s"]/clock["exchange_s"]) or abs(reference-statistics.mean(quiet))>1e-10:
        raise ValueError("pre-pulse reference differs from native boundary reads")
    errors, support, statuses = [], [], []
    slots = list(range(clock["first_hold_s"],clock["stop_s"]))
    for slot,frame,sensor,write in zip(slots,h,sensors,writes,strict=True):
        if (frame["slot"],sensor["slot"],write["slot"]) != (slot,slot,slot):
            raise ValueError("slot mismatch")
        if not 0 < write["pause_s"]-write["native"]["time_s"]<=.2+1e-7:
            raise ValueError("stale frequency")
        t=write["native"]["time_s"]
        if t not in points or abs(points[t]-write["native"]["hz"])>1e-8:
            raise ValueError("input differs from native trace")
        err = reference-points[t]
        expected = math.copysign(rule["absolute_kw"],err) if slot>=clock["event_start_s"] and abs(err)>rule["deadband_hz"] else 0.
        if abs(expected-frame["request_kw"])>1e-8:
            raise ValueError("relay not explained by prior frequency")
        device = frame["device"]
        if abs(device["start_s"]-(slot-clock["first_hold_s"]))>1e-7 or abs(device["end_s"]-device["start_s"]-1)>1e-7:
            raise ValueError("actual native feeder clock differs")
        if feeder=="123":
            records=device["devices"]
            if device["veto"] or len(records)!=4 or len({r["id"] for r in records})!=4:
                raise ValueError("invalid storage evidence")
            for r in records:
                if (abs(r["p_kw"]-expected/4)>.05 or abs(r["q_kvar"])>.05
                        or abs(r["delta_kwh"]-r["expected_kwh"])>.00005
                        or abs((r["soc_after"]-r["soc_before"])*200-r["delta_kwh"])>1e-8
                        or (expected and expected*r["delta_kwh"]>=0)
                        or r["amps"]>r["current_ceiling_a"] or r["vmin_pu"]<.95 or r["vmax_pu"]>1.05):
                    raise ValueError("invalid signed native storage readback")
        else:
            actual_p,actual_q = device["pulse_signed_kw_kvar"]
            if (abs(actual_p-(10-expected))>.05 or abs(actual_q)>.05 or not device["snapshot"]["converged"]
                    or any(w["measured_kva"]>w["rating_kva"] for w in device["snapshot"]["windings"])):
                raise ValueError("invalid flexible-load delivery or native winding evidence")
        external = sum(p["kw"] for p in cfg["transmission_load_pulses"] if p["from_slot"]<=slot<p["until_slot"])
        d = [x-y for x,y in zip(frame["pcc"],frame["initial_pcc"],strict=True)]
        for i,unit in enumerate(("kw","kvar")):
            value = anchor[unit]+d[i]+(external if i==0 else 0.)
            if abs(write[unit]-value)>1e-7:
                raise ValueError("verified native write differs from PCC projection")
            errors.append(abs(sensor["sensor"][i]-sensors[0]["sensor"][i]-(write[unit]-writes[0][unit])))
        actual = -d[0]
        if abs(actual-expected)>limits["max_pcc_vs_command_error_kw"]:
            raise ValueError("PCC/actuator mismatch")
        if expected*external>0 and abs(actual)>=limits["min_same_sign_pcc_delivery_kw"] and actual*external>0:
            statuses.append(1 if external>0 else -1)
        support.append(actual)
    if set(statuses)!={-1,1} or max(errors)>limits["max_sensor_increment_residual_kw_kvar"]:
        raise ValueError("two-way physical delivery unresolved")
    return dict(feeder=rows[0]["feeder"],git=rows[0]["git"],holds=len(h),passed=True,
                min_measured_pcc_support_kw=min(support),max_measured_pcc_support_kw=max(support),
                max_sensor_increment_residual_kw_kvar=max(errors),source_sha256=hashlib.sha256(report.read_bytes()).hexdigest())


def run(feeder, reference, report):
    cfg = json.loads(CONFIG.read_bytes())
    clock=cfg["clock"]
    rule=cfg["actuation"]
    src=json.loads(SCENARIO.read_bytes())["source_anchor"]
    anchor=PQ(src["kw"],src["kvar"])
    exchange=import_receiver(ROOT,json.loads(MANIFEST.read_bytes())["runtime"])
    with report.open("x") as stream:
        def emit(row):
            stream.write(json.dumps(row,default=_json_default,allow_nan=False)+"\n")
            stream.flush()
        emit(dict(phase="manifest",feeder=feeder,git=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
                  protocol=hashlib.sha256(CONFIG.read_bytes()).hexdigest()))
        try:
            with power_port(feeder,anchor,reference) as (initial,advance):
                hz_ref=None
                observations = []
                def dispatch(slot,pause,sample):
                    nonlocal hz_ref
                    observations.append(sample)
                    if slot==clock["event_start_s"]:
                        quiet=[x.hz for x in observations if pause-rule["reference_window_s"]<x.time_s<pause]
                        if len(quiet)!=round(rule["reference_window_s"]/clock["exchange_s"]):
                            raise ValueError("incomplete exchange-boundary quiet interval")
                        hz_ref=statistics.mean(quiet)
                        emit(dict(phase="reference",hz=hz_ref))
                    error = 0. if hz_ref is None else hz_ref-sample.hz
                    request=math.copysign(rule["absolute_kw"],error) if abs(error)>rule["deadband_hz"] else 0.
                    pcc,start,end,device=advance(request)
                    if abs(start-(slot-clock["first_hold_s"]))>1e-7 or abs(end-start-1)>1e-7:
                        raise ValueError("DSS interval clock")
                    pulse=sum(p["kw"] for p in cfg["transmission_load_pulses"] if p["from_slot"]<=slot<p["until_slot"])
                    emit(dict(phase="port",slot=slot,request_kw=request,initial_pcc=initial,pcc=pcc,device=device))
                    return anchor.kw+pcc[0]-initial[0]+pulse,anchor.kvar+pcc[1]-initial[1]
                exchange(clock,dispatch,emit)
            emit(dict(phase="complete"))
        except Exception as exc:
            emit(dict(phase="failed",reason=repr(exc)))
            raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["run","verify"])
    parser.add_argument("report",type=Path)
    parser.add_argument("--feeder",choices=["34","123"])
    parser.add_argument("--reference34",type=Path)
    args=parser.parse_args()
    os.umask(0o077)
    if args.mode=="run":
        if not args.feeder or (args.feeder=="34" and args.reference34 is None):
            parser.error("run needs --feeder and IEEE34 needs --reference34")
        run(args.feeder,args.reference34,args.report)
    print(json.dumps(verify(args.report),allow_nan=False))
