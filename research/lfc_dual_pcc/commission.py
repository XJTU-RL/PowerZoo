"""Commission native Case A resources and the finite-action capacity oracle before learning."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from contextlib import ExitStack

from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference
from scripts.check_lfc34_flexible_load import ROOT,native
from .case_a_plant import CONFIG,feeder,certify
from .score_stage0 import score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage0",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    cfg=json.loads(CONFIG.read_bytes())
    if not score(args.stage0)["stage0_passed"]:
        raise ValueError("stage 0 did not pass")
    ref=ObservedReference.read_report(args.reference.read_bytes())
    with ExitStack() as stack:
        d=native.archived_deck(ROOT,Path(stack.enter_context(tempfile.TemporaryDirectory())))
        plants={s:stack.enter_context(feeder(d,ref,cfg,s)) for s in cfg["sites"]}
        traces={s:[] for s in plants}
        for tick in range(8):
            for s,f in plants.items():
                states=f.candidates()
                valid=[(i,r) for i,r in enumerate(states) if not r["veto"]]
                idx,_=max(valid,key=lambda item:sum(cfg["actions_battery_pv_kw"][item[0]]))
                initial={"soc":f.soc(),"pcc":f.last["pcc"],"voltage":f.last["voltage"]}
                prefix=list(f.history)
                r=f.execute(idx)
                traces[s].append(dict(tick=tick,action=idx,pcc_support_kw=r["delivered_kw"],soc=r["soc_after"],
                             energy_limited=any(x["veto"]==["energy"] for x in states),
                             ramp_limited=any(x["veto"]==["ramp"] for x in states),
                             voltage_ok=all(not ("voltage" in x["veto"]) for x in states),
                             initial=initial,prefix=prefix))
        certs={}
        for s,tr in traces.items():
            certs[s]=certify(d,ref,cfg,s,tr[0]["prefix"],tr[0]["initial"],[tr[0]["action"],tr[1]["action"]])
            for r in tr:
                del r["initial"],r["prefix"]
        a,b=traces["A"],traces["B"]
        passed=(a[0]["pcc_support_kw"]>b[0]["pcc_support_kw"]+2
                and b[0]["ramp_limited"] and b[-1]["pcc_support_kw"]>8
                and a[-1]["energy_limited"] and b[-1]["pcc_support_kw"]>a[-1]["pcc_support_kw"]+4
                and all(r["voltage_ok"] for tr in traces.values() for r in tr))
        data=dict(passed=passed,scope="two native DSS instances; Case A capacity/energy/ramp, not learning",
                  config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),reference_sha256=ref.sha256,
                  traces=traces,initial_certificates=certs)
        with args.out.open("x") as dst:
            json.dump(data,dst,allow_nan=False,indent=2);dst.write("\n")
        if not passed: raise ValueError("Case A fast/limited-energy versus slow/sustainable mechanism not observed")
        print(json.dumps({"passed":passed,"initial":[a[0]["pcc_support_kw"],b[0]["pcc_support_kw"]],
                          "final":[a[-1]["pcc_support_kw"],b[-1]["pcc_support_kw"]]}))


if __name__=="__main__":
    main()
