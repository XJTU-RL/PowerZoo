"""Integrate measured master P/Q against the delayed report and the original DSS interval."""
import argparse
import json
import math
from pathlib import Path

from .stage0 import CONFIG


def traces(path):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    if rows[-1]["phase"]!="complete":
        raise ValueError("incomplete source")
    return ([r for r in rows if r["phase"]=="hold"],
            {r["port"]:r for r in rows if r["phase"]=="physical_trace"})


def score(folder):
    cfg=json.loads(CONFIG.read_bytes())
    zero,zs=traces(folder/"zero_a.jsonl")
    result={}
    for name in cfg["cases"]:
        holds,sensors=traces(folder/(name+".jsonl"))
        byport={}
        for p in cfg["ports"]:
            channel={}
            for j,k in enumerate(("p_kw","q_kvar")):
                signal=[]
                for (t,v),(tz,vz) in zip(sensors[p][k],zs[p][k],strict=True):
                    if abs(t-tz)>1e-7 or not math.isfinite(v-vz):
                        raise ValueError("nonaligned paired master logs")
                    signal.append((t,v-vz))
                actual=written=local=covered_target=unobserved=0.
                max_hold_error=0.
                for h,z in zip(holds,zero,strict=True):
                    lo,hi=h["write_s"],h["end_s"]
                    # Native fixed electrical steps. Interpolate only the integration
                    # endpoints in a bracketing pair; never extrapolate missing samples.
                    integral=coverage=0.
                    for (a,va),(b,vb) in zip(signal,signal[1:]):
                        left,right=max(a,lo),min(b,hi)
                        if right<=left:
                            continue
                        if b-a>2/60+1e-7 or b<=a:
                            raise ValueError("missing native sensor coverage")
                        yl=va+(vb-va)*(left-a)/(b-a)
                        yr=va+(vb-va)*(right-a)/(b-a)
                        integral+=(yl+yr)*(right-left)/2
                        coverage+=right-left
                    missing=hi-lo-coverage
                    if abs(missing)>1e-7:
                        if h is not holds[-1] or abs(missing-1/60)>1e-7 or abs(signal[-1][0]-(hi-missing))>1e-7:
                            raise ValueError("unexpected gap in native physical coverage")
                        unobserved+=missing
                    command=(h["target"][p][j]-z["target"][p][j])*(hi-lo)
                    on_measured_support=(h["target"][p][j]-z["target"][p][j])*coverage
                    covered_target+=on_measured_support/3600
                    error=abs(integral-on_measured_support)/coverage
                    max_hold_error=max(error,max_hold_error)
                    if error>cfg["gates"]["physical_delta_abs_kw_kvar"]:
                        raise ValueError(f"{name}/{p}/{k}: mean integral residual {error}")
                    actual+=integral/3600
                    written+=command/3600
                    d,base=h["feeders"][p],z["feeders"][p]
                    local+=(d["pcc"][j]-base["pcc"][j])*(d["end_s"]-d["start_s"])/3600
                if abs(written-local)>1e-5:
                    raise ValueError("delayed return failed interval ledger")
                channel[k]={"dss":local,"applied":written,"sensor_on_observed_span":actual,"target_on_observed_span":covered_target,"unobserved_terminal_seconds":unobserved,"max_hold_mean_residual":max_hold_error}
            byport[p]=channel
        result[name]=byport
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    result=score(args.folder)
    with args.output.open("x") as fp:
        json.dump(result,fp,indent=2,allow_nan=False)
        fp.write("\n")
