"""Independently reconstruct the strict post-write sample windows from native Sensor2/7 logs."""
import argparse
import bisect
import json
import math
from pathlib import Path


def check_file(path):
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    if not rows or rows[-1]["phase"]!="complete" or any(r["phase"]=="failed" for r in rows):
        raise ValueError("incomplete native batch")
    step=1/60
    eps=1e-7
    result={}
    for port in ("A","B"):
        trace,=[r for r in rows if r["phase"]=="physical_trace" and r["port"]==port]
        windows=[r for r in rows if r["phase"]=="sensor" and r["port"]==port]
        if len(windows)!=rows[-1]["holds"]:
            raise ValueError("missing native windows")
        channels={}
        for j,channel in enumerate(("p_kw","q_kvar")):
            times,values=zip(*trace[channel])
            if (not all(math.isfinite(x) for x in times+values)
                    or any(b<=a for a,b in zip(times,times[1:]))):
                raise ValueError("invalid physical timeseries")
            first_after,first_used=[],[]
            for hold in windows:
                start,end=hold["write_s"],hold["end_s"]
                idx=bisect.bisect_right(times,start+eps)
                if idx>=len(times) or not start+eps<times[idx]<end-eps or times[idx]-start>2*step+eps:
                    raise ValueError("no fresh native sample after actual write")
                left=max(idx,bisect.bisect_left(times,end-.2-(step+eps)))
                right=bisect.bisect_left(times,end-eps)
                t=times[left:right]
                if len(t)<2 or not all(start+eps<x<end-eps for x in t):
                    raise ValueError("physical window contains a pause boundary")
                mean=math.fsum(values[left:right])/len(t)
                if abs(mean-hold["pq"][j])>1e-8:
                    raise ValueError("logged sensor mean does not match raw physical window")
                first_after.append(times[idx]-start)
                first_used.append(t[0]-start)
            channels[channel]={"first_native_offset_s":[min(first_after),max(first_after)],
                               "first_used_in_tail_offset_s":[min(first_used),max(first_used)]}
        result[port]=channels
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    result={case:check_file(args.folder/(case+".jsonl"))
            for case in ("zero_a","zero_b","a_only","b_only","joint")}
    with args.output.open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write("\n")
