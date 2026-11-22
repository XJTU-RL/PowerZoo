"""Case A episodes with a single native transmission clock and two independent DSS contexts."""
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

import numpy as np

from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference
from envs.lfc.native_feeder_123 import PQ
from envs.lfc.receiver_123 import read_causal
from scripts.check_lfc34_flexible_load import ROOT,native
from scripts.check_lfc_123_coupling import SCENARIO,_copy_model,_poll,_check_measurement
from scripts.lfc_bus8_receiver import _advance,_instrument,_native_frequency,_physical_sensor,_verify_prefixes
from .case_a_plant import feeder,certify
from .hierarchy import attribution,local_report,slow,fast
from .stage0 import CONFIG as WIRING,write_ports
from .topology import check


def native_episode(model,cfg,method,event,reference,seed,greedy,emit, *, coordinator=None):
    import matlab.engine
    rng=np.random.default_rng(seed)
    wire=json.loads(WIRING.read_bytes())
    src=json.loads(SCENARIO.read_bytes())
    check(ROOT/src["model"],wire)
    ports=wire["ports"]
    clock=cfg["clock"]
    first,stop=clock["first_hold_s"],clock["stop_s"]
    slots=cfg["window_slots"]
    anchors={s:v["anchor_kw_kvar"] for s,v in ports.items()}
    ref=ObservedReference.read_report(reference.read_bytes())
    with ExitStack() as stack:
        tmp=Path(stack.enter_context(tempfile.TemporaryDirectory()))
        deck=native.archived_deck(ROOT,tmp)
        plants={s:stack.enter_context(feeder(deck,ref,cfg,s)) for s in ports}
        base={s:p.base[:] for s,p in plants.items()}
        if coordinator is not None:
            coordinator.initialize(plants, emit)
        name=_copy_model(src,tmp)
        eng=matlab.engine.start_matlab()
        stack.callback(eng.quit)
        eng.cd(str(tmp),nargout=0)
        eng.load_system(name,nargout=0)
        _instrument(eng,name,src["sensor"])
        if abs(float(eng.eval("Ts",nargout=1))-1/60)>1e-7: raise ValueError("master time base")
        eng.set_param(name,"StopTime",str(stop+1),nargout=0)
        eng.set_param(f"{name}/pause_time","Value",str(first),nargout=0)
        write_ports(eng,name,ports,anchors)
        eng.set_param(name,"SimulationCommand","start",nargout=0)
        _poll(eng,name,"paused",timeout_s=160)
        start=float(eng.get_param(name,"SimulationTime"))
        if not first-1e-7<=start<=first+2/60+1e-7: raise ValueError("initial pause")
        previous=start-1
        pending=[]
        last_bids={s:1. for s in ports}
        initial_reports={s:local_report(p.last,p) for s,p in plants.items()}
        req={s:0. for s in ports}
        histories={s:[] for s in ports}
        windows={}
        decisions=[]
        frames=[]
        prefixes=[]
        ref_hz=None
        for slot in range(first,stop):
            raw=_native_frequency(eng)
            prefixes.append(raw)
            obs=read_causal(lambda:raw,previous,start,clock)
            if slot==slots[0]:
                ref_hz=statistics.mean(s.hz for s in raw if start-2 < s.time_s < start)
            ref_now=obs.hz if ref_hz is None else ref_hz
            error=ref_now-obs.hz
            available=[f for f in frames if f["arrival_s"]<start]
            used=available[-1] if available else None
            if slot>=first+2 and (used is None or used["slot"]!=slot-2):
                raise RuntimeError("misaligned report")
            arrived=used["reports"] if used else initial_reports
            age=start-used["end_s"] if used else 0.
            eligible=[x for x in pending if x[0]<start]
            if eligible: last_bids=eligible[-1][1]
            if coordinator is None and slot in slots:
                req,bids,records=slow(model,cfg,method,obs.hz,ref_now,arrived,last_bids,age,rng,greedy)
                pending.append((start+clock["report_latency_s"],bids))
                for r in records:
                    affected=slot+cfg["slow_s"] if r["role"].startswith("rep") else slot
                    if affected in slots:
                        decisions.append(dict(**r,at_s=start,window=affected))
                windows[str(slot)]={"request":req.copy(),"initial":{s:{"soc":p.soc(),"pcc":p.last["pcc"],
                    "voltage":p.last["voltage"],"prefix":p.history[:]} for s,p in plants.items()},
                    "actions":{s:[] for s in ports},"outputs":{s:[] for s in ports}}
            active=slots[0]<=slot<slots[-1]+cfg["slow_s"]
            w=slots[0]+((slot-slots[0])//cfg["slow_s"])*cfg["slow_s"]
            selected={}
            for s,p in (plants.items() if coordinator is None else ()):
                if active:
                    idx,r=fast(model,cfg,method,s,req[s],error,local_report(p.last,p),p.candidates(),rng,greedy)
                    if r: decisions.append(dict(**r,at_s=start,window=w))
                else: idx=0
                selected[s]=idx
            if coordinator is not None:
                req, selected = coordinator.dispatch(slot, start, obs.time_s, error, plants)
            pulse=event["kw"] if event["start"]<=slot<event["stop"] else 0.
            targets={s:[a+((used["pcc"][s][j]-base[s][j]) if used else 0.)
                        +(pulse if s=="A" and j==0 else 0.) for j,a in enumerate(anchors[s])] for s in ports}
            write_ports(eng,name,ports,targets)
            if eng.get_param(name,"SimulationStatus")!="paused" or float(eng.get_param(name,"SimulationTime"))!=start:
                raise RuntimeError("two-port transaction advanced clock")
            end=_advance(eng,name,start,slot+1,clock)
            actual={s:p.execute(selected[s]) for s,p in plants.items()}
            if active and coordinator is None:
                for s,row in actual.items():
                    windows[str(w)]["actions"][s].append(selected[s])
                    windows[str(w)]["outputs"][s].append(row["delivered_kw"])
            frame=dict(slot=slot,write_s=start,end_s=end,arrival_s=end+clock["report_latency_s"],used_slot=used["slot"] if used else None,
                    received_s=(used["arrival_s"] if used else None),frequency=[obs.time_s,obs.hz],event_kw=pulse,
                    pcc={s:r["pcc"] for s,r in actual.items()},reports={s:local_report(actual[s],p) for s,p in plants.items()},
                    target=targets,request=req.copy() if active or coordinator is not None else {s:0. for s in ports},actions=selected,
                    delivered={s:r["delivered_kw"] for s,r in actual.items()},actual=actual)
            frames.append(frame)
            if coordinator is not None:
                coordinator.receive(frame, plants)
            emit({"phase":"hold",**frame})
            for s in ports: histories[s].append(selected[s])
            previous,start=start,end
        eng.set_param(name,"SimulationCommand","stop",nargout=0)
        _poll(eng,name,"stopped")
        hz=_native_frequency(eng)
        _verify_prefixes(prefixes,hz)
        trace=[(s.time_s,s.hz) for s in hz]
        if trace[-1][0]<stop-.200001 or any(abs(b[0]-a[0]-.2)>1e-7 for a,b in zip(trace,trace[1:])):
            raise ValueError("frequency cadence or coverage")
        emit({"phase":"native_frequency","reference_hz":ref_hz,"samples":trace})
        sensors={}
        for s in ports:
            signals=_physical_sensor(eng,name,dict(src["sensor"],block=ports[s]["sensor"]))
            emit(dict(phase="physical_native",port=s,**signals))
            signals["frequency_hz"]=trace
            policy=src["warmup"].copy()
            for k,v in zip(("p_abs_tol_kw","q_abs_tol_kvar"),anchors[s]):
                policy[k]=max(100.,.01*abs(v))
            gaps={"power":1/60+1e-7,"frequency":.2+1e-7}
            _check_measurement(PQ(*anchors[s]),signals,policy,first,3,dict(gaps,power=2/60+1e-7),settled=True)
            sensors[s]=[]
            for f in frames:
                m=_check_measurement(PQ(*f["target"][s]),signals,policy,f["end_s"],.2,gaps,
                                     settled=False,hold=(f["write_s"],f["end_s"]))
                sensors[s].append([m["mean_p_kw"],m["mean_q_kvar"]])
            emit(dict(phase="sensor_means",port=s,means=sensors[s]))
        # Attribution runs only after all policy actions, from reconstructed native states.
        for key,win in windows.items():
            origin=int(key)
            targets=[f for f in frames if f["used_slot"] in (origin,origin+1)]
            if len(targets)!=2: raise ValueError("missing realized return window")
            lo,hi=targets[0]["write_s"],targets[-1]["end_s"]
            frequencies=[v for t,v in trace if lo<t<hi]
            if not frequencies: raise ValueError("empty delayed frequency window")
            win["frequency_reward"]=-statistics.mean(abs(ref_hz-v) for v in frequencies)/cfg["frequency_scale_hz"]
            win["return_window_s"]=[lo,hi]
            win["sites"]={}
            for s in ports:
                y=min(win["outputs"][s])
                cert={"certified":False,"reason":"inference path"}
                if method=="learned" and not greedy:
                    initial=win["initial"][s]
                    try:
                        cert=certify(deck,ref,cfg,s,initial["prefix"],initial,win["actions"][s])
                        if max(abs(a-b) for a,b in zip(win["outputs"][s],cert["actual_action_replay_kw"]))>cfg["gates"]["cert_kw"] or y>cert["capacity_kw"]+cfg["gates"]["cert_kw"]:
                            raise ValueError("counterfactual does not cover measured delivery")
                    except (ValueError,RuntimeError) as exc:
                        cert={"certified":False,"reason":repr(exc)}
                result={"gap_kw":max(0.,win["request"][s]-y),"y_kw":y}
                if cert["certified"]:
                    result=attribution(win["request"][s],cert["capacity_kw"],y)
                win["sites"][s]={**result,**cert}
            emit(dict(phase="window",slot=origin,**win))
        for d in decisions: emit(dict(phase="decision",**d))
        return windows,decisions,frames,trace,sensors,ref_hz
