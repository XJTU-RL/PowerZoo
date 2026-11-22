"""One bounded 50 Hz reinitialization diagnostic in isolated in-memory plant copies."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import tempfile
from types import SimpleNamespace

from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference
from scripts.check_lfc34_flexible_load import ROOT, native
from scripts.check_lfc34_operating_point import sample
from scripts.check_lfc_123_coupling import SCENARIO, _copy_model, _poll
from scripts.lfc_bus8_receiver import _instrument, _native_frequency, _physical_sensor
from .case_a_plant import CONFIG, feeder
from .driver import dump, report, sha
from .stage0 import CONFIG as WIRING, write_ports


def main():
    import matlab.engine
    p = argparse.ArgumentParser(description=__doc__)
    for item in ("reference","inventory","out"):
        p.add_argument(item, type=Path)
    args = p.parse_args()
    args.out.mkdir(exist_ok=False)
    cfg = json.loads(CONFIG.read_bytes())
    src = json.loads(SCENARIO.read_bytes())
    spec = json.loads(WIRING.read_bytes())["ports"]
    rows = json.loads(args.inventory.read_bytes())
    if rows["model_sha256"]!=src["model_sha256"] or rows["preload"]!=["Ts = 1/60;"]:
        raise ValueError("frequency inventory source")
    ref = ObservedReference.read_report(args.reference.read_bytes())
    with report(args.out/"probe.jsonl") as emit, ExitStack() as stack:
        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        deck = native.archived_deck(ROOT,tmp)
        feeder_ok = True
        for site in cfg["sites"]:
            with feeder(deck,ref,cfg,site) as plant:
                # Operate the original passive impedances at a different frequency;
                # do not rewrite their rated-frequency data or recompute physical taps.
                plant.ctx.Text.Command = "Edit Vsource.source basefreq=50 frequency=50"
                plant.ctx.Text.Command = "Set frequency=50"
                plant.c.Solution.SolveNoControl()
                state = sample(SimpleNamespace(dss=plant.ctx,batteries={},diagnostic_mode="standard"))
                v = [r["v_pu"] for r in state.get("voltage_phase_pu",{}).values()]
                measured = [float(native._prop(plant.c,"Vsource.source",k)) for k in ("frequency","basefreq")]
                ok = (state["converged"] and state["network_power_balance_ok"] and len(v)==95 and
                      min(v)>=.95 and max(v)<=1.05 and measured==[50.,50.] and plant.c.Solution.Frequency==50.)
                feeder_ok &= ok
                emit(dict(phase="dss50", site=site,ok=ok,source_hz=measured,solution_hz=plant.c.Solution.Frequency,
                          voltage_range=[min(v),max(v)] if v else [],pcc=[state.get("pcc_import_kw"),state.get("pcc_import_kvar")]))
        name = _copy_model(src,tmp)
        eng = matlab.engine.start_matlab()
        stack.callback(eng.quit)
        eng.cd(str(tmp),nargout=0)
        eng.load_system(name,nargout=0)
        _instrument(eng,name,src["sensor"])
        # Apply named electrical/rotating-machine mask parameters, never arbitrary numbers.
        fields = {"FRated","Freq0","freq","Fn","f0","frequency","frequencyindice","fundamental","fMechanical0","fElectrical0"}
        changes = []
        for r in rows["parameters"]:
            for k,v in r["parameters"].items():
                if k in fields and v=="60":
                    if not r["sid"].isdigit():
                        raise ValueError("block SID")
                    changes.append((r["sid"],k))
        if set(k for _,k in changes)!=fields:
            raise ValueError("incomplete 60 Hz parameter inventory")
        for sid,k in changes:
            eng.eval(f"h=Simulink.ID.getHandle('{name}:{sid}'); assert(strcmp(get_param(h,'{k}'),'60'));"
                     f"set_param(h,'{k}','50'); assert(strcmp(get_param(h,'{k}'),'50'));", nargout=0)
        eng.set_param(name,"PreLoadFcn","Ts = 1/50;",nargout=0)
        eng.eval("Ts=1/50;",nargout=0)
        anchors = {s:v["anchor_kw_kvar"] for s,v in spec.items()}
        write_ports(eng,name,spec,anchors)
        eng.set_param(name+"/pause_time","Value","1000",nargout=0)
        eng.set_param(name,"StopTime","22",nargout=0)
        emit(dict(phase="simulink_config",changed=changes,Ts=float(eng.eval("Ts")),fixed_step=eng.get_param(name,"FixedStep")))
        eng.set_param(name,"SimulationCommand","start",nargout=0)
        _poll(eng,name,"stopped",timeout_s=220)
        hz = [(t.time_s,t.hz) for t in _native_frequency(eng)]
        tail = [v for t,v in hz if 19<=t<=22]
        grid_ok = (len(tail)>=14 and max(abs(v-50) for v in tail)<=.01 and
                   max(abs(b-a)/.2 for a,b in zip(tail,tail[1:]))<=.002)
        emit(dict(phase="frequency",native=hz,tail_ok=grid_ok))
        ports = {}
        for s in spec:
            z = _physical_sensor(eng,name,dict(src["sensor"],block=spec[s]["sensor"]))
            means = [statistics.mean(v for t,v in z[k] if 19<=t<=22) for k in ("p_kw","q_kvar")]
            ports[s] = means
            emit(dict(phase="port_equilibrium",site=s,measured=means,anchor=anchors[s]))
            grid_ok &= all(abs(a-b)<=max(100.,.01*abs(b)) for a,b in zip(means,anchors[s]))
        dump(args.out/"summary.json",dict(candidate_feeder_equilibrium=feeder_ok, candidate_grid_equilibrium=grid_ok,
                  frequency_tail_hz=[min(tail),max(tail)],ports=ports,formal_50hz=False, model_sha256=sha(ROOT/src["model"]),
                  remaining="machine-specific field/torque/load-flow comparison, 50 Hz feeder resource/port pulse and causal gates"))
        emit(dict(phase="complete",feeder_ok=feeder_ok,grid_ok=grid_ok,formal_50hz=False))


if __name__=="__main__":
    main()
