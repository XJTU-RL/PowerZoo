"""Case A native plant and exact finite-action max-min reference over two 1 s steps."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
from types import SimpleNamespace

from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference, RATINGS
from scripts.check_lfc34_flexible_load import DESIGN, FlexibleLoad, native
from scripts.check_lfc34_operating_point import sample, phases_and_terminals
from scripts.check_lfc34_tap_hold import read_taps
from .master_dispatch import feasible_actions


CONFIG=Path(__file__).with_name("case_a.json")


class Feeder:
    def __init__(self,ctx,deck,reference,cfg,site):
        self.ctx,self.cfg,self.site=ctx,cfg,site
        neutral=FlexibleLoad(ctx,deck,reference,json.loads(DESIGN.read_bytes()))
        self.c,self.taps=neutral.circuit,neutral.taps
        self.storage,self.pv=cfg["storage"],cfg["pv"]
        self.previous=0.
        self.history=[]
        self.s=0
        st,p=self.storage,self.pv
        for device in st["devices"]:
            ctx.Text.Command=(f"Edit {device} enabled=yes DispMode=external state=idling "
                              f"%stored={100*st['initial_soc'][site]:.12g} kvar=0")
            for key,target in (("kWRated",st["kw"]),("kWhRated",st["kwh"]),("%reserve",100*st["reserve"]),
                               ("%EffDischarge",100*st["eta"])):
                if abs(float(native._prop(self.c,device,key))-target)>1e-8:
                    raise ValueError("native storage model does not match design")
        ctx.Text.Command=(f"New {p['device']} phases=3 bus1={p['bus']} conn=delta kv={p['kv']} "
                          f"kVA={p['kva']} Pmpp={p['kw']} irradiance=1 %Pmpp=50 %cutin=0 %cutout=0 pf=1")
        self.last=self.probe(0, observe=True)
        if self.last["veto"] or any(abs(v-st["initial_soc"][site])>1e-8 for v in self.last["soc"]):
            raise ValueError(f"initial native state rejected: {self.last['veto']}")
        self.base=self.last["pcc"]

    def soc(self):
        values=[]
        for name in self.storage["devices"]:
            self.c.Storages.Name=name.split(".")[1]
            values.append(float(self.c.Storages.puSOC))
        return values

    def precheck(self,idx):
        b,p=self.cfg["actions_battery_pv_kw"][idx]
        total=b+p
        if total-self.previous>self.cfg["ramp_up"][self.site]+1e-8 or self.previous-total>self.cfg["ramp_down"]+1e-8:
            return "ramp"
        remaining=[max(0,v-self.storage["reserve"])*self.storage["kwh"] for v in self.soc()]
        if b/(len(remaining)*3600*self.storage["eta"])>min(remaining)+1e-10:
            return "energy"
        return None

    def probe(self,idx, *, observe=False):
        veto=self.precheck(idx)
        if veto:
            return {"veto":[veto]}
        b,p=self.cfg["actions_battery_pv_kw"][idx]
        st,pv=self.storage,self.pv
        before=self.soc()
        per=b/len(before)
        for device in st["devices"]:
            mode="discharging" if b else "idling"
            self.ctx.Text.Command=f"Edit {device} %discharge={100*per/st['kw']:.13g} state={mode} kvar=0"
        self.ctx.Text.Command=f"Edit {pv['device']} %Pmpp={100*(pv['base_kw']+p)/pv['kw']:.13g}"
        self.c.Solution.SolveNoControl()
        raw=sample(SimpleNamespace(dss=self.ctx,batteries={},diagnostic_mode="standard"))
        if not raw["converged"] or not raw["network_power_balance_ok"]:
            raise RuntimeError("native solve or power balance failed")
        if before!=self.soc() or read_taps(self.c)!=self.taps or abs(self.c.Solution.dblHour*3600-self.s)>1e-6:
            raise RuntimeError("preview changed native time, taps or SOC")
        dev=raw["enabled_device_terminal_measurements"]
        batt=[-dev[d.lower()][0]["p_kw"] for d in st["devices"]]
        actual_pv=-dev[pv["device"].lower()][0]["p_kw"]
        volts={k:v["v_pu"] for k,v in raw["voltage_phase_pu"].items()}
        if len(volts)!=95 or not all(map(math.isfinite,volts.values())):
            raise ValueError("invalid phase census")
        veto=[]
        if not all(self.cfg["gates"]["voltage"][0]<=v<=self.cfg["gates"]["voltage"][1] for v in volts.values()):
            veto.append("voltage")
        if any(abs(v-per)>self.cfg["gates"]["power_abs_kw"] for v in batt) or abs(actual_pv-(pv["base_kw"]+p))>self.cfg["gates"]["power_abs_kw"]:
            veto.append("terminal")
        if any(dev[d.lower()][0]["s_kva"]>st["kw"]+1e-5 for d in st["devices"]) or dev[pv["device"].lower()][0]["s_kva"]>pv["kva"]+1e-5:
            veto.append("inverter")
        for xf in ("subxf","xfm1"):
            for i,term in enumerate(phases_and_terminals(native._active(self.c,"Transformer."+xf)),1):
                _,kva,kv=RATINGS[f"{xf}:{i}"]
                amp=[x["terminal_current_amp"] for x in term["conductors"] if x["node"] in (1,2,3)]
                if len(amp)!=3 or max(amp)>kva/(math.sqrt(3)*kv) or term["s_kva"]>kva:
                    veto.append("winding")
        phase = {}
        if observe and self.cfg.get("phase_graph"):
            from .phase_graph import terminal_injections
            phase["phase_pq"] = terminal_injections(dev)
        return dict(**phase,veto=veto,soc=before,pcc=[raw["pcc_import_kw"],raw["pcc_import_kvar"]],voltage=volts,
                    battery_kw=batt,pv_kw=actual_pv,balance=[raw["network_power_residual_kw"],raw["network_power_residual_kvar"]])

    def candidates(self):
        return [self.probe(i) for i in range(len(self.cfg["actions_battery_pv_kw"]))]

    def execute(self,idx):
        row=self.probe(idx, observe=True)
        if row["veto"]:
            raise ValueError(f"inadmissible selected action {idx}: {row['veto']}")
        self.c.Solution.FinishTimeStep()
        self.s+=1
        now=self.soc()
        for v,w,kw in zip(row["soc"],now,row["battery_kw"],strict=True):
            residual=(w-v)*self.storage["kwh"]+kw/(3600*self.storage["eta"])
            if abs(residual)>self.cfg["gates"]["energy_abs_kwh"] or w<self.storage["reserve"]-1e-8:
                raise ValueError("actual native energy balance or reserve failed")
        if abs(self.c.Solution.dblHour*3600-self.s)>1e-6 or read_taps(self.c)!=self.taps:
            raise ValueError("actual native clock/tap mismatch")
        self.previous=sum(self.cfg["actions_battery_pv_kw"][idx])
        self.history.append(idx)
        row.update(action=idx,end_local_s=self.s,soc_after=now,delivered_kw=self.base[0]-row["pcc"][0])
        self.last=row
        return row


@contextmanager
def feeder(deck,reference,cfg,site):
    import dss
    ctx=dss.DSS.NewContext()
    ctx.AllowChangeDir=False
    try:
        yield Feeder(ctx,deck,reference,cfg,site)
    finally:
        ctx.ClearAll()


def certify(deck,reference,cfg,site,prefix,initial_state,executed, *, collect_paths=False):
    """Exhaust all feasible two-step paths for this finite action library only."""
    if cfg["slow_s"]!=2 or len(executed)!=2:
        raise ValueError("certification requires exactly two completed fast steps")
    best=-math.inf
    best_actions=None
    paths=0
    replay=None
    admissible=[]
    for first in range(len(cfg["actions_battery_pv_kw"])):
        with feeder(deck,reference,cfg,site) as f:
            for idx in prefix:
                f.execute(idx)
            state=f.last
            if (max(abs(a-b) for a,b in zip(f.soc(),initial_state["soc"],strict=True))>1e-8 or
                    max(abs(a-b) for a,b in zip(state["pcc"],initial_state["pcc"],strict=True))>cfg["gates"]["parity_kw"] or
                    set(state["voltage"])!=set(initial_state["voltage"]) or
                    max(abs(v-initial_state["voltage"][k]) for k,v in state["voltage"].items())>1e-6 or f.history!=prefix):
                raise ValueError("counterfactual reconstruction differs from observed window start")
            if f.probe(first)["veto"]:
                continue
            if collect_paths:
                table=cfg["actions_battery_pv_kw"]
                zero_first=first in feasible_actions(table,f.candidates(),0.)
            one=f.execute(first)["delivered_kw"]
            candidates=f.candidates()
            if collect_paths:
                zero_second=feasible_actions(table,candidates,0.)
            for second,row in enumerate(candidates):
                if row["veto"]:
                    continue
                two=f.base[0]-row["pcc"][0]
                sustainable=min(one,two)
                paths+=1
                if collect_paths:
                    admissible.append(dict(actions=[first,second],support_kw=sustainable,
                                           zero_request_allowed=zero_first and second in zero_second))
                if [first,second]==executed:
                    replay=[one,two]
                if sustainable>best:
                    best,best_actions=sustainable,[first,second]
    if not paths or replay is None:
        raise ValueError("executed path missing from certified finite window")
    # Independently execute both winning actions, including the final energy step.
    with feeder(deck,reference,cfg,site) as f:
        for idx in prefix:
            f.execute(idx)
        witness=[f.execute(idx)["delivered_kw"] for idx in best_actions]
    if abs(min(witness)-best)>cfg["gates"]["cert_kw"]:
        raise ValueError("finite upper bound did not match safe reference execution")
    result = dict(certified=True,capacity_kw=best,paths=paths,witness=best_actions,witness_kw=witness,
                  actual_action_replay_kw=replay,scope="finite discrete two-step upward support only")
    if collect_paths:
        result["action_paths"]=admissible
    return result
