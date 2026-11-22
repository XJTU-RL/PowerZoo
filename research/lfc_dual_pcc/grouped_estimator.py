"""One frozen small network and a training-group conservative lookup on identical as-of inputs."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import time

import torch
from torch import nn
from .driver import dump, setup, sha, state_digest
from .grouped_metrics import calibration, summary
from .grouped_source import PROTOCOL, read

FEATURES=("usable_dc_kwh","min_soc","battery_kw","pv_kw","pcc_import_kw","pcc_import_kvar",
          "voltage_lower_margin_pu","voltage_upper_margin_pu","previous_nominal_kw","ramp_up_kw_s",
          "ramp_down_kw_s","load_multiplier","irradiance")
METRICS=Path(__file__).with_name("grouped_metrics.py")


def git():
    return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()


def split(directory,name):
    sealed=read(directory/"seal.json")
    src=directory/(name+".json")
    if sha(src)!=sealed["splits"][name]["sha256"] or sealed["protocol_sha256"]!=sha(PROTOCOL):
        raise ValueError("modified dataset or study")
    entries=read(src)
    if {e["group"] for e in entries}!=set(read(PROTOCOL)["groups"][name]):
        raise ValueError("group leakage")
    pairs=[]
    for e in entries:
        if set(e["x"])!=set(FEATURES):
            raise ValueError("unexpected estimator inputs")
        x=[e["x"][k] for k in FEATURES]
        x[0]=math.log1p(100*x[0])
        for q in e["queries"]:
            for h in read(PROTOCOL)["hold_s"]:
                pairs.append({"key":[e["group"],e["slot"],e["site"],h,q["request_kw"]],
                      "x":[*x,h/10],"request":q["request_kw"]/10,"kw":e["capacity"][str(h)]["kw"],
                      "time_class":q["confirmed_reach_s"][str(h)] or 0})
    return entries,pairs


def transform(pairs,scale=None):
    x=torch.tensor([r["x"] for r in pairs],dtype=torch.float32)
    if scale is None:
        mu=x.mean(0);sd=x.std(0,unbiased=False)
        sd=torch.where(sd<1e-6,torch.ones_like(sd),sd)
        scale={"mean":mu.tolist(),"std":sd.tolist()}
    x=((x-torch.tensor(scale["mean"]))/torch.tensor(scale["std"])).clamp(-5,5)
    req=torch.tensor([[r["request"]] for r in pairs],dtype=torch.float32)
    return x,req,scale


class Estimator(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.scale=cfg["power_scale_kw"]
        self.hidden=nn.Sequential(nn.Linear(len(FEATURES)+1,cfg["hidden"]),nn.Tanh())
        self.power=nn.Linear(cfg["hidden"],1)
        self.arrival=nn.Linear(cfg["hidden"]+1,4)

    def forward(self,x,request):
        hidden=self.hidden(x)
        return self.scale*self.power(hidden).sigmoid().squeeze(-1),self.arrival(torch.cat([hidden,request],1))


def model_predictions(model,x,r):
    with torch.no_grad():
        kw,logits=model(x,r)
        prob=logits.softmax(1)
        return [{"power_kw":float(v),"time_class":int(q.argmax()),"p_witness":float(1-q[0])} for v,q in zip(kw,prob,strict=True)]


def saved_table(pairs,x):
    return [{"key":row["key"],"kw":row["kw"],"time":row["time_class"],"x":features.tolist()}
            for row,features in zip(pairs,x,strict=True)]


def lookup_predictions(table,pairs,x,n):
    result=[]
    for case,f in zip(pairs,x.tolist(),strict=True):
        _,_,site,h,req=case["key"]
        nearest={}
        for t in table:
            g,slot,s,ht,rt=t["key"]
            if (site,h,req)!=(s,ht,rt):continue
            dist=sum((a-b)**2 for a,b in zip(t["x"],f,strict=True))
            if g not in nearest or (dist,slot)<nearest[g][:2]:
                nearest[g]=(dist,slot,t)
        selected=[r[2] for r in sorted(nearest.values(),key=lambda v:(v[0],v[1],v[2]["key"][0]))[:n]]
        if len(selected)!=n:raise ValueError("not enough independent neighbors")
        times=[v["time"] for v in selected]
        result.append({"power_kw":min(t["kw"] for t in selected),"time_class":max(times) if all(times) else 0,
                       "p_witness":sum(int(t>0) for t in times)/n})
    return result


def train(data,out):
    study=read(PROTOCOL); cfg=study["fit"]; setup(cfg["seed"])
    _,tr=split(data,"train");_,va=split(data,"validation")
    x,r,normal=transform(tr); xv,rv,_=transform(va,normal)
    y=torch.tensor([a["kw"] for a in tr]); times=torch.tensor([a["time_class"] for a in tr])
    model=Estimator(cfg); before=state_digest(model)
    original=torch.cat([p.detach().flatten().clone() for p in model.parameters()])
    optim=torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"])
    steps=[]
    start=time.monotonic()
    for i in range(cfg["epochs"]):
        optim.zero_grad()
        est,logits=model(x,r)
        regression=((est-y)/cfg["power_scale_kw"]).square().mean()
        timing=nn.functional.cross_entropy(logits,times)
        loss=regression+cfg["reach_weight"]*timing
        loss.backward()
        grad=math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters()))
        if not math.isfinite(grad) or grad<=0:raise ValueError("invalid training gradient")
        optim.step();steps.append({"update":i+1,"loss":float(loss.detach()),"gradient_norm":grad})
    model.eval(); after=state_digest(model)
    delta=float((torch.cat([p.detach().flatten() for p in model.parameters()])-original).norm())
    if before==after or not math.isfinite(delta) or delta==0:raise ValueError("no real update")
    table=saved_table(tr,x)
    pred={"mlp":model_predictions(model,xv,rv),"lookup":lookup_predictions(table,va,xv,cfg["rule_neighbors"])}
    calibrations={a:calibration(va,v,cfg["derate_grid_kw"],cfg["maximum_validation_overpromise_rate"]) for a,v in pred.items()}
    pack={"weights":model.state_dict(),"normalizer":normal,"lookup_table":table,"calibration":calibrations,
          "git":git(),"model_code_sha":sha(Path(__file__)),"metrics_code_sha":sha(METRICS),
          "study_sha":sha(PROTOCOL),"seal_sha":sha(data/"seal.json"),"torch_runtime":str(torch.__version__),
          "model_state_sha":after,"seed":cfg["seed"]}
    with (out/"estimator.pt").open("xb") as fp:torch.save(pack,fp)
    dump(out/"updates.json",steps)
    dump(out/"training.json",{"git":git(),"study_sha":sha(PROTOCOL),"train_groups":study["groups"]["train"],
          "validation_groups":study["groups"]["validation"],"seed":cfg["seed"],"updates":len(steps),
          "initial_state_sha":before,"trained_state_sha":after,"parameter_delta_l2":delta,
          "parameter_count":sum(p.numel() for p in model.parameters()),"elapsed_s":time.monotonic()-start,
          "start_loss":steps[0]["loss"],"final_loss":steps[-1]["loss"],"checkpoint_sha256":sha(out/"estimator.pt"),
          "calibrations":calibrations,
          "validation":{a:{k:v for k,v in summary(va,z,calibrations[a]["derate_kw"]).items() if k!="power_errors_kw"}
                        for a,z in pred.items()}})


def evaluate(data,checkpoint,out):
    study=read(PROTOCOL);cfg=study["fit"]; setup(cfg["seed"])
    ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if (ck["git"]!=git() or ck["model_code_sha"]!=sha(Path(__file__)) or ck["metrics_code_sha"]!=sha(METRICS)
            or ck["study_sha"]!=sha(PROTOCOL) or ck["seal_sha"]!=sha(data/"seal.json")
            or ck["torch_runtime"]!=str(torch.__version__) or ck["seed"]!=cfg["seed"]):
        raise ValueError("checkpoint code, split, version or seed mismatch")
    model=Estimator(cfg);model.load_state_dict(ck["weights"]);model.eval()
    if state_digest(model)!=ck["model_state_sha"]:raise ValueError("model readback mismatch")
    entries,test=split(data,"test"); x,r,_=transform(test,ck["normalizer"])
    preds={"mlp":model_predictions(model,x,r),"lookup":lookup_predictions(ck["lookup_table"],test,x,cfg["rule_neighbors"])}
    result={}; prediction_rows=[]
    for name,pred in preds.items():
        cal=ck["calibration"][name]
        by={}
        for group in ["all",*study["groups"]["test"]]:
            ids=[i for i,t in enumerate(test) if group=="all" or t["key"][0]==group]
            a=[test[i] for i in ids];b=[pred[i] for i in ids]
            both={}
            for variant,offset in (("raw",0.),("calibrated",cal["derate_kw"])):
                s=summary(a,b,offset);e=s.pop("power_errors_kw")
                s["fraction_within_validation_p90_error"]=sum(abs(v)<=cal["val_p90_abs_error_kw"] for v in e)/len(e)
                both[variant]=s
            by[group]=both
        result[name]={"derating":cal,"scores":by}
        for t,v in zip(test,pred,strict=True):
            prediction_rows.append({"model":name,"key":t["key"],"witness_kw":t["kw"],"witness_reach_class":t["time_class"],
                                    "derated_kw":max(0.,v["power_kw"]-cal["derate_kw"]),**v})
    dump(out/"predictions.json",prediction_rows)
    dump(out/"evaluation.json",{"git":git(),"model_state_sha":state_digest(model),"checkpoint_sha256":sha(checkpoint),
          "test_groups":study["groups"]["test"],"counts":{"independent":len(study["groups"]["test"]),"states":len(entries)},
          "actual_native_violations":sum(e["accepted_violations"] for e in entries),"methods":result,
          "scope":"overcommit is relative to the finite witnessed catalogue. The estimator does not actuate equipment. No global infeasibility or calibrated population guarantee."})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["train","evaluate"]);p.add_argument("--data",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();os.umask(0o077);a.out.mkdir(parents=True,exist_ok=False)
    if a.mode=="train":train(a.data,a.out)
    else:evaluate(a.data,a.checkpoint,a.out)
