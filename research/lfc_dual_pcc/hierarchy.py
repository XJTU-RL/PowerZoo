"""Two-rate heterogeneous actor-critic and training-only feasible-delivery credit."""
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def attribution(request,capacity,measured):
    if not all(map(math.isfinite,(request,capacity,measured))) or request<0:
        raise ValueError("Case A requires finite upward requests")
    c=max(0.,capacity)
    allowed=min(request,c)
    upper=request-allowed
    lower=max(0.,allowed-measured)
    credit=max(0.,measured-allowed)
    upper_net=max(0.,upper-credit)
    gap=max(0.,request-measured)
    if not math.isclose(upper_net+lower,gap,abs_tol=1e-7):
        raise ArithmeticError("shortfall credit balance")
    return dict(request_kw=request,capacity_kw=capacity,y_kw=measured,capped_kw=allowed,
                upper_raw_kw=upper,lower_kw=lower,over_credit_kw=credit,upper_kw=upper_net,
                reverse_kw=max(0.,-measured),excess_over_request_kw=max(0.,measured-request),gap_kw=gap)


class Head(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.features=nn.Sequential(nn.Linear(12,24),nn.Tanh(),nn.Linear(24,16),nn.Tanh())
        self.policy=nn.Linear(16,n)
        self.value=nn.Linear(16,1)

    def forward(self,x):
        z=self.features(x)
        return self.policy(z),self.value(z).squeeze(-1)


class Hierarchy(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.heads=nn.ModuleDict({k:Head(n) for k,n in (
            ("master",len(cfg["slow_fractions"])),("repA",len(cfg["rep_multipliers"])),
            ("repB",len(cfg["rep_multipliers"])),("fastA",len(cfg["actions_battery_pv_kw"])),
            ("fastB",len(cfg["actions_battery_pv_kw"])))})

    def action(self,role,obs,mask,bias,rng,greedy):
        if len(obs)!=12 or not all(map(math.isfinite,obs)) or not any(mask):
            raise ValueError("invalid causal observation or empty feasible set")
        with torch.no_grad():
            logits,_=self.heads[role](torch.tensor(obs,dtype=torch.float32))
            dist=Categorical(logits=(logits+torch.tensor(bias)).masked_fill(~torch.tensor(mask),-1e9))
            idx=int(dist.probs.argmax()) if greedy else int(rng.choice(len(mask),p=dist.probs.double().numpy()/float(dist.probs.double().sum())))
        return idx,dict(role=role,obs=obs,mask=mask,bias=bias,action=idx)


def local_report(row,plant):
    st=plant.storage
    available=sum(max(0.,v-st["reserve"])*st["kwh"] for v in plant.soc())
    return dict(y=row.get("delivered_kw",0.),energy=available,ramp=plant.ramp if hasattr(plant,"ramp") else plant.cfg["ramp_up"][plant.site],
                soc=min(plant.soc()),min_v=min(row["voltage"].values()),max_v=max(row["voltage"].values()))


def obs_local(error,request,report,age,other=0.):
    return [float(v) for v in (max(-3,min(3,error/.0002)),request/18,report["y"]/18,
            min(1.,report["energy"]/.015),min(1.,report["energy"]/100),report["ramp"]/10,
            report["soc"],(report["min_v"]-.95)*10,(1.05-report["max_v"])*10,
            min(age/5,1.),other/18,1.)]


def slow(policy,cfg,method,hz,ref,reports,bids,age,rng,greedy):
    error=ref-hz
    total=min(cfg["max_request_kw"],cfg["droop_kw_hz"]*max(0.,error-1e-5))
    if method=="none":
        total=0.
    obs=obs_local(error,total,reports["A"],age,reports["B"]["y"])
    # The master reads both last-arrived bids and sustainable energy summaries.
    obs[4]=min(1.,reports["B"]["energy"]/.015)
    obs[6]=bids["A"]
    obs[7]=bids["B"]
    records=[]
    if method=="learned":
        idx,record=policy.action("master",obs,[True]*3,[0.]*3,rng,greedy)
        records.append(record)
        split=cfg["slow_fractions"][idx]
    elif method=="rule":
        # Equal information: delayed capacity, observed delivery, local ramp.
        room={s:min(10.,2.+reports[s]["energy"]*3600*.95/(2*cfg["slow_s"]),
                    max(0.,reports[s]["y"])+reports[s]["ramp"]*cfg["slow_s"]) for s in ("A","B")}
        split=room["A"]/max(1e-8,room["A"]+room["B"])
    else:
        split=.5
    weights={"A":split*bids["A"],"B":(1-split)*bids["B"]}
    if method!="learned":
        weights={"A":split,"B":1-split}
    requests={s:total*weights[s]/sum(weights.values()) for s in weights}
    fresh={}
    for site in ("A","B"):
        fresh[site]=1.
        if method=="learned":
            o=obs_local(error,requests[site],reports[site],age)
            idx,record=policy.action("rep"+site,o,[True]*2,[0.,0.],rng,greedy)
            records.append(record)
            fresh[site]=cfg["rep_multipliers"][idx]
    return requests,fresh,records


def fast(policy,cfg,method,site,request,error,report,feasible,rng,greedy):
    if method=="droop":
        request=min(cfg["max_request_kw"]/2,cfg["droop_kw_hz"]*.5*max(0.,error-1e-5))
    mask=[not x["veto"] and (request>.01 or i==0) for i,x in enumerate(feasible)]
    nominal=[sum(x) for x in cfg["actions_battery_pv_kw"]]
    biases=[-abs(v-request)/3. for v in nominal]
    if method=="learned":
        return policy.action("fast"+site,obs_local(error,request,report,0.),mask,biases,rng,greedy)
    if not any(mask):
        raise ValueError("no safe deterministic actuator")
    return max((i for i in range(len(mask)) if mask[i]),key=lambda i:biases[i]),None


def update(model,decisions,windows,cfg,enabled):
    optim=torch.optim.Adam(model.parameters(),lr=cfg["study"]["learning_rate"])
    before={k:torch.cat([p.detach().flatten() for p in h.parameters()]).clone() for k,h in model.heads.items()}
    losses=[]
    advantages={}
    for d in decisions:
        w=windows[str(d["window"])]
        role=d["role"]
        shares=[]
        for site in ("A","B"):
            if role!="master" and role[-1]!=site:
                continue
            m=w["sites"][site]
            factor=m["gap_kw"]*.5
            if enabled and m["certified"]:
                factor=m["lower_kw"] if role.startswith("fast") else m["upper_kw"]
            shares.append(factor)
        penalty=sum(shares)/len(shares)
        target=w["frequency_reward"]-cfg["study"]["credit_weight"]*penalty/cfg["study"]["credit_scale_kw"]
        logits,val=model.heads[role](torch.tensor(d["obs"],dtype=torch.float32))
        dist=Categorical(logits=(logits+torch.tensor(d["bias"])).masked_fill(~torch.tensor(d["mask"]),-1e9))
        advantage=target-val.detach()
        loss=-dist.log_prob(torch.tensor(d["action"]))*advantage+.5*(val-target).square()
        losses.append(loss)
        advantages.setdefault(role,[]).append(float(advantage))
    if not losses: raise ValueError("no on-policy decisions")
    optim.zero_grad()
    loss=torch.stack(losses).mean()
    loss.backward()
    grad={k:math.sqrt(sum(float(p.grad.square().sum()) for p in h.parameters() if p.grad is not None)) for k,h in model.heads.items()}
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optim.step()
    delta={k:float((torch.cat([p.detach().flatten() for p in h.parameters()])-before[k]).norm()) for k,h in model.heads.items()}
    if not all(math.isfinite(v) and v>0 for v in [*grad.values(),*delta.values()]):
        raise RuntimeError("nonfinite or absent role update")
    return dict(loss=float(loss.detach()),gradient_norm=grad,parameter_delta_l2=delta,advantages=advantages)
