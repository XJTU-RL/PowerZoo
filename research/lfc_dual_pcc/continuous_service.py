"""Bounded slow allocation and causal one-hop urgency for the engineering trial."""
import math

import torch
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F

from .clock_graph import MLP, slow_input, tensor
from .equal_service import SERVICE_KW, EqualServiceCoordinator, EqualServicePolicy, service_kw
from .master_dispatch import SITES


def allocation_interval(request, limits):
    if not all(math.isfinite(v) and v >= 0. for v in (request, limits["A"], limits["B"])):
        raise ValueError("invalid service or capacity")
    total = min(request, limits["A"]+limits["B"])
    hi = min(total,limits["A"])
    lo = min(hi,max(0.,total-limits["B"]))
    return total, lo, hi


def freshness(packet, site, now, fast_s):
    if packet is None or not packet.valid(now):
        return 0.
    data = packet.data[site]
    if type(data.get("witness")) is not int or data["duration_s"] < fast_s:
        return 0.
    return (packet.expiry-now)/(packet.expiry-packet.sampled)


class ContinuousMaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = MLP(37,2)
        self.critic = MLP(37,1)

    def distribution(self, x):
        a,b = (1.+F.softplus(self.actor(x))).unbind()
        return Beta(a,b)

    def decide(self, x, lo, hi, total, greedy, rule):
        latent = logp = None
        if hi-lo <= 1e-9:
            a = lo
        elif rule:
            a = min(hi,max(lo,total*.5))
        else:
            dist = self.distribution(x)
            latent = (dist.mean if greedy else dist.sample()).detach()
            if not 0. < float(latent) < 1.:
                raise ValueError("latent outside open Beta support")
            logp = dist.log_prob(latent).detach()
            a = lo+float(latent)*(hi-lo)
        return a, dict(args=(x,), latent=latent, interval_kw=(lo,hi), allocation_a_kw=a, old_logp=logp,
                       old_value=self.critic(x).squeeze().detach())

    def evaluate_sample(self, record):
        x, = record["args"]
        latent = record["latent"]
        lo,hi = record["interval_kw"]
        if latent is not None and (hi-lo <= 1e-9 or not 0.<float(latent)<1.
                or not math.isclose(record["allocation_a_kw"],lo+float(latent)*(hi-lo),abs_tol=1e-9)):
            raise ValueError("invalid latent or mismatched recorded allocation map")
        lp = self.distribution(x).log_prob(latent) if latent is not None else None
        return lp, self.critic(x).squeeze()


class ContinuousServicePolicy(EqualServicePolicy):
    def __init__(self):
        super().__init__()
        self.heads["master"] = ContinuousMaster()


class ContinuousServiceCoordinator(EqualServiceCoordinator):
    def __init__(self, model, cfg, greedy, *, fixed_master=False, ablation=None):
        super().__init__(model,cfg,greedy,fixed_master=fixed_master)
        self.ablation = ablation

    def command_urgency(self, error):
        return 0. if self.ablation == "downlink" else error

    def master_allocation(self, error, reports, limits, slot, now, freq_time, sequences):
        request = service_kw(slot)
        total, lo, hi = allocation_interval(request,limits)
        confidence = [freshness(reports[s],s,now,self.cfg["fast_s"]) for s in SITES]
        observed_confidence = [0.,0.] if self.ablation == "uplink" else confidence
        x = torch.cat((slow_input(error,self.cfg["frequency_scale_hz"],reports,now),
                       tensor([v/SERVICE_KW for v in (request,lo,hi,request-total)]+observed_confidence)))
        a,sample = self.model.heads["master"].decide(x,lo,hi,total,self.greedy,self.fixed_master)
        b = min(limits["B"],max(0.,total-a))
        self.records.append(dict(role="master",slot=slot,command=slot,outputs=sample))
        self.emit(dict(phase="policy",role="master",slot=slot,command=slot,consumed_s=now,
                       reports=sequences,frequency_sample_s=freq_time,x=x.tolist(),
                       opportunity=hi-lo>1e-9, feasible_a_kw=[lo,hi], requested_kw=request,
                       allocatable_kw=total, capacity_shortfall_kw=request-total,
                       one_step_freshness=dict(zip(SITES,confidence,strict=True)),
                       latent_u=float(sample["latent"]) if sample["latent"] is not None else None,
                       behavior_logp=float(sample["old_logp"]) if sample["old_logp"] is not None else None))
        return dict(A=a,B=b),request
