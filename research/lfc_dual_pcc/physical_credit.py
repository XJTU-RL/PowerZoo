"""Matched, one-window PPO-style updates with conserved physical shortfall credit."""
import math

import torch

from .hierarchy import attribution


ROLES_BY_SITE = {"A":("feederA","batteryA","pvA"), "B":("feederB","batteryB","pvB")}


def credit_budget(sites, exposures, *, attributed):
    """Distribute the same discounted kW-seconds across decision samples."""
    result = {key:0. for key in exposures}
    site_ledger = {}
    master, = [k for k in exposures if k[0]=="master"]
    mapped = {s:[(role,slot) for role in roles for slot in (master[1]+1,master[1]+2)]
              for s,roles in ROLES_BY_SITE.items()}
    if set(sites)!=set(mapped) or set(exposures)!={master,*mapped["A"],*mapped["B"]}:
        raise ValueError("invalid window or unknown role in credit exposure")
    for keys in mapped.values():
        for slot in (master[1]+1,master[1]+2):
            if len({exposures[k] for k in keys if k[1]==slot})!=1:
                raise ValueError("same-step local samples must use identical exposure")
    for site, row in sites.items():
        req, actual, cap = (row[k] for k in ("request_kw","sustainable_kw","capacity_kw"))
        gap = max(0.,req-actual)
        upper = attribution(req,cap,actual)["upper_kw"] if attributed and cap is not None else gap/2.
        lower = gap-upper
        local = mapped[site]
        weight = sum(exposures[k] for k in local)
        duration = exposures[master]
        if weight<=0. or duration<=0.:
            raise ValueError("missing positive master or local exposure")
        result[master] += upper*duration
        for k in local:
            result[k] += lower*duration*exposures[k]/weight
        site_ledger[site] = dict(gap_kw=gap,upper_kw=upper,lower_kw=lower,discounted_s=duration,
                                 penalty_kw_s=gap*duration,confirmed=cap is not None)
    if not math.isclose(sum(result.values()),sum(r["penalty_kw_s"] for r in site_ledger.values()),abs_tol=1e-8):
        raise ArithmeticError("discounted penalty is not conserved")
    return result,site_ledger


def matched_update(model, behavior, returns, penalty, lr):
    """One shared batch. Credit transfers enter the loss before actor and value gradients."""
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    before = {k:p.detach().clone() for k,p in model.named_parameters()}
    losses = []
    examples = []
    for key, record in behavior.items():
        role,slot = key
        lp,value = model.heads[role].evaluate_sample(record)
        ratio = (lp-record["old_logp"]).exp()
        target = returns[key]-.25*penalty[key]/10.
        adv = target-record["old_value"]
        losses.append(-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)+.5*(value-target).square())
        examples.append(dict(role=role,slot=slot,target=target,ratio=float(ratio.detach()),penalty_kw_s=penalty[key]))
    loss = torch.stack(losses).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    norms = {}
    for role,head in model.heads.items():
        norms[role] = {
            kind:math.sqrt(sum(float(p.grad.square().sum()) for p in sub.parameters() if p.grad is not None))
            for kind,sub in (("actor",head.actor),("critic",head.critic))
        }
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    opt.step()
    deltas = {k:float((p.detach()-before[k]).norm()) for k,p in model.named_parameters()}
    return dict(optimizer_steps=1,loss=float(loss.detach()),examples=examples,gradient_l2=norms,parameter_delta=deltas)
