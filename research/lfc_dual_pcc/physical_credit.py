"""Matched, one-window PPO-style updates with conserved physical shortfall credit."""
import math

import torch

from .hierarchy import attribution


ROLES_BY_SITE = {"A":("feederA","batteryA","pvA"), "B":("feederB","batteryB","pvB")}


def nested_bound(sites, caps, tolerance):
    """Same total target and objective for observed, fixed-share and relaxed-share paths."""
    original = {s:sites[s]["request_kw"] for s in ROLES_BY_SITE}
    if any(original[s]<0. or original[s]>caps[s]+tolerance for s in original):
        raise ValueError("observed assignment outside the shared capacity evidence")
    total = sum(original.values())
    lo,hi = max(0.,total-caps["B"]),min(total,caps["A"])
    for s in original:
        match = next((r for r in sites[s]["certificate"]["action_paths"]
                      if r["actions"]==sites[s]["actions"] and (original[s]>0 or r["zero_request_allowed"])),None)
        if match is None or abs(match["support_kw"]-sites[s]["sustainable_kw"])>tolerance:
            raise ValueError("observed execution outside the fixed evidence domain")

    def best(split):
        witnesses,loss = {},{}
        for s,req in split.items():
            options = [r for r in sites[s]["certificate"]["action_paths"] if req>0 or r["zero_request_allowed"]]
            if not options:
                raise ValueError("missing conditional witness at the assignment")
            witnesses[s] = max(options,key=lambda r:r["support_kw"])
            loss[s] = max(0.,req-witnesses[s]["support_kw"])
        return dict(request_kw=split,witnesses=witnesses,site_loss_kw=loss,loss_kw=sum(loss.values()))

    fixed = best(original)
    # The exact original assignment is included explicitly, including endpoint rounding.
    samples = [fixed]+[best({"A":a,"B":total-a}) for a in (lo+(hi-lo)*u for u in (0.,.25,.5,.75,1.))]
    relaxed = min(samples,key=lambda r:r["loss_kw"])
    observed = {s:max(0.,req-sites[s]["sustainable_kw"]) for s,req in original.items()}
    if relaxed["loss_kw"]>fixed["loss_kw"]+tolerance or any(
            fixed["site_loss_kw"][s]>observed[s]+tolerance for s in observed):
        raise ValueError("conditional loss nesting violated")
    fixed_total = sum(min(fixed["site_loss_kw"][s],observed[s]) for s in observed)
    unavoidable = min(fixed_total,relaxed["loss_kw"])
    for s in observed:
        share = min(fixed["site_loss_kw"][s],observed[s])
        sites[s]["nested_upper_kw"] = share*(1.-.5*unavoidable/fixed_total) if fixed_total else 0.
    return dict(fixed=fixed,relaxed=relaxed,observed_kw=sum(observed.values()),sampled_assignments=len(samples),
                potentially_dispatch_avoidable_kw=fixed_total-unavoidable,
                device_avoidable_kw=sum(observed.values())-fixed_total,conditional_residual_kw=unavoidable)


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
        upper = gap/2.
        if attributed and cap is not None:
            upper = row["nested_upper_kw"] if "nested_upper_kw" in row else attribution(req,cap,actual)["upper_kw"]
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
