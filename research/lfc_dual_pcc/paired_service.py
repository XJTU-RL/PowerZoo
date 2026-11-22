"""A small service-only preference update on restored native choice pairs."""
from itertools import combinations
import math

import torch
from torch.nn import functional as F

from .clock_graph import tensor
from .equal_service import SERVICE_SLOTS
from .revision_protocol import REVISED_KW
from .service_tradeoff import PROBES, TOLERANCE_KW, allowed_splits, service_terms


LEARNING_RATE = .5
SOURCE_CODE = "d5ae02d6fd1c927d2ac625cb9a64e789ecdfb5cb"
DISCOUNT_PER_SECOND = .95
MIN_CHANGED_WINDOWS = 2


def consumption_clock(origin, physical):
    positions = (origin+1, origin+2)
    if not all(k in physical for k in positions):
        raise ValueError("one of the certified action slots was never consumed")
    first, second = (physical[k] for k in positions)
    if first["slot"] >= second["slot"] or first["end_s"] > second["write_s"]+1e-8:
        raise ValueError("out of order physical consumers")
    start = first["write_s"]
    clock = []
    for action_slot, frame in zip(positions, (first, second), strict=True):
        lo, hi = frame["write_s"], frame["end_s"]
        if frame["used_slot"]!=action_slot or not all(math.isfinite(v) for v in (lo,hi)) or hi<=lo:
            raise ValueError("invalid consumption join or duration")
        weight = DISCOUNT_PER_SECOND**(lo-start)*(hi-lo)
        clock.append(dict(action_slot=action_slot,physical_slot=frame["slot"],start_s=lo,end_s=hi,weight_s=weight))
    return clock


def paired_window(witness, physical, frequency_scale):
    origin = witness["slot"]
    enabled = allowed_splits(witness["allocatable_kw"],*witness["interval_kw"],witness["reports"],
                             witness["frequency_error_hz"],frequency_scale)
    clock = consumption_clock(origin,physical)
    trials = {r["latent"]:r for r in witness["alternatives"]}
    if len(trials)!=len(witness["alternatives"]) or set(trials)!=set(PROBES):
        raise ValueError("missing or duplicate certified alternatives")
    costs, plain, deliveries = {}, {}, {}
    for action,u in enumerate(PROBES):
        if not enabled[action]:
            continue
        trial = trials[u]
        if not math.isclose(sum(trial["request_kw"].values()), REVISED_KW, rel_tol=0., abs_tol=1e-9):
            raise ValueError("certified alternatives have different service requests")
        delivery = [sum(site.values()) for site in trial["delivered_kw"]]
        if len(delivery)!=len(clock) or not all(math.isfinite(v) for v in delivery):
            raise ValueError("candidate measurements do not cover both physical consumers")
        penalties = [sum(service_terms(REVISED_KW,kw)) for kw in delivery]
        weights = [row["weight_s"] for row in clock]
        durations = [row["end_s"]-row["start_s"] for row in clock]
        costs[action] = sum(cost*weight for cost,weight in zip(penalties,weights,strict=True))/sum(weights)
        plain[action] = sum(cost*dt for cost,dt in zip(penalties,durations,strict=True))/sum(durations)
        deliveries[action] = delivery
    pairs = []
    for a,b in combinations(costs,2):
        difference = costs[b]-costs[a]
        # Require twice the engineering resolution to give a pair a direction.
        resolved = abs(difference) > 2*TOLERANCE_KW
        pairs.append(dict(a=a,b=b,service_advantage_a_over_b_kw=difference,
                          undiscounted_advantage_a_over_b_kw=plain[b]-plain[a],resolved=resolved,
                          weight=min(1.,abs(difference)/REVISED_KW) if resolved else 0.))
    return dict(slot=origin,clock=clock,mask=enabled,costs_kw=costs,delivery_kw=deliveries,
                alternatives=trials,pairs=pairs,x=witness["observation"])


def fit_bias(model, data):
    """Differentiate paired margins once; keep the actor feature map frozen."""
    layer = model.heads["master"].actor.layers[-1]
    before = layer.bias.detach().clone()
    pairs = [(w,p) for w in data for p in w["pairs"]]
    if not pairs or not any(p["resolved"] for _,p in pairs):
        raise ValueError("no resolved service preference among effective alternatives")
    scores = {w["slot"]:model.heads["master"].actor(tensor(w["x"])) for w in data}
    loss = []
    for row,pair in pairs:
        z = scores[row["slot"]]
        margin = z[pair["a"]]-z[pair["b"]]
        direction = 1 if pair["service_advantage_a_over_b_kw"] > 0 else -1
        loss.append(pair["weight"]*F.softplus(-direction*margin))
    objective = torch.stack(loss).mean()
    gradient, = torch.autograd.grad(objective,layer.bias)
    with torch.no_grad():
        layer.bias.sub_(LEARNING_RATE*gradient)
    return dict(optimizer_steps=1,loss=float(objective.detach()),learning_rate=LEARNING_RATE,
                pairs=len(pairs),resolved=sum(p["resolved"] for _,p in pairs),sum_weights=sum(p["weight"] for _,p in pairs),
                bias_before=before.tolist(),gradient=gradient.tolist(),bias_after=layer.bias.tolist())


def post_service_swing(frames, frequency):
    physical = {f["used_slot"]:f for f in frames if f["used_slot"] is not None}
    release_fast_slot = SERVICE_SLOTS[-1]+3
    pivot = physical[release_fast_slot]["write_s"]
    end = pivot+4.
    values = [hz for t,hz in frequency if pivot<t<=end]
    if len(values)<2 or frequency[-1][0]<end-1e-7:
        raise ValueError("incomplete four-second post-service meter window")
    return dict(post_service_write_s=pivot,post_service_window_end_s=end,
                post_service_sample_count=len(values),post_service_swing_hz=max(values)-min(values))
