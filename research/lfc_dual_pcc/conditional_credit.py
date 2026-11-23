"""Matched PPO updates from paired, offline intervention returns."""
from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn

from .clock_graph import MLP, Message, tensor, update
from .driver import state_digest


@dataclass(frozen=True)
class MeterStep:
    source_slot: int
    source_decision_s: float
    command: int
    start_s: float
    end_s: float
    team_reward_rate: float


@dataclass(frozen=True)
class Branch:
    role: str
    slot: int
    decision_s: float
    action: int
    state: tuple
    mask: tuple
    prefix: str
    factual_trace: str
    pending_queue: str
    policy: str
    noise: str
    reward_spec: str
    active: Message | None
    issued: Message | None
    meter: tuple[MeterStep, ...]


@dataclass(frozen=True)
class Pair:
    factual: Branch
    alternative: Branch
    # The pair's action support is fixed before sampling the factual action.
    support: tuple[int, int]


def message_timing(m):
    return m.sequence, m.sampled, m.arrival, m.effective, m.expiry, m.urgency_hz


def branch_returns(pair):
    a, b = pair.factual, pair.alternative
    for field in ("role", "slot", "decision_s", "state", "mask", "prefix", "factual_trace", "pending_queue",
                  "policy", "noise", "reward_spec", "active"):
        if getattr(a, field) != getattr(b, field):
            raise ValueError(f"unmatched intervention boundary: {field}")
    if (not all((a.prefix, a.factual_trace, a.pending_queue, a.policy, a.noise, a.reward_spec, a.state, a.meter))
            or not math.isfinite(a.decision_s) or not all(map(math.isfinite, a.state))
            or a.action == b.action or tuple(sorted(pair.support)) != tuple(sorted((a.action,b.action)))
            or any(not 0 <= i < len(a.mask) or not a.mask[i] for i in pair.support)):
        raise ValueError("invalid action support or missing intervention provenance")
    if a.role == "master":
        if a.issued is None or b.issued is None or a.issued.data == b.issued.data:
            raise ValueError("slow treatment must replace the newly issued command")
        if (message_timing(a.issued) != message_timing(b.issued) or a.issued.sequence != a.slot
                or a.issued.sampled != a.decision_s
                or not a.decision_s <= a.issued.arrival < a.issued.expiry
                or not a.decision_s < a.issued.effective < a.issued.expiry):
            raise ValueError("slow treatment changed communication timing")
    elif (a.issued != b.issued or a.active is None or not a.active.valid(a.decision_s)):
        raise ValueError("fast treatment must keep the effective slow command")
    t0 = a.meter[0].start_s
    last = t0
    first = None
    totals = [0., 0.]
    for x, y in zip(a.meter, b.meter, strict=True):
        if (x.start_s, x.end_s) != (y.start_s, y.end_s):
            raise ValueError("counterfactual returns have different physical clocks")
        if (not all(map(math.isfinite, (x.start_s, x.end_s, x.team_reward_rate, y.team_reward_rate,
                                       x.source_decision_s, y.source_decision_s)))
                or x.source_decision_s > x.start_s or y.source_decision_s > y.start_s
                or x.start_s < a.decision_s or not math.isclose(x.start_s,last,rel_tol=0.,abs_tol=1e-8)
                or x.end_s <= x.start_s):
            raise ValueError("invalid or gapped physical return window")
        last = x.end_s
        target = a.issued.sequence if a.role == "master" else a.active.sequence
        treated = x.command == target and (a.role == "master" or x.source_slot == a.slot)
        if first is None and treated:
            commands = (a.issued,b.issued) if a.role == "master" else (a.active,b.active)
            if ((x.source_slot,x.source_decision_s,x.command) != (y.source_slot,y.source_decision_s,y.command)
                    or (a.role != "master" and x.source_decision_s != a.decision_s)
                    or not all(m.valid(x.source_decision_s) for m in commands)):
                raise ValueError("paired intervention did not reach the same first eligible consumer")
            first = x.start_s
        if first is None and x != y:
            raise ValueError("branch diverged before the intervention reached the meter")
        weight = .95**(x.start_s-t0)*(x.end_s-x.start_s)
        totals[0] += weight*x.team_reward_rate
        totals[1] += weight*y.team_reward_rate
    if first is None:
        raise ValueError("intervention has no consumed outcome")
    return tuple(totals)


def fit_and_update(policy, records, grouped, pairs, lr, *, factual_trace, factual_meters):
    """Fit both centralized critics on identical outcomes, then run real PPO."""
    if not pairs:
        raise ValueError("no certified paired labels")
    fingerprint = state_digest(policy)
    decisions = {(r["role"],r["slot"]):r["outputs"] for r in records}
    grouped_keys = {id(r):(role,slot) for role,rows in grouped.items() for slot,_,_,r in rows}
    rows = []
    seen = set()
    widths = {}
    reward_spec = pairs[0].factual.reward_spec
    for pair in pairs:
        a = pair.factual
        returns = branch_returns(pair)
        record = decisions.get((a.role,a.slot))
        key = grouped_keys.get(id(record))
        if (key is None or key[0] != a.role or key in seen or a.policy != fingerprint or a.reward_spec != reward_spec
                or a.factual_trace != factual_trace or factual_meters.get((a.role,a.slot)) != a.meter
                or int(record["action"]) != a.action or tuple(record["mask"].tolist()) != a.mask):
            raise ValueError("branch factual row does not match a unique on-policy sample")
        shape = len(a.state),len(a.mask)
        if widths.setdefault(a.role, shape) != shape:
            raise ValueError("inconsistent centralized critic shape")
        seen.add(key)
        with torch.no_grad():
            head = policy.heads[a.role]
            lp, _ = head.evaluate_sample(record)
            if lp is None or not torch.isclose(lp,record["old_logp"],rtol=0.,atol=1e-7):
                raise ValueError("recorded behavior probability no longer matches")
            logits=head.logits(record["args"],record.get("logit_offsets"))
            weights=logits[list(pair.support)].softmax(0)
        rows.append((key,tensor(a.state),pair,tensor(returns),weights))
    if "master" not in widths or len(widths) < 2:
        raise ValueError("both slow and fast intervention evidence are required")
    critic = nn.ModuleDict({role:MLP(*shape) for role,shape in widths.items()})
    result = {}
    for mode in ("independent_q","paired_q"):
        q = deepcopy(critic)
        optimizer = torch.optim.Adam(q.parameters(),lr=lr)
        losses = []
        for (role,_), x, pair, targets, _ in rows:
            actions = [pair.factual.action,pair.alternative.action]
            e = q[role](x)[actions]-targets
            mean, contrast = (e[0]+e[1])/2, (e[0]-e[1])/2
            # Independent MSE equals mean²+contrast². Weight the contrast more
            # while preserving total weight across the two orthogonal terms.
            losses.append(e.square().mean() if mode=="independent_q"
                          else (2*mean.square()+4*contrast.square())/3)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(),5.,error_if_nonfinite=True)
        optimizer.step()
        advantages = {}
        with torch.no_grad():
            for key,x,pair,_,weights in rows:
                v=q[key[0]](x)[list(pair.support)]
                advantages[key]=float(v[list(pair.support).index(pair.factual.action)]-(weights*v).sum())
        clone = deepcopy(policy)
        actor = update(clone,grouped,lr,conditional_advantages=advantages)
        result[mode] = dict(model=clone,before=fingerprint,after=state_digest(clone),critic_steps=1,
                            critic_loss=float(loss.detach()),actor_update=actor,
                            advantage_by_role_slot=advantages)
    return result
