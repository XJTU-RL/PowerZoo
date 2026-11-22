"""60 Hz engineering protocol for comparing dispatch under one external service request."""
import math
import statistics

import torch

from .clock_graph import ClockGraphPolicy, Coordinator, dense, slow_input
from .driver import state_digest


SHARES_A = (.2, .5, .8)
FIXED_SHARE = 1
SERVICE_KW = 10.
SERVICE_SLOTS = (24, 26, 28, 30, 32, 34)
TRACKING_WEIGHT = 1.
LOWER_SOURCE = "83d30ce189f58b5b6b4a7d43c1bf826154fa9968"


def service_kw(command_slot):
    return SERVICE_KW if command_slot in SERVICE_SLOTS else 0.


def split_candidates(total, limits):
    """Enumerate projected commands and keep one action per physical command vector."""
    if not all(math.isfinite(x) and x >= 0 for x in (total, limits["A"], limits["B"])):
        raise ValueError("invalid external demand or capacity")
    dispatched = min(total, limits["A"] + limits["B"])
    lo, hi = max(0., dispatched-limits["B"]), min(dispatched, limits["A"])
    a = [min(hi, max(lo, share*dispatched)) for share in SHARES_A]
    commands = [dict(A=x, B=min(limits["B"], max(0., dispatched-x))) for x in a]
    enabled = [False]*len(commands)
    distinct = []
    # Keep the fixed comparator available when its projection aliases another action.
    for i in (FIXED_SHARE, 0, 2):
        if not any(math.isclose(a[i], previous, rel_tol=0., abs_tol=1e-9) for previous in distinct):
            enabled[i] = True
            distinct.append(a[i])
    return commands, enabled


class EqualServicePolicy(ClockGraphPolicy):
    def __init__(self):
        super().__init__()
        self.heads["master"] = dense(32, len(SHARES_A))

    def freeze_lower(self, weights):
        for role, head in self.heads.items():
            if role != "master":
                prefix = f"heads.{role}."
                head.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)})
                head.requires_grad_(False)

    def lower_digest(self):
        lower = torch.nn.ModuleDict({role:head for role,head in self.heads.items() if role != "master"})
        return state_digest(lower)


def freeze_model(path, key):
    base = torch.load(path, map_location="cpu", weights_only=True)
    original = ClockGraphPolicy()
    original.load_state_dict(base["weights"])
    if (base["key"]["git"] != LOWER_SOURCE or state_digest(original) != base["digest"]
            or any(base["key"][k] != key[k] for k in
                   ("config", "wiring", "reference", "mechanism", "versions", "frequency_hz"))):
        raise ValueError("incompatible frozen lower checkpoint")
    model = EqualServicePolicy()
    model.freeze_lower(base["weights"])
    return model


class EqualServiceCoordinator(Coordinator):
    def __init__(self, model, cfg, greedy, *, fixed_master=False):
        super().__init__(model, cfg, greedy, fixed_master=fixed_master, tracking_logit_weight=TRACKING_WEIGHT)

    def choose(self, role, inputs, mask, slot, source, now, *, logit_offsets=None, **details):
        if role == "master":
            return super().choose(role, inputs, mask, slot, source, now, logit_offsets=logit_offsets, **details)
        with torch.no_grad():
            action, _ = self.model.heads[role].decision(inputs, mask, True, logit_offsets=logit_offsets)
        self.emit(dict(phase="policy", role=role, slot=slot, consumed_s=now, action=action,
                       command=source, mask=mask, logit_offsets=logit_offsets, **details))
        return action

    def master_allocation(self, error, reports, limits, slot, now, freq_time, sequences):
        request = service_kw(slot)
        choices, distinct = split_candidates(request, limits)
        mask = [i == FIXED_SHARE for i in range(len(choices))] if self.fixed_master else distinct
        x = torch.cat((slow_input(error, self.cfg["frequency_scale_hz"], reports, now),
                       torch.tensor([request/SERVICE_KW])))
        action = self.choose("master", (x,), mask, slot, slot, now, reports=sequences,
                             x=x.tolist(), frequency_sample_s=freq_time, distinct_commands=sum(distinct))
        return choices[action], request

    def returns(self, frames, trace, ref):
        physical = {f["used_slot"]: f for f in frames if f["used_slot"] is not None}
        holding = {f["slot"]: f for f in frames}
        samples = []
        for r in self.records:
            origin = r["command"]
            total, elapsed = 0., 0.
            for slot in (origin+1, origin+2):
                source, measured = holding[slot], physical[slot]
                dt = measured["end_s"]-measured["write_s"]
                hz = [abs(v-ref) for t,v in trace if measured["write_s"] < t < measured["end_s"]]
                gap = max(0., service_kw(origin)-sum(source["delivered"].values()))
                score = -statistics.mean(hz)/self.cfg["frequency_scale_hz"] - gap/SERVICE_KW
                total += .95**elapsed*score*dt
                elapsed += dt
                self.emit(dict(phase="master_feedback", command=origin, action_slot=slot,
                               physical_slot=measured["slot"], unmet_kw=gap, duration_s=dt))
            samples.append((origin, elapsed, total, r["outputs"]))
        return {"master":samples}
