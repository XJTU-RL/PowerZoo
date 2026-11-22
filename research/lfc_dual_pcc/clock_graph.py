"""Causal two-rate actor-critic with DSS-masked heterogeneous device policies."""
from dataclasses import asdict, dataclass
import statistics

import torch
from torch import nn
from torch.distributions import Categorical

from .phase_graph import PhaseGraph


@dataclass(frozen=True)
class Message:
    sequence: int
    sampled: float
    arrival: float
    effective: float
    expiry: float
    data: dict

    def valid(self, now):
        return self.sampled <= self.arrival < now and self.effective <= now < self.expiry


def latest(queue, now):
    return max((m for m in queue if m.valid(now)), key=lambda m: (m.sampled, m.sequence), default=None)


def timing(msg, now):
    if msg is None:
        return [0.] * 4
    return [1., now-msg.sampled, now-msg.arrival, msg.expiry-now]


class MLP(nn.Module):
    def __init__(self, width, outputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, 24), nn.Tanh(), nn.Linear(24, outputs))

    def forward(self, x):
        return self.layers(x)


class GraphReadout(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.embed = nn.Linear(18, 16)
        self.message = nn.Linear(16*4, 16)
        self.out = MLP(32+6, outputs)

    def forward(self, x, adj, pool, context):
        h = self.embed(x).tanh()
        neighbors = torch.matmul(adj, h)
        h = self.message(torch.cat([h, *neighbors.unbind(0)], dim=1)).tanh()
        return self.out(torch.cat([h[pool].mean(0), h.mean(0), context]))


class PolicyValue(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor, self.critic = actor, critic

    def decision(self, args, mask, greedy):
        logits = self.actor(*args)
        allowed = torch.as_tensor(mask, dtype=torch.bool)
        if not allowed.any() or not torch.isfinite(logits).all():
            raise ValueError("empty feasible action set or invalid logits")
        dist = Categorical(logits=logits.masked_fill(~allowed, -torch.inf))
        action = dist.probs.argmax() if greedy else dist.sample()
        return int(action), dict(args=args, mask=allowed, action=action, old_logp=dist.log_prob(action).detach(),
                                 old_value=self.critic(*args).squeeze().detach())


def dense(n, k):
    return PolicyValue(MLP(n, k), MLP(n, 1))


class ClockGraphPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict({"master": dense(31, 9)})
        for s in ("A", "B"):
            self.heads["feeder"+s] = dense(14, 6)
            for kind, k in (("battery", 3), ("pv", 2)):
                self.heads[kind+s] = PolicyValue(GraphReadout(k), GraphReadout(1))


def tensor(xs):
    return torch.as_tensor(xs, dtype=torch.float32)


def offer(plant, dt, full=True):
    """Finite safe next-step witness. Duration is exactly the verified sample period."""
    rows = plant.candidates()
    allowed = [(plant.base[0]-r["pcc"][0], i) for i, r in enumerate(rows) if not r["veto"]]
    kw, idx = max(allowed)
    conservative = [(p,i) for p,i in allowed if p<=max(0.,kw)*.75+1e-9]
    bid, bid_idx = (kw,idx) if full or not conservative else max(conservative)
    blocked = {reason for row in rows for reason in row["veto"]}
    battery_kw = plant.cfg["actions_battery_pv_kw"][idx][0]
    # Equal device split. This conditional SOC horizon is not a network guarantee.
    remaining = min(plant.soc())-plant.storage["reserve"]
    energy_s = (max(0.,remaining)*plant.storage["kwh"]*len(plant.soc())*3600*plant.storage["eta"]/battery_kw
                if battery_kw else None)
    return dict(p_kw=plant.last["pcc"][0], q_kvar=plant.last["pcc"][1],
                delivered_kw=plant.base[0]-plant.last["pcc"][0], available_kw=max(0., kw),
                duration_s=dt, energy_only_s=energy_s, witness=idx, bid_kw=max(0.,bid), bid_witness=bid_idx,
                ramp_kw_s=plant.cfg["ramp_up"][plant.site],
                ramp_limited="ramp" in blocked, energy_limited="energy" in blocked,
                network_limited=bool(blocked-{"ramp","energy"}))


def report_features(data):
    return [data["p_kw"]/2000, data["q_kvar"]/500, data["delivered_kw"]/10,
            data["available_kw"]/10, data["duration_s"]/2, data["ramp_kw_s"]/10,
            float(data["ramp_limited"]), float(data["energy_limited"]), float(data["network_limited"])]


def slow_input(error, scale, report, now):
    values = [error/scale, *timing(report, now)]
    for s in ("A", "B"):
        d = report.data[s]
        req = d["consumed_request_kw"]
        values += report_features(d) + [req/10, (req-d["delivered_kw"])/10,
                                         1. if d["energy_only_s"] is None else min(60.,d["energy_only_s"])/60,
                                         d["bid_kw"]/10]
    return tensor(values)


class Coordinator:
    def __init__(self, model, cfg, greedy):
        self.model, self.cfg, self.greedy = model, cfg, greedy
        self.commands, self.reports, self.records = [], [], []
        self.offer_choices = {}
        self.report_consumer = {}

    def initialize(self, plants, emit):
        self.emit = emit
        self.graphs = {s: PhaseGraph(p) for s, p in plants.items()}
        emit(dict(phase="graphs", graphs={s: g.summary for s, g in self.graphs.items()}))

    def choose(self, role, inputs, mask, slot, source, now, **details):
        idx, outputs = self.model.heads[role].decision(inputs, mask, self.greedy)
        self.records.append(dict(role=role, slot=slot, command=source, outputs=outputs))
        self.emit(dict(phase="policy", role=role, slot=slot, consumed_s=now, action=idx,
                       command=source, mask=mask, **details))
        return idx

    def dispatch(self, slot, now, freq_time, error, plants):
        cfg = self.cfg
        scale = cfg["frequency_scale_hz"]
        if not freq_time < now:
            raise ValueError("policy consumed a future or coincident frequency")
        slots = cfg["window_slots"]
        if slot in slots:
            msg = latest(self.reports, now)
            if msg is None or msg.sequence != slot-2:
                raise ValueError("incomplete past telemetry at slow tick")
            x = slow_input(error, scale, msg, now)
            self.report_consumer[msg.sequence] = slot
            a = self.choose("master", (x,), [True]*9, slot, slot, now,
                            report=asdict(msg), x=x.tolist(), frequency_sample_s=freq_time)
            frac = (.2, .5, .8)[a//3]
            residual = sum(d["consumed_request_kw"]-d["delivered_kw"] for d in msg.data.values())
            total = min(cfg["max_request_kw"], max(0., error*cfg["droop_kw_hz"]*(.75,1.,1.25)[a%3]
                                                  + (0.,.15,.30)[a%3]*residual))
            allocation = {"A": total*frac, "B": total*(1-frac)}
            cmd = Message(slot, now, now+cfg["clock"]["report_latency_s"], slot+1., slot+3., allocation)
            self.commands.append(cmd)
            self.emit(dict(phase="dispatch", **asdict(cmd)))
        cmd = latest(self.commands, now)
        if slots[0]+1 <= slot <= slots[-1]+2 and (cmd is None or cmd.sequence != slots[0]+2*((slot-slots[0]-1)//2)):
            raise ValueError("command absent or consumed outside intended ZOH")
        if cmd is None:
            return {s: 0. for s in plants}, {s: 0 for s in plants}
        actions = {}
        for s, p in plants.items():
            req = cmd.data[s]
            graph = self.graphs[s]
            x = graph.features(p, req)
            data = self.reports[-1].data[s]  # local measurement, no uplink transit
            if self.reports[-1].sampled > now:
                raise ValueError("future local sample")
            obs = tensor([error/scale, req/10, *timing(cmd, now), *report_features(data)[:8]])
            bid_consumable = slot+2 in slots
            mask = [bid_consumable or i%2==1 for i in range(6)]
            rep = self.choose("feeder"+s, (obs,), mask, slot, cmd.sequence, now,
                              command_times=timing(cmd, now), x=obs.tolist(), local_sample_s=self.reports[-1].sampled)
            self.offer_choices[slot,s] = bool(rep%2)
            target = req*(.65, 1., 1.25)[rep//2]
            # Device heads see their feeder agent's effective target and command lifetime.
            x[:, -1] = target/10
            context = tensor([target/10, error/scale, *timing(cmd, now)])
            self.emit(dict(phase="device_input", site=s, slot=slot, command=cmd.sequence,
                           x=x.tolist(), context=context.tolist()))
            rows = p.candidates()
            table = cfg["actions_battery_pv_kw"]
            feasible = [(b,v) for (b,v),r in zip(table,rows) if not r["veto"] and (req>0 or b+v==0)]
            bm = [any(b==k for b,_ in feasible) for k in (0,4,8)]
            args = (x, graph.adj, graph.pools[0], context)
            b = (0,4,8)[self.choose("battery"+s, args, bm, slot, cmd.sequence, now, target_kw=target)]
            pm = [(b,k) in feasible for k in (0,2)]
            args = (x, graph.adj, graph.pools[1], context)
            v = (0,2)[self.choose("pv"+s, args, pm, slot, cmd.sequence, now, target_kw=target)]
            actions[s] = table.index([b,v])
        return cmd.data.copy(), actions

    def receive(self, frame, plants):
        now = frame["end_s"]
        # Bind the meter to the command that actually drove these devices. A later
        # issued command can already exist when the slow actor finally receives it.
        used = latest(self.commands,frame["write_s"])
        data = {}
        for s,p in plants.items():
            req = used.data[s] if used else 0.
            if req != frame["request"][s]:
                raise ValueError("feeder telemetry command join")
            data[s] = dict(**offer(p,self.cfg["fast_s"],self.offer_choices.get((frame["slot"],s),True)), consumed_request_kw=req,
                           consumed_command=used.sequence if used else None, consumed_s=frame["write_s"])
        msg = Message(frame["slot"], now, frame["arrival_s"], now, now+3., data)
        self.reports.append(msg)
        self.emit(dict(phase="telemetry", **asdict(msg)))

    def returns(self, frames, trace, ref):
        physical = {f["used_slot"]: f for f in frames if f["used_slot"] is not None}
        records = {}
        for r in self.records:
            if r["role"] == "master":
                continue
            k = r["slot"]
            if k in records:
                continue
            out = physical[k]
            hz = [abs(v-ref) for t,v in trace if out["write_s"] < t < out["end_s"]]
            src = next(f for f in frames if f["slot"] == k)
            loss = {s: abs(src["request"][s]-src["delivered"][s])/10 for s in ("A","B")}
            records[k] = (-statistics.mean(hz)/self.cfg["frequency_scale_hz"], loss, out["end_s"]-out["write_s"])
            self.emit(dict(phase="consumed_return", command=r["command"], action_slot=k,
                           consumed_slot=out["slot"], interval=[out["write_s"],out["end_s"]],
                           frequency_reward=records[k][0], local_error=loss))
        master = {}
        for r in self.records:
            if r["role"] == "master":
                keys = sorted({q["slot"] for q in self.records if q["command"]==r["command"] and q["role"]!="master"})
                if len(keys)!=self.cfg["slow_s"]:
                    raise ValueError("partial command window")
                elapsed, reward = 0., 0.
                for k in keys:
                    freq, loss, dt = records[k]
                    reward += .95**elapsed * (freq-.25*sum(loss.values())) * dt
                    elapsed += dt
                master[r["command"]] = (reward,elapsed,keys[-1])
        grouped = {}
        for r in self.records:
            role = r["role"]
            offer_consumer = None
            if role == "master":
                reward, elapsed, end_slot = master[r["command"]]
            else:
                end_slot = r["slot"]
                f, loss, elapsed = records[end_slot]
                reward = (f-.5*loss[role[-1]])*elapsed
                if role.startswith("feeder") and end_slot in self.report_consumer:
                    offer_consumer = self.report_consumer[end_slot]
                    future, duration, last = master[offer_consumer]
                    delay = physical[last]["end_s"]-physical[end_slot]["end_s"]
                    reward += .25*.95**delay*future/duration
            self.emit(dict(phase="learning_target",role=role,action_slot=r["slot"],command=r["command"],
                           last_physical_consumer=physical[end_slot]["slot"],elapsed_s=elapsed,
                           offer_consuming_master=offer_consumer,reward=reward))
            grouped.setdefault(role, []).append((end_slot,elapsed,reward,r["outputs"]))
        return grouped


def update(policy, grouped, lr):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    batches = {}
    for role, samples in grouped.items():
        returns = 0.
        for slot,elapsed,reward,record in reversed(samples):
            returns = reward+.95**elapsed*returns
            batches.setdefault(slot,[]).append((role,returns,record))
    before = {k: p.detach().clone() for k,p in policy.named_parameters()}
    gradients, steps = {}, []
    for slot, batch in sorted(batches.items()):
        terms, evidence = [], []
        for role,target,r in batch:
            head = policy.heads[role]
            logits = head.actor(*r["args"]).masked_fill(~r["mask"],-torch.inf)
            lp = Categorical(logits=logits).log_prob(r["action"])
            ratio = (lp-r["old_logp"]).exp()
            adv = target-r["old_value"]
            value = head.critic(*r["args"]).squeeze()
            terms.append(-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)+.5*(value-target).square())
            evidence.append(dict(role=role,target=target,advantage=float(adv),ratio=float(ratio.detach())))
        loss = torch.stack(terms).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(),5.,error_if_nonfinite=True)
        for k,p in policy.named_parameters():
            if p.grad is not None:
                gradients[k] = max(gradients.get(k,0.),float(p.grad.norm()))
        opt.step()
        steps.append(dict(last_action_slot=slot,loss=float(loss.detach()),gradient_norm=float(norm),samples=evidence))
    return dict(loss=sum(s["loss"] for s in steps),steps=steps,samples={k:len(v) for k,v in grouped.items()}, gradients=gradients,
                deltas={k:float((p.detach()-before[k]).norm()) for k,p in policy.named_parameters()})
