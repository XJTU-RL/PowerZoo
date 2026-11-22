"""Recompute message joins, update counts and nonzero graph updates from native traces."""
import argparse
import json
from pathlib import Path

from .driver import dump, sha


def audit(directory):
    path = directory/"raw.jsonl"
    raw = [json.loads(l) for l in path.read_text().splitlines()]
    summary = json.loads((directory/"summary.json").read_bytes())
    if sha(path)!=summary["raw_sha256"] or raw[-1]["phase"]!="complete":
        raise ValueError("incomplete or changed collection")
    def select(phase):
        return [r for r in raw if r["phase"]==phase]
    reports = {r["sequence"]:r for r in select("telemetry")}
    holds = {r["slot"]:r for r in select("hold")}
    commands = {r["sequence"]:r for r in select("dispatch")}
    returns = {r["action_slot"]:r for r in select("consumed_return")}
    targets = {(r["role"],r["action_slot"]):r for r in select("learning_target")}
    decisions = select("policy")
    if len(holds)!=24 or len(commands)!=8 or len(decisions)!=104 or len(targets)!=104:
        raise ValueError("incomplete native decisions")
    for d in decisions:
        now,slot,role = d["consumed_s"],d["slot"],d["role"]
        if role=="master":
            if d["frequency_sample_s"] >= now:
                raise ValueError("noncausal slow frequency")
            per_site = d.get("reports")
            for i, s in enumerate(("A", "B")):
                offset = 5 + 13*i
                if per_site is None:
                    sequence = d["report"]["sequence"]
                else:
                    sequence = per_site[s]
                    presence, age = d["x"][1+2*i:3+2*i]
                    if sequence is None:
                        if presence != 0 or age != 0 or any(d["x"][offset:offset+13]):
                            raise ValueError("missing report represented as a measurement")
                        continue
                    if presence != 1 or abs(age-(now-reports[sequence]["sampled"])) > 1e-5:
                        raise ValueError("incorrect per-feeder report timing")
                r = reports[sequence]
                if not (r["sampled"] <= r["arrival"] < now and r["effective"] <= now < r["expiry"]):
                    raise ValueError("noncausal slow report")
                data = r["data"][s]
                src = holds[r["sequence"]]
                if (data["consumed_s"]!=src["write_s"]
                        or data["consumed_request_kw"]!=src["request"][s]
                        or [data["p_kw"],data["q_kvar"]]!=src["pcc"][s]
                        or abs(d["x"][offset+10]-(src["request"][s]-src["delivered"][s])/10)>1e-5
                        or abs(d["x"][offset+12]-data["bid_kw"]/10)>1e-5):
                    raise ValueError("slow feature and source command/meter disagree")
        else:
            c = commands[d["command"]]
            if not c["arrival"]<now or not c["effective"]<=now<c["expiry"]:
                raise ValueError("device acted before command arrival or after expiry")
            r = returns[slot]
            physical = holds[r["consumed_slot"]]
            if r["command"]!=d["command"] or physical["used_slot"]!=slot or r["interval"]!=[physical["write_s"],physical["end_s"]]:
                raise ValueError("return accrues to wrong command or interval")
    for slot in commands:
        k = [i for i,r in returns.items() if r["command"]==slot]
        elapsed = sum(returns[i]["interval"][1]-returns[i]["interval"][0] for i in k)
        if len(k)!=2 or abs(targets["master",slot]["elapsed_s"]-elapsed)>1e-8:
            raise ValueError("slow window duration")
    return summary, dict(decisions=len(decisions),master=len(commands),fast_windows=len(returns),
                         native_holds=len(holds),checked_targets=len(targets))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training","evaluation","out"):
        p.add_argument(name,type=Path)
    p.add_argument("--fixed-eval", type=Path, help="paired evaluation with the same frozen lower policy")
    args = p.parse_args()
    train, train_count = audit(args.training)
    held, eval_count = audit(args.evaluation)
    opt = train["optimizer"]
    if held.get("master_policy", "learned") != "learned":
        raise ValueError("audit the frozen learned master before comparing the fixed evaluation arm")
    if (train["after"]!=held["before"] or held["before"]!=held["after"] or train["event"]==held["event"]
            or train["key"]!=held["key"] or train["before"]==train["after"]
            or train["checkpoint_sha256"]!=held["checkpoint_sha256"] or held["optimizer"] is not None):
        raise ValueError("evaluation checkpoint or held-out split")
    categories = {}
    for role,count in train["counts"].items():
        seen = sum(any(s["role"]==role for s in step["samples"]) for step in opt["steps"])
        if count!=seen or seen!=(8 if role=="master" else 16):
            raise ValueError("wrong number of actual optimizer steps per role")
        categories[role] = {}
        for kind in ("actor","critic"):
            prefix = f"heads.{role}.{kind}."
            keys = [k for k in opt["deltas"] if k.startswith(prefix)]
            if not keys or not any(opt["deltas"][k]>0 for k in keys):
                raise ValueError("inactive policy/value network")
            categories[role][kind] = sum(opt["deltas"][k]**2 for k in keys)**.5
            if role.startswith(("battery","pv")):
                for layer in ("embed.weight","message.weight"):
                    k = prefix+layer
                    if min(opt["gradients"][k],opt["deltas"][k])<=0:
                        raise ValueError("graph layer absent from update")
    result = dict(train=train_count, held_out=eval_count, optimizer_steps=len(opt["steps"]),
                  module_change_l2=categories, checkpoint_sha256=train["checkpoint_sha256"],
                  source=train["key"]["git"], formal_50hz=False)
    if args.fixed_eval:
        fixed, fixed_count = audit(args.fixed_eval)
        matched = ("mode", "event", "seed", "key", "before", "after", "checkpoint_sha256")
        if (fixed.get("master_policy") != "fixed" or fixed["optimizer"] is not None
                or any(fixed[k] != held[k] for k in matched) or fixed_count != eval_count):
            raise ValueError("fixed comparison must use the same evaluation event, checkpoint and trace coverage")
        keys = ("iae_hz_s", "nadir_hz", "net_shortfall_kwh", "measured_grid_positive_excess_kwh")
        result["master_comparison"] = {
            "learned": {k: held["metrics"][k] for k in keys},
            "fixed": {k: fixed["metrics"][k] for k in keys},
            "learned_minus_fixed": {k: held["metrics"][k]-fixed["metrics"][k] for k in keys},
        }
    dump(args.out, result)


if __name__=="__main__":
    main()
