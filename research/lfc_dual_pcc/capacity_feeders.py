"""Native IEEE34 feeder validation of capacity commitments and fast dispatch traces."""
import argparse
from contextlib import ExitStack
from importlib.metadata import version
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import tempfile

import numpy as np

from envs.lfc.diagnostics.ieee34_bus8 import DECK_TREE, ObservedReference
from scripts.check_lfc34_flexible_load import ROOT, native
from research.lfc_dual_pcc.capacity_policy import CapacityPolicy
from research.lfc_dual_pcc.case_a_plant import CONFIG, feeder


def finite_vector(values, count, name):
    if len(values) != count or not all(math.isfinite(v) for v in values):
        raise ValueError(f"invalid {name} vector")
    return values


def feeder_config(energy_kws, ramp_up):
    """Convert energy above reserve, on the storage side, into native device SOC."""
    cfg = json.loads(CONFIG.read_bytes())
    count = len(energy_kws)
    finite_vector(energy_kws, count, "initial energy")
    finite_vector(ramp_up, count, "ramp")
    if count < 2 or min(energy_kws) < 0 or min(ramp_up) <= 0:
        raise ValueError("expected at least two feeders with nonnegative energy and positive ramp")
    storage = cfg["storage"]
    stored_kws = len(storage["devices"]) * storage["kwh"] * 3600
    sites = [f"pcc{i+1}" for i in range(count)]
    storage["initial_soc"] = {s: storage["reserve"] + e / stored_kws
                              for s, e in zip(sites, energy_kws, strict=True)}
    if max(storage["initial_soc"].values()) > 1:
        raise ValueError("initial stored energy exceeds device rating")
    cfg["ramp_up"] = dict(zip(sites, ramp_up, strict=True))
    return cfg, sites


def nominal_table(deck, reference):
    """Measure every candidate at one reference state with all actions admissible."""
    cfg, sites = feeder_config([1000., 1000.], [10., 10.])
    with feeder(deck, reference, cfg, sites[0]) as plant:
        table = []
        for nominal, row in zip(cfg["actions_battery_pv_kw"], plant.candidates(), strict=True):
            if row["veto"]:
                raise ValueError(f"reference action {nominal} rejected: {row['veto']}")
            table.append({"battery_pv_kw": nominal, "support_kw": plant.base[0] - row["pcc"][0],
                          "q_support_kvar": plant.base[1] - row["pcc"][1]})
        return table


class NativePolicyControl:
    """Greedy online inference with measured SOC, feasible sets and PCC response."""

    def __init__(self, episode, weights):
        n = episode["n_sites"]
        seed = episode["train_seed"]
        self.arm = episode["arm"]
        if self.arm not in {"rule", "fixed_lower", "joint", "paired", "joint_matched"}:
            raise ValueError(f"unknown intervention {self.arm}")
        self.policy = CapacityPolicy(n, weights[str(n)][str(seed)][self.arm])
        self.capacity = None
        self.ramp = np.array([episode["ramp_up_kw_s"]])

    def dispatch(self, plants, index, request, activation):
        if not math.isfinite(request) or request < 0 or not 0 <= activation <= 1:
            raise ValueError("invalid external activation or capacity requirement")
        values = []
        valid = []
        energies = []
        previous = []
        for plant in plants:
            storage = plant.storage
            energy = sum(max(0., x-storage["reserve"]) for x in plant.soc()) * storage["kwh"] * 3600
            energies.append(energy)
            previous.append(plant.last.get("delivered_kw", 0.))
            rows = plant.candidates()
            valid.append([not row["veto"] for row in rows])
            values.append([plant.base[0] - row["pcc"][0] if not row["veto"] else 0. for row in rows])
        values = np.array([values])
        valid = np.array([valid])
        energy = np.array([energies])
        previous = np.array([previous])
        if index % 2 == 0:
            limits = np.where(valid, values, 0.).max(axis=-1)
            self.capacity, _ = self.policy.slow(np.array([request]), energy, self.ramp,
                                                previous, limits, rule=self.arm == "rule")
        target = activation * self.capacity
        actions, _ = self.policy.fast(target, self.capacity, energy, self.ramp, previous,
                                      valid, values, rule=self.arm in {"rule", "fixed_lower"})
        return actions[0].tolist(), self.capacity[0].tolist(), target[0].tolist()


def replay_episode(deck, reference, episode, table, weights=None):
    count = episode["n_sites"]
    cfg, sites = feeder_config(episode["energy_kws"], episode["ramp_up_kw_s"])
    if count != len(sites) or not episode["steps"]:
        raise ValueError("episode has inconsistent feeder count or empty trajectory")
    result = {key: episode[key] for key in (
        "n_sites", "arm", "event_seed", "train_seed", "energy_kws", "ramp_up_kw_s")}
    steps = []
    errors = []
    matched = 0
    control = NativePolicyControl(episode, weights) if weights is not None else None
    with ExitStack() as stack:
        plants = [stack.enter_context(feeder(deck, reference, cfg, site)) for site in sites]
        result["base_kw_kvar"] = [p.base for p in plants]
        for i, step in enumerate(episode["steps"]):
            actions = finite_vector(step["action"], count, "action")
            caps = finite_vector(step["capacity_kw"], count, "capacity")
            targets = finite_vector(step["target_kw"], count, "target")
            request, activation = step["request_kw"], step["activation"]
            if control is not None:
                actions, caps, targets = control.dispatch(plants, i, request, activation)
            matched += sum(a == b for a,b in zip(actions, step["action"], strict=True))
            if any(type(a) is not int or not 0 <= a < len(cfg["actions_battery_pv_kw"]) for a in actions):
                raise ValueError("invalid native action index")
            if any(t < -1e-9 or c < t-1e-9 for c, t in zip(caps, targets, strict=True)):
                raise ValueError("activation violates its capacity commitment")
            rows = [p.execute(a) for p, a in zip(plants, actions, strict=True)]
            delivery = [r["delivered_kw"] for r in rows]
            errors.extend(abs(v-table[a]["support_kw"]) for v,a in zip(delivery, actions, strict=True))
            steps.append({
                "end_s": i + 1, "request_kw": request, "activation": activation,
                "action": actions, "capacity_kw": caps, "target_kw": targets,
                "delivered_kw": delivery,
                "q_support_kvar": [p.base[1]-r["pcc"][1] for p,r in zip(plants,rows,strict=True)],
                "min_soc_after": [min(r["soc_after"]) for r in rows],
                "min_v_pu": min(min(r["voltage"].values()) for r in rows),
                "max_v_pu": max(max(r["voltage"].values()) for r in rows),
                "max_abs_balance_kw_kvar": max(abs(x) for r in rows for x in r["balance"]),
            })
    aggregate_error = [sum(s["delivered_kw"]) - s["request_kw"] * s["activation"] for s in steps]
    result.update(steps=steps, metrics={
        "physical_actions": count * len(steps), "reduced_action_matches": matched,
        "max_reduced_delivery_error_kw": max(errors),
        "mean_reduced_delivery_error_kw": fmean(errors),
        "mean_external_abs_error_kw": fmean(abs(e) for e in aggregate_error),
        "unmet_wh": sum(max(0., -e) for e in aggregate_error) / 3.6,
        "excess_wh": sum(max(0., e) for e in aggregate_error) / 3.6,
        "min_phase_v_pu": min(s["min_v_pu"] for s in steps),
        "max_phase_v_pu": max(s["max_v_pu"] for s in steps),
        "max_abs_balance_kw_kvar": max(s["max_abs_balance_kw_kvar"] for s in steps),
    })
    return result


def heldout_replays(templates, weights, event_seeds):
    """Reconstruct held-out input signals and frozen surrogate comparators by seed."""
    from research.lfc_dual_pcc.run_capacity_cases import batch, rollout, scenario
    for template in templates:
        count, arm, seed = (template[k] for k in ("n_sites", "arm", "train_seed"))
        policy = CapacityPolicy(count, weights[str(count)][str(seed)][arm])
        for event in event_seeds:
            events = batch([scenario(np.random.default_rng(event), count)])
            evaluation_arm = "joint" if arm in {"paired", "joint_matched"} else arm
            _, _, _, steps = rollout(policy, events, evaluation_arm, trace=True)
            yield {"n_sites": count, "arm": arm, "train_seed": seed, "event_seed": event,
                   "energy_kws": events[0][0].tolist(), "ramp_up_kw_s": events[1][0].tolist(),
                   "steps": steps}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, help="omit to measure only the reference response table")
    parser.add_argument("--weights", type=Path, help="recompute slow and fast decisions from native measurements")
    parser.add_argument("--event-seeds", nargs="+", type=int, help="generate other held-out events from their seeds")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.weights and not args.episodes:
        parser.error("--weights requires --episodes")
    if args.event_seeds and not args.weights:
        parser.error("--event-seeds requires --weights")
    reference = ObservedReference.read_report(args.reference.read_bytes())
    result = {"deck_tree": DECK_TREE, "case_config_sha256": digest(CONFIG),
              "operating_reference_sha256": digest(args.reference),
              "dss_python_version": version("dss-python"),
              "mode": "independent IEEE34 OpenDSS feeders, fixed substation voltage, 1 s step"}
    with tempfile.TemporaryDirectory() as scratch:
        deck = native.archived_deck(ROOT, Path(scratch))
        result["nominal_table"] = nominal_table(deck, reference)
        if args.episodes:
            result["source_sha256"] = digest(args.episodes)
            result["inference"] = "online on native feeder states" if args.weights else "fixed-action replay"
            weights = None
            if args.weights:
                weights = json.loads(args.weights.read_bytes())
                result["weights_sha256"] = digest(args.weights)
            episodes = json.loads(args.episodes.read_bytes())["episodes"]
            if args.event_seeds:
                episodes = heldout_replays(episodes, weights, args.event_seeds)
            result["episodes"] = [replay_episode(deck, reference, e, result["nominal_table"], weights)
                                  for e in episodes]
            if not result["episodes"]:
                raise ValueError("empty replay")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
