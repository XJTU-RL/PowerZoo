"""Closed-loop first-window interventions and zero-weight controls for the capacity trials."""
import argparse
import json
from pathlib import Path

import numpy as np

from .capacity_policy import CapacityPolicy
from .run_capacity_cases import HOLDOUT, METRICS, TRAIN_SEEDS, batch, rollout, scenario, write

# Independent synthetic paired events: hold exogenous signals, ramp and first-slot
# feasible sets fixed, exchange only the initial storage reserve margins of A and B.
INTERVENTION_EVENTS = tuple(range(9100, 9132))
EPS = 1.e-8


def swap_events(n_sites, rich_a):
    events = batch(scenario(np.random.default_rng(i), n_sites) for i in INTERVENTION_EVENTS)
    energy = [48., 10.5] if rich_a else [10.5, 48.]
    if n_sites == 3:
        energy.append(32.)
    events[0][:] = energy
    events[1][:] = [6., 7.5, 4.][:n_sites]
    return events


def compare(policy, events, force):
    report, rewards, _, _ = rollout(policy, events, "joint", first_choice=force)
    return rewards[:2].sum(axis=0), report[:, METRICS.index("episode_return")]


def effect(policy, rich):
    events = swap_events(policy.n_sites, rich)
    if policy.n_sites == 2:
        changes = [1, 3]  # shift A from the 25% to 75% point of its feasible interval
    else:
        changes = [[0, 1, 1], [2, 1, 1]]  # alter only A's bid and rerun the clearing
    low = compare(policy, events, changes[0])
    high = compare(policy, events, changes[1])
    return tuple(b - a for a, b in zip(low, high, strict=True))


def diagnose(source, output):
    with (source / "weights.json").open() as stream:
        weights = json.load(stream)
    result = {"intervention_seeds": INTERVENTION_EVENTS, "by_size": {}}
    for n in (2, 3):
        events = batch(scenario(np.random.default_rng(i), n) for i in HOLDOUT)
        untrained = CapacityPolicy(n)
        zero, _, _, _ = rollout(untrained, events, "joint")
        rows = []
        for seed in TRAIN_SEEDS:
            policy = CapacityPolicy(n, weights[str(n)][str(seed)]["joint"])
            early_poor, full_poor = effect(policy, False)
            early_rich, full_rich = effect(policy, True)
            for j, event in enumerate(INTERVENTION_EVENTS):
                rows.append(dict(train_seed=seed, event_seed=event,
                                 first_return_delta_poor=float(early_poor[j]),
                                 first_return_delta_rich=float(early_rich[j]),
                                 total_return_delta_poor=float(full_poor[j]),
                                 total_return_delta_rich=float(full_rich[j])))
        poor = np.array([r["total_return_delta_poor"] for r in rows])
        rich = np.array([r["total_return_delta_rich"] for r in rows])
        first_poor = np.array([r["first_return_delta_poor"] for r in rows])
        first_rich = np.array([r["first_return_delta_rich"] for r in rows])
        reversed_sign = ((poor > EPS) & (rich < -EPS)) | ((poor < -EPS) & (rich > EPS))
        result["by_size"][n] = dict(
            untrained_mean=dict(zip(METRICS, zero.mean(axis=0).tolist(), strict=True)),
            untrained_per_event=[dict(seed=seed, **dict(zip(METRICS, vals.tolist(), strict=True)))
                                 for seed, vals in zip(HOLDOUT, zero, strict=True)],
            full_reversal=int(reversed_sign.sum()),
            reversal_with_equal_first_effect=int(
                (reversed_sign & (np.abs(first_rich - first_poor) < EPS)).sum()),
            favors_higher_bid_only_when_rich=int(((poor < -EPS) & (rich > EPS)).sum()),
            equal_first_effect=int((np.abs(first_rich - first_poor) < EPS).sum()),
            avg_delta_poor=float(poor.mean()), avg_delta_rich=float(rich.mean()), rows=rows)
    write(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.source, args.out)
