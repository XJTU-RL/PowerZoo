"""Paired policy gradients on commitment and execution decisions with common disturbances."""
import argparse
from pathlib import Path

import numpy as np

from .capacity_policy import CapacityPolicy
from .run_capacity_cases import (METRICS, N_WINDOWS, TRAIN_SEEDS, adam_ascent, batch,
                                 rollout, scenario, train, write)


# Disjoint from development, the original comparison, and the intervention study.
EVALUATION_SEEDS = tuple(range(9200, 9232))


class NoiseTape:
    """Give each decision a fixed random draw even after a physical trajectory diverges."""

    def __init__(self, draws):
        self.draws = iter(draws)

    def random(self, shape):
        u = next(self.draws)
        if shape != u.shape:
            raise ValueError("policy sampler and common noise tape differ")
        return u


def paired_tapes(rng, n, size):
    # One independent intervention per initial state. Split the minibatch evenly
    # between levels; within a level sample the decision time and site uniformly.
    ids = np.arange(size)
    upper = (ids % 2 == 0)
    tick = 3 * rng.integers(N_WINDOWS, size=size)
    tick += np.where(upper, 0, rng.integers(1, 3, size=size))
    site = rng.integers(n, size=size)
    factual, alternate = [], []
    for t in range(3 * N_WINDOWS):
        central = n == 2 and t % 3 == 0
        u = rng.random((size,) if central else (size, n))
        v = u.copy()
        selected = ids[tick == t]
        index = selected if central else (selected, site[selected])
        v[index] = rng.random(len(selected))
        factual.append(u)
        alternate.append(v)
    return factual, alternate, tick, site


def paired_gradients(policy, events, rng):
    size = len(events[0])
    if not policy.per_site_credit or size < 2 or size % 2:
        raise ValueError("paired gradients require individual actor scores and an even batch")
    u, v, tick, site = paired_tapes(rng, policy.n_sites, size)
    ra, rewards_a, scores_a, _ = rollout(policy, events, "joint", rng=NoiseTape(u))
    rb, rewards_b, scores_b, _ = rollout(policy, events, "joint", rng=NoiseTape(v))
    if len(scores_a) != 3 * N_WINDOWS or len(scores_b) != len(scores_a):
        raise ValueError("policy scores must cover all slow and fast decisions")
    returns_diff = (rewards_a - rewards_b)[::-1].cumsum(axis=0)[::-1]
    gradients = {key: np.zeros_like(value) for key, value in policy.weights.items()}
    for t, (a, b) in enumerate(zip(scores_a, scores_b, strict=True)):
        level, reward_index, score = a
        if (level, reward_index) != b[:2]:
            raise ValueError("reward and action clocks differ")
        ids = np.flatnonzero(tick == t)
        if not len(ids):
            continue
        central = level == "upper" and policy.n_sites == 2
        select = ids if central else (ids, site[ids])
        ds = score[select] - b[2][select]
        # E[(G-G')(score-score')/2 | predecision history, other current
        # actions] gives the selected actor's policy gradient. Restore the sum
        # across shared-parameter actors by inverse site selection probability.
        factor = 1 if central else policy.n_sites
        gradients[level] += factor * (returns_diff[reward_index, ids, None] * ds).sum(axis=0) / size
        # size/2 interventions per level and the symmetric 1/2 cancel.
    return gradients, float((ra[:, -1].mean() + rb[:, -1].mean()) / 2)


def paired_train(n_sites, seed, updates, batch_size):
    policy = CapacityPolicy(n_sites)
    policy.per_site_credit = True
    env_rng = np.random.default_rng(seed)
    action_rng = np.random.default_rng(np.random.SeedSequence([seed, 97]))
    moments = {key: np.zeros_like(w) for key, w in policy.weights.items()}
    second = {key: np.zeros_like(w) for key, w in policy.weights.items()}
    curve = []
    for update in range(1, updates + 1):
        events = batch(scenario(env_rng, n_sites) for _ in range(batch_size))
        grads, ret = paired_gradients(policy, events, action_rng)
        for level, grad in grads.items():
            adam_ascent(policy, level, grad, moments, second, update)
        if update == 1 or update % 25 == 0 or update == updates:
            curve.append(dict(update=update, stochastic_return=ret))
    policy.per_site_credit = False
    return policy, curve


def experiment(out, updates=150, size=64):
    if updates < 1 or size < 2 or size % 2:
        raise ValueError("positive updates and an even minibatch of at least two are required")
    out.mkdir(parents=True, exist_ok=False)
    records, replays, weights, curves = [], [], {}, {}
    for n in (2, 3):
        weights[n], curves[n] = {}, {}
        events = batch(scenario(np.random.default_rng(s), n) for s in EVALUATION_SEEDS)
        replay_events = batch([scenario(np.random.default_rng(8100), n)])
        for seed in TRAIN_SEEDS:
            weights[n][seed], curves[n][seed] = {}, {}
            for arm in ("rule", "joint_matched", "paired"):
                if arm == "paired":
                    policy, curve = paired_train(n, seed, updates, size)
                elif arm == "joint_matched":
                    policy, curve = train(n, seed, "joint", updates, 2 * size)
                else:
                    policy, curve = CapacityPolicy(n), []
                w = {k: v.tolist() for k, v in policy.weights.items()}
                weights[n][seed][arm], curves[n][seed][arm] = w, curve
                model = CapacityPolicy(n, w)
                mode = "rule" if arm == "rule" else "joint"
                raw, _, _, _ = rollout(model, events, mode)
                for event, values in zip(EVALUATION_SEEDS, raw, strict=True):
                    records.append(dict(n_sites=n, arm=arm, train_seed=seed, event_seed=event,
                                        **dict(zip(METRICS, values.tolist(), strict=True))))
                if seed == TRAIN_SEEDS[0] and arm != "rule":
                    _, _, _, steps = rollout(model, replay_events, mode, trace=True)
                    replays.append(dict(n_sites=n, arm=arm, train_seed=seed, event_seed=8100,
                                        energy_kws=replay_events[0][0].tolist(),
                                        ramp_up_kw_s=replay_events[1][0].tolist(), steps=steps))
    write(out/"weights.json", weights)
    write(out/"replay.json", dict(episodes=replays))
    write(out/"episodes.json", records)
    write(out/"learning_curves.json", curves)
    summary = {}
    for n in (2, 3):
        summary[n] = {}
        for arm in ("rule", "joint_matched", "paired"):
            rows = [r for r in records if r["n_sites"] == n and r["arm"] == arm]
            summary[n][arm] = {key: float(np.mean([r[key] for r in rows])) for key in METRICS}
    write(out/"summary.json", dict(updates=updates, paired_batch_size=size, matched_batch_size=2*size,
                                   trajectories_per_update=2*size, holdout=EVALUATION_SEEDS,
                                   train_seeds=TRAIN_SEEDS, means=summary))
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--updates", default=150, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    args = parser.parse_args()
    experiment(args.out, args.updates, args.batch_size)
