"""Train and evaluate the two- and multi-feeder capacity mechanisms with matched seeds."""
import argparse
import json
from pathlib import Path

import numpy as np

from .capacity_policy import CapacityPolicy
from .case_a_capacity_constants import ACTIONS, DELIVERY_KW

N_WINDOWS = 6
TRAIN_SEEDS = (11, 23, 37)
HOLDOUT = tuple(range(8100, 8132))
ARMS = ("rule", "fixed_lower", "joint")
LOWER_TRACKING_COST = .35
METRICS = ("system_mae_kw", "local_mae_kw", "unmet_wh", "excess_wh",
           "uncommitted_mean_kw", "battery_used_wh", "episode_return")


def scenario(rng, n_sites):
    """Six two-second reserve products; the actual activation arrives each second."""
    if n_sites == 2:
        energy = np.array([14., 62.])
        ramp = np.array([10., 4.])
    elif n_sites >= 3:
        repeats, extras = divmod(n_sites, 3)
        types = np.tile(np.arange(3), repeats)
        if extras:
            # The unpaired types vary by event, so a four-site case does not
            # systematically duplicate the same flexibility profile.
            types = np.concatenate([types, rng.choice(3, extras, replace=False)])
        energy = np.array([14., 62., 30.])[types]
        ramp = np.array([10., 4., 6.])[types]
    else:
        raise ValueError("capacity allocation requires at least two sites")
    order = rng.permutation(n_sites)
    energy = (energy * rng.uniform(.65, 1.35, n_sites))[order]
    ramp = ramp[order]
    capacity = n_sites * rng.uniform(3.1, 5.2, N_WINDOWS)
    activation = rng.uniform(.30, .75, (N_WINDOWS, 2))
    activation[rng.integers(0, 2)] = rng.uniform(.80, 1., 2)
    activation[rng.integers(3, 6)] = rng.uniform(.85, 1., 2)
    return energy, ramp, capacity, activation


def batch(events):
    return tuple(np.stack(items) for items in zip(*events, strict=True))


def admissible(energy, ramp, previous_nominal):
    kw = ACTIONS.sum(axis=1)
    return ((ACTIONS[:, 0] / .95 <= energy[..., None] + 1.e-10) &
            (kw - previous_nominal[..., None] <= ramp[..., None] + 1.e-10) &
            (previous_nominal[..., None] - kw <= 10. + 1.e-10))


def rollout(policy, events, arm, *, rng=None, trace=False, first_choice=None):
    initial_energy, ramp, requests, activations = events
    energy = initial_energy.copy()
    previous_nominal = np.zeros_like(energy)
    previous = np.zeros_like(energy)
    report = np.zeros((len(energy), len(METRICS)))
    history, credits, steps = [], [], []
    for k in range(N_WINDOWS):
        valid = admissible(energy, ramp, previous_nominal)
        limits = np.where(valid, DELIVERY_KW, 0.).max(axis=-1)
        capacity, slow_score = policy.slow(requests[:, k], energy, ramp, previous, limits,
                                           rule=arm == "rule", rng=rng, force=first_choice if k == 0 else None)
        if slow_score is not None:
            credits.append(("upper", len(history), slow_score))
        report[:, 4] += requests[:, k] - capacity.sum(axis=1)
        for alpha in activations[:, k, :].T:
            valid = admissible(energy, ramp, previous_nominal)
            target = alpha[:, None] * capacity
            action, fast_score = policy.fast(target, capacity, energy, ramp, previous, valid,
                                             DELIVERY_KW, rule=arm != "joint", rng=rng)
            if fast_score is not None:
                credits.append(("lower", len(history), fast_score))
            if not np.take_along_axis(valid, action[..., None], axis=-1).all():
                raise ArithmeticError("infeasible dispatch")
            delivered = DELIVERY_KW[action]
            battery = ACTIONS[action, 0]
            remaining = energy - battery / .95
            if remaining.min() < -1.e-9:
                raise ArithmeticError("violated storage reserve")
            # The external signal remains binding when the initial clearing is scarce.
            error = requests[:, k] * alpha - delivered.sum(axis=1)
            local = np.abs(target - delivered).sum(axis=1)
            reward = -np.abs(error) - LOWER_TRACKING_COST * local
            history.append(reward)
            report[:, 0] += np.abs(error)
            report[:, 1] += local / policy.n_sites
            report[:, 2] += np.maximum(error, 0.) / 3.6
            report[:, 3] += np.maximum(-error, 0.) / 3.6
            report[:, 5] += battery.sum(axis=1) / 3.6
            report[:, 6] += reward
            if trace:
                steps.append(dict(action=action[0].tolist(), capacity_kw=capacity[0].tolist(),
                                  target_kw=target[0].tolist(), delivered_kw=delivered[0].tolist(),
                                  request_kw=float(requests[0, k]), activation=float(alpha[0])))
            energy, previous, previous_nominal = remaining, delivered, ACTIONS[action].sum(axis=-1)
    report[:, :2] /= 2 * N_WINDOWS
    report[:, 4] /= N_WINDOWS
    return report, np.array(history), credits, steps


def adam_ascent(policy, level, gradient, moments, second, update):
    moments[level] = .9 * moments[level] + .1 * gradient
    second[level] = .999 * second[level] + .001 * gradient**2
    m = moments[level] / (1. - .9**update)
    v = second[level] / (1. - .999**update)
    policy.weights[level] += .035 * m / (np.sqrt(v) + 1.e-8)


def train(n_sites, seed, arm, updates, batch_size):
    policy = CapacityPolicy(n_sites)
    moments = {key: np.zeros_like(v) for key, v in policy.weights.items()}
    second = {key: np.zeros_like(v) for key, v in policy.weights.items()}
    env_rng = np.random.default_rng(seed)
    action_rng = np.random.default_rng(np.random.SeedSequence([seed, 97]))
    curve = []
    for update in range(1, updates + 1):
        events = batch(scenario(env_rng, n_sites) for _ in range(batch_size))
        _, rewards, credits, _ = rollout(policy, events, arm, rng=action_rng)
        returns = rewards[::-1].cumsum(axis=0)[::-1]
        gradients = {key: np.zeros_like(v) for key, v in policy.weights.items()}
        for level, t, score in credits:
            # Leave-one-episode-out baseline is independent of that episode's actions.
            advantage = (returns[t] - returns[t].mean()) * batch_size / (batch_size - 1)
            gradients[level] += (score * advantage[:, None]).mean(axis=0)
        for level, gradient in gradients.items():
            steps = N_WINDOWS if level == "upper" else N_WINDOWS * 2
            gradient /= steps
            adam_ascent(policy, level, gradient, moments, second, update)
        if update == 1 or update % 25 == 0 or update == updates:
            curve.append(dict(update=update, stochastic_return=float(returns[0].mean())))
    return policy, curve


def write(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def experiment(out, updates, batch_size):
    if updates < 1 or batch_size < 2:
        raise ValueError("positive updates and batch of at least two episodes required")
    out.mkdir(parents=True, exist_ok=False)
    weights, profiles, curves, records, replays = {}, {}, {}, [], []
    for n in (2, 3):
        events = batch(scenario(np.random.default_rng(seed), n) for seed in HOLDOUT)
        weights[n], profiles[n], curves[n] = {}, {}, {}
        all_metrics = {arm: [] for arm in ARMS}
        for seed in TRAIN_SEEDS:
            weights[n][seed], curves[n][seed] = {}, {}
            for arm in ARMS:
                if arm == "rule":
                    policy, curve = CapacityPolicy(n), []
                else:
                    policy, curve = train(n, seed, arm, updates, batch_size)
                checkpoint = {key: v.tolist() for key, v in policy.weights.items()}
                weights[n][seed][arm], curves[n][seed][arm] = checkpoint, curve
                restored = CapacityPolicy(n, checkpoint)
                metrics, _, _, steps = rollout(restored, events, arm, trace=seed == TRAIN_SEEDS[0])
                all_metrics[arm].append(metrics)
                for event, values in zip(HOLDOUT, metrics, strict=True):
                    records.append(dict(n_sites=n, train_seed=seed, event_seed=event, arm=arm,
                                        **dict(zip(METRICS, values.tolist(), strict=True))))
                if steps:
                    replays.append(dict(n_sites=n, arm=arm, train_seed=seed, event_seed=HOLDOUT[0],
                                        energy_kws=events[0][0].tolist(), ramp_up_kw_s=events[1][0].tolist(),
                                        steps=steps))
        for arm, trials in all_metrics.items():
            means = np.mean(trials, axis=1)
            profiles[n][arm] = dict(mean=dict(zip(METRICS, means.mean(axis=0).tolist(), strict=True)),
                                    by_train_seed={s: dict(zip(METRICS, m.tolist(), strict=True))
                                                   for s, m in zip(TRAIN_SEEDS, means, strict=True)})
        print(n, {arm: round(profiles[n][arm]["mean"]["system_mae_kw"], 3) for arm in ARMS},
              flush=True)
    write(out / "summary.json", dict(updates=updates, batch_size=batch_size, train_seeds=TRAIN_SEEDS,
                                     holdout=HOLDOUT, steps_per_episode=N_WINDOWS * 2, metrics=profiles))
    write(out / "weights.json", weights)
    write(out / "episodes.json", records)
    write(out / "replay.json", dict(episodes=replays))
    write(out / "learning_curves.json", curves)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    experiment(args.out, args.updates, args.batch_size)
