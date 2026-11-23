"""Zero-shot transfer of the three-site capacity policies to larger populations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .capacity_policy import CapacityPolicy
from .run_capacity_cases import ARMS, METRICS, TRAIN_SEEDS, batch, rollout, scenario, write

SCALE_EVENTS = tuple(range(10200, 10232))


def evaluate(source, out):
    checkpoint_bytes = source.read_bytes()
    trained = json.loads(checkpoint_bytes)["3"]
    result = dict(trained_sites=3, holdout=SCALE_EVENTS, train_seeds=TRAIN_SEEDS,
                  weights_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(), means={}, rows=[])
    for n in (4, 6):
        events = batch(scenario(np.random.default_rng(s), n) for s in SCALE_EVENTS)
        # Permuting stations must preserve the same physical outcome up to relabeling.
        order = np.random.default_rng(7300 + n).permutation(n)
        permuted = (events[0][:, order], events[1][:, order], *events[2:])
        means = {}
        for arm in ARMS:
            all_seeds = []
            for seed in TRAIN_SEEDS:
                policy = CapacityPolicy(n, trained[str(seed)][arm])
                metrics, _, _, _ = rollout(policy, events, arm)
                shuffled, _, _, _ = rollout(policy, permuted, arm)
                np.testing.assert_allclose(shuffled, metrics, rtol=1.e-11, atol=1.e-9)
                system = metrics[:, METRICS.index("system_mae_kw")]
                all_seeds.extend(system.tolist())
                for event, value in zip(SCALE_EVENTS, system, strict=True):
                    result["rows"].append(dict(n_sites=n, train_seed=seed, event_seed=event,
                                               arm=arm, system_mae_kw=float(value),
                                               normalized_system_mae_kw=float(value/n)))
            mean = float(np.mean(all_seeds))
            means[arm] = dict(system_mae_kw=mean, normalized_system_mae_kw=mean/n)
        result["means"][n] = means
    write(out, result)
    return result["means"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(evaluate(args.weights, args.out), flush=True)
