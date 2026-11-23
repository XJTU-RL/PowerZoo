import numpy as np
import pytest

from .capacity_policy import CapacityPolicy
from .run_capacity_cases import N_WINDOWS, batch, scenario
from .train_capacity_counterfactual import paired_gradients, paired_tapes


@pytest.mark.parametrize("n", [2, 3])
def test_only_one_selected_draw_changes_per_event(n):
    size = 64
    u, v, tick, site = paired_tapes(np.random.default_rng(14), n, size)
    changed = np.zeros((size, 3 * N_WINDOWS), int)
    for k, (left, right) in enumerate(zip(u, v, strict=True)):
        diff = (left != right).reshape(size, -1)
        changed[:, k] = diff.sum(axis=1)
        selected = np.flatnonzero(k == tick)
        if diff.shape[1] > 1:
            assert diff[selected, site[selected]].all()
    assert (changed.sum(axis=1) == 1).all()
    assert changed[np.arange(size), tick].all()
    assert (tick[::2] % 3 == 0).all()
    assert (tick[1::2] % 3 != 0).all()


def test_individual_score_and_balanced_levels_required():
    rng = np.random.default_rng(12)
    p = CapacityPolicy(3)
    with pytest.raises(ValueError, match="individual"):
        paired_gradients(p, batch(scenario(rng, 3) for _ in range(2)), rng)
    p.per_site_credit = True
    with pytest.raises(ValueError, match="even"):
        paired_gradients(p, batch(scenario(rng, 3) for _ in range(3)), rng)
