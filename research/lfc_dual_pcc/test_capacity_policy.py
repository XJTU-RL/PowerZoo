import numpy as np
import pytest

from research.lfc_dual_pcc.capacity_policy import CapacityPolicy, clear_capacity
from research.lfc_dual_pcc.case_a_capacity_constants import ACTIONS, DELIVERY_KW
from research.lfc_dual_pcc.run_capacity_cases import admissible, batch, rollout, scenario


def test_clearing_with_saturation_and_shortage():
    limits = np.array([[1., 2., 8.], [1., 4., 0.]])
    weights = np.array([[8., 3., 1.], [9., 1., 2.]])
    actual = clear_capacity(np.array([6., 9.]), limits, weights)
    np.testing.assert_allclose(actual, [[1., 2., 3.], [1., 4., 0.]])
    order = [1, 2, 0]
    shuffled = clear_capacity(np.array([6., 9.]), limits[:, order], weights[:, order])
    np.testing.assert_allclose(shuffled, actual[:, order])


def test_feasible_sets_use_storage_efficiency_and_nominal_ramp():
    energy = np.array([[4., 4. / .95]])
    valid = admissible(energy, np.array([[10., 4.]]), np.zeros_like(energy))
    assert valid[0, 0].tolist() == [True, True, False, False, False, False]
    assert valid[0, 1].tolist() == [True, True, True, False, False, False]


@pytest.mark.parametrize(("n", "low", "high"), [(2, 0, 4), (3, [0, 1, 1], [2, 1, 1])])
def test_first_choice_changes_feasible_allocation_and_recomputes_execution(n, low, high):
    policy = CapacityPolicy(n)
    events = batch([scenario(np.random.default_rng(9123), n)])
    snapshots = [x.copy() for x in events]
    traces = [rollout(policy, events, "joint", trace=True, first_choice=force)[3]
              for force in (low, high)]
    first, second = (trace[0] for trace in traces)
    assert first["capacity_kw"][0] < second["capacity_kw"][0]
    assert sum(first["capacity_kw"]) == pytest.approx(sum(second["capacity_kw"]))
    assert any(a["action"] != b["action"] for a, b in zip(*traces, strict=True))
    for trace in traces:
        energy = events[0].copy()
        previous = np.zeros_like(energy)
        for step in trace:
            action = np.array(step["action"])
            assert admissible(energy, events[1], previous)[0, np.arange(n), action].all()
            np.testing.assert_allclose(DELIVERY_KW[action], step["delivered_kw"])
            energy -= ACTIONS[action, 0] / .95
            previous = ACTIONS[action].sum(axis=-1)[None, :]
    for old, now in zip(snapshots, events, strict=True):
        np.testing.assert_array_equal(old, now)


@pytest.mark.parametrize(("demand", "limits", "weights"), [
    ([-1.], [[1.]], [[1.]]), ([2.], [[-.1]], [[1.]]), ([np.nan], [[1.]], [[1.]]),
    ([2.], [[1.]], [[0.]]), ([2.], [[1.]], [[np.inf]]), ([1., 2.], [[1.]], [[1.]])])
def test_clearing_rejects_invalid_contract_inputs(demand, limits, weights):
    with pytest.raises(ValueError):
        clear_capacity(demand, limits, weights)


def test_float_clearing_at_saturation_and_zero_demand():
    limits = np.array([[1.e-13, .1, .2, .3], [1.e-13, .1, .2, .3]])
    weights = np.array([[10., 3., .4, 7.], [10., 3., .4, 7.]])
    out = clear_capacity(np.array([0., limits[1].sum()]), limits, weights)
    np.testing.assert_array_equal(out[0], 0.)
    np.testing.assert_allclose(out[1], limits[1], atol=1.e-16, rtol=1.e-13)


def test_sampling_never_chooses_leading_or_trailing_masked_actions():
    from research.lfc_dual_pcc.capacity_policy import _choice

    class EdgeSampler:
        def random(self, shape):
            return np.array([0., np.nextafter(1., 0.)]).reshape(shape)

    mask = np.tile([False, True, True, True, False], (2, 1))
    features = np.tile(np.arange(5.)[:, None], (2, 1, 1))
    choices, score = _choice(features, np.zeros(1), np.zeros((2, 5)), mask, EdgeSampler())
    np.testing.assert_array_equal(choices, [1, 3])
    np.testing.assert_allclose(score[:, 0], [-1., 1.])
