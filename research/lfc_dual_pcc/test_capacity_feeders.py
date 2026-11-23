"""Regression tests for the native evaluation units and intervention labels."""
import numpy as np
import pytest

from research.lfc_dual_pcc.capacity_feeders import feeder_config, heldout_replays
from research.lfc_dual_pcc.capacity_policy import CapacityPolicy
from research.lfc_dual_pcc.run_capacity_cases import batch, rollout, scenario


def test_two_device_energy_and_three_distinct_local_limits():
    cfg, sites = feeder_config([0., 14., 62.], [4., 6., 10.])
    storage = cfg["storage"]
    for site, energy, ramp in zip(sites, (0., 14., 62.), (4., 6., 10.), strict=True):
        soc = storage["initial_soc"][site]
        stored_kws = (soc - storage["reserve"]) * len(storage["devices"]) * storage["kwh"] * 3600
        assert stored_kws == pytest.approx(energy, abs=1e-9)
        assert cfg["ramp_up"][site] == ramp
    with pytest.raises(ValueError):
        feeder_config([0., -1.], [4., 10.])


@pytest.mark.parametrize("arm", ["joint_matched", "paired"])
def test_alternative_learning_arms_keep_their_fast_policy(arm):
    params = {"upper": [0.] * 5, "lower": [0., 0., -1.e3, 0., 0., 0.]}
    weights = {"2": {"11": {arm: params}}}
    [case] = heldout_replays([{"n_sites": 2, "arm": arm, "train_seed": 11}], weights, [8101])
    policy = CapacityPolicy(2, params)
    events = batch([scenario(np.random.default_rng(8101), 2)])
    joint = rollout(policy, events, "joint", trace=True)[-1]
    frozen = rollout(policy, events, "fixed_lower", trace=True)[-1]
    assert joint != frozen
    assert case["steps"] == joint


def test_external_load_pulse_and_feedback_are_independent():
    from research.lfc_dual_pcc.run_capacity_closed_loop import feedback, grid_input
    specs = {s: {"anchor_kw_kvar": [20., 20.]} for s in "ABC"}
    applied = {"A": [1.5, .2], "B": [2., .3]}
    assert grid_input(specs, 30., applied) == {"A": [48.5, 19.8], "B": [18., 19.7], "C": [20., 20.]}
    assert feedback(60., 59.9999) == pytest.approx(.5)
    assert feedback(60., 60.001) == 0.
    assert feedback(60., 59.999) == 1.
