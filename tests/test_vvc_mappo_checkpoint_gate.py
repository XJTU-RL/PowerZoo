"""Pure tests of the opt-in gate; no checkpoint deserialization or DSS."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from envs.powergrid.checkpoint import _validate_replay_algo


def make(algo="mappo"):
    opts = {
        "algo": {"share_param": False, "use_policy_active_masks": True},
        "model": {
            "hidden_sizes": [128, 128],
            "use_recurrent_policy": False,
            "use_naive_recurrent_policy": False,
        },
    }
    cfg = SimpleNamespace(main_args={"algo": algo}, algo_args=opts)
    return cfg, {"config_kind": "resolved", **deepcopy(opts)}


def test_happo_default_does_not_reread_snapshot():
    cfg, _ = make("happo")
    _validate_replay_algo(cfg, "vvc", None, False, lambda: pytest.fail())


def test_mappo_default_rejected_and_vvc_opt_in_passes():
    cfg, frozen = make()
    with pytest.raises(ValueError, match="HAPPO"):
        _validate_replay_algo(cfg, "vvc", "vvc", False, lambda: frozen)
    _validate_replay_algo(cfg, "vvc", "vvc", True, lambda: frozen)


@pytest.mark.parametrize("flag", [1, "true", None])
def test_malformed_opt_in(flag):
    cfg, frozen = make()
    with pytest.raises(ValueError, match="boolean"):
        _validate_replay_algo(cfg, "vvc", "vvc", flag, lambda: frozen)


@pytest.mark.parametrize(
    "family,expected", [("vvc", None), ("smartgrid", "vvc"), ("smartgrid", "smartgrid")]
)
def test_rejects_nonexplicit_vvc(family, expected):
    cfg, frozen = make()
    with pytest.raises(ValueError, match="explicit VVC"):
        _validate_replay_algo(cfg, family, expected, True, lambda: frozen)


@pytest.mark.parametrize("share", [True, "group", "false", None, 0])
def test_rejects_shared_or_nonboolean_saved_flag(share):
    cfg, frozen = make()
    frozen["algo"]["share_param"] = share
    with pytest.raises(ValueError, match="share_param"):
        _validate_replay_algo(cfg, "vvc", "vvc", True, lambda: frozen)


def test_rejects_configuration_default_filling_policy_flags():
    cfg, frozen = make()
    del frozen["model"]["use_recurrent_policy"]
    with pytest.raises(ValueError, match="policy model flags"):
        _validate_replay_algo(cfg, "vvc", "vvc", True, lambda: frozen)


def test_disallows_other_algo_under_opt_in():
    cfg, frozen = make("shom")
    with pytest.raises(ValueError, match="HAPPO"):
        _validate_replay_algo(cfg, "vvc", "vvc", True, lambda: frozen)
