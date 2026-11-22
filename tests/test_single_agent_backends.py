"""集中控制轻量回归：使用内存环境及合成剖面，不启动物理仿真。"""

import json
from dataclasses import dataclass
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from envs.cbess.profile_store import ProfileStore
from utils.centralized_cli import SingleAgentAdapter, load_model
from utils.config_pipeline import ConfigLoader, UnifiedConfig
from utils.single_agent.backends import (
    collect_ldcppo_rollout,
    estimate_ldcpo_budgets,
    validate_backend,
)
from utils.single_agent.binding import (
    binding_path,
    fingerprint,
    make_binding,
    validate_binding,
    write_binding,
)
from utils.single_agent.costs import COST_KEYS, extract_costs


@pytest.mark.parametrize(
    "family, backend",
    [("dsr", "ldcppo"), ("dsr", "sb3_sac"), ("dsr", "sb3_td3"), ("vvc", "ldcppo")],
)
def test_reject_incompatible_family_before_building_environment(family, backend):
    with pytest.raises(ValueError):
        validate_backend(backend, family)


@pytest.mark.parametrize("backend", ["sb3_sac", "sb3_td3", "ldcppo"])
def test_quantized_box_only_supported_by_ppo(backend):
    env = SimpleNamespace(
        action_space=gym.spaces.Box(-1.0, 1.0, (1,)),
        core=SimpleNamespace(agent_spaces=[gym.spaces.Discrete(2)]),
    )
    validate_backend("sb3_ppo", "vvc", env)
    with pytest.raises(ValueError, match="量化"):
        validate_backend(backend, "cbess", env)


@pytest.mark.parametrize("backend", ["sb3_sac", "sb3_td3"])
def test_off_policy_config_loads_for_cbess(backend):
    cfg = ConfigLoader().load(algo=backend, env="cbess")
    assert cfg.algo_args["algo"]["learning_starts"] == 100
    assert cfg.env_args["split"] == "train"


@pytest.mark.parametrize(
    "info",
    [
        {},
        {"cost_voltage": 1},
        {"costs": {"VD": 1, "VN": 2}},
        {"costs": {"VD": 1, "VN": 2, "LO": float("nan")}},
    ],
)
def test_missing_or_invalid_cost_rejected(info):
    with pytest.raises(ValueError, match="成本|通道"):
        extract_costs(info)


class RolloutEnv:
    action_space = gym.spaces.Box(-1.0, 1.0, (1,))
    zero_control_action = np.array([-1.0], dtype=np.float32)

    def __init__(self, boundary, costs=None):
        self.boundary = boundary
        self.costs = dict.fromkeys(COST_KEYS, 2.0) if costs is None else costs

    def reset(self):
        self.steps = 0
        return np.array([0.0]), {}

    def step(self, action):
        assert self.steps == 0 or self.boundary == "collector"
        assert self.action_space.contains(action)
        self.applied_action = action.copy()
        self.steps += 1
        return (
            np.array([self.steps]),
            3.0,
            self.boundary == "terminal",
            self.boundary == "time_limit",
            {"costs": self.costs},
        )


@pytest.mark.parametrize(
    "boundary, horizon, bootstrap, expected",
    [
        ("terminal", 5, True, 0.0),
        ("time_limit", 5, True, 1.0),
        ("collector", 2, True, 2.0),
        ("time_limit", 5, False, 0.0),
    ],
)
def test_rollout_bootstraps_only_nonterminal_boundaries(
    boundary, horizon, bootstrap, expected
):
    env = RolloutEnv(boundary)
    agent = SimpleNamespace(
        cfg=SimpleNamespace(bootstrap_at_episode_end=bootstrap),
        act=lambda obs: (np.array([2.0], dtype=np.float32), -0.5),
        values=lambda obs: dict.fromkeys(("R", *COST_KEYS), float(obs[0])),
    )
    states, actions, logps, rewards, costs, last_values = collect_ldcppo_rollout(
        env, agent, horizon
    )
    count = 2 if boundary == "collector" else 1
    assert len(states) == len(rewards) == len(costs["VD"]) == count
    assert actions.tolist() == [[2.0]] * count
    assert logps.tolist() == [-0.5] * count
    assert env.applied_action.tolist() == [1.0]
    assert last_values == dict.fromkeys(("R", *COST_KEYS), expected)


@pytest.mark.parametrize("boundary", ["terminal", "time_limit"])
def test_budget_stops_at_first_boundary_and_rejects_missing_costs(boundary):
    cfg = SimpleNamespace(gamma=0.9, cost_budget_fraction=dict.fromkeys(COST_KEYS, 0.5))
    assert estimate_ldcpo_budgets(RolloutEnv(boundary), cfg, 2, 5) == dict.fromkeys(
        COST_KEYS, 1.0
    )
    with pytest.raises(ValueError, match="约束通道"):
        estimate_ldcpo_budgets(RolloutEnv(boundary, {"VD": 0}), cfg, 2, 5)


@pytest.mark.parametrize(
    "info, expected",
    [
        ({"TimeLimit.truncated": True}, (False, True)),
        ({"terminated": True, "truncated": False}, (True, False)),
        ({"terminated": True, "truncated": True}, (True, True)),
        ({}, (True, False)),
    ],
)
def test_adapter_preserves_core_boundaries(info, expected):
    space = gym.spaces.Box(-1.0, 1.0, (1,))
    core = SimpleNamespace(
        agent_spaces=[space],
        observation_space=space,
        step=lambda _: (np.array([0.0]), 1.0, True, info),
    )
    assert SingleAgentAdapter(core).step(np.zeros(1))[2:4] == expected


def test_mixed_profile_manifest_rejected_before_opening_archive(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"split": {"a": "train", "b": "test"}})
    )

    def forbid_archive(*args, **kwargs):
        pytest.fail("混合剖面包不应打开")

    monkeypatch.setattr(np, "load", forbid_archive)
    with pytest.raises(ValueError, match="只包含 train"):
        ProfileStore(tmp_path, split="train")


@dataclass
class Scenario:
    split: str = "train"
    seed: int = 1


def bound_env(split="train", scenario_hash="train_sha", dimension=2):
    return SimpleNamespace(
        config=Scenario(split),
        split=split,
        scenario_sha256=scenario_hash,
        observation_space=gym.spaces.Box(-np.inf, np.inf, (dimension,)),
        action_space=gym.spaces.Box(-1.0, 1.0, (1,)),
    )


def binding_for(env, backend="ldcppo", episodes=2, gamma=0.99, device="CBESS.6"):
    cfg = UnifiedConfig(
        main_args={"env": "cbess"},
        env_args={"split": env.split},
        algo_args={"train": {"total_episodes": episodes}, "algo": {"gamma": gamma}},
    )
    return make_binding(backend, cfg, env, {"action_schema": [device]})


def test_binding_validates_resume_and_explicit_validation(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    training = binding_for(bound_env())
    write_binding(checkpoint, training)
    validate_binding(checkpoint, training, resume=True)
    extended = binding_for(bound_env(), episodes=3)
    validate_binding(checkpoint, extended, resume=True)
    with pytest.raises(ValueError, match="eval_config"):
        validate_binding(checkpoint, extended)

    valid = binding_for(bound_env("validation", "validation_sha"))
    with pytest.raises(ValueError, match="eval-scenario"):
        validate_binding(checkpoint, valid)
    validate_binding(checkpoint, valid, eval_scenario_sha256="validation_sha")
    with pytest.raises(ValueError, match="仅用于评估"):
        validate_binding(checkpoint, valid, resume=True)

    for binding, mismatch in [
        (binding_for(bound_env(), backend="sb3_ppo"), "backend"),
        (binding_for(bound_env(dimension=3)), "observation"),
        (binding_for(bound_env(), device="CBESS.11"), "physics"),
        (binding_for(bound_env(), gamma=0.9), "resume_config"),
        (binding_for(bound_env(scenario_hash="changed")), "场景"),
    ]:
        with pytest.raises(ValueError, match=mismatch):
            validate_binding(checkpoint, binding, resume=True)
    checkpoint.write_bytes(b"wrong weights")
    with pytest.raises(ValueError, match="文件摘要"):
        validate_binding(checkpoint, training)
    binding_path(checkpoint).unlink()
    with pytest.raises(ValueError, match="缺少绑定"):
        validate_binding(checkpoint, training)


def test_new_ppo_model_cannot_fall_back_when_binding_missing(tmp_path):
    model = tmp_path / "ppo"
    model.with_suffix(".physics.json").write_text('{"single_agent_binding_version": 1}')
    with pytest.raises(ValueError, match="缺少 checkpoint 绑定"):
        load_model(model, bound_env())


def test_ldcppo_rejects_multidiscrete_even_for_cbess():
    env = SimpleNamespace(action_space=gym.spaces.MultiDiscrete([2, 3]))
    with pytest.raises(ValueError, match="离散"):
        validate_backend("ldcppo", "cbess", env)


def test_ldcppo_sample_logp_matches_policy_before_command_clipping():
    import torch

    from utils.single_agent.backends import make_ldcppo

    env = RolloutEnv("terminal")
    env.action_space = gym.spaces.Box(-1e-8, 1e-8, (1,))
    env.observation_space = gym.spaces.Box(-1.0, 1.0, (1,))
    cfg = ConfigLoader().load(
        experiment="configs/experiments/cbess_ldcppo_quickstart.yaml"
    )
    torch.manual_seed(11)
    policy = make_ldcppo(cfg, env)
    states, actions, logps, rewards, costs, _ = collect_ldcppo_rollout(env, policy, 4)
    with torch.no_grad():
        distribution = policy.actor.distribution(torch.as_tensor(states))
        expected = distribution.log_prob(torch.as_tensor(actions)).sum(-1).numpy()
    np.testing.assert_allclose(logps, expected)
    assert np.abs(actions[0, 0]) > env.action_space.high[0]
    np.testing.assert_array_equal(
        env.applied_action, np.clip(actions[0], env.action_space.low, env.action_space.high)
    )
    assert rewards.tolist() == [3.0]
    assert all(values.tolist() == [2.0] for values in costs.values())

def test_legacy_ppo_accepts_a_resolved_config(tmp_path, monkeypatch):
    from utils.single_agent import backends

    model = tmp_path / "old_ppo"
    manifest = {"action_schema_sha256": "old"}
    model.with_suffix(".physics.json").write_text(json.dumps(manifest))
    env = SimpleNamespace(get_physics_manifest=lambda: manifest)
    cfg = ConfigLoader().load(experiment="configs/experiments/vvc_13bus_smoke.yaml")

    def load(checkpoint, *, env, device):
        assert checkpoint == model.with_suffix(".zip") and device == "cpu"
        return env

    monkeypatch.setattr(backends, "sb3_class", lambda _: SimpleNamespace(load=load))
    assert load_model(model, env, cfg=cfg) is env


def test_binding_hash_accepts_continuous_action_marker():
    continuous = {"env": {"bat_act_num": float("inf")}}
    discrete = {"env": {"bat_act_num": 33}}
    assert fingerprint(continuous) != fingerprint(discrete)
    with pytest.raises(ValueError):
        fingerprint({"env": {"bat_act_num": float("nan")}})
