"""Pure action and artifact gates; no solver or training."""
import json
from hashlib import sha256
from types import SimpleNamespace
import sys

import gymnasium as gym
import numpy as np
import pytest

from envs.powergrid.adapters import CentralizedAdapter
from utils.centralized_cli import load_model
from envs.vvc.vvc.vvc_config import VVCConfig


class Core:
    config = VVCConfig()
    observation_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
    agent_spaces = [gym.spaces.Discrete(n) for n in [2, 2, *([33] * 8)]]

    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        return np.zeros(2, dtype=np.float32), -3., False, {}

    def get_physics_manifest(self):
        return {"task": type(self.config).__name__, "action_schema_sha256": "ordered-devices"}

    def close(self):
        pass


@pytest.mark.parametrize("action", [[0]*9, [0]*11, [0]*9+[float("nan")],
                                    [0]*9+[1.5], [0]*9+[33], [-1]+[0]*9])
def test_discrete_invalid_action_never_reaches_core(action):
    core = Core()
    with pytest.raises(ValueError):
        CentralizedAdapter(core, "discrete").step(action)
    assert not core.actions


def test_discrete_is_device_product_and_continuous_default_stays_quantized():
    core = Core()
    discrete = CentralizedAdapter(core, "discrete")
    assert discrete.action_space.nvec.tolist() == [2, 2, *([33]*8)]
    action = [1, 0, 17, 15, 16, 16, 16, 16, 18, 14]
    discrete.step(np.array(action, dtype=np.int64))
    assert core.actions[-1] == action
    continuous = CentralizedAdapter(core)
    continuous.step([2*(a+.5)/s.n-1 for a,s in zip(action, core.agent_spaces)])
    assert core.actions[-1] == action


def test_discrete_rejects_mixed_device_spaces():
    core = Core()
    core.agent_spaces = [gym.spaces.Discrete(2), gym.spaces.Box(-1,1,(1,))]
    with pytest.raises(ValueError, match="discrete"):
        CentralizedAdapter(core, "discrete")


@pytest.fixture
def saved_model(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "stable_baselines3", SimpleNamespace(
        PPO=SimpleNamespace(load=lambda *a, **k: calls.append((a, k)) or "loaded")))
    env = CentralizedAdapter(Core(), "discrete")
    prefix = tmp_path / "ppo"
    prefix.with_suffix(".zip").write_bytes(b"stand-in archive; load is mocked")
    prefix.with_suffix(".environment.yaml").write_text(
        json.dumps({"config_kind":"resolved", "run":{"env":"vvc"}}))
    prefix.with_suffix(".physics.json").write_text(json.dumps(env.core.get_physics_manifest()))
    return prefix, env, calls


def write_binding(prefix, env):
    metadata = {"algorithm":"SB3 PPO", "action_space_type":"discrete",
                "action_nvec":env.action_space.nvec.tolist(),
                "files_sha256":{prefix.with_suffix(s).name:
                    sha256(prefix.with_suffix(s).read_bytes()).hexdigest()
                    for s in (".zip", ".environment.yaml", ".physics.json")}}
    prefix.with_suffix(".sb3.json").write_text(json.dumps(metadata))
    return metadata


@pytest.mark.parametrize("task", ["DSRConfig", "VVCConfig", "SmartGridConfig", "DistrictDispatchConfig"])
@pytest.mark.parametrize("optional_metadata", [False, True])
def test_legacy_discrete_restore_only_requires_valid_physics(saved_model, task, optional_metadata):
    prefix, env, calls = saved_model
    physics = {**env.core.get_physics_manifest(), "task":task}
    env.core.get_physics_manifest = lambda: physics
    prefix.with_suffix(".physics.json").write_text(json.dumps(physics))
    # Legacy callers must not start requiring or parsing an optional new manifest.
    if optional_metadata:
        prefix.with_suffix(".sb3.json").write_text("not a new-format manifest")
    assert load_model(prefix, env) == "loaded"
    assert calls[0][1] == {"env":env, "device":"cpu"}


@pytest.mark.parametrize("broken", ["missing", "mismatch"])
def test_legacy_restore_still_rejects_bad_physics(saved_model, broken):
    prefix, env, calls = saved_model
    path = prefix.with_suffix(".physics.json")
    if broken == "missing":
        path.unlink()
    else:
        path.write_text(json.dumps({"task":"other"}))
    with pytest.raises(ValueError):
        load_model(prefix, env)
    assert not calls


def test_strict_vvc_restore_accepts_complete_artifacts(saved_model):
    prefix, env, calls = saved_model
    write_binding(prefix, env)
    assert load_model(prefix, env, require_vvc_binding=True) == "loaded"
    assert len(calls) == 1


@pytest.mark.parametrize("hashes", [None, {}, [], {"ppo.physics.json":"0"*64},
                                    {"../ppo.zip":"0"*64}])
def test_strict_vvc_rejects_incomplete_or_unsafe_hash_map(saved_model, hashes):
    prefix, env, calls = saved_model
    metadata = write_binding(prefix, env)
    metadata["files_sha256"] = hashes
    prefix.with_suffix(".sb3.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="files_sha256"):
        load_model(prefix, env, require_vvc_binding=True)
    assert not calls


@pytest.mark.parametrize("suffix", [".zip", ".environment.yaml", ".physics.json"])
@pytest.mark.parametrize("fault", ["missing_hash", "missing_file", "replaced_file"])
def test_strict_vvc_binds_every_required_file(saved_model, suffix, fault):
    prefix, env, calls = saved_model
    metadata = write_binding(prefix, env)
    path = prefix.with_suffix(suffix)
    if fault == "missing_hash":
        del metadata["files_sha256"][path.name]
        prefix.with_suffix(".sb3.json").write_text(json.dumps(metadata))
    elif fault == "missing_file":
        path.unlink()
    else:
        path.write_bytes(b"replacement")
    with pytest.raises(ValueError):
        load_model(prefix, env, require_vvc_binding=True)
    assert not calls


@pytest.mark.parametrize("fault", ["missing_metadata", "missing_hashes", "extra_hash",
                                   "bad_digest", "wrong_mode", "wrong_nvec", "wrong_algorithm",
                                   "unresolved_config", "bad_physics"])
def test_strict_vvc_rejects_other_binding_faults(saved_model, fault):
    prefix, env, calls = saved_model
    metadata = write_binding(prefix, env)
    if fault == "missing_metadata":
        prefix.with_suffix(".sb3.json").unlink()
    else:
        if fault == "missing_hashes":
            del metadata["files_sha256"]
        elif fault == "extra_hash":
            metadata["files_sha256"]["unrelated.zip"] = "0"*64
        elif fault == "bad_digest":
            metadata["files_sha256"]["ppo.zip"] = "not-sha256"
        elif fault == "wrong_mode":
            metadata["action_space_type"] = "continuous"
        elif fault == "wrong_nvec":
            metadata["action_nvec"] = [2]
        elif fault == "wrong_algorithm":
            metadata["algorithm"] = "other"
        elif fault in ("unresolved_config", "bad_physics"):
            suffix = ".environment.yaml" if fault == "unresolved_config" else ".physics.json"
            path = prefix.with_suffix(suffix)
            path.write_text(json.dumps({"config_kind":"experiment", "task":"other"}))
            metadata["files_sha256"][path.name] = sha256(path.read_bytes()).hexdigest()
        prefix.with_suffix(".sb3.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_model(prefix, env, require_vvc_binding=True)
    assert not calls


@pytest.mark.parametrize("fault", ["DSRConfig", "SmartGridConfig", "vvc", "continuous", "non_boolean_flag"])
def test_strict_binding_is_an_explicit_vvc_discrete_contract(saved_model, fault):
    prefix, env, calls = saved_model
    write_binding(prefix, env)
    if fault in ("DSRConfig", "SmartGridConfig", "vvc"):
        env.core.get_physics_manifest = lambda: {"task":fault}
    elif fault == "continuous":
        env.mode = "continuous"
    with pytest.raises(ValueError):
        load_model(prefix, env, require_vvc_binding="yes" if fault == "non_boolean_flag" else True)
    assert not calls
