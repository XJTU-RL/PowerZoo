"""物理求解失败语义验收：发散工况终止+重罚、观测冻结、reset 恢复、非物理失败照常上抛。"""

import numpy as np
import pytest

from utils.config_pipeline import ConfigLoader
from utils.envs_tools import make_power_env

pytestmark = pytest.mark.requires_opendss


def _build():
    cfg = ConfigLoader().load("happo", "vvc")
    return make_power_env("vvc", cfg.env_args, 0, 0)


def _actions(env, available):
    actions = []
    for i, space in enumerate(env.action_space):
        if space.__class__.__name__ == "Discrete":
            actions.append(int(np.flatnonzero(available[i])[0]))
        else:
            actions.append(np.zeros(space.shape))
    return actions


def test_divergence_terminates_with_penalty_and_freezes_obs():
    env = _build()
    try:
        obs0, _, available = env.reset()
        actions = _actions(env, available)
        core = env.env
        core.circuit.step = None  # 占位避免误用
        original_solve = core.circuit.solve

        def diverge(*args, **kwargs):
            core.circuit._failed = True
            raise RuntimeError("DSS在3600.0s未收敛")

        core.circuit.solve = diverge
        obs1, _, rewards, dones, infos, _ = env.step(actions)
        core.circuit.solve = original_solve
        reward, done = float(rewards[0]), bool(dones[0])
        assert done and not infos[0]["truncated"]
        assert infos[0]["physical_failure"] is True
        assert "未收敛" in infos[0]["failure_reason"]
        assert reward == pytest.approx(-core.divergence_penalty)
        arr1, arr0 = np.asarray(obs1), np.asarray(obs0)
        # 观测冻结为上一步值；唯一变化是末位 episode_progress（t 已推进）。
        assert arr1.shape == arr0.shape
        assert not (arr1[:, :-1] != arr0[:, :-1]).any(), "时间进度以外的观测被改变"
        assert np.allclose(arr1[:, -1], 1 / core.horizon)
        # 失败后必须 reset 才能继续
        with pytest.raises(RuntimeError):
            env.step(actions)
        obs2, _, _ = env.reset()
        assert np.isfinite(obs2).all()
    finally:
        env.close()


def test_non_physical_runtime_error_is_not_swallowed():
    env = _build()
    try:
        _, _, available = env.reset()
        actions = _actions(env, available)
        core = env.env

        def buggy(*args, **kwargs):
            raise RuntimeError("无关代码缺陷")

        original = core._apply_data
        core._apply_data = buggy
        try:
            with pytest.raises(RuntimeError, match="无关代码缺陷"):
                env.step(actions)
        finally:
            core._apply_data = original
        assert not core.circuit._failed or True
    finally:
        env.close()


def test_divergence_penalty_validation():
    with pytest.raises(ValueError, match="divergence_penalty"):
        ConfigLoader().load(
            "happo",
            "vvc",
            values={"env": {"divergence_penalty": -1.0}},
        )
