"""SmartGrid任务与公共物理层、集中/多智能体动作及模型重放的联合验收。"""

import json
from pathlib import Path

import numpy as np
import pytest

from utils.config_pipeline import ConfigLoader
from utils.config_validation import build_dataclass, environment_class
from utils.physics_manifest import validate_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def core(values=None):
    from envs.smartgrid.base_env.runtime_env import Env

    cfg = ConfigLoader().load(
        "happo",
        "smartgrid",
        values={
            "env": {
                "max_episode_steps": 24,
                "data": {"episodes_per_split": 6},
                **(values or {}),
            }
        },
    )
    return Env(str(ROOT), build_dataclass(environment_class("smartgrid"), cfg.env_args))


def hold(env):
    return (
        [1] * env.cap_num
        + [
            round(
                (r.initial - r.mintap) / (r.maxtap - r.mintap) * (env.reg_act_num - 1)
            )
            for r in env.circuit.regulators.values()
        ]
        + [env.bat_act_num // 2] * env.bat_num
        + ([x for _ in env.pv_names for x in (1, 0)] if env.pv_control_enabled else [])
        + [0] * len(env.svg_names)
    )


def test_half_available_power_preserves_native_nameplate():
    env = core()
    try:
        env.reset(load_profile_idx=0)
        action = hold(env)
        start = env.cap_num + env.reg_num + env.bat_num
        action[start::2] = [0] * env.pv_num  # Box 0对应可用AC功率的一半
        _, _, _, info = env.step(action)
        for name, dev in env.circuit.pvs.items():
            trace = info["control_trace"]["PVSystem." + name]
            assert trace["requested"]["kw"] == pytest.approx(
                env.available_pv_kw[name] * 0.5
            )
            assert (
                float(env.circuit.read_property("PVSystem." + name, "Pmpp")) == dev.pmpp
            )
            assert trace["measured"]["kw"] <= trace["applied"]["kw"] * 1.001
        assert abs(info["active_balance_error_kw"]) < 0.05
    finally:
        env.close()


@pytest.mark.parametrize("value", [-1, 1])
def test_svg_task_uses_actual_q_and_matches_single_agent(value):
    from envs.smartgrid.single_agent.adapter import SingleAgentVVCEnv
    from envs.smartgrid.base_env.marl_adapter import VVCEnv

    env = core(
        {"dss_file": "network_with_svg.dss", "svg_elements": ["Generator.svg834"]}
    )
    single = SingleAgentVVCEnv(env.config)
    multi = VVCEnv(env)
    try:
        env.reset(load_profile_idx=0)
        single.reset(options={"scenario_index": 0})
        action = hold(env)
        action[-1] = value
        offset, per_agent, encoded = 0, [], []
        for space, size in zip(env.agent_spaces, single._sizes):
            piece = action[offset : offset + size]
            per_agent.append(piece)
            encoded.extend(
                [2 * (piece[0] + 0.5) / space.n - 1] if hasattr(space, "n") else piece
            )
            offset += size
        ma = multi.step(per_agent)
        sa = single.step(np.array(encoded))
        assert ma[0][0] == pytest.approx(sa[0], abs=1e-5)
        assert ma[2][0, 0] == pytest.approx(sa[1], abs=1e-6)
        info = ma[4][0]
        assert info["svg_reactive_power_kvar"] * value > 0
        assert abs(info["active_balance_error_kw"]) < 0.05
        assert len(env.agent_spaces) == 20 and len(sa[0]) == 155
    finally:
        multi.close()
        single.close()


def test_manifest_rejects_changed_model_engine_or_action_binding(tmp_path):
    env = core()
    try:
        manifest = env.get_physics_manifest()
        model_dir = tmp_path / "models" / "checkpoint_episode_10"
        model_dir.mkdir(parents=True)
        with pytest.raises(ValueError, match="缺少"):
            validate_checkpoint(model_dir, manifest)
        path = tmp_path / "physics.manifest.json"
        path.write_text(json.dumps(manifest))
        validate_checkpoint(model_dir, manifest)
        for key in (
            "model_sha256",
            "engine",
            "action_schema_sha256",
            "master",
            "overlays",
            "step_seconds",
        ):
            changed = dict(manifest, **{key: "different"})
            with pytest.raises(ValueError, match=key):
                validate_checkpoint(model_dir, changed)
    finally:
        env.close()


def test_full_day_is_native_86400_seconds_and_idle_soc_is_resettable():
    env = core({"max_episode_steps": 1440})
    try:
        env.reset(load_profile_idx=0)
        action = hold(env)
        for _ in range(1440):
            _, reward, done, info = env.step(action)
            assert np.isfinite(reward) and info["dss_converged"]
        assert done and info["time_s"] == pytest.approx(86400, abs=1e-5)
        assert all(b.soc == b.initial_soc for b in env.circuit.batteries.values())
        env.reset(load_profile_idx=0)
        assert env.circuit.time_s == 0
    finally:
        env.close()


def test_smoothing_uses_executed_settings_including_storage_and_svg():
    env = core(
        {"dss_file": "network_with_svg.dss", "svg_elements": ["Generator.svg834"]}
    )
    try:
        action = hold(env)
        env.step(action)
        previous = env.applied_control_vector.copy()
        action[-1] = 1
        action[env.cap_num + env.reg_num] = 32
        env.step(action)
        current = env.applied_control_vector
        assert current != previous
        assert env.reward_func.previous_actions == current
        assert current[-1] > 0 and current[env.cap_num + env.reg_num] > 0
    finally:
        env.close()


def test_snapshot_reports_native_devices_branches_and_integrated_energy():
    from envs.smartgrid import Env
    from envs.smartgrid.base_env.runtime_env import Env as RuntimeEnv
    from envs.smartgrid.render.data.snapshot_assembler import assemble_snapshot
    from envs.smartgrid.render.data.circuit_data_extractor import (
        compute_episode_circuit_summary,
    )

    assert Env is RuntimeEnv
    env = core(
        {"dss_file": "network_with_svg.dss", "svg_elements": ["Generator.svg834"]}
    )
    try:
        action = hold(env)
        action[-1] = -1
        _, reward, _, info = env.step(action)
        snapshot = assemble_snapshot(env, 1, infos=info)
        json.dumps(snapshot, allow_nan=False)
        devices = snapshot["devices"]
        assert {k: len(v) for k, v in devices.items()} == {
            "capacitors": 2,
            "regulators": 6,
            "batteries": 2,
            "pvs": 9,
            "svgs": 1,
        }
        assert devices["svgs"]["svg834"]["reactive_power_kvar"] == pytest.approx(
            info["svg_reactive_power_kvar"]
        )
        assert snapshot["lines"]["l1"]["length_km"] == pytest.approx(2.58 * 0.3048)
        assert snapshot["lines"]["l1"]["loading_pct"] >= 0
        assert snapshot["circuit"]["grid_import_kw"] == pytest.approx(
            info["total_power_kw"]
        )
        for bus in snapshot["buses"].values():
            assert set(bus["phase_nodes"]) <= {1, 2, 3}
            assert len(bus["phase_nodes"]) == len(bus["v_mag_pu"])
        summary = compute_episode_circuit_summary([snapshot, snapshot])
        assert summary["total_loss_kwh"] == pytest.approx(
            info["power_loss_kw"] * 2 / 60
        )
    finally:
        env.close()


def test_render_runner_keeps_continuous_actions_and_counts_team_reward_once():
    from envs.smartgrid.render.engine.episode_runner import SmartGridEpisodeRunner

    env = core()
    runner = SmartGridEpisodeRunner(env.config, project_root=str(ROOT))
    try:
        # PV连续设定不可在渲染路径被整帧取整。
        action = hold(env)
        start = env.cap_num + env.reg_num + env.bat_num
        action[start:] = [x for _ in env.pv_names for x in (0.37, -0.13)]
        data = runner.run_episode(
            seed=1,
            max_steps=2,
            action_overrides={1: np.asarray([action]), 2: np.asarray([action])},
        )
        trace = data.snapshots[1]["info"]["control_trace"]
        assert runner._env.circuit.pvs["pv834"].fraction == pytest.approx(0.685)
        assert trace["PVSystem.pv834"]["requested"]["kvar"] < 0
        assert data.total_reward == pytest.approx(
            sum(s["info"]["reward_total"] for s in data.snapshots[1:])
        )
        assert data.episode_length == 2
    finally:
        runner.close()
        env.close()


def test_stress_scan_changes_real_data_fields_and_rejects_unused_parameters():
    from envs.smartgrid.render.engine.stress_test_runner import (
        SmartGridStressTestRunner,
    )

    env = core()
    try:
        runner = SmartGridStressTestRunner(env.config)
        cfg = runner._apply_params_to_config(
            env.config, {"load_mult": 1.2, "irradiance_scale": 0.5, "lambda_init": 2}
        )
        assert (cfg.data.load_scale, cfg.data.irradiance_scale, cfg.lambda_init) == (
            1.2,
            0.5,
            2,
        )
        assert env.config.data.load_scale == 1
        with pytest.raises(ValueError, match="不支持"):
            runner._apply_params_to_config(env.config, {"pv_ratio": 1.5})
    finally:
        env.close()


def test_manual_override_uses_bound_device_spaces():
    from envs.smartgrid.render.engine.manual_override import SmartGridManualOverride
    from envs.smartgrid.render.engine.episode_runner import SmartGridEpisodeRunner

    env = core(
        {"dss_file": "network_with_svg.dss", "svg_elements": ["Generator.svg834"]}
    )
    runner = SmartGridEpisodeRunner(env.config, project_root=str(ROOT))
    try:
        runner._init_env(seed=1)
        override = SmartGridManualOverride.from_env(runner._env)
        override.set_agent_override(19, [-0.37])
        assert override.has_overrides()
        assert "Generator.svg834" in override.get_action_labels(19)[0]
        actions = override.apply(runner._random_actions())
        assert actions[19, 0] == pytest.approx(-0.37)
        with pytest.raises(ValueError, match="超出"):
            override.set_agent_override(0, [5])
        runner._env_step(actions)
        assert runner._env.circuit.trace["Generator.svg834"]["requested"][
            "kvar"
        ] == pytest.approx(-37, abs=1e-4)
    finally:
        runner.close()
        env.close()


@pytest.mark.parametrize("sign", [-1, 1])
def test_night_pv_reactive_capability_is_an_explicit_native_assumption(sign):
    env = core()
    try:
        env.reset(load_profile_idx=5)
        action = hold(env)
        start = env.cap_num + env.reg_num + env.bat_num
        action[start + 1 :: 2] = [sign * 0.5] * env.pv_num
        _, _, _, info = env.step(action)
        assert info["irradiance_kw_m2"] == 0
        assert abs(info["pv_power_kw"]) < 1e-5
        assert env.circuit.read_property("PVSystem.pv834", "VarFollowInverter") == "No"
        assert (
            sum(-env.circuit.powers("PVSystem." + n)[1] for n in env.pv_names) * sign
            > 500
        )
    finally:
        env.close()
