"""VVC共享物理、经典目标、原生档位及单/多智能体的实际验收。"""

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pytest

from envs.powergrid.config import DataConfig
from envs.vvc.runtime_env import Env
from envs.vvc.vvc.vvc_config import VVCConfig
from utils.config_pipeline import ConfigLoader
from utils.config_validation import build_dataclass

ROOT = Path(__file__).resolve().parents[1]


def make_core(system="13Bus", **changes):
    cfg = ConfigLoader().load(
        "happo", "vvc", system_ref=system, values={"env": changes}
    )
    return Env(str(ROOT), build_dataclass(VVCConfig, cfg.env_args))


def hold(env):
    return (
        [int(env.circuit.capacitor_state(n)[0]) for n in env.cap_names]
        + [
            round(
                (r.initial - r.mintap) / (r.maxtap - r.mintap) * (env.reg_act_num - 1)
            )
            for r in env.circuit.regulators.values()
        ]
        + [env.bat_act_num // 2] * env.bat_num
    )


@pytest.mark.parametrize(
    "system,counts", [("13Bus", (2, 3, 1)), ("34Bus", (2, 6, 2)), ("123Bus", (4, 7, 4))]
)
def test_three_cases_actual_devices_and_full_native_day(system, counts):
    env = make_core(system)
    try:
        assert (env.cap_num, env.reg_num, env.bat_num) == counts
        env.reset(load_profile_idx=0)
        action = hold(env)
        for _ in range(24):
            _, reward, done, info = env.step(action)
            assert np.isfinite(reward) and info["dss_converged"]
            assert abs(info["active_balance_error_kw"]) < 0.05
        assert done and info["time_s"] == 86400
        assert all(b.soc == b.initial_soc for b in env.circuit.batteries.values())
    finally:
        env.close()


def test_regulator_does_not_rewrite_numtaps_or_source_winding():
    env = make_core()
    try:
        c = env.circuit
        reg = c.regulators[env.reg_names[0]]
        c.native.Transformers.Name = reg.transformer
        c.native.Transformers.Wdg = 1
        source_tap = float(c.native.Transformers.Tap)
        action = hold(env)
        action[env.cap_num] = 0
        env.step(action)
        assert c.regulator_tap(env.reg_names[0]) == pytest.approx(reg.mintap)
        c.native.Transformers.Wdg = 1
        assert c.native.Transformers.Tap == source_tap
        c.native.Transformers.Wdg = reg.winding
        assert c.native.Transformers.NumTaps == 32
    finally:
        env.close()


def test_classic_reward_uses_actual_tap_steps_and_terminal_energy():
    env = make_core(
        max_episode_steps=2,
        data={"step_seconds": 60},
        power_w=0,
        cap_w=2,
        reg_w=3,
        dis_w=0,
        soc_w=5,
    )
    try:
        action = hold(env)
        action[0] = 1 - action[0]
        action[env.cap_num] += 1
        _, reward, done, info = env.step(action)
        assert info["ctrl_reward"] == pytest.approx(-5)
        assert reward == pytest.approx(info["vol_reward"] - 5)
        assert not done
        action[env.cap_num + env.reg_num] = 32
        _, _, done, info = env.step(action)
        deviation = sum(
            abs(b.soc - b.initial_soc) for b in env.circuit.batteries.values()
        )
        assert done and info["ctrl_reward"] == pytest.approx(-5 * deviation)
    finally:
        env.close()


def test_vvc_and_smartgrid_share_transition_without_task_inheritance(tmp_path):
    from envs.smartgrid.base_env.env_config import SmartGridConfig
    from envs.smartgrid.base_env.runtime_env import Env as SmartGrid
    from utils.physics_manifest import validate_checkpoint

    data = DataConfig(step_seconds=60, noise=0)
    vvc = Env(
        str(ROOT),
        VVCConfig(
            system_name="34Bus_PV",
            dss_folder="node_systems/34Bus_PV",
            data=data,
            overlays=["pv_plans/aggressive.dss"],
        ),
    )
    smart = SmartGrid(
        str(ROOT),
        SmartGridConfig(
            max_episode_steps=24,
            data=data,
            pv_plan="aggressive",
            pv_control_enabled=False,
            use_lagrangian_reward=False,
        ),
    )
    try:
        assert not isinstance(vvc, SmartGrid)
        vvc.reset(load_profile_idx=0)
        smart.reset(load_profile_idx=0)
        action = hold(vvc)
        action[-1] = 32
        for _ in range(3):
            vvc.step(action)
            smart.step(action)
            assert vvc.physical_state.grid_import == pytest.approx(
                smart.physical_state.grid_import, abs=1e-7
            )
            assert vvc.physical_state.storage_soc == pytest.approx(
                smart.physical_state.storage_soc, abs=1e-9
            )
            assert vvc.physical_state.losses == pytest.approx(
                smart.physical_state.losses, abs=1e-7
            )
        a, b = vvc.get_physics_manifest(), smart.get_physics_manifest()
        assert a["action_schema_sha256"] == b["action_schema_sha256"]
        (tmp_path / "physics.manifest.json").write_text(json.dumps(a))
        with pytest.raises(ValueError, match="task"):
            validate_checkpoint(tmp_path, b)
    finally:
        vvc.close()
        smart.close()


def test_vvc_centralized_and_multiagent_have_same_physics_and_team_reward():
    from envs.powergrid.adapters import CentralizedAdapter, DeviceMARLAdapter

    a, b = make_core(), make_core()
    single, multi = CentralizedAdapter(a), DeviceMARLAdapter(b)
    try:
        single.reset(options={"scenario_index": 0})
        b.reset(load_profile_idx=0)
        action = hold(a)
        action[-1] = 32
        encoded = np.asarray(
            [2 * (x + 0.5) / s.n - 1 for x, s in zip(action, a.agent_spaces)],
            dtype=np.float32,
        )
        sa = single.step(encoded)
        ma = multi.step([[v] for v in action])
        np.testing.assert_allclose(sa[0], ma[0][0], atol=1e-6)
        assert sa[1] == pytest.approx(ma[2][0, 0])
        assert np.all(ma[2] == ma[2][0])
        assert a.circuit.time_s == 3600 and b.circuit.time_s == 3600
    finally:
        single.close()
        multi.close()


def test_vvc_config_rejects_retired_unused_knobs_and_oversized_day():
    from utils.config_validation import ConfigError

    for values in (
        {"scale": 2},
        {"useS": True},
        {"load_noise": True},
        {"node_size": 500},
        {"dss_act": False},
        {"max_episode_steps": 25},
    ):
        with pytest.raises((ConfigError, ValueError)):
            VVCConfig.from_env_args(values)
    assert asdict(VVCConfig())["data"]["step_seconds"] == 3600


def test_episode_physical_metrics_integrate_dt_and_count_team_once():
    from envs.vvc.vvc_logger import VVCLogger

    env = make_core()
    try:
        _, _, _, info = env.step(hold(env))
        record = VVCLogger._empty()
        VVCLogger._accumulate(record, [info] * 6)
        VVCLogger._accumulate(record, np.asarray([info] * 6, dtype=object))
        assert record["steps"] == 2
        assert record["loss_kwh"] == pytest.approx(2 * info["power_loss_kw"])
        assert record["reward"] == pytest.approx(2 * info["reward_total"])
    finally:
        env.close()


def test_shared_snapshots_and_vvc_active_power_chart_preserve_units():
    from envs.vvc.render.engine.episode_runner import VVCEpisodeRunner
    from envs.vvc.render.viz.plotly.power_flow_diagram import create_power_flow_diagram
    from envs.vvc.render.viz.plotly.device_schedule_chart import create_device_schedule

    env = make_core()
    runner = VVCEpisodeRunner(env.config, project_root=str(ROOT))
    try:
        action = hold(env)
        action[-1] = 32
        data = runner.run_episode(
            max_steps=2,
            action_overrides={1: np.asarray([action]), 2: np.asarray([action])},
        )
        snapshot = data.snapshots[1]
        fig = create_power_flow_diagram(snapshot)
        values = list(fig.data[0].x)
        assert sum(values) == pytest.approx(0, abs=0.05)
        assert values[0] == pytest.approx(snapshot["info"]["total_power_kw"])
        assert "kvar" not in fig.to_json().lower()
        schedule = create_device_schedule(data.snapshots)
        assert any(
            any(abs(float(v)) > 1 for v in trace.y)
            for trace in schedule.data
            if str(trace.name).startswith("Power")
        )
        assert data.total_reward == pytest.approx(
            sum(s["info"]["reward_total"] for s in data.snapshots[1:])
        )
    finally:
        runner.close()
        env.close()


def test_vvc_continuous_storage_uses_real_native_power_and_soc():
    from envs.powergrid.adapters import DeviceMARLAdapter

    env = make_core(bat_act_num=float("inf"))
    try:
        wrapper = DeviceMARLAdapter(env)
        action = hold(env)[: -env.bat_num] + [0.3] * env.bat_num
        wrapper.step([[v] for v in action])
        battery = next(iter(env.circuit.batteries.values()))
        assert battery.soc == pytest.approx(0.5 - 30 / 0.95 / 200, abs=1e-7)
        assert env._info()["battery_discharge_kw"] == pytest.approx(30, rel=1e-6)
    finally:
        env.close()


def test_vvc_stress_changes_actual_generated_load_and_no_unused_pv_ratio():
    from envs.vvc.render.engine.stress_test_runner import VVCStressTestRunner

    env = make_core(max_episode_steps=2, data={"step_seconds": 60})
    runner = VVCStressTestRunner(env.config)
    try:
        config = runner._apply_params_to_config(
            env.config, {"load_mult": 1.2, "noise": 0.0}
        )
        changed = Env(str(ROOT), config)
        try:
            assert config.data.load_scale == 1.2 and config.data.noise == 0
            assert changed.current_row[1] > env.current_row[1]
        finally:
            changed.close()
        with pytest.raises(ValueError, match="不支持"):
            runner._apply_params_to_config(env.config, {"pv_ratio": 1.5})
    finally:
        env.close()


def test_vvc_retired_files_have_recoverable_byte_exact_archive():
    import hashlib

    manifest = json.loads(
        (ROOT / "archive/physics_refactor_vvc/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for record in manifest["files"]:
        assert (
            hashlib.sha256((ROOT / record["archive"]).read_bytes()).hexdigest()
            == record["sha256"]
        )
        if not record.get("replacement"):
            assert not (ROOT / record["original"]).exists()
