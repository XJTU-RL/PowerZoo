"""真实数据契约和 DSS 行为验收；不以随机奖励有数值代替控制效果。"""

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import pytest

from envs.smartgrid.data_process.dataset import (
    DataConfig,
    ScenarioDataset,
    generate_dataset,
    prepare_dataset,
)

ROOT = Path(__file__).resolve().parents[1]


def test_generation_reproducible_and_splits_independent(tmp_path):
    cfg = DataConfig(seed=0, episodes_per_split=6)
    a = generate_dataset(tmp_path / "a", cfg, 360)
    b = generate_dataset(tmp_path / "b", cfg, 360)
    assert a.read_bytes() == b.read_bytes()
    manifests = json.loads(a.read_text())
    assert len(manifests["episodes"]) == 18
    hashes = [v["sha256"] for v in manifests["episodes"]]
    assert len(set(hashes)) == 18
    train = ScenarioDataset(a, "train", 360, 60)
    assert {e[0]["scenario"] for e in train.episodes} == {
        "sunny",
        "cloudy",
        "cloud_ramp",
        "evening_peak",
        "high_pv",
        "night",
    }
    assert np.all(train.episodes[5][1][:, 4] == 0)
    assert train.episodes[0][1][:, 4].max() > 0.5
    assert train.episodes[0][1][:, 5].max() > 30
    with pytest.raises(FileExistsError):
        generate_dataset(tmp_path / "a", cfg, 360)


@pytest.mark.parametrize(
    "damage",
    ["units", "csv", "time", "nan", "negative", "length", "duplicate", "digest"],
)
def test_dataset_rejects_damage(tmp_path, damage):
    path = generate_dataset(tmp_path / "data", DataConfig(episodes_per_split=6), 24)
    manifest = json.loads(path.read_text())
    entry = manifest["episodes"][0]
    file = path.parent / entry["file"]
    if damage == "units":
        manifest["units"]["irradiance"] = "W/m2"
    elif damage == "duplicate":
        manifest["episodes"].append(entry)
    elif damage == "length":
        entry["steps"] = 23
    elif damage == "digest":
        entry["sha256"] = "0" * 64
    else:
        lines = file.read_text().splitlines()
        row = lines[2].split(",")
        if damage == "csv":
            lines[0] = "bad_header"
        elif damage == "time":
            row[0] = str(float(row[0]) + 1)
        elif damage == "nan":
            row[4] = "nan"
        elif damage == "negative":
            row[4] = "-1"
        lines[2] = ",".join(row)
        file.write_text("\n".join(lines) + "\n")
        entry["sha256"] = sha256(file.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        ScenarioDataset(path, "train", 24, 60)


def test_manifest_pin_and_files_entry(tmp_path):
    path = generate_dataset(tmp_path / "data", DataConfig(episodes_per_split=6), 24)
    cfg = DataConfig(source="files", path=str(path))
    assert prepare_dataset(cfg, 24).path == path
    cfg.manifest_sha256 = "bad"
    with pytest.raises(ValueError, match="摘要"):
        prepare_dataset(cfg, 24)
    with pytest.raises(ValueError, match="参数"):
        prepare_dataset(DataConfig(path=str(path), seed=42), 24)


def test_full_day_and_sampling_interval(tmp_path):
    path = generate_dataset(
        tmp_path / "data", DataConfig(step_seconds=300, episodes_per_split=6), 288
    )
    data = ScenarioDataset(path, "test", 288, 300)
    values = data.episodes[0][1]
    assert values[0, 0] == 0
    assert values[-1, 0] == 86100
    assert values[0, 4] == 0 and values[144, 4] > 0


def make_core(extra=None):
    from utils.config_pipeline import ConfigLoader
    from utils.config_validation import build_dataclass
    from envs.smartgrid.base_env.env_config import SmartGridConfig
    from envs.smartgrid.base_env.runtime_env import Env

    fields = {"max_episode_steps": 24, "data": {"episodes_per_split": 6}}
    fields.update(extra or {})
    cfg = ConfigLoader().load(
        "happo",
        "smartgrid",
        values={"env": fields},
    )
    return Env(str(ROOT), build_dataclass(SmartGridConfig, cfg.env_args))


def neutral(core):
    a = [1] * core.cap_num
    a += [
        round((r.initial - r.mintap) / (r.maxtap - r.mintap) * (core.reg_act_num - 1))
        for r in core.circuit.regulators.values()
    ]
    a += [core.bat_act_num // 2] * core.bat_num
    a += [v for _ in core.pv_names for v in (1, 0)]
    return np.array(a, dtype=float)


def test_native_power_units_curtailment_q_load_and_clock():
    core = make_core()
    try:
        core.reset(load_profile_idx=0)
        a = neutral(core)
        _, _, _, full = core.step(a)
        assert 100 < full["pv_power_kw"] < 1200  # 原生 kW，不能再除1000
        assert (
            full["voltage_min_pu"] > 0.5
        )  # PV864 单相接线修复，不能过滤零节点掩盖错误
        assert full["time_s"] == pytest.approx(core.profile[0, 0] + 60)
        assert full["total_power_kw"] == pytest.approx(
            full["total_load_kw"]
            + full["power_loss_kw"]
            - full["pv_power_kw"]
            + full["battery_charge_kw"]
            - full["battery_discharge_kw"],
            rel=1e-5,
            abs=0.02,
        )
        for name, base in core.circuit.loads.items():
            c = core.circuit.native
            c.Loads.Name = name
            group = core.dataset.group_for(name, core.config.data.load_groups)
            assert c.Loads.kW == pytest.approx(base.kw * core.profile[0, 1 + group])
            assert c.Loads.kvar == pytest.approx(base.kvar * core.profile[0, 1 + group])
        core.reset(load_profile_idx=0)
        start = core.cap_num + core.reg_num + core.bat_num
        a[start::2] = -1
        _, _, _, off = core.step(a)
        assert abs(off["pv_power_kw"]) < full["pv_power_kw"] * 0.01
        core.reset(load_profile_idx=0)
        a = neutral(core)
        a[start + 1 :: 2] = 0.5
        core.step(a)
        q = sum(-core.circuit.powers("PVSystem." + n)[1] for n in core.pv_names)
        assert q > 10
        core.reset(load_profile_idx=0)
        a[start + 1 :: 2] = -0.5
        core.step(a)
        assert (
            sum(-core.circuit.powers("PVSystem." + n)[1] for n in core.pv_names) < -10
        )
    finally:
        core.close()


def test_storage_energy_and_natural_terminal():
    core = make_core()
    try:
        core.reset(load_profile_idx=0)
        a = neutral(core)
        start = core.cap_num + core.reg_num
        a[start : start + core.bat_num] = core.bat_act_num - 1
        _, _, _, info = core.step(a)
        assert info["battery_discharge_kw"] > 100
        for name, bat in core.circuit.batteries.items():
            power = -core.circuit.powers("Storage." + name)[0]
            assert bat.soc == pytest.approx(
                bat.initial_soc - power / 60 / bat.eta_discharge / bat.max_kwh
            )
        core.reset(load_profile_idx=0)
        a[start : start + core.bat_num] = 0
        core.step(a)
        assert all(b.soc > b.initial_soc for b in core.circuit.batteries.values())
        for _ in range(23):
            _, reward, done, info = core.step(neutral(core))
            assert np.isfinite(reward)
        assert done and info["terminated"] and not info["TimeLimit.truncated"]
        with pytest.raises(RuntimeError, match="结束"):
            core.step(neutral(core))
    finally:
        core.close()


def test_contexts_isolated_models_read_only_and_errors_propagate():
    import os

    folder = ROOT / "node_systems/34Bus_PV"
    before = {
        str(p): sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob("*")
        if p.is_file()
    }
    cwd = os.getcwd()
    first = make_core()
    second = None
    try:
        baseline = first._info()["total_load_kw"]
        second = make_core()
        second.step(neutral(second))
        assert first._info()["total_load_kw"] == baseline
        second.close()
        first.step(neutral(first))
        with pytest.raises(ValueError):
            first.step([float("nan")])
        with pytest.raises(ValueError, match="越界"):
            first.reset(load_profile_idx=1000)
        assert os.getcwd() == cwd
    finally:
        first.close()
        if second:
            second.close()
    assert before == {
        str(p): sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob("*")
        if p.is_file()
    }


def test_single_and_multi_share_same_physics():
    from envs.smartgrid.single_agent.adapter import SingleAgentVVCEnv
    from envs.smartgrid.base_env.marl_adapter import VVCEnv

    core = make_core()
    single = SingleAgentVVCEnv(core.config)
    multi = VVCEnv(core, core.config)
    try:
        core.reset(load_profile_idx=0)
        single.reset(options={"scenario_index": 0})
        a = neutral(core)
        action, offset = [], 0
        for space, size in zip(single.core.agent_spaces, single._sizes):
            if hasattr(space, "n"):
                action.append(2 * (a[offset] + 0.5) / space.n - 1)
            else:
                action.extend(a[offset : offset + size])
            offset += size
        per_agent = []
        offset = 0
        for size in single._sizes:
            per_agent.append(a[offset : offset + size])
            offset += size
        ma = multi.step(per_agent)
        sa = single.step(np.array(action))
        assert ma[0][0] == pytest.approx(sa[0], abs=1e-5)
        assert ma[2][0, 0] == pytest.approx(sa[1], abs=1e-6)
    finally:
        multi.close()
        single.close()


def test_reward_components_computed_once(monkeypatch):
    core = make_core()
    try:
        calls = []
        original = core.reward_func.control_reward

        def wrapped(*args):
            calls.append(1)
            return original(*args)

        monkeypatch.setattr(core.reward_func, "control_reward", wrapped)
        _, reward, _, info = core.step(neutral(core))
        assert len(calls) == 1
        assert reward == pytest.approx(
            info["reward_main"] - info.get("lambda", 0) * info["cost_voltage"]
        )
    finally:
        core.close()


@pytest.mark.parametrize(
    "plan,count",
    [(None, 9), ("none", 0), ("conservative", 5), ("optimized", 7), ("aggressive", 9)],
)
def test_pv_plans_have_real_devices_and_energized_phases(plan, count):
    core = make_core({"pv_plan": plan})
    try:
        assert core.pv_num == count
        assert core._info()["voltage_min_pu"] > 0.5
        core.step(neutral(core))
    finally:
        core.close()


def test_training_and_evaluation_log_actual_episode_metrics(tmp_path):
    import io
    from types import SimpleNamespace

    from envs.smartgrid.logging.smartgrid_logger import SmartGridLogger
    from envs.smartgrid.logging.visualization_manager import VisualizationManager

    log = SmartGridLogger.__new__(SmartGridLogger)
    log.algo_args = {
        "train": {"n_rollout_threads": 1, "episode_length": 99},
        "eval": {"n_eval_rollout_threads": 1},
    }
    log.viz_manager = None
    log.train_episode_cost_voltage = np.zeros(1)
    log.train_episode_lambda = np.zeros(1)
    log.done_episodes_cost_voltage, log.done_episodes_lambda = [], []
    log.init(1)
    log._init_eval_collectors()
    infos = None
    for i in range(2):
        infos = [
            [
                {
                    "power_loss_kw": 10 + i * 10,
                    "total_load_kw": 100,
                    "powerloss_reward": -2,
                    "control_reward": -0.3,
                    "pv_reward": 0.4,
                    "cost_voltage": 0.02,
                    "lambda": 1,
                    "voltage_violation_rate": i * 0.5,
                }
            ]
        ]
        rewards = np.array([[[-3.0]]])
        dones = np.array([[i == 1]])
        log.per_step(
            (None, None, rewards, dones, infos, None, None, None, None, None, None)
        )
        log.eval_per_step((None, None, rewards, dones, infos, None))
    log.eval_thread_done(0)
    assert log.done_episodes_power_loss_kw == [15]
    assert log.done_episodes_cost_voltage == pytest.approx([0.02])
    assert log.done_episodes_ctrl_reward == pytest.approx([-0.6])
    assert log.done_episodes_voltage_violation_rate == [0.25]
    assert log._calculate_power_loss_percentage() == 15
    np.testing.assert_allclose(log.eval_power_loss_kw_episode_rewards[0][0], [[15]])
    np.testing.assert_allclose(log.eval_ctrl_episode_rewards[0][0], [[-0.6]])
    recorded = {}
    log.writer = SimpleNamespace(
        add_scalar=lambda key, value, step: recorded.update({key: value})
    )
    log.episode, log.total_num_steps = 1, 2
    log.log_file, log.eval_env_reward_infos = io.StringIO(), {}
    log.eval_log(1)
    assert recorded["eval_power_loss_kw_average"] == 15
    log.train_episode_steps[0] = 3
    log._clear_done_episodes()
    assert log.train_episode_steps[0] == 3  # 日志间隔不能重置正在进行的回合

    viz = VisualizationManager(str(tmp_path), plot_interval=1000)
    viz.update({"reward_total": -3, "reward_main": -2})
    assert viz.data_buffers["total_reward"][-1] == -3
    viz.update_episode_end(np.float32(-6), np.float32(0.02), np.float32(1))
    viz._save_statistics()
    saved = json.loads((tmp_path / "statistics_ep1.json").read_text())
    assert saved["episode_stats"][0]["reward"] == -6


def test_regulator_actions_must_match_native_taps():
    with pytest.raises(ValueError, match="NumTaps"):
        make_core({"regulator": {"action_num": 65, "control_enabled": True}})
