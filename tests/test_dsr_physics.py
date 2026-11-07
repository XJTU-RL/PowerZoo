"""真实故障恢复：原生供电、相别、容量、动作掩码与可穷举上限。"""

from itertools import product
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from utils.config_pipeline import ConfigLoader
from utils.config_validation import build_dataclass, environment_class
from envs.dsr.core.dsr_core import DSRCoreEnv
from envs.powergrid.topology import energized_phases, operating_graph

ROOT = Path(__file__).resolve().parents[1]


def configuration(family="dsr", **changes):
    cfg = ConfigLoader().load("happo", family, values={"env": changes})
    return build_dataclass(environment_class(family), cfg.env_args)


@pytest.fixture
def core():
    env = DSRCoreEnv(configuration())
    yield env
    env.close()


def test_transformer_source_and_real_tie_restore_actual_delivery(core):
    info = core._info()
    assert ("root", 1) in energized_phases(core.circuit)
    assert ("a", 1) not in energized_phases(core.circuit)
    assert info["delivered_kw"] == pytest.approx(50, abs=0.001)
    assert info["restored_ratio"] == pytest.approx(1 / 3)
    assert core.available_actions()[0][2] == 0 and core.available_actions()[0][6] == 1
    _, _, _, info = core.step([6, 0, 0])
    assert info["delivered_kw"] == pytest.approx(150, abs=0.001)
    assert info["restored_ratio"] == pytest.approx(1.0)
    assert info["time_s"] == 300 and info["operations"] == 1
    assert abs(info["active_balance_error_kw"]) < 0.05
    assert not core.circuit.connection("Line.feed_a")["enabled"]


def test_permanent_fault_cannot_be_reclosed_and_invalid_frame_is_atomic(core):
    before = core.positions()
    _, _, _, info = core.step([2, 1, 0])
    assert info["invalid_actions"] == 1 and info["operations"] == 0
    assert core.positions() == before
    assert core.circuit.time_s == 300
    with pytest.raises(ValueError, match="不可用"):
        core.circuit.apply({"Line.feed_a": {"closed": 1}})


def test_zero_time_trial_restores_positions_and_clock(core):
    before = core.positions()
    trial = core.evaluate_positions({"Line.tie": True})
    assert trial["restored_ratio"] == pytest.approx(1.0)
    assert core.positions() == before and core.circuit.time_s == 0
    assert core._info()["restored_ratio"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("episode,upper", [(0, 1.0), (1, 1.0), (2, 0.0), (3, 0.0)])
def test_exhaustive_small_network_upper_bound(core, episode, upper):
    core.reset(load_profile_idx=episode)
    best = 0.0
    for values in product((False, True), repeat=len(core.controls)):
        trial = core.evaluate_positions(dict(zip(core.controls, values)))
        if trial is not None:
            best = max(best, trial["restored_ratio"])
    assert best == pytest.approx(upper, abs=1e-6)
    assert core.circuit.time_s == 0


def test_radial_mask_rejects_real_loop(tmp_path):
    cfg = configuration()
    data = json.loads((ROOT / cfg.data.path).read_text())
    data["episodes"][0]["unavailable"] = []
    path = tmp_path / "events.json"
    path.write_text(json.dumps(data))
    cfg.data.path = str(path)
    env = DSRCoreEnv(cfg)
    try:
        assert env.available_actions()[0][6] == 0
        assert not operating_graph(env.circuit, {"Line.tie": True}).graph["radial"]
        _, _, _, info = env.step([6, 0, 0])
        assert info["invalid_actions"] == 1 and not env.positions()["Line.tie"]
    finally:
        env.close()


def test_enabled_is_not_delivery_and_unserved_energy_uses_seconds(core):
    assert core.circuit.connection("Load.critical")["enabled"]
    assert core.load_metrics["Load.critical"]["delivered_kw"] == 0
    _, _, _, info = core.step([0, 0, 0])
    assert info["unserved_kwh"] == pytest.approx(100 * 300 / 3600)
    assert info["weighted_service_unserved_kwh"] == pytest.approx(3 * 100 * 300 / 3600)


def test_missing_phase_is_not_complete_service(core):
    core.circuit.command("Open Line.feed_b term=1 cond=2")
    core._settle()
    meter = core.load_metrics["Load.ordinary"]
    assert ("b", 2) not in core.energized
    assert not meter["source_connected"]
    assert meter["qualified_served_kw"] == 0
    assert meter["delivered_kw"] > 0


def test_overloaded_supply_path_is_not_qualified_service(core):
    core.circuit.command("Edit Line.feed_b normamps=1")
    core._settle()
    meter = core.load_metrics["Load.ordinary"]
    assert meter["delivered_kw"] > 49
    assert meter["source_connected"] and not meter["capacity_ok"]
    assert meter["qualified_served_kw"] == 0


def test_grid_following_pv_is_not_blackstart_source(tmp_path):
    model = tmp_path / "model"
    shutil.copytree(ROOT / "node_systems/RestorationSmall", model)
    with (model / "network.dss").open("a") as file:
        file.write(
            "\nNew PVSystem.pv bus1=a phases=3 kv=4.16 kva=50 pmpp=50 irradiance=1 %cutin=0 %cutout=0\n"
        )
    env = DSRCoreEnv(configuration(dss_folder=str(model)))
    try:
        env.reset(load_profile_idx=2)
        assert not env.energized
        assert not env.circuit.connection("PVSystem.pv")["enabled"]
        assert env._info()["delivered_kw"] < 1e-6
        assert env._info()["restored_ratio"] == 0
        env.step([0, 0, 0])
        assert env._info()["restored_ratio"] == 0
    finally:
        env.close()


def test_ieee13_actual_alternate_path_and_unrecoverable_transformer():
    env = DSRCoreEnv(configuration("dsr_13bus"))
    try:
        initial = env._info()["restored_ratio"]
        _, _, _, info = env.step([4, 0, 0])
        assert initial < 0.3 and info["restored_ratio"] > 0.999
        assert abs(info["active_balance_error_kw"]) < 0.05
        assert ("634", 1) in env.energized
        env.reset(load_profile_idx=1)
        assert not env.load_metrics["Load.634a"]["source_connected"]
        assert not env.circuit.connection("Transformer.xfm1")["enabled"]
        assert env._info()["restored_ratio"] < 0.9
    finally:
        env.close()


def test_individual_masks_do_not_silently_accept_a_joint_cycle(tmp_path):
    model = tmp_path / "model"
    shutil.copytree(ROOT / "node_systems/RestorationSmall", model)
    with (model / "network.dss").open("a") as file:
        file.write("\nOpen Line.feed_a term=1\n")
    cfg = configuration(
        dss_folder=str(model),
        control_groups=[
            {"name": "feeder", "controls": ["Line.feed_a"]},
            {"name": "tie", "controls": ["Line.tie"]},
        ],
    )
    data = json.loads((ROOT / cfg.data.path).read_text())
    data["episodes"][0]["unavailable"] = []
    path = tmp_path / "events.json"
    path.write_text(json.dumps(data))
    cfg.data.path = str(path)
    env = DSRCoreEnv(cfg)
    try:
        assert env.available_actions()[0][2] == env.available_actions()[1][2] == 1
        before = env.positions()
        _, _, _, info = env.step([2, 2])
        assert info["invalid_actions"] == 2 and info["operations"] == 0
        assert env.positions() == before
    finally:
        env.close()


def test_real_discrete_inference_masks_have_batch_axis(core, tmp_path):
    import torch
    from envs.render_common.engine.inference_engine import (
        InferenceEngine,
        _DEFAULT_POLICY_ARGS,
    )
    from models.policy_models.stochastic_policy import StochasticPolicy

    args = {**_DEFAULT_POLICY_ARGS, "hidden_sizes": [8, 8]}
    for i, space in enumerate(core.agent_spaces):
        policy = StochasticPolicy(
            args, core.observation_space, space, torch.device("cpu")
        )
        torch.save(policy.state_dict(), tmp_path / f"actor_agent{i}.pt")
    engine = InferenceEngine(
        str(tmp_path), 3, [core.observation_space.shape[0]] * 3, [1] * 3, hidden_size=8
    )
    engine.load_actors(args, action_spaces=core.agent_spaces)
    masks = [np.zeros(s.n, dtype=np.float32) for s in core.agent_spaces]
    masks[0][6] = 1
    masks[1][0] = masks[2][0] = 1
    frame = engine.infer(
        [core.get_obs()] * 3, deterministic=True, available_actions=masks
    )
    assert frame[:, 0].tolist() == [6, 0, 0]
    assert core.step(frame.reshape(-1))[3]["restored_ratio"] == pytest.approx(1.0)


def test_switch_requests_validate_whole_frame_before_native_mutation(core):
    original = core.circuit.read_property("Load.ordinary", "kW")
    with pytest.raises(ValueError):
        core.circuit.apply({"Load.ordinary": {"kw": 1.0}, "Line.feed_a": {"closed": 1}})
    assert core.circuit.read_property("Load.ordinary", "kW") == original
    assert core.circuit.time_s == 0


def test_fault_manifest_is_frozen_and_unsupported_8500_is_not_advertised(tmp_path):
    from envs.dsr.core.scenario import RestorationDataset
    from training_ui.core.registry import ENV_REGISTRY

    cfg = configuration()
    dataset = RestorationDataset(cfg.data)
    cfg.data.manifest_sha256 = "0" * 64
    with pytest.raises(ValueError, match="摘要"):
        RestorationDataset(cfg.data)
    assert dataset.episodes and "dsr_8500node" not in ENV_REGISTRY
    with pytest.raises(FileNotFoundError):
        ConfigLoader().load("happo", "dsr_8500node")


def test_disabled_load_node_refs_do_not_change_observation_shape_or_phases():
    from envs.powergrid.snapshot import assemble_snapshot

    env = DSRCoreEnv(configuration("dsr_13bus"))
    try:
        shape = env.get_obs().shape
        schema = env.get_physics_manifest()["observation_schema_sha256"]
        env.reset(load_profile_idx=1)
        env.step([0, 0, 1])
        assert env.get_obs().shape == shape
        assert env.circuit.connection("Load.634a")["nodes"][0] == 1
        assert env._fields()["634.1.in_solution"] == 0
        assert env.get_physics_manifest()["observation_schema_sha256"] == schema
        env.circuit.apply({"Load.634b": {"enabled": 0}, "Load.634c": {"enabled": 0}})
        env._settle()
        assert env.physical_state.voltages_pu["634"] == {}
        assert env.get_obs().shape == shape
        snapshot = assemble_snapshot(env, 1, None, None, env._info())
        assert snapshot["buses"]["634"]["active_in_solution"] is False
        assert snapshot["buses"]["634"]["v_mag_pu"] == []
    finally:
        env.close()


def test_completely_inactive_network_has_no_fabricated_voltage(tmp_path):
    from envs.dsr.dsr_logger import DSRLogger
    from envs.powergrid.snapshot import assemble_snapshot

    model = tmp_path / "model"
    model.mkdir()
    (model / "network.dss").write_text(
        "New Circuit.test basekv=12.47 pu=1 bus1=s phases=3\nNew Load.a bus1=s phases=3 kv=12.47 kw=100 kvar=10\nSet VoltageBases=[12.47]\n"
    )
    data = json.loads((ROOT / "data/restoration/small/manifest.json").read_text())
    data["episodes"] = [
        {
            "id": "all_off",
            "split": "train",
            "unavailable": ["Vsource.source"],
            "load_multiplier": 1.0,
        }
    ]
    path = tmp_path / "events.json"
    path.write_text(json.dumps(data))
    cfg = configuration(
        dss_folder=str(model),
        data={"path": str(path)},
        control_groups=[{"name": "load", "controls": ["Load.a"]}],
    )
    cfg.priorities = {}
    env = DSRCoreEnv(cfg)
    try:
        shape = env.get_obs().shape
        _, reward, _, info = env.step([1])
        assert np.isfinite(reward) and env.get_obs().shape == shape
        assert info["inactive_phase_count"] == 3 and info["delivered_kw"] == 0
        assert info["voltage_min_pu"] is None and info["voltage_max_pu"] is None
        snapshot = assemble_snapshot(env, 1, None, None, info)
        assert snapshot["circuit"]["v_min_pu"] is None
        record = DSRLogger._empty()
        DSRLogger._accumulate(record, [info])
        assert record["voltage_min_pu"] is None
        json.dumps(record, allow_nan=False)
    finally:
        env.close()


def test_centralized_discrete_and_cooperative_frames_have_same_physics(core):
    from envs.powergrid.adapters import CentralizedAdapter

    other = DSRCoreEnv(configuration())
    central = CentralizedAdapter(other, action_space_type="discrete")
    try:
        core.reset(load_profile_idx=0)
        central.reset(options={"scenario_index": 0})
        one = core.step([6, 0, 0])
        two = central.step(np.array([6, 0, 0]))
        assert np.allclose(one[0], two[0])
        assert one[1] == pytest.approx(two[1])
        assert one[3]["delivered_kw"] == pytest.approx(two[4]["delivered_kw"])
    finally:
        central.close()


def test_legacy_restoration_sources_are_recoverable():
    from hashlib import sha256

    manifest = json.loads(
        (ROOT / "archive/physics_refactor_dsr/manifest.json").read_text()
    )
    assert len(manifest["files"]) == 55
    for entry in manifest["files"]:
        assert (
            sha256((ROOT / entry["archive"]).read_bytes()).hexdigest()
            == entry["sha256"]
        )
        if entry["disposition"] == "retired":
            assert not (ROOT / entry["original"]).exists()


def test_reference_switch_transfer_opens_before_closing(tmp_path, monkeypatch):
    cfg = configuration()
    data = json.loads((ROOT / cfg.data.path).read_text())
    data["episodes"][0]["unavailable"] = []
    path = tmp_path / "events.json"
    path.write_text(json.dumps(data))
    cfg.data.path = str(path)
    env = DSRCoreEnv(cfg)
    try:
        command = env.circuit.command

        def checked(text):
            command(text)
            assert operating_graph(env.circuit).graph["radial"]

        monkeypatch.setattr(env.circuit, "command", checked)
        # 调用方故意先列闭合，再列断开；真实下发必须先开后合。
        trial = env.evaluate_positions({"Line.tie": True, "Line.feed_a": False})
        assert trial["restored_ratio"] == pytest.approx(1.0)
    finally:
        env.close()
