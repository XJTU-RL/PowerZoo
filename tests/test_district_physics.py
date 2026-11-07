"""区域调度直接核对原生端口、能量、价格与信息契约。"""

import numpy as np
import pytest

from utils.config_pipeline import ConfigLoader
from utils.config_validation import build_dataclass, environment_class
from envs.district_dispatch.core.dispatch_core import DistrictDispatchCore


def configuration(**changes):
    cfg = ConfigLoader().load("happo", "district_dispatch", values={"env": changes})
    return build_dataclass(environment_class("district_dispatch"), cfg.env_args)


def actions(env, storage=0.0, pv=1.0, q=0.0):
    result = []
    for region in env.config.regions:
        for element in region.controls:
            result.extend(
                [pv, q]
                if element.startswith("PVSystem.")
                else [storage if element.startswith("Storage.") else q]
            )
    return np.array(result, dtype=float)


@pytest.fixture
def core():
    env = DistrictDispatchCore(configuration())
    yield env
    env.close()


def test_actual_regions_transformers_and_signed_balance(core):
    assert [s.shape[0] for s in core.agent_spaces] == [5, 6, 3]
    assert core.neighbors == [[1], [0, 2], [1]]
    assert not any("exchange" in n.lower() for n in core.circuit.native.AllElementNames)
    for _ in range(4):
        _, _, _, info = core.step(actions(core, storage=0.5, q=-0.2))
        meters = list(info["regions"].values())
        assert max(abs(x) for m in meters for x in m["balance_error"]) < 0.05
        assert sum(m["load"][0] for m in meters) == pytest.approx(
            core.circuit.total_load_power()[0], abs=1e-7
        )
        assert sum(m["boundary_import"][0] for m in meters) == pytest.approx(
            -core.crossing_losses[0], abs=1e-6
        )
        assert sum(m["internal_loss"][0] for m in meters) + core.crossing_losses[
            0
        ] == pytest.approx(core.physical_state.losses[0], abs=1e-6)
        assert info["regions"]["midstream"]["shunt"][1] > 0
        assert abs(info["active_balance_error_kw"]) < 0.05
    # downstream与其余区域实际通过变压器连接，而非虚拟848—890交换。
    xfmr = core.physical_state.branches["Transformer.xfm1"]
    assert [b.split(".")[0] for b in xfmr["buses"]] == ["832", "888"]
    assert core.meters[2]["boundary_import"][1] == pytest.approx(-xfmr["powers"][1][1])


@pytest.mark.parametrize("power", [-0.5, 0.5])
def test_native_energy_and_clock_once(core, power):
    before = core._energy()
    _, _, _, info = core.step(actions(core, storage=power))
    assert info["time_s"] == 900
    for name, dev in core.circuit.batteries.items():
        p = core.injections["storage." + name][0]
        assert p * power > 0
        # 原生含1%待机损耗；独立核对放/充电与损耗DC流入。
        idle = dev.max_kw * 0.01
        dc_out = (
            (p + idle) / dev.eta_discharge if p > 0 else (p + idle) * dev.eta_charge
        )
        assert core._energy()[name] == pytest.approx(
            before[name] - dc_out * 0.25, abs=0.05
        )
    energy = core._energy()
    core.circuit.solve(native_controls=True)
    assert core._energy() == energy and core.circuit.time_s == 900
    core.reset(load_profile_idx=0)
    assert core.circuit.time_s == 0 and core._energy() == before


def test_prices_change_money_not_physics(core):
    other = DistrictDispatchCore(
        configuration(market={"base_price_yuan_kwh": 1.0, "carbon_price_yuan_kg": 0.5})
    )
    try:
        _, reward, _, one = core.step(actions(core))
        _, reward2, _, two = other.step(actions(other))
        assert two["total_power_kw"] == pytest.approx(one["total_power_kw"], abs=1e-7)
        assert two["grid_bill_yuan"] == pytest.approx(2 * one["grid_bill_yuan"])
        assert two["carbon_cost_yuan"] == pytest.approx(10 * one["carbon_cost_yuan"])
        assert reward2 != pytest.approx(reward)
        assert -reward * core.config.costs.scale_yuan == pytest.approx(
            one["total_cost_yuan"]
        )
    finally:
        other.close()


def test_native_der_response_and_terminal_energy(core):
    core.t = core.horizon - 1
    core.circuit.native.Solution.dblHour = core.profile[core.t, 0] / 3600
    _, _, done, info = core.step(actions(core, storage=0.5, pv=-1.0, q=0.3))
    assert done and info["terminal_energy_cost_yuan"] > 0
    assert abs(info["pv_power_kw"]) < 0.01
    assert core.injections["generator.svg_mid"][1] > 0
    for name, pv in core.circuit.pvs.items():
        assert float(core.circuit.read_property("PVSystem." + name, "Pmpp")) == pv.pmpp
    with pytest.raises(RuntimeError, match="结束"):
        core.step(actions(core))


def test_regional_actor_and_global_critic_observations(core):
    from envs.district_dispatch.district_dispatch_env import DistrictDispatchEnv

    wrapper = DistrictDispatchEnv.__new__(DistrictDispatchEnv)
    from envs.powergrid.adapters import DeviceMARLAdapter

    DeviceMARLAdapter.__init__(wrapper, core)
    obs, share, _ = wrapper.reset()
    assert np.shape(obs) == (3, 25)
    assert np.shape(share) == (3, 187)
    result = wrapper.step([np.zeros(6)] * 3)
    assert np.all(result[2] == result[2][0])
    assert np.shape(result[0]) == (3, 25) and np.shape(result[1]) == (3, 187)
    fields = core._local_fields(2)
    assert not any("pv_d0" in name for name in fields)
    assert not any("upstream" in name for name in fields)
    assert any("neighbor.midstream" in name for name in fields)
    other = DistrictDispatchCore(configuration(use_neighbor_obs=False))
    try:
        assert (
            other.local_observation_space.shape[0]
            < core.local_observation_space.shape[0]
        )
        assert other.agent_spaces == core.agent_spaces
        assert (
            other.get_physics_manifest()["observation_schema_sha256"]
            != core.get_physics_manifest()["observation_schema_sha256"]
        )
    finally:
        other.close()


@pytest.mark.parametrize(
    "failure",
    ["duplicate_bus", "missing_bus", "missing_der", "duplicate_der", "wrong_owner"],
)
def test_bindings_are_actual_and_exclusive(failure):
    cfg = configuration()
    if failure == "duplicate_bus":
        cfg.regions[1].buses.append(cfg.regions[0].buses[0])
    elif failure == "missing_bus":
        cfg.regions[0].buses.pop()
    elif failure == "missing_der":
        cfg.regions[0].controls.pop()
    elif failure == "duplicate_der":
        cfg.regions[1].controls.append(cfg.regions[0].controls[0])
    else:
        a, b = cfg.regions[:2]
        a.controls[-1], b.controls[-2] = b.controls[-2], a.controls[-1]
    with pytest.raises(ValueError):
        DistrictDispatchCore(cfg)


def test_invalid_frame_does_not_mutate_clock_or_load(core):
    before = core.circuit.read_property("Load.ev_d0", "kW")
    bad = actions(core)
    bad[-1] = np.nan
    with pytest.raises(ValueError):
        core.step(bad)
    assert (
        core.circuit.time_s == 0
        and core.circuit.read_property("Load.ev_d0", "kW") == before
    )


def test_native_source_zero_index_is_valid(core):
    assert core.circuit.native.SetActiveElement("Vsource.source") == 0
    core.circuit.require_element("Vsource.source")
    with pytest.raises(ValueError):
        core.circuit.require_element("Load.missing")


def test_centralized_and_regional_actions_produce_same_physics(core):
    from envs.powergrid.adapters import CentralizedAdapter

    other = DistrictDispatchCore(configuration())
    central = CentralizedAdapter(other)
    try:
        core.reset(load_profile_idx=0)
        central.reset(options={"scenario_index": 0})
        for _ in range(4):
            frame = actions(core, storage=0.3, q=-0.2)
            _, reward, _, one = core.step(frame)
            _, reward2, _, _, two = central.step(frame)
            assert reward == pytest.approx(reward2)
            assert one["storage_energy_kwh"] == two["storage_energy_kwh"]
            assert one["total_power_kw"] == pytest.approx(
                two["total_power_kw"], abs=1e-7
            )
    finally:
        central.close()


def test_region_membership_is_bound_into_checkpoint_schema(core):
    cfg = configuration()
    cfg.regions[0].buses.remove("830")
    cfg.regions[1].buses.append("830")
    other = DistrictDispatchCore(cfg)
    try:
        assert (
            other.get_physics_manifest()["observation_schema_sha256"]
            != core.get_physics_manifest()["observation_schema_sha256"]
        )
    finally:
        other.close()


def test_regional_manual_labels_and_replay_project_real_actor_spaces(core):
    from envs.powergrid.manual_override import DeviceManualOverride
    from envs.district_dispatch.render.engine.episode_runner import EpisodeRunner

    manual = DeviceManualOverride(core)
    assert len(manual.get_action_labels(1)) == 6
    manual.set_agent_override(1, [1, 0, 1, 0, 0, 0.2])
    assert manual.has_overrides()
    runner = EpisodeRunner(core.config)
    try:
        runner._env = runner._create_env()
        _, _, _, _, info, _ = runner._env_step(np.full((3, 6), 3.0))
        assert info[0]["dss_converged"]
        assert all(
            v <= 1 for values in info[0]["normalized_actions"].values() for v in values
        )
    finally:
        runner.close()


def test_retired_district_sources_are_recoverable_and_not_on_active_path():
    from hashlib import sha256
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "archive/physics_refactor_district/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(manifest["files"]) == 64
    for record in manifest["files"]:
        assert (
            sha256((root / record["archive"]).read_bytes()).hexdigest()
            == record["sha256"]
        )
        if record["disposition"] == "retired":
            assert not (root / record["original"]).exists()
    assert not (root / "envs/district_dispatch/core/power_exchange.py").exists()


def test_export_bill_and_team_logger_count_actual_energy_once():
    from envs.district_dispatch.district_dispatch_logger import DistrictDispatchLogger

    core = DistrictDispatchCore(
        configuration(max_episode_steps=24, data={"load_scale": 0.1})
    )
    try:
        _, _, _, info = core.step(actions(core))
        assert info["total_power_kw"] < 0 and info["grid_bill_yuan"] < 0
        energy = -info["total_power_kw"] * info["step_seconds"] / 3600
        assert -info["grid_bill_yuan"] == pytest.approx(
            energy * info["sell_price_yuan_kwh"]
        )
        record = DistrictDispatchLogger._empty()
        DistrictDispatchLogger._accumulate(record, [info] * 3)
        assert record["steps"] == 1
        assert record["grid_import_kwh"] == 0
        assert record["grid_export_kwh"] == pytest.approx(energy)
        assert record["net_grid_import_kwh"] == pytest.approx(-energy)
        assert record["total_cost_yuan"] == info["total_cost_yuan"]
    finally:
        core.close()


def test_global_state_contains_actual_native_mechanical_state(core):
    name, reg = next(iter(core.circuit.regulators.items()))
    initial = core.circuit.regulator_tap(name)
    increment = (reg.maxtap - reg.mintap) / reg.numtaps
    target = (
        initial - increment
        if initial - increment >= reg.mintap
        else initial + increment
    )
    core.circuit.apply({"RegControl." + name: {"tap": target}})
    core.circuit.solve()
    core._observe()
    assert core._global_fields()["RegControl." + name + ".tap_pu"] == pytest.approx(
        target
    )
    for cap in core.circuit.capacitors:
        assert "Capacitor." + cap + ".state" in core._global_fields()


def test_replay_default_root_does_not_depend_on_launch_directory(
    core, monkeypatch, tmp_path
):
    from envs.district_dispatch.render.engine.episode_runner import EpisodeRunner

    monkeypatch.chdir(tmp_path)
    runner = EpisodeRunner(core.config)
    try:
        episode = runner.run_episode(seed=2, max_steps=2)
        assert episode.episode_length == 2
        assert episode.snapshots[-1]["info"]["dss_converged"]
    finally:
        runner.close()


@pytest.mark.parametrize(
    "extra,message",
    [
        ("New Reactor.r bus1=source phases=3 kv=12.47 kvar=30", "尚未验收"),
        (
            "New Capacitor.c bus1=source phases=3 kv=12.47 kvar=30 numsteps=3",
            "多级电容",
        ),
    ],
)
def test_unimplemented_physical_semantics_are_rejected(tmp_path, extra, message):
    (tmp_path / "model.dss").write_text(
        "New Circuit.test basekv=12.47 pu=1 phases=3 bus1=source\nNew Load.l bus1=source phases=3 kv=12.47 kw=100 kvar=20\nNew Storage.b bus1=source phases=3 kv=12.47 kwrated=50 kwhrated=200 %stored=50 %reserve=10\n"
        + extra
        + "\nSet VoltageBases=[12.47]\n",
        encoding="utf-8",
    )
    cfg = configuration(
        dss_folder=str(tmp_path),
        dss_file="model.dss",
        regions=[{"name": "only", "buses": ["source"], "controls": ["Storage.b"]}],
    )
    with pytest.raises(ValueError, match=message):
        DistrictDispatchCore(cfg)


def test_bill_is_integrated_with_actual_interval_hours(tmp_path):
    (tmp_path / "model.dss").write_text(
        """New Circuit.test basekv=12.47 pu=1 phases=3 bus1=source
New Line.l bus1=source bus2=bus phases=3 r1=.1 x1=.1 r0=.2 x0=.2 length=1 units=km
New Load.load bus1=bus phases=3 kv=12.47 kw=100 kvar=20 model=1
New Storage.bat bus1=bus phases=3 kv=12.47 kwrated=50 kwhrated=200 %stored=50 %reserve=10 %idlingkw=0 state=idling dispmode=external
Set VoltageBases=[12.47]
""",
        encoding="utf-8",
    )
    cores = [
        DistrictDispatchCore(
            configuration(
                max_episode_steps=24,
                data={"step_seconds": dt},
                dss_folder=str(tmp_path),
                dss_file="model.dss",
                regions=[
                    {
                        "name": "only",
                        "buses": ["source", "bus"],
                        "controls": ["Storage.bat"],
                    }
                ],
            )
        )
        for dt in (900, 3600)
    ]
    try:
        bills = []
        reference = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 25.0])
        for core in cores:
            core.circuit.native.Solution.dblHour = reference[0] / 3600
            core.profile = np.tile(reference, (core.horizon, 1))
            core.profile[:, 0] = (
                reference[0] + np.arange(core.horizon) * core.config.data.step_seconds
            )
            bill = 0
            for _ in range(3600 // core.config.data.step_seconds):
                _, _, _, info = core.step(actions(core))
                bill += info["grid_bill_yuan"]
            bills.append(bill)
        assert bills[0] == pytest.approx(bills[1], abs=0.001)
    finally:
        for core in cores:
            core.close()
