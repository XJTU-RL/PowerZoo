"""顺序定价的原生量测、独立收益、精确响应与真实SB3轨迹契约。"""

import copy
import json
from pathlib import Path
import numpy as np
import pytest
from utils.config_pipeline import ConfigLoader
from utils.config_validation import ConfigError
from envs.stackelberg.pricing_env import SequentialPricingEnv


def config(case="stackelberg", **env):
    return ConfigLoader().load("sequential_ppo", case, values={"env": env})


@pytest.fixture
def grid():
    env = SequentialPricingEnv(config().env_args)
    yield env
    env.close()


def test_this_price_is_observed_before_response_without_engine_time(grid):
    before = grid.circuit.time_s
    old = grid.leader_observation().copy()
    obs = grid.publish_price(4)
    assert old[4] == 0 and obs[0][-3] == 1
    assert obs[1][-3] == 1 and grid.circuit.time_s == before
    with pytest.raises(RuntimeError):
        grid.publish_price(0)
    _, rewards, done, info = grid.respond([0, 1])
    assert not done and len(set(rewards)) == 3
    assert info["time_s"] == before + 3600
    assert grid.leader_observation()[4] == 1


def test_simultaneous_step_and_old_team_runner_rejected(grid):
    from utils.environment_registry import environment_spec

    with pytest.raises(RuntimeError, match="顺序任务"):
        grid.step([0, 0, 0])
    with pytest.raises(ConfigError, match="sequential_ppo"):
        ConfigLoader().load("happo", "stackelberg")
    with pytest.raises(ConfigError, match="同步协作"):
        ConfigLoader().load("sequential_ppo", "vvc")
    with pytest.raises(ValueError, match="顺序任务"):
        environment_spec("stackelberg").create(grid.config, 0)
    with pytest.raises(RuntimeError):
        grid.respond([0, 0])


@pytest.mark.parametrize("bad", [[0], [0, 3], [float("nan"), 0], [0.5, 0]])
def test_bad_response_does_not_consume_time_or_change_loads(grid, bad):
    grid.publish_price(0)
    before = grid.circuit.time_s
    with pytest.raises(ValueError):
        grid.respond(bad)
    assert grid.circuit.time_s == before
    assert grid._delivered() == pytest.approx(grid.baseline_kw, abs=1e-6)


def test_ledger_uses_actual_kwh_yuan_and_conserves_transfers(grid):
    grid.publish_price(2)
    _, u, _, i = grid.respond([0, 1])
    energy = np.array(i["consumer_energy_kwh"])
    assert energy == pytest.approx(np.array(i["consumer_delivered_kw"]))
    assert i["consumer_bills_yuan"] == pytest.approx(energy * 0.6)
    reduction = np.maximum(0, np.array(i["consumer_baseline_kw"]) - energy)
    discomfort = (
        0.5 * np.array([1, 1.5]) * reduction**2 / np.array(i["consumer_baseline_kw"])
    )
    assert i["consumer_discomfort_yuan"] == pytest.approx(discomfort)
    assert u[0] == pytest.approx(
        sum(i["consumer_bills_yuan"])
        - i["wholesale_cost_yuan"]
        - i["network_cost_yuan"]
    )
    assert u.sum() == pytest.approx(
        -i["wholesale_cost_yuan"] - i["network_cost_yuan"] - discomfort.sum()
    )
    assert abs(i["balance_error_kw"]) < 0.05
    assert i["grid_import_kw"] == pytest.approx(energy.sum() + i["losses_kw"], abs=0.05)


def test_price_has_cash_effect_and_true_analytic_follower_response(grid):
    full = [2, 2]
    low = grid.evaluate_response(0, full)[1]
    high = grid.evaluate_response(1, full)[1]
    assert high["consumer_bills_yuan"] == pytest.approx(
        np.array(low["consumer_bills_yuan"]) * 2
    )
    assert high["wholesale_cost_yuan"] == pytest.approx(
        low["wholesale_cost_yuan"], abs=1e-5
    )
    response = []
    for p, price in enumerate(grid.config.retail_prices_yuan_per_kwh):
        _, equilibria = grid.exact_response_table(p)
        # 恒PQ小例可独立解析，beta/2*(1-f)^2+price*f，等价元/kWh成本。
        expected = tuple(
            int(
                np.argmin(
                    [
                        price * f + 0.5 * c.discomfort_yuan_per_kwh * (1 - f) ** 2
                        for f in grid.config.load_fractions
                    ]
                )
            )
            for c in grid.config.consumers
        )
        assert equilibria == [expected]
        assert grid.unilateral_gains(p, expected) == pytest.approx([0, 0], abs=1e-7)
        response.append(expected)
    assert response[0] != response[-1]


def test_counterfactual_restores_native_state_time_and_soc(grid):
    before = grid.circuit.time_s
    loads = grid._delivered().copy()
    reference = grid.exact_reference()
    assert grid.circuit.time_s == before
    assert grid._delivered() == pytest.approx(loads, abs=1e-5)
    assert reference["best"]["price_index"] == 4
    assert reference["best"]["utilities_yuan"][0] == max(
        r["utilities_yuan"][0] for r in reference["prices"]
    )
    assert grid.phase == "leader"
    with pytest.raises(ValueError):
        grid.evaluate_response(999, [0, 0])
    assert grid._delivered() == pytest.approx(loads, abs=1e-5)


def test_unit_duration_changes_cash_by_same_duration(grid):
    other = SequentialPricingEnv(
        config(data={"step_seconds": 1800}, max_episode_steps=24).env_args
    )
    try:
        grid.reset(load_profile_idx=0)
        other.reset(load_profile_idx=0)
        a = grid.evaluate_response(2, [1, 2])[1]
        b = other.evaluate_response(2, [1, 2])[1]
        # 同一合成种子首行一致；只改变工程时长。
        assert a["consumer_baseline_kw"] == pytest.approx(b["consumer_baseline_kw"])
        assert a["consumer_energy_kwh"] == pytest.approx(
            np.array(b["consumer_energy_kwh"]) * 2
        )
        assert a["utilities_yuan"] == pytest.approx(
            np.array(b["utilities_yuan"]) * 2, abs=1e-5
        )
    finally:
        other.close()


def test_native_discomfort_and_network_cost_settings_have_real_consumers(grid):
    cfg = copy.deepcopy(config().env_args)
    cfg["consumers"][0]["discomfort_yuan_per_kwh"] = 2
    other = SequentialPricingEnv(cfg)
    try:
        grid.reset(load_profile_idx=0)
        other.reset(load_profile_idx=0)
        a = grid.evaluate_response(1, [0, 2])[1]
        b = other.evaluate_response(1, [0, 2])[1]
        assert b["consumer_discomfort_yuan"][0] == pytest.approx(
            2 * a["consumer_discomfort_yuan"][0]
        )
        assert b["consumer_bills_yuan"] == pytest.approx(a["consumer_bills_yuan"])
        other.config.voltage_min = 1.01
        other.config.voltage_max = 1.02
        cost = other.evaluate_response(1, [0, 2])[1]["network_cost_yuan"]
        other.config.network_cost_yuan_per_hour *= 2
        assert cost > 0
        assert other.evaluate_response(1, [0, 2])[1][
            "network_cost_yuan"
        ] == pytest.approx(cost * 2)
    finally:
        other.close()


def test_each_actual_actor_value_and_buffer_keeps_own_reward(grid, tmp_path):
    from envs.stackelberg.training import make_policies, collect_episode

    cfg = config()
    models = make_policies(cfg, grid, "cpu", tmp_path)
    trace = []
    totals, records = collect_episode(grid, models, training=True, trace=trace)
    raw = np.array([r["utilities_yuan"] for r in records])
    assert len({id(m.rollout_buffer) for m in models}) == 3
    assert len({id(m.policy.value_net) for m in models}) == 3
    for i, model in enumerate(models):
        assert model.rollout_buffer.full
        assert model.rollout_buffer.rewards[:, 0] == pytest.approx(raw[:, i], rel=1e-6)
        assert np.isfinite(model.rollout_buffer.returns).all()
        if i:
            assert model.rollout_buffer.observations[:, 0, -3] == pytest.approx(
                [r["retail_price_yuan_per_kwh"] for r in records]
            )
        model.train()
        model.get_env().close()
    assert totals == pytest.approx(raw.sum(axis=0))
    assert not np.array_equal(
        models[0].rollout_buffer.returns, models[1].rollout_buffer.returns
    )


def test_ieee13_actual_load_bindings_and_fixed_storage():
    env = SequentialPricingEnv(config("stackelberg_13bus").env_args)
    try:
        env.reset(load_profile_idx=0)
        soc = env.circuit.batteries["batt1"].soc
        before = env.circuit.time_s
        env.exact_reference()
        assert env.circuit.time_s == before
        assert env.circuit.batteries["batt1"].soc == soc
        env.publish_price(1)
        _, _, _, i = env.respond([0, 1])
        assert abs(i["balance_error_kw"]) < 0.05
        assert env.circuit.batteries["batt1"].soc == pytest.approx(soc)
        assert (
            len(env.circuit.loads) == 15 and len(env.circuit.read_state().branches) > 10
        )
    finally:
        env.close()


def test_configuration_rejects_fictional_or_duplicated_devices(grid):
    cfg = copy.deepcopy(config().env_args)
    cfg["consumers"][0]["loads"] = ["business"]
    with pytest.raises(ValueError, match="恰好归属"):
        SequentialPricingEnv(cfg)
    for key in ("ess_config", "async_config", "n_consumer_agents", "monitoring_config"):
        with pytest.raises(ConfigError, match=key):
            config(**{key: {}})


def test_pricing_model_cannot_bill_disconnected_load_as_demand(tmp_path):
    import shutil

    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "node_systems/PricingSmall", tmp_path / "model")
    with (tmp_path / "model/network.dss").open("a", encoding="utf-8") as f:
        f.write("\nOpen Line.feed_a term=1\n")
    with pytest.raises(ValueError, match="有效主网源"):
        SequentialPricingEnv(config(dss_folder=str(tmp_path / "model")).env_args)


def test_all_old_pricing_files_preserved_and_unused():
    from hashlib import sha256

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "archive/physics_refactor_stackelberg/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for item in manifest["files"]:
        assert (
            sha256((root / item["archive"]).read_bytes()).hexdigest() == item["sha256"]
        )
        if item["disposition"] == "retired":
            assert not (root / item["original"]).exists()


def test_real_checkpoint_roundtrip_and_contract_failure(tmp_path, monkeypatch):
    from utils.training_setup import prepare_run
    from envs.stackelberg.training import (
        make_policies,
        collect_episode,
        save_checkpoint,
        load_checkpoint,
    )

    cfg = prepare_run(config())
    env = SequentialPricingEnv(cfg.env_args)
    models = make_policies(cfg, env, "cpu", tmp_path)
    folder = tmp_path / "checkpoint"
    try:
        collect_episode(env, models, training=True)
        for model in models:
            model.train()
        save_checkpoint(cfg, env, models, folder)
    finally:
        env.close()
        for model in models:
            model.get_env().close()
    # 实际换到仓库外启动恢复，禁止依赖cwd猜模型路径。
    monkeypatch.chdir(tmp_path)
    restored, policies = load_checkpoint(folder)
    try:
        _, records = collect_episode(restored, policies, training=False)
        assert len(records) == 24 and all(p.num_timesteps == 24 for p in policies)
    finally:
        restored.close()
        for policy in policies:
            policy.get_env().close()
    path = folder / "physics_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["economics"]["network_cost_yuan_per_hour"] += 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="契约不一致"):
        load_checkpoint(folder)
