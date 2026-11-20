"""Offline wiring tests; these do not exercise the DSS engine."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.export_vvc_baselines import _inventory, _run


class Circuit:
    step_seconds = 3600
    model_sha256 = "physics"
    capacitors = {"cap": (1,)}
    batteries = {"bat": SimpleNamespace(soc=0.5)}
    regulators = {
        f"r{i}": SimpleNamespace(transformer=f"tx{i}", tap_winding=2) for i in range(6)
    }

    def __init__(self):
        self.native = SimpleNamespace(
            AllElementNames=["Vsource.source", "Storage.bat", "Capacitor.cap"]
            + [f"RegControl.{name}" for name in self.regulators],
            ActiveCktElement=SimpleNamespace(Enabled=True),
        )
        self._failed = False
        self.tap = 1.0
        self.events = []
        self.fail = False

    def require_element(self, el):
        self.native.ActiveCktElement.Enabled = el != "RegControl.r5"

    def read_property(self, el, key):
        assert key == "maxtapchange"
        return "16"

    def capacitor_state(self, name):
        return self.capacitors[name]

    def regulator_tap(self, name):
        return self.tap if name == "r0" else 1.0

    def powers(self, el):
        assert el == "Storage.bat"
        return (0.0, 0.0)

    def begin_interval(self, t):
        self.events.append(("begin", t))

    def apply(self, request):
        self.events.append(("apply", request))

    def solve(self, *, native_controls):
        self.events.append(("solve", native_controls))
        if self.fail:
            self._failed = True
            raise RuntimeError("failure")
        if native_controls:
            self.tap = 1.0125

    def finish_interval(self):
        self.events.append(("finish",))


class Core:
    horizon = 2
    profile = {(0, 0): 0.0, (1, 0): 3600.0}
    entry = {"id": "val-2", "split": "validation"}
    dataset = SimpleNamespace(sha256="manifest")
    config = SimpleNamespace(
        max_episode_steps=2,
        voltage_min=0.95,
        voltage_max=1.05,
        data=SimpleNamespace(manifest_sha256="manifest"),
    )

    def __init__(self):
        self.circuit = Circuit()

    def seed(self, seed):
        assert seed == 17

    def reset(self, *, load_profile_idx):
        self._episode_index = load_profile_idx
        self.t = 0

    def get_physics_manifest(self):
        return "saved"

    def _apply_data(self, i):
        self.circuit.events.append(("data", i))

    def _observe(self):
        self.circuit.events.append(("observe",))

    def _info(self):
        return {
            "dss_converged": True,
            "step_seconds": 3600,
            "power_loss_kw": 4,
            "voltage_min_pu": 0.94,
            "voltage_max_pu": 1.0,
            "voltage_violation_count": 2,
            "scenario_id": self.entry["id"],
            "dataset_sha256": self.dataset.sha256,
            "physics_model_sha256": self.circuit.model_sha256,
        }

    def _divergence_transition(self, exc):
        return None, None, True, {"physical_failure": True}


def compare(saved, live):
    assert saved == live


@pytest.mark.parametrize("native", [False, True])
def test_baseline_ownership_and_interval_order(native):
    core = Core()
    output = _run(core, "saved", compare, 2, 17, native)
    for i in range(2):
        assert core.circuit.events[i * 6 : (i + 1) * 6] == [
            ("begin", i * 3600.0),
            ("data", i),
            ("apply", {"Storage.bat": {"kw": 0.0}}),
            ("solve", native),
            ("finish",),
            ("observe",),
        ]
    assert output["dss_owned_regulator_count"] == (5 if native else 0)
    assert output["initial_device_state"]["tap_pu"]["r0"] == 1.0
    assert output["last_converged_device_state"]["tap_pu"]["r0"] == (
        1.0125 if native else 1.0
    )
    assert output["metrics"]["voltage_violation_interval_hours"] == 2


def test_must_not_finish_failed_interval_or_rank_energy():
    core = Core()
    core.circuit.fail = True
    output = _run(core, "saved", compare, 2, 17, True)
    assert core.circuit.events[-1] == ("solve", True)
    assert output["records"] == [{"physical_failure": True}]
    assert output["metrics"]["energy_kwh"] is None


def test_unexpected_controller_fails_closed():
    circuit = Circuit()
    circuit.native.AllElementNames.append("StorageController.extra")
    with pytest.raises(ValueError, match="unaudited"):
        _inventory(circuit)


def test_duplicate_enabled_tap_writer_rejected():
    circuit = Circuit()
    circuit.regulators = dict(circuit.regulators)
    circuit.regulators["r1"] = circuit.regulators["r0"]
    with pytest.raises(ValueError, match="ambiguous"):
        _inventory(circuit)


def test_unowned_tap_change_rejected():
    core = Core()
    original = core.circuit.solve

    def solve(*, native_controls):
        original(native_controls=native_controls)
        core.circuit.tap = 0.975

    core.circuit.solve = solve
    with pytest.raises(RuntimeError, match="unowned"):
        _run(core, "saved", compare, 2, 17, False)


def test_json_accepts_native_numpy_endpoint_scalars():
    core = Core()
    circuit = core.circuit
    circuit.capacitors = {"cap": (np.int32(1), np.int32(0))}
    circuit.batteries = {"bat": SimpleNamespace(soc=np.float32(0.5))}
    circuit.regulators = {
        name: SimpleNamespace(transformer=reg.transformer, tap_winding=np.int32(2))
        for name, reg in circuit.regulators.items()
    }
    tap = circuit.regulator_tap
    circuit.regulator_tap = lambda name: np.float32(tap(name))
    result = json.loads(
        json.dumps(_run(core, "saved", compare, 2, 17, True), allow_nan=False)
    )
    assert result["initial_device_state"] == {
        "capacitors": {"cap": [1, 0]},
        "storage_soc": {"bat": 0.5},
        "tap_pu": {name: 1.0 for name in circuit.regulators},
    }
    assert result["regcontrol_inventory"][0]["tap_winding"] == 2
    assert result["last_converged_device_state"]["capacitors"] == {"cap": [1, 0]}


def test_paired_reset_policy_fingerprint_is_sensitive_to_observations_and_masks():
    from scripts.vvc_device_metrics import reset_policy_input_sha256

    core = SimpleNamespace(observation_mode="global", agent_spaces=(1, 2),
                           get_obs=lambda: [1, 2],
                           available_actions=lambda: ([1, 0], None))
    digest = reset_policy_input_sha256(core)
    assert digest == reset_policy_input_sha256(core)
    core.available_actions = lambda: ([0, 1], None)
    assert digest != reset_policy_input_sha256(core)
    core.available_actions = lambda: ([1, 0], None)
    core.get_obs = lambda: [2, 1]
    assert digest != reset_policy_input_sha256(core)


def test_paired_trace_identity_ignores_timestamp_relabel_and_format():
    from scripts.export_vvc_baselines import _trace_key

    source = [[0, 1, -0., 2], [3600, 1.3, 1, 2]]
    relabel = [[86400, 1., 0, 2.], [90000, 1.3, 1., 2.]]
    assert _trace_key(source) == _trace_key(relabel)
    relabel[1][-1] = 3.
    assert _trace_key(source) != _trace_key(relabel)


@pytest.mark.parametrize("arm_count", [4, 5])
def test_paired_gate_refuses_reset_and_different_physical_ticks(arm_count):
    from copy import deepcopy
    from scripts.export_vvc_baselines import _check_pair

    identity = dict(split="validation", seed=17, scenario_id="valid-1", scenario_index=1,
                    dataset_sha256="data", physics_model_sha256="stock")
    base = dict(identity, reset_policy_input_sha256="capture", device_endpoints=[{"step": 0}],
                records=[{"dss_converged": True, "step_seconds": 3600}],
                metrics={"eligible_for_day_ranking": True})
    rows = [deepcopy(base) for _ in range(arm_count)]
    assert _check_pair(rows, identity)
    rows[2]["reset_policy_input_sha256"] = "different"
    with pytest.raises(ValueError, match="observation/action"):
        _check_pair(rows, identity)
    rows[2]["reset_policy_input_sha256"] = "capture"
    rows[2]["records"][0]["step_seconds"] = 300
    with pytest.raises(ValueError, match="durations"):
        _check_pair(rows, identity)
    rows[2]["metrics"]["eligible_for_day_ranking"] = False
    assert not _check_pair(rows, identity)


def test_checkpoint_binding_reports_only_verified_digests_and_budget():
    from scripts.export_vvc_baselines import _verified_checkpoint_report

    actor = {"config.resolved.yaml": "a", "physics.manifest.json": "b", "actor_agent0.pt": "c"}
    details = (["private path"], {"train": {"num_env_steps": 48000, "episode_length": 24,
                                           "n_rollout_threads": 1}},
               {}, 17, actor)
    binding = _verified_checkpoint_report(details, actor)
    assert binding == {"training_seed": 17, "checkpoint_files_sha256": actor,
                       "configured_training": {"num_env_steps": 48000, "episode_length": 24,
                                               "n_rollout_threads": 1}}
    with pytest.raises(ValueError, match="digests differ"):
        _verified_checkpoint_report(details, {**actor, "actor_agent0.pt": "other"})


@pytest.mark.parametrize("fault", ["missing_sb3_digest", "replaced_sb3_manifest"])
def test_paired_plan_still_binds_sb3_manifest_before_scenario_access(tmp_path, monkeypatch, fault):
    from hashlib import sha256
    from scripts import export_vvc_baselines as export

    manifest = tmp_path / "manifest.json"
    manifest.write_text("not consumed by this gate")
    digest = sha256(manifest.read_bytes()).hexdigest()
    physics = {"model_sha256":"model", "action_schema_sha256":"actions",
               "observation_schema_sha256":"observations", "overlays":[]}
    env = {"system_name":"34Bus", "dss_folder":"node_systems/34Bus",
           "dss_file":"network.dss", "max_episode_steps":24,
           "data":{"split":"train", "path":str(manifest),
                   "manifest_sha256":digest, "step_seconds":3600}}
    config = {"env":env, "train":{"num_env_steps":24, "episode_length":24,
                                   "n_rollout_threads":1}}
    # Existing MARL digest gates are covered elsewhere; isolate the optional fifth arm.
    monkeypatch.setattr(export, "_checkpoint_inputs", lambda *args: ([], config, physics, 1, {}))
    def no_scenarios(*args):
        raise AssertionError("must reject manifest binding before reading any scenario")
    monkeypatch.setattr(export, "_paired_sources", no_scenarios)
    prefix = tmp_path / "ppo"
    for suffix in (".zip", ".environment.yaml", ".physics.json", ".sb3.json"):
        prefix.with_suffix(suffix).write_text("original")
    hashes = {prefix.with_suffix(s).name:sha256(prefix.with_suffix(s).read_bytes()).hexdigest()
              for s in (".zip", ".environment.yaml", ".physics.json", ".sb3.json")}
    if fault == "missing_sb3_digest":
        del hashes["ppo.sb3.json"]
    else:
        prefix.with_suffix(".sb3.json").write_text("replacement")
    plan = {"split":"validation", "environment_seeds":[17], "scenario_indices":[0],
            "manifest":str(manifest), "manifest_sha256":digest,
            "checkpoints":{"happo":{"path":str(tmp_path),"sha256":{}},
                           "mappo":{"path":str(tmp_path),"sha256":{}},
                           "ppo":{"path":str(prefix),"sha256":hashes}}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="PPO checkpoint or sidecar digests differ"):
        export.run_paired_plan(path)


@pytest.mark.parametrize("has_ppo,rows,expected", [
    (False, [{"all_four_full_day":True}], 0),
    (False, [{"all_four_full_day":False}], 1),
    (True, [{"all_four_full_day":True,"all_five_full_day":False}], 1),
    (True, [{"all_four_full_day":True,"all_five_full_day":True}], 0),
    (True, [{"all_four_full_day":True}], 1),
    (True, [{"all_four_full_day":True,"all_five_full_day":True},
            {"all_four_full_day":True,"all_five_full_day":False}], 1),
])
def test_paired_cli_exit_requires_every_selected_arm(monkeypatch, capsys, has_ppo, rows, expected):
    from scripts import export_vvc_baselines as export
    report = {"checkpoints":{"ppo":{}} if has_ppo else {}, "results":rows}
    monkeypatch.setattr(export, "run_paired_plan", lambda path:report)
    assert export.main(["--paired-plan", "unused.json"]) == expected
    assert json.loads(capsys.readouterr().out) == report
