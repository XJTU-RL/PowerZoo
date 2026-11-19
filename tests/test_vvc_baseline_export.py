"""Offline wiring tests; these do not exercise the DSS engine."""

from types import SimpleNamespace

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
