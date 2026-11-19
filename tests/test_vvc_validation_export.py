"""Exercise validation export glue without training or an OpenDSS engine."""

from dataclasses import dataclass
from hashlib import sha256
from types import SimpleNamespace

import pytest

from scripts import export_vvc_validation as export


@dataclass
class Data:
    split: str
    path: str
    manifest_sha256: str


@dataclass
class Config:
    data: Data
    max_episode_steps: int = 1
    smartds_series: dict | None = None
    voltage_min: float = 0.95
    voltage_max: float = 1.05


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    digest = sha256(manifest.read_bytes()).hexdigest()
    config = Config(Data("train", str(manifest), digest))
    closed = []

    class Runner:
        def __init__(self, validation, engine, *, scenario_index):
            assert validation.data.split == "validation"
            assert config.data.split == "train"
            assert engine.config is config
            self.index = scenario_index
            self.benchmark_identity = {
                "scenario_id": f"val-{scenario_index}",
                "scenario_index": scenario_index,
                "split": "validation",
                "dataset_sha256": digest,
            }

        def run_episode(self, seed, max_steps):
            assert (seed, max_steps) == (17, 1)
            if self.index == 2:
                step = {"step": 1, "physical_failure": True}
            else:
                identity = dict(self.benchmark_identity)
                if self.index == 6:
                    identity["scenario_id"] = "wrong"
                step = {
                    "step": 1,
                    "info": {
                        **identity,
                        "dss_converged": True,
                        "step_seconds": 900,
                        "power_loss_kw": 4.0,
                        "voltage_violation_count": 1,
                        "voltage_min_pu": 0.94,
                        "voltage_max_pu": 1.01,
                    },
                }
            return SimpleNamespace(snapshots=[{"step": 0}, step], episode_length=1)

        def close(self):
            closed.append(self.index)

    def load(checkpoint, *, device, expected_family):
        assert (checkpoint, device, expected_family) == ("model", "cpu", "vvc")
        return SimpleNamespace(config=config)

    monkeypatch.setattr(export, "_runtime", lambda: (load, Runner))
    return manifest, config, closed


def test_validation_nonzero_indices_and_native_failure(runtime):
    manifest, _, closed = runtime
    result = export.export_validation("model", manifest, [3, 2], seed=17)
    healthy, failed = result["episodes"]
    assert healthy["scenario_id"] == "val-3"
    assert len(healthy["records"]) == 1
    assert healthy["metrics"]["energy_kwh"] == 1
    assert healthy["metrics"]["voltage_violation_interval_hours"] == 0.25
    assert healthy["metrics"]["voltage_max_exceedance_pu"] == pytest.approx(0.01)
    assert healthy["metrics"]["day_ranking_refusal"] == "not_24_hours"
    assert failed["metrics"]["status"] == "solver_failure"
    assert failed["metrics"]["energy_kwh"] is None
    assert closed == [3, 2]


def test_rejects_changed_hash_and_nonvalidation(runtime):
    manifest, config, closed = runtime
    config.data.manifest_sha256 = ""
    with pytest.raises(ValueError, match="matching SHA256"):
        export.export_validation("model", manifest, [3], seed=17)
    with pytest.raises(ValueError, match="only validation"):
        export.export_validation("model", manifest, [3], seed=17, split="test")
    assert not closed


def test_rejects_inconsistent_native_identity_and_closes(runtime):
    manifest, _, closed = runtime
    with pytest.raises(ValueError, match="scenario_id"):
        export.export_validation("model", manifest, [6], seed=17)
    assert closed == [6]


def test_rejects_duplicate_indices(runtime):
    manifest, _, _ = runtime
    with pytest.raises(ValueError, match="distinct"):
        export.export_validation("model", manifest, [3, 3], seed=17)


def test_mappo_option_is_forwarded_only_when_explicit(runtime, monkeypatch):
    manifest, _, _ = runtime
    load, runner = export._runtime()

    def with_opt_in(checkpoint, *, allow_mappo, **kwargs):
        assert allow_mappo is True
        return load(checkpoint, **kwargs)

    monkeypatch.setattr(export, "_runtime", lambda: (with_opt_in, runner))
    export.export_validation("model", manifest, [3], seed=17, allow_mappo=True)
    with pytest.raises(ValueError, match="boolean"):
        export.export_validation("model", manifest, [3], seed=17, allow_mappo="true")
