"""验收状态、证据校验与顺序测试，使用纯内存回调。"""

from dataclasses import replace
import json

import pytest

from utils.acceptance import (
    AcceptanceBlocked,
    AcceptanceChecks,
    STAGES,
    run_acceptance,
    write_report,
)
from utils.capabilities import Preflight


@pytest.fixture
def preflight():
    return Preflight("happo", "vvc", "13Bus", "marl")


@pytest.fixture
def evidence():
    return {
        "physical_readback": {
            "converged": True,
            "active_balance_error_kw": 0.01,
            "balance_tolerance_kw": 0.05,
            "device_ids": ["RegControl.a"],
            "readback": {
                "RegControl.a": {
                    "quantity": "tap",
                    "requested": 1.0,
                    "applied": 1.0,
                    "measured": 1.0,
                    "tolerance": 1e-6,
                    "matched": True,
                }
            },
        },
        "minimal_update": {
            "steps": 4,
            "updates": 1,
            "parameters_before_sha256": "a" * 64,
            "parameters_after_sha256": "b" * 64,
            "training_scenarios": {"train_000": "c" * 64},
        },
        "save_reload": {"checkpoint_sha256": "d" * 64, "prediction_match": True},
        "independent_replay": {
            "split": "validation",
            "steps": 24,
            "expected_steps": 24,
            "terminated": True,
            "truncated": False,
            "physical_failure": False,
            "scenario_id": "validation_000",
            "scenario_sha256": "e" * 64,
            "checkpoint_sha256": "d" * 64,
        },
    }


def hooks(evidence, calls):
    def get(stage):
        calls.append(stage)
        entry = evidence[stage]
        if isinstance(entry, Exception):
            raise entry
        return entry

    return AcceptanceChecks(
        **{stage: lambda name=stage: get(name) for stage in STAGES[1:]}
    )


def test_reports_all_stages_in_order(preflight, evidence, tmp_path):
    calls = []
    report = run_acceptance(preflight, hooks(evidence, calls))
    assert report["status"] == "passed"
    assert calls == list(STAGES[1:])
    destination = tmp_path / "report.json"
    write_report(report, destination)
    assert list(json.loads(destination.read_text())["stages"]) == list(STAGES)
    with pytest.raises(FileExistsError):
        write_report(report, destination)


def test_dry_run_and_failed_preflight_never_call_runtime(preflight, evidence):
    calls = []
    report = run_acceptance(preflight)
    assert report["status"] == "not_run"
    assert report["stages"]["preflight"]["status"] == "passed"
    failed = run_acceptance(
        replace(preflight, issues=("unsupported",)), hooks(evidence, calls)
    )
    assert failed["status"] == "failed" and not calls
    assert failed["stages"]["physical_readback"]["status"] == "not_run"


@pytest.mark.parametrize(
    ("stage", "key", "value"),
    [
        ("physical_readback", "active_balance_error_kw", float("nan")),
        ("physical_readback", "active_balance_error_kw", 0.2),
        ("physical_readback", "readback", {}),
        ("minimal_update", "updates", 0),
        ("minimal_update", "parameters_after_sha256", "a" * 64),
        ("save_reload", "prediction_match", False),
        ("independent_replay", "scenario_id", "train_000"),
        ("independent_replay", "scenario_sha256", "c" * 64),
        ("independent_replay", "split", "test"),
        ("independent_replay", "checkpoint_sha256", "f" * 64),
        ("independent_replay", "truncated", True),
        ("independent_replay", "physical_failure", True),
        ("independent_replay", "steps", 23),
    ],
)
def test_rejects_missing_or_inconsistent_evidence(
    preflight, evidence, stage, key, value
):
    calls = []
    evidence[stage][key] = value
    report = run_acceptance(preflight, hooks(evidence, calls))
    assert report["status"] == "failed"
    assert report["stages"][stage]["status"] == "failed"
    assert calls == list(STAGES[1 : STAGES.index(stage) + 1])


def test_blocked_stage_preserves_previous_evidence(preflight, evidence):
    evidence["minimal_update"] = AcceptanceBlocked("missing dependency")
    report = run_acceptance(preflight, hooks(evidence, []))
    assert report["status"] == "blocked"
    assert report["stages"]["physical_readback"]["status"] == "passed"
    assert report["stages"]["minimal_update"]["detail"] == "missing dependency"
    assert report["stages"]["save_reload"]["status"] == "not_run"


def test_failed_numeric_readback_cannot_be_overridden_by_flag(preflight, evidence):
    evidence["physical_readback"]["readback"]["RegControl.a"]["measured"] = 0.95
    report = run_acceptance(preflight, hooks(evidence, []))
    failed = report["stages"]["physical_readback"]
    assert failed["status"] == "failed"
    assert failed["evidence"]["readback"]["RegControl.a"]["measured"] == 0.95
    assert report["stages"]["minimal_update"]["status"] == "not_run"
