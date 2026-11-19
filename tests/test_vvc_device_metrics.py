"""Device metrics consume measurements, never actions or control traces."""

import pytest

from scripts.vvc_benchmark_metrics import summarize_episode
from scripts.vvc_device_metrics import summarize_devices


def sample(step, cap, tap, storage):
    return {
        "step": step,
        "devices": {
            "capacitor_stages": {"a": cap, "b": [step % 2]},
            "regulator_tap_pu": {"r1": tap, "r2": 1.0 + step * 0.00000025},
            "storage_injection_kw": dict(zip(("s1", "s2"), storage)),
        },
    }


def record(hours):
    return {
        "dss_converged": True,
        "step_seconds": 3600 * hours,
        "power_loss_kw": 0,
        "voltage_violation_count": 0,
        "voltage_min_pu": 1,
        "voltage_max_pu": 1,
    }


def test_reset_is_compared_not_integrated_and_storage_is_not_netted():
    samples = [
        sample(0, [0, 1], 1.0, [1000, -1000]),
        sample(1, [1, 0], 1.1, [-4, 4]),
        sample(2, [1, 0], 1.10000001, [2, -1]),
    ]
    records = [record(6), record(18)]
    assert summarize_devices(samples, records, summarize_episode(records, 2)) == {
        "status": "valid",
        "capacitor_stage_endpoint_changes": 4,
        "regulator_endpoint_changes": 1,
        "storage_charge_endpoint_estimate_kwh": 42,
        "storage_discharge_endpoint_estimate_kwh": 60,
    }


@pytest.mark.parametrize(
    "records, expected, reason",
    [
        ([record(6), {"physical_failure": True}], 2, "solver_failure"),
        ([record(6)], 2, "incomplete"),
        ([record(6)], 1, "not_24_hours"),
    ],
)
def test_ineligible_day_has_no_device_totals(records, expected, reason):
    samples = [sample(0, [0], 1.0, [0, 0]), sample(1, [1], 1.1, [3, -4])]
    result = summarize_devices(samples, records, summarize_episode(records, expected))
    assert result.pop("status") == reason
    assert all(value is None for value in result.values())


@pytest.mark.parametrize(
    "case, match",
    [
        ("membership", "membership"),
        ("shape", "stage vector"),
        ("nonbinary", "stage vector"),
        ("missing_reset", "align"),
        ("unaligned", "endpoint step"),
    ],
)
def test_reject_device_evidence_that_does_not_align(case, match):
    samples = [sample(0, [0], 1.0, [0, 0]), sample(1, [1], 1.1, [3, -4])]
    if case == "membership":
        samples[1]["devices"]["storage_injection_kw"].pop("s1")
    if case == "shape":
        samples[1]["devices"]["capacitor_stages"]["a"].append(1)
    if case == "nonbinary":
        samples[1]["devices"]["capacitor_stages"]["a"] = [2]
    if case == "missing_reset":
        samples.pop(0)
    if case == "unaligned":
        samples[0]["step"] = 1
    records = [record(24)]
    with pytest.raises(ValueError, match=match):
        summarize_devices(samples, records, summarize_episode(records, 1))
