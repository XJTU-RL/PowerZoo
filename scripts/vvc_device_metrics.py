"""Observed IEEE 34Bus device endpoints, not controller operation logs."""

import math


TAP_ABS_TOL_PU = 1e-6


def measure_devices(circuit):
    return {
        "capacitor_stages": {
            name: [int(x) for x in circuit.capacitor_state(name)]
            for name in circuit.capacitors
        },
        "regulator_tap_pu": {
            name: float(circuit.regulator_tap(name)) for name in circuit.regulators
        },
        "storage_injection_kw": {
            name: -float(circuit.powers(f"Storage.{name}")[0])
            for name in circuit.batteries
        },
    }


def summarize_devices(samples, records, episode_metrics):
    """Require t=0 and one measured endpoint for each successful interval."""
    fields = ("capacitor_stages", "regulator_tap_pu", "storage_injection_kw")
    completed = episode_metrics["steps_completed"]
    if not samples or len(samples) != completed + 1:
        raise ValueError(
            "device samples must align with completed intervals plus reset"
        )
    initial = samples[0]["devices"]
    for step, row in enumerate(samples):
        if row["step"] != step or set(row["devices"]) != set(fields):
            raise ValueError("invalid device endpoint step or fields")
        devices = row["devices"]
        for key in fields:
            if set(devices[key]) != set(initial[key]):
                raise ValueError(f"device membership changed for {key}")
        for name, states in devices[fields[0]].items():
            if (
                not states
                or len(states) != len(initial[fields[0]][name])
                or any(
                    type(state) is not int or state not in (0, 1) for state in states
                )
            ):
                raise ValueError("invalid capacitor stage vector")
        for key in fields[1:]:
            if any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in devices[key].values()
            ):
                raise ValueError(f"invalid device measurement for {key}")
    eligible = episode_metrics["eligible_for_day_ranking"]
    result = {
        "status": "valid" if eligible else episode_metrics["day_ranking_refusal"],
        "capacitor_stage_endpoint_changes": None,
        "regulator_endpoint_changes": None,
        "storage_charge_endpoint_estimate_kwh": None,
        "storage_discharge_endpoint_estimate_kwh": None,
    }
    if not eligible:
        return result
    cap, tap, charge, discharge = 0, 0, [], []
    for i in range(completed):
        before = samples[i]["devices"]
        after = samples[i + 1]["devices"]
        dt = records[i]["step_seconds"] / 3600
        cap += sum(
            a != b
            for name, values in before[fields[0]].items()
            for a, b in zip(values, after[fields[0]][name])
        )
        tap += sum(
            not math.isclose(
                value, after[fields[1]][name], rel_tol=0, abs_tol=TAP_ABS_TOL_PU
            )
            for name, value in before[fields[1]].items()
        )
        charge.extend(max(-p, 0) * dt for p in after[fields[2]].values())
        discharge.extend(max(p, 0) * dt for p in after[fields[2]].values())
    result.update(
        capacitor_stage_endpoint_changes=cap,
        regulator_endpoint_changes=tap,
        storage_charge_endpoint_estimate_kwh=math.fsum(charge),
        storage_discharge_endpoint_estimate_kwh=math.fsum(discharge),
    )
    return result


def device_output(samples, records, episode_metrics):
    return {
        "device_endpoints": samples,
        "device_metrics": summarize_devices(samples, records, episode_metrics),
    }
