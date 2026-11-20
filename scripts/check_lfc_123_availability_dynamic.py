"""Opt-in two-hold OpenDSS-only trial of scratch prospective vs live delivery."""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

from envs.lfc.damped_droop_123 import load_protocol as load_damped
from envs.lfc.droop_123_draft import load_protocol as load_fixed
from envs.lfc.native_123_availability import (
    DeviceDispatchMismatch, SPEC, _assert_parity, _device_voltages, _near, collect, envelope, prospective,
    verify_device_envelope,
)
from envs.lfc.native_feeder_123 import Native123Feeder, PQ, stock_probe_spec
from scripts.check_lfc_123_coupling import SCENARIO as TRANSPORT


def _point(op):
    devs = stock_probe_spec()[0]["storage_devices"]
    return {
        "clock_s": op.clock_s,
        "soc": op.soc,
        "pcc_kw": op.source_pq.kw,
        "pcc_kvar": op.source_pq.kvar,
        "storage": [
            {
                "id": d["id"],
                "kw": p.kw,
                "kvar": p.kvar,
                "voltage_pu": _device_voltages(op, d),
                "conductor_a": {
                    str(x.node): x.amps for x in op.conductors if x.element == d["id"]
                },
            }
            for d, p in zip(devs, op.storage_pq, strict=True)
        ],
    }


def _pq_near(a, b, gap):
    return _near(a.kw, b.kw, gap) and _near(a.kvar, b.kvar, gap)


def _hold(live, previous, command, spec, emit):
    damped, fixed = load_damped(), load_fixed()
    control, storage = damped["controller"], damped["storage"]
    tol = spec["parity_abs"]
    before = collect(live, spec)
    emit({"phase": "hold_start", "command_kw": command, "prior_kw": previous,
          "live_start": _point(before)})
    predicted = prospective(live, previous, command, spec)
    if (
        not _near(predicted.clock_s, before.clock_s, tol["seconds"])
        or len(predicted.soc) != len(before.soc)
        or not all(_near(x, y, tol["soc"]) for x, y in zip(predicted.soc, before.soc))
    ):
        raise RuntimeError("prospective result was not at the actual pre-step state")
    emit({"phase": "hold_predicted_no_finish", "prediction": _point(predicted)})

    # Exactly one sanctioned real step: this driver calls FinishTimeStep once.
    # The scratch context never calls it. This trial does not retry a hold.
    row = live.step(command)
    actual = collect(live, spec)
    emit({"phase": "hold_observed_after_finish", "actual": _point(actual)})
    if (
        not _near(row.start_s, before.clock_s, tol["seconds"])
        or not _near(actual.clock_s, before.clock_s + control["exchange_s"], tol["seconds"])
        or not _near(row.end_s, actual.clock_s, tol["seconds"])
    ):
        raise RuntimeError("actual native interval did not advance exactly once")
    if not _near(row.active_balance_error_kw, 0, fixed["gates"]["feeder_balance_kw"]):
        raise RuntimeError("actual native PCC active power balance failed")
    if not _pq_near(row.feeder_import, actual.source_pq, tol["power_kw_or_kvar"]):
        raise RuntimeError("actual PCC no longer matches the accepted interval solution")
    if tuple(x.device for x in row.storage) != tuple(live.storage_ids):
        raise RuntimeError("unexpected actual terminal identities")
    # Compare actual electrical solution to prospective at same electrical
    # inputs, while testing the intentionally different clock/SOC separately.
    aligned = replace(actual, clock_s=before.clock_s, soc=before.soc)
    _assert_parity(predicted, aligned, spec)
    verify_device_envelope(aligned, command, spec, damped, fixed)
    ledger = []
    for i, reading in enumerate(row.storage):
        expected = (
            -actual.storage_pq[i].kw * control["exchange_s"]
            / (3600 * storage["discharge_efficiency"])
        )
        observed = (actual.soc[i] - before.soc[i]) * storage["capacity_kwh"]
        if (
            not _pq_near(reading.measured_injection, actual.storage_pq[i], tol["power_kw_or_kvar"])
            or not _near(reading.soc_before, before.soc[i], tol["soc"])
            or not _near(reading.soc_after, actual.soc[i], tol["soc"])
            or not storage["reserve_soc"] <= actual.soc[i] <= 1
            or not _near(observed, expected, fixed["storage"]["energy_abs_tol_kwh"])
        ):
            raise RuntimeError(f"{reading.device}: actual native energy ledger failed")
        ledger.append({"id": reading.device, "observed_delta_kwh": observed,
                       "expected_delta_kwh": expected, "residual_kwh": observed - expected})
    emit({"phase": "hold_verified", "ledger": ledger, "start_s": row.start_s,
          "end_s": row.end_s, "active_balance_kw": row.active_balance_error_kw})
    return actual


def trial(live, spec, emit):
    zero = dict.fromkeys(live.storage_ids, 0)
    joint = dict.fromkeys(live.storage_ids, 25)
    _hold(live, zero, joint, spec, emit)
    before = collect(live, spec)
    # Negative control: zero must NOT be accepted as the preceding delivered
    # operating point after the 4x25 kW step. No live step on this check.
    try:
        prospective(live, zero, zero, spec)
    except DeviceDispatchMismatch as err:
        if type(err) is not DeviceDispatchMismatch or err.__cause__ is not None:
            raise
        _assert_parity(before, collect(live, spec), spec)
        emit({"phase": "stale_reference_rejected", "at_s": before.clock_s,
              "reason": str(err)})
    else:
        raise RuntimeError("bad nonzero-time reference was accepted")
    _hold(live, joint, zero, spec, emit)
    return {"status": "two_native_holds_verified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with open(args.report, "x", encoding="utf-8",
              opener=lambda path, flags: os.open(path, flags, 0o600)) as result:
        def emit(record):
            result.write(json.dumps(record, allow_nan=False) + "\n")
            result.flush()

        emit({"phase": "source_manifest", "sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest()})
        target = json.loads(TRANSPORT.read_bytes())["source_anchor"]
        try:
            with Native123Feeder(PQ(target["kw"], target["kvar"])) as live:
                outcome = trial(live, envelope(), emit)
        except Exception as err:
            outcome = {"status": "failed", "reason": f"{type(err).__name__}: {err}"}
        emit({"phase": "outcome", **outcome})
    print(json.dumps(outcome))
    if outcome["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
