"""Run DSS-only 123 LFC feeder probe. Run as python -m scripts.check_lfc_123_native."""
from dataclasses import asdict
import json
import math

from envs.lfc.native_feeder_123 import Native123Feeder, PQ, stock_probe_spec


def run_probe():
    spec, scenario_sha256 = stock_probe_spec()
    target = PQ(spec["fixture_t0"]["kw"], spec["fixture_t0"]["kvar"])
    with Native123Feeder(target) as driver:
        reports = []
        trajectories = []
        for run in range(2):
            settled = driver.reset(target) if run else driver.calibration
            rows = []
            for pulse in spec["input_pulses"]:
                kw = dict(zip(driver.storage_ids, pulse["storage_kw"], strict=True))
                row = driver.step(kw)
                if not math.isclose(row.start_s, pulse["start_s"], abs_tol=1e-6):
                    raise RuntimeError("wrong interval start")
                if not math.isclose(row.end_s - row.start_s, 1, abs_tol=1e-6):
                    raise RuntimeError("DSS must advance by one second once")
                if abs(row.active_balance_error_kw) > 0.05:
                    raise RuntimeError("active power balance error exceeds 50 W")
                for item in row.storage:
                    injection = item.measured_injection.kw
                    if abs(injection - kw[item.device]) > 0.05:
                        raise RuntimeError(f"wrong signed active injection {item}")
                    if abs(item.measured_injection.kvar) > 0.05:
                        raise RuntimeError(f"nonzero reactive injection {item}")
                    dev = driver.session.batteries[item.device.split(".")[1]]
                    hours = (row.end_s - row.start_s) / 3600
                    expected = (
                        -injection * dev.eta_charge
                        if injection <= 0
                        else -injection / dev.eta_discharge
                    ) * hours
                    measured = (item.soc_after - item.soc_before) * dev.max_kwh
                    if abs(measured - expected) > 2e-5:
                        raise RuntimeError(
                            f"storage energy residual {measured-expected:g} kWh: {item}"
                        )
                rows.append(row)
            base, dis, chg, idle = rows
            for zero in (base, idle):
                for component in ("kw", "kvar"):
                    if (
                        abs(
                            getattr(zero.equivalent_transmission_load, component)
                            - getattr(target, component)
                        )
                        > 0.05
                    ):
                        raise RuntimeError("zero-dispatch load anchor drifted")
            if not dis.feeder_import.kw < base.feeder_import.kw < chg.feeder_import.kw:
                raise RuntimeError("source import responds in wrong direction")
            trajectories.append([asdict(r) for r in rows])
            reports.append(asdict(settled))
        if trajectories[0] != trajectories[1] or reports[0] != reports[1]:
            raise RuntimeError("two reset-then-replay runs are not identical")
        return dict(
            status="ok",
            scope=spec["scope"],
            real_matlab_equilibrium_validated=False,
            scenario_sha256=scenario_sha256,
            deck_model_sha256=driver.session.model_sha256,
            engine=driver.session.manifest()["engine"],
            calibration=reports[0],
            rows=trajectories[0],
            repeated_runs=2,
        )


if __name__ == "__main__":
    print(json.dumps(run_probe(), indent=2, allow_nan=False))
