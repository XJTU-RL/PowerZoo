"""Measure low-power storage absorption in the unmodified IEEE123 DSS model."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess

from envs.lfc.native_123_availability import collect, envelope
from envs.lfc.native_feeder_123 import Native123Feeder, PQ, stock_probe_spec
from scripts.check_lfc_123_availability_dynamic import _point
from scripts.check_lfc_123_coupling import SCENARIO

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "scenarios/lfc/m3/storage_absorption_v1.json"


def rating(session, device):
    fields = ("kWRated", "kWhRated", "kVA", "%reserve", "%EffCharge", "%EffDischarge",
              "%IdlingkW", "%charge", "%discharge", "%stored")
    return {key: float(session.read_property(device, key)) for key in fields}


def inspect_hold(feeder, request_kw, cfg, ratings):
    spec = envelope()
    before = collect(feeder, spec)
    interval = feeder.step(dict.fromkeys(feeder.storage_ids, request_kw))
    after = collect(feeder, spec)
    measured = _point(after)
    devices = stock_probe_spec()[0]["storage_devices"]
    veto = []
    evidence = []
    dt = interval.end_s - interval.start_s
    if abs(dt - cfg["hold_s"]) > 1e-7 or before.state != after.state:
        veto.append("clock_or_mechanical_state")
    if abs(interval.active_balance_error_kw) > cfg["max_p_balance_kw"]:
        veto.append("feeder_active_balance")
    for index, (d, row, obs) in enumerate(zip(devices, interval.storage, measured["storage"], strict=True)):
        name = d["id"]
        native = ratings[name]
        p, q = row.measured_injection.kw, row.measured_injection.kvar
        energy = (after.soc[index] - before.soc[index]) * native["kWhRated"]
        efficiency = native["%EffCharge"]/100 if p < 0 else 100/native["%EffDischarge"]
        expected = -p * efficiency * dt / 3600
        voltage = obs["voltage_pu"]
        amps = max(obs["conductor_a"].values())
        current_ceiling = native["kVA"] / (d["kv"] * (math.sqrt(3) if len(d["phases"]) == 3 else 1))
        state = feeder.session.read_property(name, "state").strip().lower()
        logical_state = "discharging" if request_kw > 0 else "charging" if request_kw < 0 else "idling"
        native_rate = float(feeder.session.read_property(name, "%charge" if request_kw < 0 else "%discharge"))
        checks = {
            "requested_applied_terminal": abs(row.applied_kw-request_kw) <= 1e-7
                and abs(p-request_kw) <= cfg["max_terminal_p_error_kw"]
                and abs(p-obs["kw"]) <= 0.001 and abs(q-obs["kvar"]) <= 0.001,
            "native_state_rate": state == logical_state
                and abs(native_rate*native["kWRated"]/100-abs(request_kw)) <= 1e-7,
            "reactive": abs(q) <= cfg["max_terminal_q_kvar"],
            "energy": abs(energy-expected) <= cfg["max_energy_error_kwh"]
                and (request_kw == 0 or energy*request_kw < 0),
            "capacity": native["%reserve"]/100 <= after.soc[index] <= 1,
            "inverter": math.hypot(p, q) <= native["kVA"] and amps <= current_ceiling,
            "voltage": min(voltage) >= spec["terminal_voltage_pu"][0]
                and max(voltage) <= spec["terminal_voltage_pu"][1],
        }
        veto.extend(f"{name}:{key}" for key, ok in checks.items() if not ok)
        evidence.append(dict(id=name, applied_kw=row.applied_kw, p_kw=p, q_kvar=q, native_state=state,
                             rate_percent=native_rate, soc_before=before.soc[index], soc_after=after.soc[index],
                             delta_kwh=energy, expected_kwh=expected, vmin_pu=min(voltage), vmax_pu=max(voltage),
                             amps=amps, current_ceiling_a=current_ceiling,
                             geometric_q_headroom_kvar=math.sqrt(max(0, native["kVA"]**2-p**2))))
    return dict(start_s=interval.start_s, end_s=interval.end_s, request_per_device_kw=request_kw,
                pcc=asdict(interval.feeder_import), balance_kw=interval.active_balance_error_kw,
                devices=evidence, veto=veto)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    target = parser.parse_args().output
    cfg = json.loads(PROTOCOL.read_bytes())
    source = json.loads(SCENARIO.read_bytes())["source_anchor"]
    anchor = PQ(source["kw"], source["kvar"])
    neutral = None
    failed = False
    with target.open("x") as stream:
        def record(data):
            stream.write(json.dumps(data, allow_nan=False) + "\n")
            stream.flush()
        record(dict(phase="protocol", git=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],
                                                                  text=True).strip(),
                    protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()))
        for index, pulse in enumerate(cfg["per_device_kw"]):
            with Native123Feeder(anchor) as feeder:
                ratings = {name: rating(feeder.session, name) for name in feeder.storage_ids}
                record(dict(phase="setup", arm=index, pulse=pulse, ratings=ratings,
                            deck_sha256=feeder.session.model_sha256))
                if any(abs(x["%stored"]/100-cfg["initial_soc"]) > 1e-9 for x in ratings.values()):
                    raise ValueError("unmatched initial SOC")
                holds = []
                for i, phase in enumerate(cfg["intervals"]):
                    requested = pulse if phase == "pulse" else 0
                    row = inspect_hold(feeder, requested, cfg, ratings)
                    if neutral is not None:
                        row["paired_pcc_kw"] = neutral[i]["pcc"]["kw"] - row["pcc"]["kw"]
                        row["paired_pcc_kvar"] = neutral[i]["pcc"]["kvar"] - row["pcc"]["kvar"]
                        if requested:
                            ratio = row["paired_pcc_kw"] / (len(ratings)*requested)
                            if not cfg["min_pcc_delivery_fraction"] <= ratio <= cfg["max_pcc_delivery_fraction"]:
                                row["veto"].append("signed_pcc_delivery")
                        elif abs(row["paired_pcc_kw"]) > cfg["max_zero_return_kw"]:
                            row["veto"].append("pcc_release")
                        if pulse == 0 and max(abs(row["paired_pcc_kw"]),abs(row["paired_pcc_kvar"])) > cfg["neutral_jitter_kw_kvar"]:
                            row["veto"].append("neutral_jitter")
                    record(dict(phase="hold", arm=index, stage=phase, **row))
                    holds.append(row)
                    if row["veto"]:
                        failed = True
                        break
                if neutral is None:
                    neutral = holds
            if failed:
                break
        record(dict(phase="decision", passed=not failed, attempted_arms=index+1))
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
