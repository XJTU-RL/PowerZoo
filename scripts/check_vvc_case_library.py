"""核对两个参考场景的原生读回与大馈线单点潮流。"""

import argparse
from contextlib import closing
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from envs.powergrid.case_inventory import dependency_closure, node_voltages
from envs.powergrid.config import DataConfig
from envs.powergrid.scenario import COLUMNS, UNITS, VERSION
from envs.powergrid.session import DSSSession
from envs.vvc.vvc.env_register import make_base_env
from envs.vvc.vvc.vvc_config import VVCConfig
from scripts.vvc_case_library import ROOT, load_catalog, stage_reference
from utils.config_pipeline import ConfigLoader


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def probe_input(output):
    """只写单条公开合成 validation 输入，不调用三分集合成器。"""
    csv = output / "validation_probe.csv"
    csv.write_bytes((",".join(COLUMNS) + "\n0,0.7,0.7,0.7,0,20\n").encode())
    manifest = output / "validation_probe.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": VERSION,
                "units": UNITS,
                "origin": "PowerZoo VVC case library synthetic 0.70 nameplate snapshot",
                "episodes": [
                    {
                        "id": "validation_probe",
                        "split": "validation",
                        "scenario": "physical_readback",
                        "file": csv.name,
                        "sha256": sha256(csv.read_bytes()).hexdigest(),
                        "steps": 1,
                        "step_seconds": 3600,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return DataConfig(
        source="files", split="validation", path=str(manifest), step_seconds=3600
    )


def phase_voltages(session):
    voltages = {
        key: value
        for key, value in node_voltages(session.dss).items()
        if key.rpartition(".")[2] in ("1", "2", "3")
    }
    require(voltages and all(map(math.isfinite, voltages.values())), "相电压读数无效")
    return voltages


def voltage_summary(voltages, low, high):
    return {
        "min_pu": min(voltages.values()),
        "max_pu": max(voltages.values()),
        "below_limit": sum(value < low for value in voltages.values()),
        "above_limit": sum(value > high for value in voltages.values()),
        "phase_nodes": len(voltages),
        "zero_voltage_nodes": [name for name, value in voltages.items() if value == 0],
    }


def line_readback(session, expected_unsourced):
    lines = [
        name
        for name in session.native.AllElementNames
        if name.lower().startswith("line.")
    ]
    if expected_unsourced is None:
        return {"line_count": len(lines), "thermal_status": "unverified"}
    closure = dependency_closure(session.folder, session.master)
    for name in closure["files"]:
        text = (session.folder / name).read_text(encoding="utf-8")
        for raw in text.splitlines():
            command = re.split(r"!|//", raw, maxsplit=1)[0]
            require(
                not re.search(
                    r"\b(?:normamps|emer(?:g)?amps|geometry|wiredata)\s*=",
                    command,
                    re.I,
                ),
                f"闭包中发现需重新审计的额定或导线定义：{name}",
            )
    require(len(lines) == expected_unsourced, "无来源额定的线路数与登记不符")
    engine_values, peak = set(), 0.0
    for name in lines:
        row = session.element_info(name)
        engine_values.add(row["normal_amps"])
        phase_currents = [
            amps
            for node, amps in zip(row["nodes"], row["currents_a"], strict=True)
            if node in (1, 2, 3)
        ]
        require(
            phase_currents and all(map(math.isfinite, phase_currents)),
            f"{name} 相电流无效",
        )
        peak = max(peak, *phase_currents)
    require(engine_values == {400.0}, "123 软件默认额定读回与锁定证据不符")
    return {
        "line_count": len(lines),
        "lines_without_sourced_limits": expected_unsourced,
        "software_default_normal_amps": sorted(engine_values),
        "peak_phase_current_a": peak,
        "thermal_status": "no_sourced_limits",
        "thermal_overload_count": None,
    }


def reference_endpoint(config, scenario):
    with closing(make_base_env(config)) as env:
        env.reset(seed=17, load_profile_idx=0)
        session = env.circuit
        storage_buses = {
            name: session.connection(f"Storage.{name}")["buses"][0]
            for name in session.batteries
        }
        require(storage_buses == scenario["storage_buses"], "储能母线映射不符")
        require(len(session.capacitors) == scenario["capacitor_count"], "电容数量不符")
        require(
            len(session.regulators) == scenario["regulator_count"], "调压器数量不符"
        )
        require(len(session.pvs) == scenario["pv_count"], "PV 数量不符")
        initial = phase_voltages(session)

        cap_name = env.cap_names[0]
        reg_name = env.reg_names[0]
        battery_name = env.bat_names[-1]
        cap_values = [session.capacitor_state(name)[0] for name in env.cap_names]
        cap_values[0] = 1 - cap_values[0]
        tap_values = []
        for name in env.reg_names:
            reg = session.regulators[name]
            tap = session.regulator_tap(name)
            index = (
                (tap - reg.mintap)
                * (config.reg_act_num - 1)
                / (reg.maxtap - reg.mintap)
            )
            require(
                math.isclose(index, round(index), abs_tol=1e-6), "初态未对齐原生档位"
            )
            tap_values.append(round(index))
        tap_values[0] += 1 if tap_values[0] < config.reg_act_num - 1 else -1
        battery_values = [config.bat_act_num // 2] * len(env.bat_names)
        battery_values[-1] = config.bat_act_num - 1
        binding = env.get_physics_manifest()
        _, _, done, info = env.step(cap_values + tap_values + battery_values)
        require(
            done and info.get("dss_converged") and not info.get("physical_failure"),
            "参考潮流未完成",
        )
        require(info["time_s"] == 3600, "区间未推进一小时")
        require(
            abs(env.physical_state.active_balance_error_kw) < 0.05, "有功平衡残差超限"
        )
        control = info["control_trace"]
        capacitor = control[f"Capacitor.{cap_name}"]
        regulator = control[f"RegControl.{reg_name}"]
        storage = control[f"Storage.{battery_name}"]
        require(capacitor["measured"] == capacitor["requested"], "电容原生读回不符")
        require(
            math.isclose(
                regulator["measured"]["tap"],
                regulator["requested"]["tap"],
                abs_tol=1e-8,
            ),
            "调压器绕组原生读回不符",
        )
        require(
            math.isclose(storage["requested"]["kw"], 100, abs_tol=1e-6)
            and math.isclose(storage["applied"]["kw"], 76, abs_tol=1e-6)
            and math.isclose(storage["measured"]["kw"], 76, abs_tol=1e-5)
            and abs(storage["measured"]["kvar"]) < 1e-5
            and math.isclose(session.batteries[battery_name].soc, 0.10, abs_tol=1e-8),
            "储能功率、端口无功或 SOC 读回不符",
        )
        return {
            "engine": session.dss.Version,
            "dss_model_sha256": binding["model_sha256"],
            "action_schema": binding["action_schema"],
            "regulator_tap_bounds": {
                name: {
                    "min": reg.mintap,
                    "max": reg.maxtap,
                    "native_intervals": reg.numtaps,
                }
                for name, reg in session.regulators.items()
            },
            "storage_ratings": binding["storage"],
            "initial_voltages": initial,
            "final_voltages": phase_voltages(session),
            "final_grid_kw_kvar": env.physical_state.grid_import,
            "final_loss_kw_kvar": env.physical_state.losses,
            "balance_error_kw": env.physical_state.active_balance_error_kw,
            "time_s": info["time_s"],
            "storage_buses": storage_buses,
            "storage_soc": env.physical_state.storage_soc,
            "changed_devices": {
                key: control[key]
                for key in (
                    f"Capacitor.{cap_name}",
                    f"RegControl.{reg_name}",
                    f"Storage.{battery_name}",
                )
            },
            "lines": line_readback(
                session, scenario.get("lines_without_sourced_limits")
            ),
        }


def reference_case(case_id, case, destination, data):
    loader = ConfigLoader()
    staged_experiment = stage_reference(case_id, destination / case_id)
    configs = [
        replace(
            VVCConfig.from_env_args(loader.load(experiment=path).env_args),
            max_episode_steps=1,
            data=data,
        )
        for path in (case["legacy_experiment"], staged_experiment)
    ]
    legacy, staged = (reference_endpoint(cfg, case["scenario"]) for cfg in configs)
    for key in (
        "action_schema",
        "regulator_tap_bounds",
        "storage_ratings",
        "storage_buses",
        "changed_devices",
    ):
        require(legacy[key] == staged[key], f"旧版与独立叠加层的 {key} 不同")
    max_voltage_deltas = {}
    for key in ("initial_voltages", "final_voltages"):
        a, b = legacy[key], staged[key]
        require(a.keys() == b.keys(), f"旧版与独立叠加层的 {key} 相节点不同")
        delta = max(abs(a[node] - b[node]) for node in a)
        require(delta <= 2e-6, f"旧版与独立叠加层的 {key} 差值超限")
        max_voltage_deltas[key] = delta
        limits = configs[1].voltage_min, configs[1].voltage_max
        legacy[key], staged[key] = (
            voltage_summary(a, *limits),
            voltage_summary(b, *limits),
        )
    for key in ("final_grid_kw_kvar", "final_loss_kw_kvar"):
        require(
            all(
                math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
                for a, b in zip(legacy[key], staged[key], strict=True)
            ),
            f"{key} 不同",
        )
    with closing(DSSSession(configs[1].dss_folder, configs[1].dss_file)) as bare:
        require(not bare.batteries and not bare.pvs, "无叠加电路包含场景资产")
        require(
            len(bare.capacitors) == case["scenario"]["capacitor_count"]
            and len(bare.regulators) == case["scenario"]["regulator_count"],
            "无叠加电路的原生设备数量不符",
        )
    return {
        "bare_storage_count": 0,
        "voltage_max_abs_deltas_pu": max_voltage_deltas,
        "legacy": legacy,
        "staged": staged,
    }


def scale_snapshot(config, multiplier):
    with closing(DSSSession(ROOT / config.dss_folder, config.dss_file)) as session:
        solution = session.native.Solution
        report = {
            "master": config.dss_file,
            "load_multiplier": multiplier,
            "load_model_counts": {},
            "pv_count": len(session.pvs),
            "storage_count": len(session.batteries),
            "regulator_count": len(session.regulators),
            "model_sha256": session.model_sha256,
            "engine": session.dss.Version,
            "snapshot_algorithm": solution.Algorithm.name,
            "snapshot_mode": solution.ModeID,
            "max_iterations": int(solution.MaxIterations),
            "weather": {"irradiance_kw_m2": 0, "temperature_c": 20},
            "initial_taps": {
                name: session.regulator_tap(name) for name in session.regulators
            },
        }
        for name in session.loads:
            model = session.read_property(f"Load.{name}", "model")
            counts = report["load_model_counts"]
            counts[model] = counts.get(model, 0) + 1
        session.apply(
            {
                f"Load.{name}": {
                    "kw": device.kw * multiplier,
                    "kvar": device.kvar * multiplier,
                }
                for name, device in session.loads.items()
            }
        )
        session.set_pv_conditions(0, 20)
        try:
            session.solve()
        except RuntimeError as exc:
            if not str(exc).endswith("未收敛") or solution.Converged:
                raise
            report["failure_reason"] = str(exc)
        converged = bool(solution.Converged)
        report.update(
            converged=converged,
            iterations=int(solution.Iterations),
            time_s=session.time_s,
        )
        if converged:
            report["voltages"] = voltage_summary(
                phase_voltages(session), config.voltage_min, config.voltage_max
            )
            report["loss_kw_kvar"] = session.total_loss().tolist()
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="新的专用输出目录")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    catalog = load_catalog()
    data = probe_input(output)
    loader = ConfigLoader()
    report = {
        "schema_version": 1,
        "base_revision": catalog["base_revision"],
        "probe_manifest_sha256": sha256(Path(data.path).read_bytes()).hexdigest(),
        "cases": {},
    }
    report_path = output / "report.json"
    for key, case in catalog["cases"].items():
        if case["role"] == "reference":
            result = reference_case(key, case, output, data)
        elif case["role"] == "scale_power_flow_diagnostic":
            model = loader.load(algo="happo", env="vvc", system_ref=case["system_ref"])
            config = VVCConfig.from_env_args(model.env_args)
            expected = case.get("expected_nominal") != "non_convergence"
            result = {"nominal": scale_snapshot(config, 1.0)}
            if not expected:
                result["reduced"] = scale_snapshot(config, 0.85)
        else:
            continue
        report["cases"][key] = result
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        if case["role"] == "scale_power_flow_diagnostic":
            require(
                result["nominal"]["converged"] == expected
                and result.get("reduced", {"converged": True})["converged"],
                f"{key} 潮流结果与登记的诊断边界不符，见 {report_path}",
            )
        print(f"核对完成 {key}", flush=True)


if __name__ == "__main__":
    main()
