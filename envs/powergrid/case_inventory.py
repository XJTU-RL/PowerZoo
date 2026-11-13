"""外部算例结构清单：文本依赖闭包、引擎口径盘点与 A/B 结果对照。

依赖闭包按 OpenDSS 脚本语义解析 ``redirect``/``include``（含 ``~`` 续行与
反斜杠路径），逐文件记录 SHA-256；缺失依赖精确报 ``文件:行号``。盘点取自
已编译（可选已求解）的 DSS 上下文，区分“设备存在”与“原生控制器绑定”，
不涉及智能体控制——后者由系统 YAML 的 devices 段声明。

功率口径：TotalPower/Powers 为 kW（源端注入为负）；Losses 为 W，除以 1000
记 kW；电压 puVmag 以模型 voltagebases 为基值。
"""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import math
import re

import numpy as np


def _strip_comment(line):
    for marker in ("!", "//"):
        position = line.find(marker)
        if position >= 0:
            line = line[:position]
    return line.strip()


def _joined_commands(text):
    """按 ``~`` 续行拼接命令，产出 (起始行号, 命令文本)。"""
    buffer, start = [], 1
    for number, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw)
        if not buffer:
            start = number
        if not line:
            continue
        if line.endswith("~"):
            buffer.append(line[:-1].rstrip())
            continue
        buffer.append(line)
        yield start, " ".join(buffer)
        buffer = []
    if buffer:  # 文件末尾悬空续行：仍作为命令产出
        yield start, " ".join(buffer)


def dependency_closure(model_dir, master):
    """从主入口出发收集 redirect/include 与 LoadShape ``file=`` 依赖闭包。

    返回 ``{"entry", "files", "missing", "notes"}``；missing 精确到引用处的
    ``相对路径:行号``。``compile`` 命令只记入 notes（其语义是新建电路而非
    引用依赖）。数据文件（``mult=(file=..)`` 等）记为 ``kind="datafile"`` 且
    不再递归解析。路径统一用 ``/``；解析相对当前文件目录。
    """
    model_dir = Path(model_dir).resolve()
    entry = model_dir / master
    if not entry.is_file():
        raise FileNotFoundError(f"主入口不存在：{entry}")

    files, missing, notes, stack = {}, [], [], [(entry, f"{master}:1", "script")]
    visited = set()
    while stack:
        path, origin, kind = stack.pop()
        key = path.relative_to(model_dir).as_posix()
        if key in visited:
            continue
        visited.add(key)
        if not path.is_file():
            missing.append({"from": origin, "path": key, "kind": kind})
            continue
        files[key] = {
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "kind": kind,
        }
        if kind != "script":
            continue  # 数据文件（如 LoadShape CSV）不再递归解析
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, command in _joined_commands(text):
            parts = command.split(None, 1)
            head = parts[0].lower() if parts else ""
            rest = parts[1].strip().strip('"') if len(parts) > 1 else ""
            if head in ("redirect", "include"):
                target = (path.parent / rest.replace("\\", "/")).resolve()
                if not target.is_relative_to(model_dir):
                    missing.append(
                        {"from": f"{key}:{line_number}", "path": rest, "reason": "越界"}
                    )
                    continue
                stack.append((target, f"{key}:{line_number}", "script"))
            elif head == "compile":
                notes.append({"from": f"{key}:{line_number}", "command": command})
            elif head == "new":
                # LoadShape 等数据驱动元件：mult=(file=..)/qmult=(file=..)
                # 指向的 CSV 是运行依赖（SMART-DS 布局根下的 profiles/）。
                for match in re.finditer(r"file=([^)\s]+)", rest):
                    reference = match.group(1).strip('"')
                    target = (path.parent / reference.replace("\\", "/")).resolve()
                    if not target.is_relative_to(model_dir):
                        missing.append(
                            {
                                "from": f"{key}:{line_number}",
                                "path": reference,
                                "reason": "越界",
                                "kind": "datafile",
                            }
                        )
                        continue
                    stack.append(
                        (target, f"{key}:{line_number}", "datafile")
                    )
    return {"entry": master, "files": files, "missing": missing, "notes": notes}


def node_voltages(ctx):
    """全部电气节点 pu 电压，键为 ``母线.节点`` 全名。"""
    circuit = ctx.ActiveCircuit
    names, values = list(circuit.AllNodeNames), list(circuit.AllBusVmagPu)
    if len(names) != len(values) or not names:
        raise RuntimeError(
            f"节点电压口径异常：{len(names)} 个节点名对 {len(values)} 个电压值"
        )
    return dict(zip(names, map(float, values)))


def _voltage_summary(voltages, limits):
    phase = {k: v for k, v in voltages.items() if k.rpartition(".")[2] in ("1", "2", "3")}
    values = np.fromiter(phase.values(), dtype=float, count=len(phase))
    low, high = limits
    return {
        "phase_nodes": int(values.size),
        "all_nodes": len(voltages),
        "min_pu": float(values.min()),
        "max_pu": float(values.max()),
        "mean_pu": float(values.mean()),
        f"below_{low}_pu": int((values < low).sum()),
        f"above_{high}_pu": int((values > high).sum()),
    }


def _branch_loss_sum(ctx):
    """线路+变压器逐元件损耗求和（W→kW），交叉验证 Circuit.Losses 口径。"""
    circuit = ctx.ActiveCircuit
    total = np.zeros(2)
    for element in circuit.AllElementNames:
        if not element.lower().startswith(("line.", "transformer.")):
            continue
        circuit.SetActiveElement(element)
        total += np.asarray(circuit.ActiveCktElement.Losses, dtype=float)
    return (total / 1000).tolist()


def inventory(ctx, label, *, solved=True, limits=(0.95, 1.05), losses_by_elements=False):
    """对已编译（默认已求解）的 DSS 上下文做引擎口径盘点。

    ``ctx`` 为 ``dss.DSS.NewContext()`` 上下文；``solved=True`` 时额外记录
    收敛/迭代/电压/功率段，未收敛会抛错以便区分“结构可载、求解失败”。
    """
    circuit = ctx.ActiveCircuit
    solution = circuit.Solution
    from importlib.metadata import version as package_version

    def class_names(class_name):
        circuit.SetActiveClass(class_name)
        interface = ctx.ActiveClass
        # 空类的 AllNames 会返回占位值 ['NONE']，统一以 Count 挡掉。
        return list(interface.AllNames) if interface.Count else []

    def names_of(interface):
        return list(interface.AllNames) if interface.Count else []

    def element_flag(full_name):
        circuit.SetActiveElement(full_name)
        return circuit.ActiveCktElement

    report = {
        "label": label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": {
            "version": ctx.Version,
            "dss_python": package_version("dss-python"),
        },
        "circuit": {
            "name": circuit.Name,
            "frequency_hz": float(solution.Frequency),
            "num_buses": int(circuit.NumBuses),
            "num_nodes": int(circuit.NumNodes),
            "voltage_bases_kv": [
                float(v) for v in circuit.Settings.VoltageBases
            ],
        },
    }

    sources = []
    for name in class_names("Vsource"):
        obj = element_flag(f"vsource.{name}")
        properties = obj.Properties
        sources.append(
            {
                "name": name,
                "bus": obj.BusNames[0],
                "base_kv": float(properties("basekv").Val),
                "pu": float(properties("pu").Val),
                "enabled": bool(obj.Enabled),
            }
        )
    report["sources"] = sources

    loads = circuit.Loads
    load_stats = {
        "total": 0,
        "enabled": 0,
        "disabled": 0,
        "kw_enabled": 0.0,
        "kvar_enabled": 0.0,
        "kw_disabled": 0.0,
        "phases": {},
    }
    disabled_load_names = []
    for name in names_of(loads):
        loads.Name = name
        obj = element_flag(f"load.{name}")
        load_stats["total"] += 1
        phases = int(obj.NumPhases)
        load_stats["phases"][str(phases)] = load_stats["phases"].get(str(phases), 0) + 1
        if obj.Enabled:
            load_stats["enabled"] += 1
            load_stats["kw_enabled"] += loads.kW
            load_stats["kvar_enabled"] += loads.kvar
        else:
            load_stats["disabled"] += 1
            load_stats["kw_disabled"] += loads.kW
            disabled_load_names.append(name)
    report["loads"] = {
        **load_stats,
        "disabled_names": disabled_load_names,
    }

    transformers, regcontrols = circuit.Transformers, circuit.RegControls
    regulators = []
    for name in names_of(regcontrols):
        regcontrols.Name = name
        bound = regcontrols.Transformer
        transformers.Name, transformers.Wdg = bound, regcontrols.Winding
        # Properties 视图随“当前活跃元件”惰性解析，必须立即取值；
        # MinTap/MaxTap 接口在 C-API 下不可靠，改读元件属性（与 DSS 文件同源）。
        bound_properties = element_flag(f"transformer.{bound}").Properties
        mintap = float(bound_properties("mintap").Val)
        maxtap = float(bound_properties("maxtap").Val)
        tap_ratio = float(transformers.Tap)
        enabled = bool(element_flag(f"regcontrol.{name}").Enabled)
        regulators.append(
            {
                "name": name,
                "transformer": bound,
                "winding": int(regcontrols.Winding),
                "enabled": enabled,
                "vreg": float(regcontrols.ForwardVreg),
                "band": float(regcontrols.ForwardBand),
                "ptratio": float(regcontrols.PTratio),
                "maxtapchange": int(regcontrols.MaxTapChange),
                "reversible": bool(regcontrols.IsReversible),
                "monitored_bus": regcontrols.MonitoredBus,
                "numtaps": int(transformers.NumTaps),
                "mintap": mintap,
                "maxtap": maxtap,
                "tap_ratio": tap_ratio,
                "tap_number": int(regcontrols.TapNumber),
            }
        )
    report["regcontrols"] = regulators
    report["capcontrols"] = [
        {
            "name": name,
            "enabled": bool(element_flag(f"capcontrol.{name}").Enabled),
        }
        for name in class_names("CapControl")
    ]
    capacitors = circuit.Capacitors
    capacitor_list = []
    for name in names_of(capacitors):
        capacitors.Name = name
        capacitor_list.append(
            {
                "name": name,
                "enabled": bool(element_flag(f"capacitor.{name}").Enabled),
                "states": [int(s) for s in capacitors.States],
            }
        )
    report["capacitors"] = capacitor_list

    pvsystems = circuit.PVSystems
    report["pvsystems"] = [
        {"name": name, "enabled": bool(element_flag(f"pvsystem.{name}").Enabled)}
        for name in names_of(pvsystems)
    ]
    storage_list = []
    for name in names_of(circuit.Storages):
        obj = element_flag(f"storage.{name}")
        properties = obj.Properties
        storage_list.append(
            {
                "name": name,
                "enabled": bool(obj.Enabled),
                "kwrated": float(properties("kWrated").Val),
                "kwhstored": float(properties("kWhstored").Val),
                "kwhrating": float(properties("kWhrating").Val),
            }
        )
    report["storages"] = storage_list
    report["generators"] = [
        {"name": name, "enabled": bool(element_flag(f"generator.{name}").Enabled)}
        for name in names_of(circuit.Generators)
    ]

    line_stats = {"total": 0, "enabled": 0, "disabled": 0, "open_elements": 0}
    open_lines = []
    for element in circuit.AllElementNames:
        if not element.lower().startswith("line."):
            continue
        obj = element_flag(element)
        line_stats["total"] += 1
        if not obj.Enabled:
            line_stats["disabled"] += 1
            continue
        line_stats["enabled"] += 1
        if any(
            obj.IsOpen(terminal, conductor)
            for terminal in range(1, obj.NumTerminals + 1)
            for conductor in range(1, obj.NumConductors + 1)
        ):
            line_stats["open_elements"] += 1
            open_lines.append(element)
    report["lines"] = {**line_stats, "open_names": open_lines}

    buses_by_phase = {str(n): 0 for n in (1, 2, 3)}
    for bus in circuit.AllBusNames:
        circuit.SetActiveBus(bus)
        phase_nodes = sum(
            1 for node in circuit.ActiveBus.Nodes if node in (1, 2, 3)
        )
        key = str(phase_nodes) if phase_nodes in (1, 2, 3) else "0"
        buses_by_phase[key] = buses_by_phase.get(key, 0) + 1
    report["buses_by_phase_nodes"] = buses_by_phase

    report["solver"] = {
        "converged": bool(solution.Converged),
        "iterations": int(solution.Iterations),
        "control_iterations": int(solution.ControlIterations),
        "mode": solution.ModeID,
        "algorithm": int(solution.Algorithm),
        "max_iterations": int(solution.MaxIterations),
        "tolerance": float(solution.Tolerance),
        "control_mode": int(solution.ControlMode),
    }
    if solved:
        if not solution.Converged:
            raise RuntimeError(
                f"[{label}] 电路未收敛，拒绝读取电压/功率段；"
                "如只需结构清单请以 solved=False 重跑"
            )
        report["voltage_summary"] = _voltage_summary(node_voltages(ctx), limits)
        report["power"] = {
            # TotalPower 负值=源端注入；Losses 原始单位 W，统一换算 kW。
            "total_power_kw_kvar": [
                float(-circuit.TotalPower[0]),
                float(-circuit.TotalPower[1]),
            ],
            "losses_kw_kvar": (
                np.asarray(circuit.Losses, dtype=float) / 1000
            ).tolist(),
        }
        if losses_by_elements:
            report["power"]["branch_loss_sum_kw_kvar"] = _branch_loss_sum(ctx)
    return report


def compare(a, b, *, voltage_atol=2e-6, power_rtol=1e-6):
    """对照两份已求解盘点：逐节点电压、源端 P/Q、损耗、元件数与调压器档位。

    公差依据：同一引擎同一模型两处编译（A 直载 / B 经 DSSSession 临时副本），
    数值路径一致，电压差异应达浮点噪声量级（≤1e-6 pu）；超出即如实报 mismatch。
    """
    def count_diffs():
        keys = (
            ("loads", "total"),
            ("lines", "total"),
            ("regcontrols", None),
            ("capacitors", None),
            ("pvsystems", None),
            ("storages", None),
            ("generators", None),
        )
        diffs = {}
        for section, field in keys:
            left, right = a[section], b[section]
            left_value = len(left) if field is None else left[field]
            right_value = len(right) if field is None else right[field]
            if left_value != right_value:
                diffs[section] = {"a": left_value, "b": right_value}
        for key in ("num_buses", "num_nodes"):
            if a["circuit"][key] != b["circuit"][key]:
                diffs[f"circuit.{key}"] = {
                    "a": a["circuit"][key],
                    "b": b["circuit"][key],
                }
        return diffs

    voltages_a, voltages_b = a.pop("_node_voltages", {}), b.pop("_node_voltages", {})
    shared = sorted(set(voltages_a) & set(voltages_b))
    only_a = sorted(set(voltages_a) - set(voltages_b))
    only_b = sorted(set(voltages_b) - set(voltages_a))
    if shared:
        delta = np.array(
            [abs(voltages_a[k] - voltages_b[k]) for k in shared], dtype=float
        )
        worst = int(delta.argmax())
        voltage_result = {
            "compared_nodes": len(shared),
            "only_in_a": len(only_a),
            "only_in_b": len(only_b),
            "max_abs_diff_pu": float(delta.max()),
            "mean_abs_diff_pu": float(delta.mean()),
            "worst_node": shared[worst] if delta.size else None,
        }
    else:
        voltage_result = {"compared_nodes": 0, "only_in_a": len(only_a), "only_in_b": len(only_b)}

    def power_delta(field):
        left, right = a["power"][field], b["power"][field]
        scale = max(abs(left[0]), abs(left[1]), 1.0)
        return {
            "a": left,
            "b": right,
            "max_rel_diff": max(
                abs(left[0] - right[0]) / scale, abs(left[1] - right[1]) / scale
            ),
        }

    taps_a = {r["name"]: r["tap_ratio"] for r in a["regcontrols"]}
    taps_b = {r["name"]: r["tap_ratio"] for r in b["regcontrols"]}
    open_a = set(a["lines"]["open_names"])
    open_b = set(b["lines"]["open_names"])
    result = {
        "voltage_atol_pu": voltage_atol,
        "power_rtol": power_rtol,
        "node_voltages": voltage_result,
        "source_pq": power_delta("total_power_kw_kvar"),
        "losses": power_delta("losses_kw_kvar"),
        "count_diffs": count_diffs(),
        "regcontrol_taps_equal": taps_a == taps_b,
        "line_open_state_mismatches": sorted(open_a ^ open_b),
        "solver_a": a["solver"],
        "solver_b": b["solver"],
    }
    result["match"] = (
        voltage_result.get("compared_nodes", 0) > 0
        and voltage_result["only_in_a"] == 0
        and voltage_result["only_in_b"] == 0
        and voltage_result["max_abs_diff_pu"] <= voltage_atol
        and result["source_pq"]["max_rel_diff"] <= power_rtol
        and result["losses"]["max_rel_diff"] <= power_rtol
        and not result["count_diffs"]
        and result["regcontrol_taps_equal"]
        and not result["line_open_state_mismatches"]
    )
    return result
