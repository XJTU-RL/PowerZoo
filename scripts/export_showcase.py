"""导出展示页使用的 IEEE13 原生稳态扫掠，不执行策略训练。"""

import argparse
import csv
from hashlib import sha256
from importlib.metadata import version
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.powergrid.session import DSSSession


def coordinates(model_folder, buses):
    """使用模型自带坐标，保留源端及调压器辅助母线，不虚构坐标。"""
    with (model_folder / "IEEE13Node_BusXY.csv").open(encoding="utf-8") as handle:
        points = {
            row[0].strip().lower(): (float(row[1]), float(row[2]))
            for row in csv.reader(handle)
            if row
        }
    missing = set(buses) - points.keys()
    if missing:
        raise ValueError(f"母线缺少原始坐标：{sorted(missing)}")
    xs, ys = zip(*(points[name] for name in buses))
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    if width <= 0 or height <= 0:
        raise ValueError("模型坐标不能归一化")
    return [
        {
            "id": name,
            "x": (points[name][0] - min(xs)) / width,
            "y": (max(ys) - points[name][1]) / height,
        }
        for name in sorted(buses)
    ]


def export(source_commit):
    model_folder = ROOT / "node_systems/13Bus"
    controls = {
        "load_mult": [round(0.6 + i * 0.1, 1) for i in range(9)],
        "tap_step": [-4, 0, 4],
        "capacitors": [0, 1],
    }
    session = DSSSession(model_folder, "network.dss", step_seconds=3600)
    try:
        model_manifest = session.manifest()
        nodes = coordinates(model_folder, session.bus_names)
        edges = [
            {"source": a, "target": b, "element": data["element"]}
            for a, b, data in sorted(
                session.topology.edges(data=True),
                key=lambda row: (row[0], row[1], row[2]["element"]),
            )
        ]
        regulator_names = sorted(session.regulators)
        capacitor_names = sorted(session.capacitors)
        storage_names = sorted(session.batteries)
        pv_names = sorted(session.pvs)
        if not regulator_names or not capacitor_names:
            raise ValueError("算例缺少调压器或电容器")
        snapshots = []
        for load_mult, tap_step, capacitors in itertools.product(
            controls["load_mult"], controls["tap_step"], controls["capacitors"]
        ):
            # 各工况独立编译，避免控制状态或储能状态从上一工况继承。
            session.reset()
            session.command(f"Set LoadMult={load_mult:.1f}")
            requests = {
                "Capacitor." + name: {"state": capacitors}
                for name in capacitor_names
            }
            for name in regulator_names:
                regulator = session.regulators[name]
                step = (regulator.maxtap - regulator.mintap) / regulator.numtaps
                if not math.isclose(step, 0.00625, abs_tol=1e-12):
                    raise ValueError("算例调压器每档并非0.00625")
                requests["RegControl." + name] = {"tap": 1.0 + tap_step * step}
            requests.update(
                {"Storage." + name: {"kw": 0.0} for name in storage_names}
            )
            session.apply(requests)
            session.solve(native_controls=False)
            state = session.read_state()
            voltages = [v for phases in state.voltages_pu.values() for v in phases.values()]
            if not voltages or not all(math.isfinite(v) for v in voltages):
                raise ValueError("工况包含无效电压")
            if not all(state.voltages_pu.values()):
                raise ValueError("工况中存在未求解母线")
            balance = state.active_balance_error_kw
            if not math.isfinite(balance) or abs(balance) >= 0.05:
                raise ValueError(f"有功守恒误差超过50W：{balance}")
            if state.time_s != 0:
                raise ValueError("稳态扫掠不应推进仿真时间")
            snapshots.append(
                {
                    "id": f"load-{load_mult:.1f}_tap-{tap_step}_cap-{capacitors}",
                    "load_mult": load_mult,
                    "tap_step": tap_step,
                    "capacitors": capacitors,
                    "voltages": state.voltages_pu,
                    "min_pu": min(voltages),
                    "max_pu": max(voltages),
                    "loss_kw": state.losses[0],
                    "grid_kw": state.grid_import[0],
                    "balance_error_kw": balance,
                }
            )
        return {
            "schema_version": 1,
            "provenance": {
                "kind": "precomputed_opendss_steady_state_sweep",
                "source_commit": source_commit,
                "source_repository": "https://github.com/XJTU-RL/PowerZoo",
                "model_path": "node_systems/13Bus/network.dss",
                "model_sha256": model_manifest["model_sha256"],
                "model_files": model_manifest["input_files"],
                "exporter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "command": (
                    "python scripts/export_showcase.py "
                    "--output docs/showcase/data/vvc-13bus.json "
                    f"--source-commit {source_commit}"
                ),
                "dss_python_version": version("dss-python"),
                "dss_python_backend_version": version("dss-python-backend"),
                "opendss_engine": model_manifest["engine"],
                "python_version": sys.version.split()[0],
                "conditions": {
                    "simulation": "static AC snapshot; no policy training or online browser solver",
                    "native_controls": False,
                    "reset_each_snapshot": True,
                    "advance_time": False,
                    "load_scaling": "DSS LoadMult applied to native model loads",
                    "tap_setting": "all native RegControl devices share 1 + tap_step * 0.00625",
                    "capacitor_setting": "all native capacitor banks share the selected state",
                    "storage_kw": 0.0,
                    "pv_setting": "unchanged network.dss baseline; this model contains no PVSystem",
                    "voltage_limits_pu": [0.95, 1.05],
                    "active_balance_tolerance_kw": 0.05,
                },
                "devices": {
                    "regulators": regulator_names,
                    "capacitors": capacitor_names,
                    "storage": storage_names,
                    "pv": pv_names,
                },
            },
            "controls": controls,
            "nodes": nodes,
            "edges": edges,
            "snapshots": snapshots,
        }
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/showcase/data/vvc-13bus.json")
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args()
    source_commit = args.source_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = export(source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "snapshots": len(result["snapshots"]),
                "nodes": len(result["nodes"]),
                "edges": len(result["edges"]),
                "min_pu": min(s["min_pu"] for s in result["snapshots"]),
                "max_pu": max(s["max_pu"] for s in result["snapshots"]),
                "max_balance_error_kw": max(abs(s["balance_error_kw"]) for s in result["snapshots"]),
                "sha256": sha256(args.output.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
