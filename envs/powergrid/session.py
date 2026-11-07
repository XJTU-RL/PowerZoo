"""共享OpenDSS会话：不重写输入程序，原生Storage每个成功区间推进一次。"""

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
import math
import shutil
import tempfile

import dss
import networkx as nx
import numpy as np

from envs.powergrid.devices import discover, prepare, names
from envs.powergrid.state import GridState


class DSSSession:
    def __init__(
        self, model_folder, master, overlays=(), step_seconds=60, svg_elements=()
    ):
        if not math.isfinite(step_seconds) or step_seconds <= 0:
            raise ValueError("step_seconds必须为正有限值")
        self.step_seconds = float(step_seconds)
        self._closed, self._pending, self._solved, self._failed = (
            False,
            False,
            False,
            False,
        )
        self._temporary = tempfile.TemporaryDirectory(prefix="powerzoo-grid-")
        self.folder = Path(self._temporary.name).resolve()
        self.source = Path(model_folder).resolve()
        self.master, self.overlays = master, tuple(overlays)
        self.svg_elements = tuple(svg_elements)
        if len(set(self.svg_elements)) != len(self.svg_elements):
            raise ValueError("SVG元件绑定不能重复")
        self.trace = {}
        self._measurement_valid = False
        try:
            shutil.copytree(
                self.source,
                self.folder,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("logs", "__pycache__", "*.lock"),
            )
            self.input_files = {
                p.relative_to(self.folder)
                .as_posix(): sha256(p.read_bytes())
                .hexdigest()
                for p in sorted(self.folder.rglob("*"))
                if p.is_file()
            }
            self.model_sha256 = sha256(
                json.dumps(self.input_files, sort_keys=True).encode()
            ).hexdigest()
            self.dss = dss.DSS.NewContext()
            self.dss.AllowChangeDir = False
            self.reset()
        except Exception:
            self.close()
            raise

    def _input_path(self, relative):
        path = (self.folder / relative).resolve()
        if not path.is_relative_to(self.folder):
            raise ValueError("DSS输入必须位于模型目录内")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def command(self, text):
        if self._closed:
            raise RuntimeError("DSS会话已关闭")
        self._solved = False
        self._measurement_valid = False
        try:
            self.dss.Text.Command = text
        except Exception:
            self._failed = True
            raise

    def reset(self, seconds=0):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("初始时间必须为非负有限秒数")
        self._pending, self._solved, self._failed = False, False, False
        self.trace = {}
        self.dss.ClearAll()
        self.command(f'compile "{self._input_path(self.master)}"')
        for overlay in self.overlays:
            self.command(f'redirect "{self._input_path(overlay)}"')
        self.command("Set mode=snapshot controlmode=off")
        self.command("CalcVoltageBases")
        self.command(
            f"Set stepsize={self.step_seconds:.12g} number=1 maxiterations=100 tolerance=1e-8"
        )
        self.native = self.dss.ActiveCircuit
        self.native.Solution.dblHour = seconds / 3600
        # Enabled变化会重建全局NodeRef；停用元件的NodeOrder不能再作为相别权威。
        self._connections = {}
        for element in self.native.AllElementNames:
            self.require_element(element)
            obj = self.native.ActiveCktElement
            if obj.NumTerminals:
                self._connections[element.lower()] = {
                    "buses": list(obj.BusNames),
                    "nodes": list(map(int, obj.NodeOrder)),
                    "phases": int(obj.NumPhases),
                    "conductors": int(obj.NumConductors),
                    "terminals": int(obj.NumTerminals),
                }
        self.bus_phase_nodes = {name: [] for name in self.native.AllBusNames}
        for node in self.native.AllNodeNames:
            bus, _, phase = node.rpartition(".")
            if phase in ("1", "2", "3"):
                self.bus_phase_nodes[bus.lower()].append(int(phase))
        discover(self)
        self.topology = nx.MultiGraph()
        for element in sorted(self.native.AllElementNames):
            if element.lower().startswith(("line.", "transformer.")):
                self.require_element(element)
                buses = [
                    b.split(".")[0].lower()
                    for b in self.native.ActiveCktElement.BusNames
                ]
                for bus in buses[1:]:
                    self.topology.add_edge(buses[0], bus, key=element, element=element)

    def require_element(self, element):
        if self.native.SetActiveElement(element) < 0:
            raise ValueError(f"DSS设备不存在：{element}")

    def read_property(self, element, key):
        self.require_element(element)
        return self.native.ActiveDSSElement.Properties(key).Val

    # 兼容已有物理探针；正式控制走apply。
    _property = read_property

    def begin_interval(self, seconds):
        if self._closed or self._failed or self._pending:
            raise RuntimeError(
                "会话不可开始新区间；失败后须reset，成功区间只能收尾一次"
            )
        if not math.isfinite(seconds) or not math.isclose(
            seconds, self.time_s, abs_tol=1e-6
        ):
            raise ValueError(f"数据时间{seconds}与DSS时间{self.time_s}不一致")
        self._pending, self._solved = True, False
        self.trace = {}

    @property
    def time_s(self):
        return float(self.native.Solution.dblHour * 3600)

    def apply(self, requests):
        commands, trace = prepare(self, requests)
        try:
            for command in commands:
                self.command(command)
        except Exception:
            self._failed = True
            raise
        self.trace.update(trace)
        return trace

    def set_pv_conditions(self, irradiance, temperature):
        if (
            not math.isfinite(irradiance)
            or irradiance < 0
            or not math.isfinite(temperature)
        ):
            raise ValueError("PV辐照度与组件温度无效")
        for name in self.pvs:
            self.command(
                f"Edit PVSystem.{name} irradiance={irradiance:.12g} temperature={temperature:.12g}"
            )

    def available_pv_power(self):
        """当步其他设备设定下，零Q且不限有功的原生可用AC功率；不推进能量。"""
        for name in self.pvs:
            self.command(f"Edit PVSystem.{name} %Pmpp=100 kvar=0")
        self.solve()
        return {
            name: max(0.0, -float(self.powers("PVSystem." + name)[0]))
            for name in self.pvs
        }

    def solve(self, seconds=None, *, native_controls=False):
        if self._closed or self._failed:
            raise RuntimeError("会话已关闭或失败，须reset")
        if seconds is not None:
            if self._pending:
                raise RuntimeError("区间求解不能另行改写时钟")
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("求解时间无效")
            self.native.Solution.dblHour = seconds / 3600
        try:
            self.native.Solution.ControlMode = 0 if native_controls else -1
            for _ in range(12):
                self.native.Solution.SolveSnap()
                if not self.native.Solution.Converged:
                    raise RuntimeError(f"DSS在{self.time_s}s未收敛")
                if native_controls and not self.native.Solution.ControlActionsDone:
                    raise RuntimeError("DSS原生控制循环尚未完成")
                limited = False
                for name, dev in self.svgs.items():
                    element = "Generator." + name
                    self.require_element(element)
                    currents = np.asarray(self.native.ActiveCktElement.CurrentsMagAng)[
                        ::2
                    ]
                    current = float(
                        max(currents[: self.native.ActiveCktElement.NumPhases])
                    )
                    if current > dev.rated_current_a * (1 + 1e-6):
                        q = float(self.read_property(element, "kvar"))
                        reduced = q * dev.rated_current_a / current * (1 - 1e-7)
                        self.command(f"Edit {element} kvar={reduced:.12g}")
                        entry = self.trace.setdefault(
                            element, {"requested": {"kvar": q}, "applied": {"kvar": q}}
                        )
                        entry["applied"]["kvar"] = reduced
                        entry["limit_reason"] = "phase_current"
                        limited = True
                if not limited:
                    break
            else:
                raise RuntimeError("SVG实测相电流限幅迭代未收敛")
        except Exception:
            self._failed, self._solved = True, False
            raise
        self._solved = True
        self._measurement_valid = True

    def finish_interval(self):
        if not self._pending or not self._solved or self._failed:
            raise RuntimeError("只能结束已成功求解且尚未收尾的区间")
        start = self.time_s
        try:
            self.native.Solution.FinishTimeStep()
        except Exception:
            self._failed, self._solved = True, False
            raise
        self._pending, self._solved = False, False
        if not math.isclose(self.time_s, start + self.step_seconds, abs_tol=1e-6):
            self._failed = True
            raise RuntimeError("原生时间推进与区间长度不一致")
        for name, dev in self.batteries.items():
            dev.soc = float(self.read_property("Storage." + name, "%stored")) / 100
            if not math.isfinite(dev.soc) or not -1e-9 <= dev.soc <= 1 + 1e-9:
                self._failed = True
                raise RuntimeError(f"Storage.{name}原生SOC无效：{dev.soc}")
        for element, trace in self.trace.items():
            if element.startswith("RegControl."):
                dev = self.regulators[element.split(".", 1)[1]]
                self.native.Transformers.Name, self.native.Transformers.Wdg = (
                    dev.transformer,
                    dev.winding,
                )
                trace["measured"] = {"tap": float(self.native.Transformers.Tap)}
            elif element.startswith("Capacitor."):
                self.native.Capacitors.Name = element.split(".", 1)[1]
                trace["measured"] = {"state": int(self.native.Capacitors.States[0])}
            elif element.startswith("Line."):
                from envs.powergrid.topology import is_closed

                trace["measured"] = {"closed": int(is_closed(self.connection(element)))}
            else:
                p, q = (
                    self.powers(element)
                    if element.startswith("Load.")
                    else -self.powers(element)
                )
                trace["measured"] = {"kw": float(p), "kvar": float(q)}

    def powers(self, element, terminal=1):
        self.require_element(element)
        if not 1 <= terminal <= self.native.ActiveCktElement.NumTerminals:
            raise ValueError(f"{element}端口{terminal}不存在")
        values = np.asarray(self.native.ActiveCktElement.TotalPowers, dtype=float)
        result = values[2 * (terminal - 1) : 2 * terminal]
        if not np.isfinite(result).all():
            raise RuntimeError(f"{element}功率读数无效")
        return result

    def capacitor_state(self, name):
        self.native.Capacitors.Name = name
        return tuple(self.native.Capacitors.States)

    def regulator_tap(self, name):
        reg = self.regulators[name]
        self.native.Transformers.Name, self.native.Transformers.Wdg = (
            reg.transformer,
            reg.winding,
        )
        return float(self.native.Transformers.Tap)

    def element_info(self, element):
        """端口不合并、不取绝对值；电流原生A，损耗kW/kvar。"""
        self.require_element(element)
        obj = self.native.ActiveCktElement
        return {
            **self.connection(element),
            "currents_a": np.asarray(obj.CurrentsMagAng)[::2].astype(float).tolist(),
            "powers": np.asarray(obj.TotalPowers).reshape(-1, 2).astype(float).tolist(),
            "losses": (np.asarray(obj.Losses, dtype=float) / 1000).tolist(),
            "normal_amps": float(obj.NormalAmps),
        }

    def connection(self, element):
        """无需求解的原生连接、可用状态和逐端子逐导体开断状态。"""
        self.require_element(element)
        obj = self.native.ActiveCktElement
        geometry = self._connections.get(element.lower())
        if (
            geometry is None
            or list(obj.BusNames) != geometry["buses"]
            or obj.NumConductors != geometry["conductors"]
        ):
            raise RuntimeError("元件连接清单改变，须通过模型重新编译；不能沿用旧相别")
        return {
            **geometry,
            "buses": list(geometry["buses"]),
            "nodes": list(geometry["nodes"]),
            "enabled": bool(obj.Enabled),
            "open": [
                [bool(obj.IsOpen(t, c)) for c in range(1, obj.NumConductors + 1)]
                for t in range(1, obj.NumTerminals + 1)
            ],
        }

    def enable_elements(self, states):
        """整帧验证原生Enabled开关；故障不可修复等业务规则由任务约束。"""
        for element, enabled in states.items():
            self.require_element(element)
            if type(enabled) is not bool:
                raise ValueError("Enabled状态必须为布尔值")
        for element, enabled in states.items():
            self.command(f"Edit {element} enabled={'yes' if enabled else 'no'}")

    def branch_state(self):
        result = {}
        for element in sorted(self.native.AllElementNames):
            if not element.lower().startswith(("line.", "transformer.")):
                continue
            info = self.element_info(element)
            if element.lower().startswith("line."):
                currents = [
                    i
                    for i, node in zip(info["currents_a"], info["nodes"])
                    if node in (1, 2, 3)
                ]
                info["loading_pu"] = (
                    max(currents) / info["normal_amps"]
                    if currents and info["normal_amps"] > 0
                    else None
                )
            else:
                self.native.Transformers.Name = element.split(".", 1)[1]
                ratings, loading = [], []
                for winding in range(info["terminals"]):
                    self.native.Transformers.Wdg = winding + 1
                    kva, kv = float(self.native.Transformers.kVA), float(
                        self.native.Transformers.kV
                    )
                    rated_a = kva / (kv * (math.sqrt(3) if info["phases"] == 3 else 1))
                    start = winding * info["conductors"]
                    phase_currents = info["currents_a"][start : start + info["phases"]]
                    ratings.append(kva)
                    loading.append(max(phase_currents) / rated_a)
                info["winding_kva"] = ratings
                info["loading_pu"] = max(loading)
            result[element] = info
        return result

    def phase_voltages(self, name):
        if self.native.SetActiveBus(name) < 0:
            raise ValueError(f"母线不存在：{name}")
        nodes = self.native.ActiveBus.Nodes
        values = np.asarray(self.native.ActiveBus.puVmagAngle)[::2]
        result = {
            int(node): float(v) for node, v in zip(nodes, values) if node in (1, 2, 3)
        }
        if not result or not np.isfinite(list(result.values())).all():
            raise RuntimeError(f"{name}相电压读数无效")
        return result

    def bus_voltage(self, name):
        return list(self.phase_voltages(name).values())

    def total_loss(self):
        return np.asarray(self.native.Losses, dtype=float) / 1000

    def total_power(self):
        return np.asarray(self.native.TotalPower, dtype=float)

    def total_load_power(self):
        return sum((self.powers("Load." + n) for n in self.loads), start=np.zeros(2))

    def read_state(self):
        if (
            self._failed
            or not self._measurement_valid
            or not self.native.Solution.Converged
        ):
            raise RuntimeError("失败会话不能读取为正常物理状态")
        active_buses = {
            node.rpartition(".")[0].lower()
            for node in self.native.AllNodeNames
            if node.rpartition(".")[2] in ("1", "2", "3")
        }
        elements = [
            e
            for e in self.native.AllElementNames
            if e.lower().startswith(("load.", "pvsystem.", "storage.", "generator."))
        ]
        return GridState(
            self.time_s,
            {
                n: self.phase_voltages(n) if n in active_buses else {}
                for n in self.bus_names
            },
            {e: tuple(-self.powers(e)) for e in sorted(elements)},
            tuple(-self.total_power()),
            tuple(self.total_loss()),
            {n: b.soc for n, b in self.batteries.items()},
            self.branch_state(),
        )

    def manifest(self):
        return {
            "model_sha256": self.model_sha256,
            "input_files": self.input_files,
            "master": str(self.master),
            "overlays": list(self.overlays),
            "step_seconds": self.step_seconds,
            "engine": self.dss.Version,
            "storage": {
                n: {k: v for k, v in asdict(d).items() if k not in ("soc", "kw")}
                for n, d in self.batteries.items()
            },
            "pv": {
                n: {
                    k: v
                    for k, v in asdict(d).items()
                    if k not in ("fraction", "q_fraction")
                }
                for n, d in self.pvs.items()
            },
            "svg": {n: asdict(d) for n, d in self.svgs.items()},
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        if hasattr(self, "dss"):
            self.dss.ClearAll()
        self._temporary.cleanup()
