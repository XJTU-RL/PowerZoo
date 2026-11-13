"""共享OpenDSS会话：不重写输入程序，原生Storage每个成功区间推进一次。"""

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
import math
import shutil
import tempfile
import time

import dss
import networkx as nx
import numpy as np

from envs.powergrid.devices import discover, prepare, names
from envs.powergrid.state import GridState


def _derived_node_order(element_obj):
    """按 BusNames 语义推导元件节点序：显式相别后缀优先，无后缀取 1..NumPhases，
    不足 NumConductors 补中性导体 0。外部算例存在编译期即停用的元件（如英国
    MV-LV 模型被 lds_edit.dss 禁用的中压集中负荷），它们不进引擎节点序表，
    NodeOrder 会报“Nodes are not initialized”；该推导对启用元件已与原生
    NodeOrder 在 UK/8500/9500 共 7 万元件上逐一对拍全等。"""
    names = list(element_obj.BusNames)
    phases, conductors = int(element_obj.NumPhases), int(element_obj.NumConductors)
    order = []
    for name in names:
        suffix = name.partition(".")[2]
        nodes = (
            [int(x) for x in suffix.split(".")]
            if suffix
            else list(range(1, phases + 1))
        )
        order.extend(nodes[:conductors] + [0] * (conductors - len(nodes)))
    return order


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
        # 分段耗时（毫秒，累计；reset 重建基线，供基准脚本与性能诊断使用）。
        self.timings = {}
        try:
            with self._timed("init.copy"):
                self._temporary = tempfile.TemporaryDirectory(prefix="powerzoo-grid-")
                self.folder = Path(self._temporary.name).resolve()
                self.source = Path(model_folder).resolve()
                shutil.copytree(
                    self.source,
                    self.folder,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("logs", "__pycache__", "*.lock"),
                )
            with self._timed("init.sha256"):
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
            self.master, self.overlays = master, tuple(overlays)
            self.svg_elements = tuple(svg_elements)
            if len(set(self.svg_elements)) != len(self.svg_elements):
                raise ValueError("SVG元件绑定不能重复")
            self.trace = {}
            self._measurement_valid = False
            with self._timed("init.context"):
                self.dss = dss.DSS.NewContext()
                self.dss.AllowChangeDir = False
            self.reset()
        except Exception:
            self.close()
            raise

    def _timed(self, key):
        """累计分段耗时（毫秒）；不改变被测行为。"""
        from contextlib import contextmanager

        @contextmanager
        def span():
            started = time.perf_counter()
            try:
                yield
            finally:
                self.timings[key] = self.timings.get(key, 0.0) + (
                    (time.perf_counter() - started) * 1000
                )

        return span()

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
        with self._timed("reset.compile"):
            self.command(f'compile "{self._input_path(self.master)}"')
            for overlay in self.overlays:
                self.command(f'redirect "{self._input_path(overlay)}"')
            self.command("Set mode=snapshot controlmode=off")
        # 训练求解固定 Newton 快照：恒功率负荷（Model=1）语义正确。
        # 前推回代（solve mode=direct）把恒功率负荷按导纳处理，会改变负荷
        # 物理响应，只允许出现在模型文件的 compile 期初始化序列中。
        self.command("CalcVoltageBases")
        self.command(
            f"Set stepsize={self.step_seconds:.12g} number=1 maxiterations=100 tolerance=1e-8"
        )
        self.native = self.dss.ActiveCircuit
        # snapshot 求解仍会用全局 LoadMult 缩放全部负荷：若模型 compile 期
        # 设置了非 1 的乘子，会与场景层"铭牌×组倍率"的写入路径重复叠加，
        # 必须拒绝而不是静默相乘（SMART-DS 等外部算例接入的安全栅栏）。
        if not math.isclose(self.native.Solution.LoadMult, 1.0, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(
                f"模型设置了非 1 的全局负荷乘子 LoadMult="
                f"{self.native.Solution.LoadMult}；会话场景路径会把铭牌值当作"
                "唯一真值写入，二者叠加会重复计负荷。请在模型中复位该设置。"
            )
        self.native.Solution.dblHour = seconds / 3600
        # 结构缓存：同一会话的模型文件在生命周期内不变（临时目录只读副本），
        # 连接清单/相别映射/拓扑与设备铭牌跨 reset 确定性相同；首次构建后
        # 复用，避免大规模模型（数万元件）每回合重复扫描。Storage 的 SOC
        # 等运行态每次 reset 后仍从引擎重读（discover 重建设备对象）。
        cached = getattr(self, "_structure_cache", None)
        if cached is not None and cached["sha"] == self.model_sha256:
            with self._timed("reset.cache_hit"):
                self._connections = cached["connections"]
                self.bus_phase_nodes = cached["bus_phase_nodes"]
                self.topology = cached["topology"]
                self.loads, self.capacitors, self.regulators = (
                    cached["loads"], cached["capacitors"], cached["regulators"]
                )
                self.batteries, self.pvs, self.svg_elements = (
                    cached["batteries"], cached["pvs"], cached["svg_elements"]
                )
                self.bus_names = cached["bus_names"]
                # discover 的编译期副作用与运行态字段在缓存命中时必须重放：
                # 每次 compile 后引擎回到模型初始状态（DispMode/status/SOC），
                # 而缓存的设备对象可能带着上回合的运行态残留。
                for name in self.batteries:
                    self.command(f"Edit Storage.{name} DispMode=external")
                    self.batteries[name].soc = (
                        float(self.read_property("Storage." + name, "%stored")) / 100
                    )
                for dev in self.pvs.values():
                    dev.fraction, dev.q_fraction = 1.0, 0.0
                for element in self.svg_elements:
                    self.command(f"Edit {element} status=fixed")
            return
        # Enabled变化会重建全局NodeRef；停用元件的NodeOrder不能再作为相别权威。
        self._connections = {}
        with self._timed("reset.connections"):
            for element in self.native.AllElementNames:
                self.require_element(element)
                obj = self.native.ActiveCktElement
                if obj.NumTerminals:
                    # 停用元件（如外部算例禁用的中压集中负荷）不在引擎节点序表中，
                    # 改用与 NodeOrder 对拍全等的推导，避免加载合法模型即报错。
                    nodes = (
                        list(map(int, obj.NodeOrder))
                        if obj.Enabled
                        else _derived_node_order(obj)
                    )
                    self._connections[element.lower()] = {
                        "buses": list(obj.BusNames),
                        "nodes": nodes,
                        "phases": int(obj.NumPhases),
                        "conductors": int(obj.NumConductors),
                        "terminals": int(obj.NumTerminals),
                    }
        self.bus_phase_nodes = {name: [] for name in self.native.AllBusNames}
        for node in self.native.AllNodeNames:
            bus, _, phase = node.rpartition(".")
            if phase in ("1", "2", "3"):
                self.bus_phase_nodes[bus.lower()].append(int(phase))
        with self._timed("reset.discover"):
            discover(self)
        with self._timed("reset.topology"):
            self.topology = nx.MultiGraph()
            for element in sorted(self.native.AllElementNames):
                if element.lower().startswith(("line.", "transformer.")):
                    self.require_element(element)
                    buses = [
                        b.split(".")[0].lower()
                        for b in self.native.ActiveCktElement.BusNames
                    ]
                    for bus in buses[1:]:
                        self.topology.add_edge(
                            buses[0], bus, key=element, element=element
                        )
        self.bus_names = sorted(self.native.AllBusNames)
        self._structure_cache = {
            "sha": self.model_sha256,
            "connections": self._connections,
            "bus_phase_nodes": self.bus_phase_nodes,
            "topology": self.topology,
            "loads": self.loads,
            "capacitors": self.capacitors,
            "regulators": self.regulators,
            "batteries": self.batteries,
            "pvs": self.pvs,
            "svg_elements": self.svg_elements,
            "bus_names": self.bus_names,
        }

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
        with self._timed("step.apply"):
            commands, trace = prepare(self, requests)
            try:
                for command in commands:
                    self.command(command)
            except Exception:
                self._failed = True
                raise
            self.trace.update(trace)
            return trace

    def set_pv_conditions(self, irradiance, temperature, mapping=None):
        """写入 PV 辐照度/温度；mapping 逐 PV 覆盖且只写 mapping 内设备。

        mapping=None 保持历史行为：全网同一标量（SMART-DS 之前唯一路径）。
        mapping={PV名: (辐照度 kW/m², 温度 °C)} 用于逐设备天气（SMART-DS
        每台 PV 有独立倾角/朝向曲线）；映射中出现不存在的 PV 立即报错。
        """
        if mapping is None:
            if (
                not math.isfinite(irradiance)
                or irradiance < 0
                or not math.isfinite(temperature)
            ):
                raise ValueError("PV辐照度与组件温度无效")
            targets = {name: (irradiance, temperature) for name in self.pvs}
        else:
            unknown = set(mapping) - set(self.pvs)
            if unknown:
                raise ValueError(f"PV辐照映射包含不存在的设备：{sorted(unknown)}")
            targets = dict(mapping)
        for name, (irradiance_pv, temperature_pv) in targets.items():
            if (
                not math.isfinite(irradiance_pv)
                or irradiance_pv < 0
                or not math.isfinite(temperature_pv)
            ):
                raise ValueError(f"PVSystem.{name} 辐照度与组件温度无效")
            self.command(
                f"Edit PVSystem.{name} irradiance={irradiance_pv:.12g}"
                f" temperature={temperature_pv:.12g}"
            )

    def available_pv_power(self, names=None):
        """计时包装：探测含求解与恢复，单独归入 step.probe 段。"""
        with self._timed("step.probe"):
            return self._available_pv_power(names)

    def _available_pv_power(self, names=None):
        """当步其他设备设定下，零Q且不限有功的原生可用AC功率；不推进能量。

        names=None 时保持历史行为：对全部 PV 写 ``%Pmpp=100 kvar=0`` 并保持
        该设定（既有训练路径依赖此语义）。显式传入名单时只探测这些设备，
        且**探测前快照、探测后恢复**其 ``%Pmpp/kvar``——被探测设备与其他
        未列入设备的设定在探测前后完全不变（SMART-DS 引入：未接管 PV 的
        原生 pf/InvControl 行为不得被探测覆盖）。
        """
        if names is None:
            for name in self.pvs:
                self.command(f"Edit PVSystem.{name} %Pmpp=100 kvar=0")
            self.solve()
            return {
                name: max(0.0, -float(self.powers("PVSystem." + name)[0]))
                for name in self.pvs
            }
        unknown = set(names) - set(self.pvs)
        if unknown:
            raise ValueError(f"PV探测名单包含不存在的设备：{sorted(unknown)}")
        snapshot = {}
        for name in names:
            self.require_element(f"PVSystem.{name}")
            properties = self.native.ActiveCktElement.Properties
            pmpp_pct = properties("%pmpp").Val
            kvar = properties("kvar").Val
            snapshot[name] = (pmpp_pct, kvar)
            self.command(f"Edit PVSystem.{name} %Pmpp=100 kvar=0")
        self.solve()
        result = {
            name: max(0.0, -float(self.powers("PVSystem." + name)[0]))
            for name in names
        }
        for name, (pmpp_pct, kvar) in snapshot.items():
            self.command(f"Edit PVSystem.{name} %Pmpp={pmpp_pct} kvar={kvar}")
        # 恢复写入会使量测栅栏失效；补一次同工况求解，使探测对会话状态
        # 完全透明（读数合法且电路回到探测前的工作点）。
        self.solve()
        return result

    def solve(self, seconds=None, *, native_controls=False):
        with self._timed("step.solve"):
            return self._solve(seconds, native_controls=native_controls)

    def _solve(self, seconds=None, *, native_controls=False):
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
        with self._timed("step.branch_state"):
            return self._branch_state()

    def _branch_state(self):
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

    def read_state(self, light=False):
        with self._timed("step.read_state"):
            return self._read_state(light=light)

    def _read_state(self, light=False):
        converged = getattr(
            getattr(self.native, "Solution", None), "Converged", None
        )
        if self._failed or not self._measurement_valid or not converged:
            raise RuntimeError(
                "失败会话不能读取为正常物理状态："
                f"failed={self._failed} "
                f"measurement_valid={self._measurement_valid} "
                f"converged={converged}"
            )
        if light:
            # 训练步精简读取：电压用一次 AllBusVmagPu 批量按节点名分组（与
            # 逐母线 puVmagAngle 同源同值）；注入只取可控设备类（负荷类注入
            # 与全网支路明细省略，active_balance_error 置 None——完整校验与
            # 轨迹快照走默认完整读取）。
            names = self.native.AllNodeNames
            values = self.native.AllBusVmagPu
            voltages = {}
            for node, value in zip(names, values):
                bus, _, phase = node.rpartition(".")
                if phase in ("1", "2", "3"):
                    voltages.setdefault(bus.lower(), {})[int(phase)] = float(value)
            # 与完整读取同键序同覆盖（按 bus_names 排序、保留无相节点母线的
            # 空映射），保证两种模式的观测向量逐位同序。
            voltages = {n: voltages.get(n, {}) for n in self.bus_names}
            injections = {}
            for prefix in ("pvsystem.", "storage.", "generator."):
                for element in self._connections:
                    if element.startswith(prefix):
                        injections[element] = tuple(-self.powers(element))
            return GridState(
                self.time_s,
                voltages,
                injections,
                tuple(-self.total_power()),
                tuple(self.total_loss()),
                {n: b.soc for n, b in self.batteries.items()},
                None,
            )
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
