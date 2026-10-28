"""独立 DSS context；时序由 Python 明确注入，模型目录只读。"""

from pathlib import Path
from types import SimpleNamespace
import math
import re
import shutil
import tempfile

import dss
import networkx as nx
import numpy as np


class SmartGridCircuit:
    def __init__(self, model_folder, master, pv_plan, step_seconds):
        self._closed = False
        self._temporary = tempfile.TemporaryDirectory(prefix="powerzoo-smartgrid-")
        self.folder = Path(self._temporary.name)
        source = Path(model_folder).resolve()
        try:
            shutil.copytree(
                source,
                self.folder,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    "loadshape", "irradiation", "temperature", "logs", "__pycache__"
                ),
            )
            # 只转换工作副本。输入/输出时序全由本运行时负责；保留电路本身的物理参数。
            for file in (
                p for p in self.folder.rglob("*") if p.suffix.lower() == ".dss"
            ):
                lines = []
                for raw in file.read_text(encoding="utf-8-sig").splitlines():
                    line = raw.split("!", 1)[0].strip()
                    if re.match(
                        r"(?i)^(clear|solve|calcvoltagebases|set\s+(mode|stepsize|number|hour|sec))\b",
                        line,
                    ):
                        continue
                    if re.match(r"(?i)^redirect\s+.*(loadshape|pv_data)", line):
                        continue
                    if file.name.lower() != "pv_systems_base.dss" and re.match(
                        r"(?i)^redirect\s+.*(pv_systems_base|pv_plans)", line
                    ):
                        continue
                    line = re.sub(
                        r"(?i)\b(t?daily|t?yearly|t?duty)\s*=\s*[^\s]+", "", line
                    )
                    lines.append(line)
                file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.dss = dss.DSS.NewContext()
            self.dss.AllowChangeDir = False
            master_path = (self.folder / master).resolve()
            if not master_path.is_relative_to(self.folder):
                raise ValueError("dss_file 必须位于模型目录内")
            self.command(f'compile "{master_path}"')
            if pv_plan != "none":
                plan = self.folder / (
                    f"pv_plans/{pv_plan}.dss" if pv_plan else "pv_systems_base.dss"
                )
                if not plan.resolve().is_relative_to(self.folder):
                    raise ValueError("pv_plan 必须位于模型目录内")
                if not plan.is_file():
                    raise FileNotFoundError(f"PV 方案不存在：{plan}")
                self.command(f'redirect "{plan}"')
            self.command("Set mode=snapshot controlmode=off")
            self.command("CalcVoltageBases")
            self.command(f"Set stepsize={step_seconds} number=1 maxiterations=100")
            self.step_seconds = step_seconds
            self.native = self.dss.ActiveCircuit
            self._discover()
        except Exception:
            self.close()
            raise

    def command(self, text):
        self.dss.Text.Command = text

    def _names(self, collection):
        return list(collection.AllNames) if collection.Count else []

    def _property(self, element, name):
        self.native.SetActiveElement(element)
        return self.native.ActiveDSSElement.Properties(name).Val

    def _discover(self):
        c = self.native
        self.loads = {}
        for name in self._names(c.Loads):
            c.Loads.Name = name
            self.loads[name] = SimpleNamespace(kw=c.Loads.kW, kvar=c.Loads.kvar)
        self.capacitors = {name: None for name in self._names(c.Capacitors)}
        self.regulators = {}
        for name in self._names(c.RegControls):
            c.RegControls.Name = name
            transformer, winding = c.RegControls.Transformer, c.RegControls.Winding
            c.Transformers.Name = transformer
            c.Transformers.Wdg = winding
            self.regulators[name] = SimpleNamespace(
                transformer=transformer,
                winding=winding,
                mintap=c.Transformers.MinTap,
                maxtap=c.Transformers.MaxTap,
                numtaps=c.Transformers.NumTaps,
                initial=c.Transformers.Tap,
            )
        self.batteries = {}
        for name in self._names(c.Storages):
            element = "Storage." + name
            value = lambda key: float(self._property(element, key))
            capacity, rating = value("kWhRated"), value("kWRated")
            self.batteries[name] = SimpleNamespace(
                max_kw=rating,
                max_kwh=capacity,
                initial_soc=value("%stored") / 100,
                soc=value("%stored") / 100,
                reserve=value("%reserve") / 100,
                eta_charge=value("%EffCharge") / 100,
                eta_discharge=value("%EffDischarge") / 100,
                kw=0.0,
            )
            if capacity <= 0 or rating <= 0:
                raise ValueError(f"{element} 额定功率/容量必须为正")
            bat = self.batteries[name]
            if not (
                0 <= bat.reserve <= bat.initial_soc <= 1
                and 0 < bat.eta_charge <= 1
                and 0 < bat.eta_discharge <= 1
            ):
                raise ValueError(f"{element} 的 SOC/效率参数无效")
        # 旧模型可能使用 Generator 模拟电池，但它没有容量/额定功率，不能猜测其基准。
        ambiguous = [n for n in self._names(c.Generators) if n.startswith("batt")]
        if ambiguous:
            raise ValueError(
                f"电池 {ambiguous} 缺少 Storage 额定参数，请先转换 DSS 模型"
            )
        self.pvs = {}
        for name in self._names(c.PVSystems):
            c.PVSystems.Name = name
            self.pvs[name] = SimpleNamespace(
                pmpp=c.PVSystems.Pmpp,
                kva=c.PVSystems.kVArated,
                fraction=1.0,
                q_fraction=0.0,
                pt_curve=self._property("PVSystem." + name, "P-TCurve"),
            )
            if self.pvs[name].pmpp <= 0 or self.pvs[name].kva <= 0:
                raise ValueError(f"PVSystem.{name} 的 Pmpp/kVA 必须为正")
        self.bus_names = list(c.AllBusNames)
        self.topology = nx.Graph()
        for element in c.AllElementNames:
            if element.lower().startswith(("line.", "transformer.")):
                c.SetActiveElement(element)
                buses = [b.split(".")[0].lower() for b in c.ActiveCktElement.BusNames]
                if len(buses) >= 2:
                    self.topology.add_edge(*buses[:2])

    def powers(self, element):
        self.native.SetActiveElement(element)
        # TotalPowers 已经是 kW / kvar；正数表示元件吸收电能。
        return np.asarray(self.native.ActiveCktElement.TotalPowers, dtype=float)[:2]

    def bus_voltage(self, name):
        self.native.SetActiveBus(name)
        return list(np.asarray(self.native.ActiveBus.puVmagAngle)[::2])

    def total_loss(self):
        return (
            np.asarray(self.native.Losses, dtype=float) / 1000
        )  # 原生 W/var -> kW/kvar

    def total_power(self):
        return np.asarray(self.native.TotalPower, dtype=float)

    def total_load_power(self):
        return sum((self.powers("Load." + n) for n in self.loads), start=np.zeros(2))

    def solve(self, seconds):
        self.native.Solution.dblHour = seconds / 3600
        self.native.Solution.SolveSnap()
        if not self.native.Solution.Converged:
            raise RuntimeError(f"DSS 在 {seconds}s 未收敛")

    def close(self):
        if self._closed:
            return
        self._closed = True
        if hasattr(self, "dss"):
            self.dss.ClearAll()
        self._temporary.cleanup()
