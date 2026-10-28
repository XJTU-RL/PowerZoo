"""原生设备清单与物理指令。输入使用kW/kvar/Tap，不接受训练器的数组切片。"""

from dataclasses import dataclass, asdict
import math

import numpy as np


@dataclass
class StorageDevice:
    max_kw: float
    max_kwh: float
    initial_soc: float
    reserve: float
    eta_charge: float
    eta_discharge: float
    soc: float
    kw: float = 0.0


@dataclass
class PVDevice:
    pmpp: float
    kva: float
    kvar_max: float
    kvar_max_abs: float
    pt_curve: str
    fraction: float = 1.0
    q_fraction: float = 0.0


@dataclass
class SVGDevice:
    kva: float
    kvar_max: float
    kvar_max_abs: float
    rated_current_a: float


def names(collection):
    return sorted(collection.AllNames) if collection.Count else []


def discover(session):
    """数据来自已编译电路；不按文件名或母线序号推测设备。"""
    from types import SimpleNamespace

    c = session.native
    session.loads = {}
    for name in names(c.Loads):
        c.Loads.Name = name
        session.loads[name] = SimpleNamespace(
            kw=float(c.Loads.kW), kvar=float(c.Loads.kvar)
        )
    session.capacitors = {}
    for name in names(c.Capacitors):
        c.Capacitors.Name = name
        session.capacitors[name] = tuple(c.Capacitors.States)
    session.regulators = {}
    for name in names(c.RegControls):
        c.RegControls.Name = name
        transformer, winding = c.RegControls.Transformer, c.RegControls.Winding
        c.Transformers.Name, c.Transformers.Wdg = transformer, winding
        session.regulators[name] = SimpleNamespace(
            transformer=transformer,
            winding=winding,
            mintap=float(c.Transformers.MinTap),
            maxtap=float(c.Transformers.MaxTap),
            numtaps=int(c.Transformers.NumTaps),
            initial=float(c.Transformers.Tap),
        )
    session.batteries = {}
    for name in names(c.Storages):
        value = lambda key: float(session.read_property("Storage." + name, key))
        dev = StorageDevice(
            max_kw=value("kWRated"),
            max_kwh=value("kWhRated"),
            initial_soc=value("%stored") / 100,
            reserve=value("%reserve") / 100,
            eta_charge=value("%EffCharge") / 100,
            eta_discharge=value("%EffDischarge") / 100,
            soc=value("%stored") / 100,
        )
        if not (
            dev.max_kw > 0
            and dev.max_kwh > 0
            and 0 <= dev.reserve <= dev.soc <= 1
            and 0 < dev.eta_charge <= 1
            and 0 < dev.eta_discharge <= 1
        ):
            raise ValueError(f"Storage.{name}额定值、SOC或效率无效")
        # 此会话由外部策略调度；原生引擎仍负责能量状态。
        session.command(f"Edit Storage.{name} DispMode=external")
        session.batteries[name] = dev
    session.pvs = {}
    for name in names(c.PVSystems):
        c.PVSystems.Name = name
        value = lambda key: session.read_property("PVSystem." + name, key)
        dev = PVDevice(
            float(c.PVSystems.Pmpp),
            float(c.PVSystems.kVArated),
            float(value("kvarMax")),
            float(value("kvarMaxAbs")),
            value("P-TCurve"),
        )
        if dev.pmpp <= 0 or dev.kva <= 0 or min(dev.kvar_max, dev.kvar_max_abs) < 0:
            raise ValueError(f"PVSystem.{name}原生额定值无效")
        session.pvs[name] = dev
    session.bus_names = sorted(c.AllBusNames)
    session.svgs = {}
    for element in session.svg_elements:
        kind, _, name = element.partition(".")
        if kind != "Generator" or not name:
            raise ValueError(f"SVG必须显式绑定Generator原生元件：{element}")
        session.require_element(element)
        value = lambda key: session.read_property(element, key)
        if float(value("kW")) != 0 or int(value("Model")) != 1:
            raise ValueError(
                f"{element}需要零有功Model=1静态SVG；本机Model=7存在零P无功方向问题"
            )
        phases, kv, kva = int(value("phases")), float(value("kV")), float(value("kVA"))
        if phases not in (1, 3) or min(kv, kva) <= 0:
            raise ValueError(f"{element}SVG相别或额定值无效")
        if value("conn").lower() not in ("wye", "ln"):
            raise ValueError(f"{element}SVG首版需要明确的wye接线")
        session.command(f"Edit {element} status=fixed")
        session.svgs[name] = SVGDevice(
            kva,
            min(kva, float(value("maxkvar"))),
            min(kva, -float(value("minkvar"))),
            kva / (kv * (math.sqrt(3) if phases == 3 else 1)),
        )
        dev = session.svgs[name]
        if min(dev.kvar_max, dev.kvar_max_abs) <= 0:
            raise ValueError(f"{element}需要正的发出/吸收无功能力")


def prepare(session, requests):
    """先验证整帧，再写原生状态；结果记录请求与实际下发值。"""
    commands, trace = [], {}
    for element, request in requests.items():
        if not isinstance(request, dict) or not request:
            raise ValueError(f"{element}需要非空物理指令映射")
        kind, sep, name = element.partition(".")
        if not sep or not name:
            raise ValueError(f"设备ID必须是类别.名称：{element}")
        session.require_element(element)
        if not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in request.values()
        ):
            raise ValueError(f"{element}指令必须为有限数值")
        applied = dict(request)
        if kind == "Load" and name in session.loads:
            allowed = {"kw", "kvar"}
            parts = [f"{key}={value:.12g}" for key, value in request.items()]
        elif kind == "Capacitor" and name in session.capacitors:
            allowed = {"state"}
            if len(session.capacitors[name]) != 1 or request.get("state") not in (0, 1):
                raise ValueError(f"{element}当前仅实现单级二状态电容")
            parts = [f"states=[{int(request['state'])}]"]
        elif kind == "RegControl" and name in session.regulators:
            allowed = {"tap"}
            reg = session.regulators[name]
            tap = request.get("tap", math.nan)
            index = (tap - reg.mintap) / (reg.maxtap - reg.mintap) * reg.numtaps
            if not reg.mintap <= tap <= reg.maxtap or not np.isclose(
                index, round(index), atol=1e-7
            ):
                raise ValueError(f"{element}Tap不在原生档位上")
            if set(request) - allowed:
                raise ValueError(f"{element}未知指令：{set(request)-allowed}")
            commands.append(
                f"Edit Transformer.{reg.transformer} wdg={reg.winding} tap={tap:.12g}"
            )
            trace[element] = {"requested": dict(request), "applied": applied}
            continue
        elif kind == "Storage" and name in session.batteries:
            allowed = {"kw"}
            dev = session.batteries[name]
            dt = session.step_seconds / 3600
            # 防止请求跨越能量边界；实际原生状态仍只在FinishTimeStep更新。
            upper = min(
                dev.max_kw,
                max(0, dev.soc - dev.reserve) * dev.max_kwh * dev.eta_discharge / dt,
            )
            lower = -min(
                dev.max_kw, max(0, 1 - dev.soc) * dev.max_kwh / dev.eta_charge / dt
            )
            kw = float(np.clip(request.get("kw", math.nan), lower, upper))
            applied["kw"] = kw
            state = "discharging" if kw > 0 else "charging" if kw < 0 else "idling"
            rate = "%discharge" if kw >= 0 else "%charge"
            parts = [
                f"state={state}",
                f"{rate}={abs(kw)/dev.max_kw*100:.12g}",
                "kvar=0",
            ]
        elif kind == "PVSystem" and name in session.pvs:
            allowed = {"kw", "kvar"}
            if set(request) != allowed:
                raise ValueError(f"{element}需要同时指定kw和kvar")
            dev = session.pvs[name]
            kw = float(np.clip(request["kw"], 0, min(dev.pmpp, dev.kva)))
            headroom = math.sqrt(max(0, dev.kva**2 - kw**2))
            q = float(
                np.clip(
                    request["kvar"],
                    -min(dev.kvar_max_abs, headroom),
                    min(dev.kvar_max, headroom),
                )
            )
            applied = {"kw": kw, "kvar": q}
            parts = [f"%Pmpp={kw/dev.pmpp*100:.12g}", f"kvar={q:.12g}"]
        elif kind == "Generator" and name in session.svgs:
            allowed = {"kvar"}
            dev = session.svgs[name]
            q = float(
                np.clip(request.get("kvar", math.nan), -dev.kvar_max_abs, dev.kvar_max)
            )
            applied = {"kvar": q}
            parts = ["kW=0", f"kvar={q:.12g}"]
        else:
            raise ValueError(f"未实现设备物理控制：{element}")
        if set(request) - allowed:
            raise ValueError(f"{element}未知指令：{set(request)-allowed}")
        commands.append(f"Edit {element} " + " ".join(parts))
        trace[element] = {"requested": dict(request), "applied": applied}
    return commands, trace
