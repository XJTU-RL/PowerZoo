"""SMART-DS 忠实时序：逐负荷 P/Q 与逐 PV 辐照度的原始 15 分钟数据接口。

不替代 ScenarioDataset（合成/分类倍率路径原样保留）；本模块按 SMART-DS
官方发布物直接供数，供外部算例任务与基线脚本使用。

单位与时间基准（上游口径，详见 docs/smart-ds-import.md）：
- 负荷 pu 曲线 ``profiles/<shape>_pu.csv``（35040 行单列，无表头）= 负荷
  对象年峰值归一；该时刻实际 kW/kvar = DSS 铭牌 × pu[t]。中心抽头用户是
  两个负荷对象共享同一曲线、铭牌各半——逐对象写入即无重复计负荷，禁止
  把曲线再放大一倍或合并对象。
- PV 曲线 ``profiles/SFO_*.csv`` 单位 **kW/m²**（辐照度，非 pu）；引擎
  语义 PVSystem 出力 = Pmpp × irradiance（本模型无 P-TCurve/EffCurve，
  以 OpenDSS 实测核对）。
- 时间轴：当地标准时（SFO=UTC-8，全年无夏令时），35040 点 × 15 分钟，
  **按行号对齐**（上游 parquet 为区间末端标记、solar_data 为区间起点
  标记，官方示例统一按行号）。默认不插值不重采样：步长恒为原始 15 分钟
  （step_seconds=900），episode 步数=96×天数。
- 温度取自 ``solar_data/<SFO 组合>_full.csv`` 的 Temperature 列（°C，
  环境温度；无 P-TCurve 时对出力无影响，仅如实透传）。
"""

from dataclasses import dataclass
from pathlib import Path
import math
import re

import numpy as np

POINTS_PER_YEAR = 35040
INTERVAL_SECONDS = 900
STEPS_PER_DAY = 24 * 60 // 15  # 96 步/天（15 分钟原始采样）
DAYS_PER_YEAR = POINTS_PER_YEAR // STEPS_PER_DAY  # 365


def _shape_definitions(scripts):
    """从脚本序列解析 ``new loadshape.<name> ... mult=(file=..)`` 定义。

    返回 {shape名: {"mult": 相对路径或 None, "qmult": 相对路径或 None,
    "origin": "文件:行号"}}；路径相对该脚本所在目录。
    """
    shapes = {}
    for relative, text in scripts:
        for number, raw in enumerate(text.splitlines(), start=1):
            line = raw.split("!")[0].split("//")[0].strip()
            if not line:
                continue
            head = line.split(None, 1)
            if len(head) < 2 or head[0].lower() != "new":
                continue
            target = head[1].split(".", 1)
            if len(target) != 2 or target[0].lower() != "loadshape":
                continue
            name = target[1].split()[0].lower()
            entry = shapes.setdefault(
                name, {"mult": None, "qmult": None, "origin": f"{relative}:{number}"}
            )
            for key in ("mult", "qmult"):
                match = re.search(rf"{key}\s*=\s*\(\s*file\s*=\s*([^)]+)", line, re.I)
                if match:
                    entry[key] = match.group(1).strip().strip('"')
    return shapes


def _collect_scripts(model_dir, master):
    """从主入口递归收集脚本（redirect/include），返回 [(相对路径, text)]。"""
    entry = model_dir / master
    stack, seen, scripts = [entry.resolve()], set(), []
    while stack:
        path = stack.pop()
        key = path.as_posix()
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        text = path.read_text(encoding="utf-8", errors="replace")
        scripts.append((path.relative_to(model_dir).as_posix(), text))
        for line in text.splitlines():
            clean = line.split("!")[0].split("//")[0].strip()
            parts = clean.split(None, 1)
            if len(parts) == 2 and parts[0].lower() in ("redirect", "include"):
                target = (path.parent / parts[1].strip().strip('"').replace("\\", "/")).resolve()
                if target.is_relative_to(model_dir.resolve()):
                    stack.append(target)
    return scripts


@dataclass
class SmartDSTimeseries:
    """绑定 SMART-DS 布局根与入口的逐设备时序供数器。

    用法：``series = SmartDSTimeseries(layout_root, master_rel).bind(session)``
    然后 ``series.apply(session, index)`` 在每步 begin_interval 后写入该时刻
    的逐负荷绝对 P/Q 与逐 PV 辐照/温度；不推进时间或能量（收尾仍走会话）。
    """

    layout_root: Path
    master: str

    def __post_init__(self):
        self.layout_root = Path(self.layout_root).resolve()
        scripts = _collect_scripts(self.layout_root, self.master)
        if not scripts:
            raise FileNotFoundError(f"入口脚本不可读：{self.layout_root / self.master}")
        self.shapes = _shape_definitions(scripts)
        self._curves = {}
        self._temperatures = {}
        self.load_shapes = {}  # bind 后：负荷名 -> shape 名
        self.pv_shapes = {}  # bind 后：PV 名 -> shape 名
        self.nameplates = {}  # bind 后：负荷名 -> (铭牌 kW, 铭牌 kvar)

    def _curve(self, shape_name, key):
        entry = self.shapes.get(shape_name)
        if entry is None:
            raise KeyError(f"未定义的 LoadShape：{shape_name}")
        reference = entry[key]
        if not reference:
            raise KeyError(f"LoadShape.{shape_name} 缺少 {key} 数据文件（{entry['origin']}）")
        cache_key = (shape_name, key)
        if cache_key not in self._curves:
            path = self._resolve(entry["origin"], reference)
            values = np.loadtxt(path, dtype=np.float64)
            if values.ndim != 1 or values.size != POINTS_PER_YEAR:
                raise ValueError(
                    f"{path} 应为单列 {POINTS_PER_YEAR} 点，实测 {values.shape}"
                )
            if not np.isfinite(values).all() or values.min() < 0:
                raise ValueError(f"{path} 含负值或非有限值")
            self._curves[cache_key] = values
        return self._curves[cache_key]

    def _resolve(self, origin, reference):
        base = self.layout_root / Path(origin).parent
        target = (base / reference.replace("\\", "/")).resolve()
        if not target.is_relative_to(self.layout_root):
            raise ValueError(f"曲线引用越界：{reference}（来自 {origin}）")
        if not target.is_file():
            raise FileNotFoundError(f"曲线文件缺失：{target}")
        return target

    def _temperature(self, shape_name):
        """PV 曲线组合对应的 solar_data 温度列（°C）。"""
        if shape_name in self._temperatures:
            return self._temperatures[shape_name]
        mult_file = self.shapes[shape_name]["mult"]
        stem = Path(mult_file.replace("\\", "/")).stem
        solar = self.layout_root / "solar_data" / f"{stem}_full.csv"
        if not solar.is_file():
            raise FileNotFoundError(
                f"缺少辐照/温度数据 {solar}（清单发现应包含该组合）"
            )
        rows = np.genfromtxt(
            solar, delimiter=",", names=True, dtype=None, encoding="utf-8"
        )
        temperature = np.asarray(rows["Temperature"], dtype=np.float64)
        if temperature.size != POINTS_PER_YEAR:
            raise ValueError(f"{solar} Temperature 列应为 {POINTS_PER_YEAR} 点")
        self._temperatures[shape_name] = temperature
        return temperature

    def bind(self, session):
        """与已编译会话交叉校验并缓存铭牌；返回自身。

        校验：每个发现的负荷/PV 必须带 yearly 曲线且曲线已定义；负荷 pu
        曲线年峰 ≈1.0（数据完整性）；逐对象铭牌与曲线峰值口径一致。
        """
        circuit = session.native
        for name in session.loads:
            circuit.SetActiveElement(f"load.{name}")
            shape = circuit.ActiveCktElement.Properties("yearly").Val.strip().lower()
            if not shape or shape == "none":
                raise ValueError(f"负荷 {name} 未绑定 yearly 曲线（SMART-DS 应有）")
            if shape not in self.shapes:
                raise ValueError(
                    f"负荷 {name} 引用的曲线 {shape} 未在脚本中定义或文件缺失"
                )
            self.load_shapes[name] = shape
            device = session.loads[name]
            self.nameplates[name] = (device.kw, device.kvar)
            peak = float(self._curve(shape, "mult").max())
            if not math.isclose(peak, 1.0, rel_tol=0, abs_tol=1e-9):
                raise ValueError(
                    f"曲线 {shape} 年峰 {peak} ≠ 1.0：pu 口径损坏，拒绝使用"
                )
        for name in session.pvs:
            circuit.SetActiveElement(f"pvsystem.{name}")
            shape = circuit.ActiveCktElement.Properties("yearly").Val.strip().lower()
            if not shape or shape == "none":
                raise ValueError(f"PVSystem {name} 未绑定 yearly 曲线")
            if shape not in self.shapes:
                raise ValueError(
                    f"PVSystem {name} 引用的曲线 {shape} 未定义或文件缺失"
                )
            self.pv_shapes[name] = shape
            self._temperature(shape)  # 预载并校验温度列
        return self

    def mapping(self):
        """设备→曲线映射清单（供 manifest/报告记录数据映射）。"""
        return {
            "load_shapes": dict(self.load_shapes),
            "pv_shapes": dict(self.pv_shapes),
            "shape_files": {
                name: {
                    key: value
                    for key, value in entry.items()
                    if key in ("mult", "qmult", "origin")
                }
                for name, entry in self.shapes.items()
                if name in set(self.load_shapes.values())
                or name in set(self.pv_shapes.values())
            },
            "points_per_year": POINTS_PER_YEAR,
            "interval_seconds": INTERVAL_SECONDS,
            "units": {
                "load_curve": "pu_of_object_annual_peak",
                "pv_curve": "kW/m2",
                "temperature": "degC ambient",
            },
            "time_base": "当地标准时（无夏令时），按行号对齐",
        }

    def row(self, index):
        """第 index 个 15 分钟点的逐设备数值（0-based，全年 0..35039）。"""
        if not 0 <= index < POINTS_PER_YEAR:
            raise IndexError(f"时刻索引越界：{index}")
        loads = {}
        for name, shape in self.load_shapes.items():
            kw_base, kvar_base = self.nameplates[name]
            loads[name] = (
                kw_base * float(self._curve(shape, "mult")[index]),
                kvar_base * float(self._curve(shape, "qmult")[index]),
            )
        pv = {}
        for name, shape in self.pv_shapes.items():
            pv[name] = (
                float(self._curve(shape, "mult")[index]),
                float(self._temperature(shape)[index]),
            )
        return {
            "index": index,
            "time_s": index * INTERVAL_SECONDS,
            "loads": loads,
            "pv": pv,
        }

    def apply(self, session, index):
        """把该时刻逐负荷 P/Q 与逐 PV 辐照/温度写入会话；不推进时间。"""
        row = self.row(index)
        session.apply(
            {
                f"Load.{name}": {"kw": kw, "kvar": kvar}
                for name, (kw, kvar) in row["loads"].items()
            }
        )
        if row["pv"]:
            session.set_pv_conditions(
                0.0,
                0.0,
                mapping={
                    name: (irradiance, temperature)
                    for name, (irradiance, temperature) in row["pv"].items()
                },
            )
        return row


def validate_holdout_days(days):
    """按天留出门禁：日号 0..364 整数（排除 bool）、各列表无重复、
    训练与验证不相交。非法配置在数据集构造点立即失败，不延迟到
    worker 内崩溃（复审 4d8af3b5：重叠配置曾被接受，两个 split 都
    reset 到同一天，留出形同虚设）。
    """
    seen = {}
    for split, values in days.items():
        for day in values:
            # bool 是 int 子类，必须显式排除（True 曾通过整数检查）
            if isinstance(day, bool) or not isinstance(day, int):
                raise ValueError(
                    f"smartds_series {split}_days 含非整数日号：{day!r}"
                    "（须为 0..364 整数）"
                )
            if not 0 <= day < DAYS_PER_YEAR:
                raise ValueError(
                    f"smartds_series {split}_days 日号越界：{day}"
                    f"（须为 0..{DAYS_PER_YEAR - 1}）"
                )
            if day in seen and seen[day] != split:
                raise ValueError(
                    f"smartds_series train_days 与 validation_days 必须不相交："
                    f"day{day:03d} 同时出现在两个 split"
                )
            if day in seen:
                raise ValueError(
                    f"smartds_series {split}_days 含重复日号：day{day:03d}"
                )
            seen[day] = split


class SmartDSEpisodeDataset:
    """SMART-DS 逐设备时序的训练数据集：一个 episode = 一天（96×15min）。

    按天划分训练/评估留出（相邻时刻随机划分会造成泄漏，审核指出）：train_days
    与 eval_days 不相交；评估 worker（envs_tools 强制 validation split）只
    见 eval 天。逐设备数据由绑定的 SmartDSTimeseries 直接写入会话；
    profile 列仅作观测上下文（全网平均辐照/温度/负荷倍率），不参与写入。
    """

    def __init__(self, layout_root, master, days, split="train", day_indices=None):
        if split not in ("train", "validation"):
            raise ValueError(
                f"smartds_series split 只支持 train/validation，收到：{split!r}"
            )
        validate_holdout_days(days)
        self.series_source = (Path(layout_root), master)
        self.step_seconds = INTERVAL_SECONDS
        self.steps_per_episode = STEPS_PER_DAY
        self.day_indices = {
            "train": list(days.get("train", [])),
            "validation": list(days.get("validation", [])),
        }
        self.split = split
        from hashlib import sha256
        import json as _json

        digest = sha256(
            _json.dumps(
                {
                    "layout": str(layout_root),
                    "master": master,
                    "days": {k: sorted(v) for k, v in self.day_indices.items()},
                    "steps": STEPS_PER_DAY,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.sha256 = digest
        self._series = None  # 首次 bind 时构建（避免构造期读全部曲线）
        self._bind_hook = None

    # -- 与 ScenarioDataset 相同的最小接口面（control_env 消费） ----------
    @property
    def episodes(self):
        days = self.day_indices[self.split]
        return [
            (
                {
                    "id": f"day{day:03d}",
                    "day": day,
                    "scenario": f"smartds_day{day:03d}",
                },
                self._episode_profile(day),
            )
            for day in days
        ]

    def _episode_profile(self, day):
        """(96,6) 上下文矩阵：时间轴 + 全网平均负荷倍率/辐照/温度（仅观测）。"""
        series = self._ensure_series()
        rows = [series.row(day * STEPS_PER_DAY + t) for t in range(STEPS_PER_DAY)]
        profile = np.zeros((STEPS_PER_DAY, 6))
        for t, row in enumerate(rows):
            loads = np.array([v[0] for v in row["loads"].values()])
            pvs = np.array([v[0] for v in row["pv"].values()]) if row["pv"] else np.zeros(1)
            temps = np.array([v[1] for v in row["pv"].values()]) if row["pv"] else np.zeros(1)
            profile[t] = (
                t * INTERVAL_SECONDS,
                float(loads.mean() / 1e3),
                float(loads.mean() / 1e3),
                float(loads.mean() / 1e3),
                float(pvs.mean()),
                float(temps.mean()),
            )
        return profile

    def group_for(self, name, mapping):
        return 0  # 逐设备写入不使用组倍率；接口占位

    def max_episode_steps_ok(self, steps):
        return steps <= STEPS_PER_DAY

    def bind(self, session):
        """把底层 SmartDSTimeseries 绑定到会话（构造后调用一次）。"""
        if self._series is None:
            layout, master = self.series_source
            self._series = SmartDSTimeseries(layout, master).bind(session)
        return self._series

    def _ensure_series(self):
        if self._series is None:
            raise RuntimeError("SmartDSEpisodeDataset 需先 bind(session)")
        return self._series

    def apply_step(self, session, day, t):
        """写入该天第 t 步的逐设备负荷/PV 辐照（不推进时间）。"""
        return self._ensure_series().apply(session, day * STEPS_PER_DAY + t)
