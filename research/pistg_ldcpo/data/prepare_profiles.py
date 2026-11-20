"""Ausgrid Solar Home 半小时数据 → 逐日 5-min 场景剖面（profiles.npz）。

论文口径（Sec. IV-A）：以 Ausgrid Solar Home Electricity Data（300 户
屋顶光伏家庭、2010-2013 三年半小时 GC/GG）线性插值构造每母线 5-min
负荷与 PV 剖面；DER 母线夜间（20:00-6:00）叠加住宅 EV 充电。

建模选择（论文未明说部分，记入 docs/pistg-repro-deviations.md）：
- 300 户固定随机分派到 26 个 DER 母线（每组 11-12 户），日间形状真实、
  每日每母线能量重标定到 Fig.5(b) 量级（基荷 ~850 kWh、EV ~350 kWh）；
- PV 安装峰值 = 0.9 × 该母线典型日峰负荷（表 I 渗透率口径），日内出力
  形状 = 该组 GG 归一形状（保留阴天差异）；
- q = p × tanφ（pf=0.85，标准 33-bus 系统功率因数）；
- 批发电价为合成 NEM 风格日形状 × 对数正态日因子；
- 按日哈希 80/10/10 划分 train/validation/test。

用法：python -m research.pistg_ldcpo.data.prepare_profiles \
        --source data/external/ausgrid-solar-home --out <同目录>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
STEPS = 288
DER_BUSES = sorted(set(range(4, 19)) | {19, 20, 21, 22} | set(range(27, 34)))
NON_DER_LOAD_BUSES = [2, 3, 23, 24, 25, 26]  # bus1 为变电站无负荷
TAN_PHI = np.tan(np.arccos(0.85))
# DER 母线日能量标定（M2 零控制标定冻结）：基荷 1600 + EV 300 kWh/
# 母线/日，使负荷峰 ~200 kW、PV 峰 ~150-170 kW，白天反送 ~3 MW 出现
# 论文量级的过压，与 Fig.5(a) 峰值口径和 Fig.9(a) 工作区间一致；
# Fig.5(b) 的 850 与 Fig.5(a) 在论文内自相矛盾（850 kWh/日不可能有
# 200 kW 峰），以现象口径优先，偏离文档记录。
DER_DAILY_ENERGY_KWH = 1600.0
EV_DAILY_ENERGY_KWH = 300.0
EV_DAY_SIGMA = 0.15
EV_FACTOR_RANGE = (0.7, 1.2)  # 日因子截断（标定：避免极端夜压/过载日）
PRICE_DAY_SIGMA = 0.2
PV_PENETRATION = 0.9
# PV 加成（M2 标定冻结）：0.9×峰荷 的字面渗透率不足以复现论文
# Fig.5(a)/Fig.9(a) 的白天过压现象（论文 Fig.5(a) 与 (b) 的尺度自相矛盾，
# 取现象口径），等效渗透率 1.35，偏离文档记录。
PV_BOOST = 1.5
# 非 DER 母线铭牌缩放（M2 标定冻结）：0.5 抑制夜间馈线整体压降，
# 使夜压落在论文 Fig.9(a) 工作区间附近。
NON_DER_SCALE = 0.5
PV_OVERSIZE = 1.1

# 合成批发电价形状锚点（分钟, AUD/MWh）：夜间低、早晚高、正午光伏压低。
PRICE_ANCHORS = [
    (0, 40.0), (360, 40.0), (420, 65.0), (540, 75.0), (720, 35.0),
    (900, 30.0), (1020, 60.0), (1140, 110.0), (1260, 85.0), (1440, 45.0),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_year(csv_path: Path):
    """一年 (dates, customers, GC48, GG48)；要求行数=日×户严格配对。"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig", skiprows=1, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    df["Consumption Category"] = df["Consumption Category"].str.strip()
    raw_dates = df["date"].str.strip()
    try:
        df["date"] = pd.to_datetime(raw_dates, format="%d-%b-%y")
    except ValueError:
        df["date"] = pd.to_datetime(raw_dates, format="%d/%m/%Y")
    # 前五列为元数据；2011 起尾部多 Row Quality 列，按"含冒号"筛时间列。
    half_hours = [c for c in df.columns[5:] if ":" in str(c)]
    if len(half_hours) != 48:
        raise ValueError(f"{csv_path.name} 半小时列数异常：{len(half_hours)}")
    frames = {}
    all_dates = None
    customers = np.sort(df["Customer"].unique())
    for cat in ("GC", "GG"):
        sub = df[df["Consumption Category"] == cat]
        dates = sorted(sub["date"].unique())
        if all_dates is None:
            all_dates = dates
        if list(dates) != list(all_dates):
            raise ValueError(f"{csv_path.name} {cat} 与 GC 日期集合不一致")
        # 重索引到完整 (日×户) 网格：缺失行按 0 计（2012-2013 有零星缺行）。
        full = pd.MultiIndex.from_product(
            [all_dates, customers], names=["date", "Customer"]
        )
        sub = sub.set_index(["date", "Customer"]).reindex(full)
        values = sub[half_hours].apply(pd.to_numeric, errors="coerce")
        frames[cat] = np.nan_to_num(
            values.to_numpy(np.float32)
        ).reshape(len(all_dates), len(customers), 48)
    dates_out = [pd.Timestamp(d) for d in all_dates]
    return dates_out, customers, frames["GC"], frames["GG"]


def _ev_shape(steps: int = STEPS) -> np.ndarray:
    """夜间 EV 归一形状（能量积分=1）：20:00 升、平台、04:30-06:00 降。"""
    minutes = np.arange(steps) * (1440 // steps)
    shape = np.zeros(steps, dtype=np.float64)
    ramp_up = (minutes >= 1200) & (minutes < 1290)
    flat = (minutes >= 1290) | (minutes < 270)
    ramp_down = (minutes >= 270) & (minutes < 360)
    shape[ramp_up] = (minutes[ramp_up] - 1200) / 90.0
    shape[flat] = 1.0
    shape[ramp_down] = (360.0 - minutes[ramp_down]) / 90.0
    energy = shape.sum() * (1440.0 / steps / 60.0)
    return shape / energy  # kWh 权重：乘以目标日能量(kWh)即得 kW 剖面


def _price_shape(steps: int = STEPS) -> np.ndarray:
    minutes = np.arange(steps) * (1440 // steps)
    anchor_min, anchor_val = map(np.asarray, zip(*PRICE_ANCHORS))
    return np.interp(minutes, anchor_min, anchor_val)


def _interp_half_hourly(values48: np.ndarray, steps: int = STEPS) -> np.ndarray:
    """48 个半小时值（标签为区间末）→ 5-min 线性插值，锚点半小时中点。"""
    anchor_min = (np.arange(48) + 0.5) * 30.0
    minutes = (np.arange(steps) + 0.5) * (1440.0 / steps)
    return np.interp(minutes, anchor_min, values48)


def _nameplate_from_dss(case_folder: Path) -> dict[int, tuple[float, float]]:
    """从 network.dss 解析负荷铭牌（单一事实源）。"""
    nameplate = {}
    for line in (case_folder / "network.dss").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0].lower() != "new":
            continue
        name, fields = parts[1].lower(), " ".join(parts[2:])
        if not name.startswith("load."):
            continue
        bus = int(name.split("_b")[-1])
        kw = float(fields.split("kW=")[1].split()[0])
        kvar = float(fields.split("kvar=")[1].split()[0])
        nameplate[bus] = (kw, kvar)
    if sorted(nameplate) != sorted(NON_DER_LOAD_BUSES + DER_BUSES):
        raise ValueError("network.dss 负荷母线集合与场景定义不一致")
    return nameplate


def build(source: Path, case_folder: Path, out: Path) -> None:
    rng = np.random.default_rng(SEED)
    years = ["2010-2011", "2011-2012", "2012-2013"]
    dates, customers, gc_all, gg_all = [], None, [], []
    for year in years:
        y_dates, y_customers, y_gc, y_gg = _load_year(source / f"Solar home {year}.csv")
        if customers is None:
            customers = y_customers
        elif not np.array_equal(customers, y_customers):
            raise ValueError(f"{year} 客户集合与首年不一致")
        dates.extend(y_dates)
        gc_all.append(y_gc)
        gg_all.append(y_gg)
    gc_all = np.concatenate(gc_all)  # (D, 300, 48)
    gg_all = np.concatenate(gg_all)
    n_days = len(dates)

    # 固定 home→DER 母线分派：打乱后均分 26 组。
    perm = rng.permutation(len(customers))
    groups = np.array_split(perm, len(DER_BUSES))

    # 逐日、逐母线聚合（48 点），再线性插值到 288 点。
    interp = np.empty((n_days, len(DER_BUSES), STEPS), np.float32)
    pv_shape = np.empty_like(interp)
    for j, members in enumerate(groups):
        gc_bus = gc_all[:, members, :].sum(axis=1)  # (D, 48) kWh/半小时
        gg_bus = gg_all[:, members, :].sum(axis=1)
        for d in range(n_days):
            interp[d, j] = _interp_half_hourly(gc_bus[d])
            pv_shape[d, j] = _interp_half_hourly(gg_bus[d])

    # DER 母线基荷：每日能量重标定到目标 kWh（形状保持真实）。
    scale = DER_DAILY_ENERGY_KWH / (interp.sum(axis=2) * (1440.0 / STEPS / 60.0))
    p_base = interp * scale[:, :, None]  # (D, 26, 288) kW

    # EV：固定形状 × 日因子 × 目标能量。
    ev_template = _ev_shape()[None, None, :]
    ev_factor = np.clip(
        rng.lognormal(0.0, EV_DAY_SIGMA, (n_days, 1)), *EV_FACTOR_RANGE
    ).astype(np.float32)
    p_ev = ev_template * ev_factor[:, :, None] * EV_DAILY_ENERGY_KWH

    p_der = p_base + p_ev  # (D, 26, 288) kW

    # PV 安装峰值 = 0.9 × 母线典型日峰负荷（全样本日均，静态容量）。
    typical_peak = p_der.max(axis=2).mean(axis=0)  # (26,)
    pv_peak = PV_PENETRATION * PV_BOOST * typical_peak
    pv_max = pv_shape.max(axis=2)  # (D, 26)
    pv_available = np.where(
        pv_max[:, :, None] > 0,
        pv_shape * (pv_peak[None, :, None] / np.maximum(pv_max[:, :, None], 1e-9)),
        0.0,
    ).astype(np.float32)

    # 非 DER 母线：铭牌 × 全网日形状（当日全部 GC 归一到均值 1）。
    nameplate = _nameplate_from_dss(case_folder)
    system_shape48 = gc_all.sum(axis=1)  # (D, 48)
    system_288 = np.stack([_interp_half_hourly(system_shape48[d]) for d in range(n_days)])
    system_norm = system_288 / system_288.mean(axis=1, keepdims=True)

    p_load = np.zeros((n_days, STEPS, 33), np.float32)
    q_load = np.zeros_like(p_load)
    pv_out = np.zeros((n_days, STEPS, 26), np.float32)
    for j, bus in enumerate(DER_BUSES):
        p_load[:, :, bus - 1] = p_der[:, j, :]
        pv_out[:, :, j] = pv_available[:, j, :]
    for bus in NON_DER_LOAD_BUSES:
        kw, _ = nameplate[bus]
        p_load[:, :, bus - 1] = kw * NON_DER_SCALE * system_norm
    q_load[:] = p_load * np.float32(TAN_PHI)

    # 电价：合成形状 × 对数正态日因子。
    price = _price_shape()[None, :] * rng.lognormal(
        0.0, PRICE_DAY_SIGMA, (n_days, 1)
    )
    price = price.astype(np.float32)

    # 按日哈希 80/10/10 划分。
    day_keys = [d.strftime("%Y-%m-%d") for d in dates]
    split = {}
    for key in day_keys:
        h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 2**32
        split[key] = "train" if h < 0.8 else ("validation" if h < 0.9 else "test")

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "profiles.npz",
        p_load_kw=p_load,
        q_load_kvar=q_load,
        pv_available_kw=pv_out,
        price_aud_per_mwh=price,
        day_keys=np.array(day_keys),
    )
    manifest = {
        "source": {
            "zip": _sha256(source / "ausgrid.zip"),
            "years": {y: _sha256(source / f"Solar home {y}.csv") for y in years},
        },
        "seed": SEED,
        "der_buses": DER_BUSES,
        "parameters": {
            "der_daily_energy_kwh": DER_DAILY_ENERGY_KWH,
            "ev_daily_energy_kwh": EV_DAILY_ENERGY_KWH,
            "pv_penetration": PV_PENETRATION,
            "pv_inverter_oversize": PV_OVERSIZE,
            "power_factor": 0.85,
            "price": "synthetic NEM-style shape × lognormal(0,0.2) daily",
        },
        "pv_peak_kw": {str(b): round(float(v), 3) for b, v in zip(DER_BUSES, pv_peak)},
        "split": split,
        "days": n_days,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    counts = {s: sum(1 for v in split.values() if v == s) for s in
              ("train", "validation", "test")}
    print(f"生成 {n_days} 天：{counts}；DER 母线基荷均值日能量 "
          f"{float(p_base.sum(axis=2).mean() * 1440 / STEPS / 60):.0f} kWh，"
          f"PV 峰值范围 {pv_peak.min():.1f}-{pv_peak.max():.1f} kW")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("data/external/ausgrid-solar-home")
    )
    parser.add_argument(
        "--case", type=Path, default=Path("node_systems/33Bus_CBESS")
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    build(args.source, args.case, args.out or args.source)
