"""逐日场景剖面的运行时存取：训练/测试按天留出，采样确定性可复现。

数据契约（由 prepare_profiles.py 生成 profiles.npz + manifest.json）：
- p_load_kw:    (D, 288, 33) kW，有功负荷（含 DER 母线基荷+EV、非 DER 标准负荷）
- q_load_kvar:  (D, 288, 33) kvar
- pv_available_kw: (D, 288, 26) kW，DER 母线可用 PV 出力 p̃（弃光前）
- price_aud_per_mwh: (D, 288) AUD/MWh 批发价
- day_keys:     (D,) 日期字符串 YYYY-MM-DD
- 母线序 = 经典编号 1..33（列 i ↔ bus i+1）；DER 母线序 = case_meta der_buses
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DayProfile:
    """一天 288 步的全部外生剖面（步 t 覆盖 [t*300, (t+1)*300)）。"""

    day_key: str
    p_load_kw: np.ndarray  # (288, 33)
    q_load_kvar: np.ndarray  # (288, 33)
    pv_available_kw: np.ndarray  # (288, 26)
    price_aud_per_mwh: np.ndarray  # (288,)

    @property
    def price_aud_per_kwh_5min(self) -> np.ndarray:
        """步内能量价：AUD/MWh → AUD/kWh × (300/3600) h。"""
        return self.price_aud_per_mwh / 1000.0 * (300.0 / 3600.0)


class ProfileStore:
    """剖面库：按 split（train/validation/test）留出天数后抽样。"""

    def __init__(self, folder: Path, *, split: str | None = None):
        folder = Path(folder)
        self.manifest = json.loads(
            (folder / "manifest.json").read_text(encoding="utf-8")
        )
        splits = self.manifest["split"]
        if split is not None and (not splits or set(splits.values()) != {split}):
            raise ValueError(f"CBESS 需要只包含 {split} 的剖面包：{folder}")
        with np.load(folder / "profiles.npz", allow_pickle=False) as data:
            self.day_keys = [str(k) for k in data["day_keys"]]
            if len(self.day_keys) != len(splits) or set(splits) != set(self.day_keys):
                raise ValueError("manifest 按天划分与剖面天数不一致")
            self.p_load_kw = data["p_load_kw"]
            self.q_load_kvar = data["q_load_kvar"]
            self.pv_available_kw = data["pv_available_kw"]
            self.price_aud_per_mwh = data["price_aud_per_mwh"]
        self._split_index: dict[str, list[int]] = {}
        for i, key in enumerate(self.day_keys):
            self._split_index.setdefault(splits[key], []).append(i)
        unknown = set(self._split_index) - {"train", "validation", "test"}
        if unknown:
            raise ValueError(f"未知 split：{unknown}")

    def fingerprint(self) -> str:
        """为实际加载的剖面、采样顺序及静态容量生成摘要。"""
        from hashlib import sha256

        digest = sha256()
        for value in (self.manifest, self.day_keys):
            digest.update(json.dumps(value, sort_keys=True).encode())
        for array in (
            self.p_load_kw,
            self.q_load_kvar,
            self.pv_available_kw,
            self.price_aud_per_mwh,
        ):
            dtype = array.dtype.newbyteorder("<")
            digest.update(json.dumps([dtype.str, array.shape]).encode())
            digest.update(array.astype(dtype, copy=False).tobytes())
        return digest.hexdigest()

    def episode_count(self, split: str) -> int:
        return len(self._split_index[split])

    def sample_episode(self, rng: np.random.Generator, split: str) -> DayProfile:
        """有放回均匀抽一天（论文 2000 训练回合对天数池重复采样）。"""
        pool = self._split_index[split]
        idx = int(rng.integers(len(pool)))
        return self.episode(pool[idx])

    def episode(self, index: int) -> DayProfile:
        return DayProfile(
            day_key=self.day_keys[index],
            p_load_kw=self.p_load_kw[index],
            q_load_kvar=self.q_load_kvar[index],
            pv_available_kw=self.pv_available_kw[index],
            price_aud_per_mwh=self.price_aud_per_mwh[index],
        )
