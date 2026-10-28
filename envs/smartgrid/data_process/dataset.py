"""SmartGrid 的唯一时序格式、可复现合成器和严格读取器（不导入 DSS）。"""

from dataclasses import asdict
from envs.smartgrid.base_env.env_config import DataConfig, GROUPS
from hashlib import sha256
from pathlib import Path
import csv
import json
import math

import numpy as np
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[3]
VERSION = 1
UNITS = {
    "time_s": "s",
    "load_multiplier": "pu_of_DSS_load_kW_kvar",
    "irradiance": "kW/m2",
    "temperature": "degC_array",
}
SCENARIOS = ("sunny", "cloudy", "cloud_ramp", "evening_peak", "high_pv", "night")
COLUMNS = ("time_s", *("load_" + g for g in GROUPS), "irradiance", "temperature")


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _generation(config, steps):
    return {
        k: v
        for k, v in dict(asdict(config), steps=steps, version=VERSION).items()
        if k not in ("path", "source", "split", "load_groups", "manifest_sha256")
    }


def _smooth(rng, n, amplitude, correlation=0.96):
    x = np.zeros(n)
    innovations = rng.normal(0, amplitude * math.sqrt(1 - correlation**2), n)
    for i in range(1, n):
        x[i] = correlation * x[i - 1] + innovations[i]
    return x


def generate_dataset(output, config=None, steps=360):
    """生成三组独立序列，拒绝覆盖；完整日窗口不会只抽取白天。"""
    config = config or DataConfig()
    if steps < 1 or steps * config.step_seconds > 86400:
        raise ValueError("合成场景长度须在一天以内；修改 steps 或 step_seconds")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"数据输出目录必须为空：{output}")
    manifest = {
        "schema_version": VERSION,
        "units": UNITS,
        "origin": "synthetic",
        "generator": _generation(config, steps),
        "episodes": [],
    }
    for split_id, split in enumerate(("train", "validation", "test")):
        for i in range(config.episodes_per_split):
            rng = np.random.default_rng(
                np.random.SeedSequence([config.seed, split_id, i])
            )
            scenario = SCENARIOS[i % len(SCENARIOS)]
            start = (9, 9, 10, 17, 10, 0)[i % 6]
            if steps * config.step_seconds > 6 * 3600:
                start = 0
            hour = start + np.arange(steps) * config.step_seconds / 3600
            solar = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
            weather = rng.uniform(0.88, 1.0) * np.ones(steps)
            if scenario == "cloudy":
                weather *= 0.35
            elif scenario == "cloud_ramp":
                center, width = steps * 0.5, max(steps * 0.1, 1)
                weather *= 1 - 0.75 * np.exp(
                    -(((np.arange(steps) - center) / width) ** 2)
                )
            g = np.clip(solar * (weather + _smooth(rng, steps, config.noise)), 0, 1.3)
            g *= config.irradiance_scale
            # 简化 NOCT=45°C 热模型 + 5分钟热惯性；是合成假设，不是实测组件温度。
            ambient = 20 + 7 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.uniform(-4, 4)
            target = ambient + 31.25 * g
            temperature = target.copy()
            alpha = 1 - np.exp(-config.step_seconds / 300)
            for j in range(1, steps):
                temperature[j] = temperature[j - 1] + alpha * (
                    target[j] - temperature[j - 1]
                )
            common = _smooth(rng, steps, config.noise)
            evening = np.exp(-(((hour - 19) / 2.2) ** 2))
            morning = np.exp(-(((hour - 8) / 2) ** 2))
            bases = (
                0.48 + 0.3 * evening + 0.15 * morning,
                0.45 + 0.35 * np.maximum(0, np.sin(np.pi * (hour - 7) / 13)),
                0.73 + 0.07 * np.cos(2 * np.pi * hour / 24),
            )
            level = rng.uniform(0.9, 1.1) * config.load_scale
            if scenario == "evening_peak":
                level *= 1.15
            elif scenario == "high_pv":
                level *= 0.65
            loads = [
                np.clip(
                    (b + common + _smooth(rng, steps, config.noise / 2)) * level, 0.1, 3
                )
                for b in bases
            ]
            values = np.column_stack((hour * 3600, *loads, g, temperature))
            name = f"{split}_{i:03d}.csv"
            with (output / name).open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(COLUMNS)
                writer.writerows([[f"{v:.10g}" for v in row] for row in values])
            manifest["episodes"].append(
                {
                    "id": f"{split}_{i:03d}",
                    "split": split,
                    "scenario": scenario,
                    "file": name,
                    "sha256": _digest(output / name),
                    "steps": steps,
                    "step_seconds": config.step_seconds,
                }
            )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output / "manifest.json"


def prepare_dataset(config, steps):
    """显式文件优先；合成缓存按全部生成参数寻址，用锁保证只生成一次。"""
    if config.path:
        path = Path(config.path)
        path = path if path.is_absolute() else ROOT / path
    else:
        key = sha256(
            json.dumps(_generation(config, steps), sort_keys=True).encode()
        ).hexdigest()[:20]
        folder = ROOT / "results" / "smartgrid_data" / key
        folder.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(folder) + ".lock", timeout=120):
            if not (folder / "manifest.json").exists():
                generate_dataset(folder, config, steps)
        path = folder / "manifest.json"
    dataset = ScenarioDataset(path, config.split, steps, config.step_seconds)
    if config.manifest_sha256 and dataset.sha256 != config.manifest_sha256:
        raise ValueError("数据清单摘要与训练快照不符")
    if config.source == "synthetic" and dataset.manifest.get(
        "generator"
    ) != _generation(config, steps):
        raise ValueError(
            "数据集生成参数与配置不一致；选择新缓存或使用 source=files 重放"
        )
    return dataset


class ScenarioDataset:
    """manifest + CSV 是唯一数据入口；不回填、截断、循环或猜测单位。"""

    def __init__(self, path, split, steps, step_seconds):
        self.path = Path(path).resolve()
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            self.manifest.get("schema_version") != VERSION
            or self.manifest.get("units") != UNITS
        ):
            raise ValueError("数据格式版本或单位不符，先显式转换数据")
        self.sha256 = _digest(self.path)
        if (
            not isinstance(self.manifest.get("origin"), str)
            or not self.manifest["origin"]
        ):
            raise ValueError("数据清单必须声明非空 origin 来源")
        self.episodes = []
        seen, files = set(), set()
        for entry in self.manifest["episodes"]:
            file = (self.path.parent / entry["file"]).resolve()
            if entry["id"] in seen or file in files:
                raise ValueError("重复场景编号或跨集合复用文件")
            seen.add(entry["id"])
            files.add(file)
            if entry["split"] not in ("train", "validation", "test"):
                raise ValueError("未知数据集合")
            if entry["split"] != split:
                continue
            if (
                not file.is_relative_to(self.path.parent)
                or _digest(file) != entry["sha256"]
            ):
                raise ValueError(f"数据路径越界或摘要不符：{file}")
            with file.open(encoding="utf-8") as f:
                if tuple(next(csv.reader(f))) != COLUMNS:
                    raise ValueError(f"数据列不符：{file}")
            values = np.loadtxt(file, delimiter=",", skiprows=1, ndmin=2)
            if (
                entry["steps"] != steps
                or entry["step_seconds"] != step_seconds
                or values.shape != (steps, len(COLUMNS))
            ):
                raise ValueError(f"数据长度或步长不符：{file}")
            if not np.isfinite(values).all() or not np.allclose(
                np.diff(values[:, 0]), step_seconds, rtol=0, atol=1e-5
            ):
                raise ValueError(f"非有限数值或不连续时间轴：{file}")
            if (
                (values[:, 1:4] < 0).any()
                or (values[:, 1:4] > 5).any()
                or (values[:, 4] < 0).any()
                or (values[:, 4] > 2).any()
                or (values[:, 5] < -80).any()
                or (values[:, 5] > 120).any()
            ):
                raise ValueError(f"时序数值超出物理检查范围：{file}")
            values.setflags(write=False)
            self.episodes.append((entry, values))
        if not self.episodes:
            raise ValueError(f"数据集没有 {split} 场景")

    def group_for(self, name, mapping):
        if name in mapping:
            return GROUPS.index(mapping[name])
        # 未分类设备使用稳定分组，仅表示合成实验假设。
        return int(sha256(name.encode()).hexdigest()[:8], 16) % len(GROUPS)
