"""恢复故障清单；共用正式prepare_run，字段按恢复任务校验，不伪造天气CSV。"""

from hashlib import sha256
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class RestorationDataset:
    def __init__(self, config):
        path = Path(config.path)
        self.path = path if path.is_absolute() else ROOT / path
        raw = self.path.read_bytes()
        self.sha256 = sha256(raw).hexdigest()
        if config.manifest_sha256 and self.sha256 != config.manifest_sha256:
            raise ValueError("故障清单摘要与训练快照不符")
        manifest = json.loads(raw)
        expected = {"schema_version", "origin", "units", "step_seconds", "episodes"}
        if (
            set(manifest) != expected
            or manifest["schema_version"] != 1
            or not manifest["origin"]
        ):
            raise ValueError("恢复数据清单格式或来源无效")
        if (
            manifest["units"]
            != {"power": "kW", "time": "s", "load_multiplier": "ratio"}
            or manifest["step_seconds"] != config.step_seconds
        ):
            raise ValueError("恢复数据单位/操作步长不符")
        self.episodes, seen = [], set()
        for episode in manifest["episodes"]:
            if set(episode) != {"id", "split", "unavailable", "load_multiplier"}:
                raise ValueError("故障场景有缺失或未知字段")
            if episode["id"] in seen or episode["split"] not in (
                "train",
                "validation",
                "test",
            ):
                raise ValueError("重复场景编号或未知集合")
            seen.add(episode["id"])
            if (
                not isinstance(episode["unavailable"], list)
                or not all(
                    isinstance(e, str) and "." in e for e in episode["unavailable"]
                )
                or len(set(episode["unavailable"])) != len(episode["unavailable"])
            ):
                raise ValueError("不可用设备必须为唯一原生ID列表")
            scale = episode["load_multiplier"]
            if type(scale) not in (int, float) or not math.isfinite(scale) or scale < 0:
                raise ValueError("负荷倍率必须非负有限")
            if episode["split"] == config.split:
                self.episodes.append(episode)
        if not self.episodes:
            raise ValueError("恢复清单没有所选split的场景")
