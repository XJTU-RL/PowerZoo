"""物理与动作映射绑定的模型重放检查，不导入仿真或训练依赖。"""

from pathlib import Path
import json


def validate_checkpoint(model_dir, manifest):
    if manifest is None:
        return
    directory = Path(model_dir).resolve()
    candidates = [directory, directory.parent, directory.parent.parent]
    path = next(
        (
            d / "physics.manifest.json"
            for d in candidates
            if (d / "physics.manifest.json").is_file()
        ),
        None,
    )
    if path is None:
        raise ValueError(
            "模型缺少physics.manifest.json，不能证明动作顺序/物理模型与当前环境一致"
        )
    saved = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "model_sha256",
        "action_schema_sha256",
        "engine",
        "master",
        "overlays",
        "step_seconds",
        "storage",
        "pv",
        "svg",
    ):
        if saved.get(key) != manifest.get(key):
            raise ValueError(
                f"checkpoint的{key}与当前环境不一致；不能静默重放不同物理模型或动作映射"
            )
