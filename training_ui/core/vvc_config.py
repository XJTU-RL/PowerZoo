"""VVC 工作台的标准馈线选择和 DSS 入口校验。"""

from pathlib import Path

from training_ui.core.config_builder import PROJECT_ROOT
from utils.config_pipeline import read_yaml
from utils.config_validation import ConfigError

STANDARD_FEEDERS = ("13Bus", "34Bus", "123Bus")
YAML_FEEDER = "由 YAML 指定"


def _system_section(document):
    env = document.get("env", {})
    if not isinstance(env, dict):
        raise ConfigError("env 必须是映射")
    if (
        "system_ref" in document
        and "system_ref" in env
        and document["system_ref"] != env["system_ref"]
    ):
        raise ConfigError("顶层 system_ref 与 env.system_ref 不一致，请先统一或保留一处。")
    return env


def select_feeder(document, feeder):
    """修改原引用的位置，保留其他配置。"""
    env = _system_section(document)
    if "system_ref" in document or "system_ref" not in env:
        document["system_ref"] = feeder
    if "system_ref" in env:
        env["system_ref"] = feeder
    return document


def load_short_preset(feeder):
    if feeder not in STANDARD_FEEDERS:
        raise ConfigError("请先选择 13Bus、34Bus 或 123Bus。")
    preset = read_yaml(
        PROJECT_ROOT / "configs/experiments" / f"vvc_{feeder.lower()}_smoke.yaml"
    )
    return preset.pop("run")["algo"], preset


def describe_feeder(document, env, feeder):
    """依据已解析的配置确认标准馈线和实际 DSS 入口。"""
    overrides = _system_section(document)
    system_name = env["system_name"]
    folder = (PROJECT_ROOT / (env["dss_folder"] or f"node_systems/{system_name}")).resolve()
    master = Path(env["dss_file"])
    entry = (folder / master).resolve()
    source_label = "（由 YAML 指定）" if feeder == YAML_FEEDER else ""
    summary = f"实际系统 {system_name}{source_label} | DSS 入口 {entry}"
    if master.is_absolute() or not entry.is_relative_to(folder):
        raise ConfigError(f"DSS 入口必须是模型目录内的相对路径。\n{summary}")

    overlays = env["overlays"]
    if feeder in STANDARD_FEEDERS:
        expected = read_yaml(PROJECT_ROOT / "configs/systems" / f"{feeder}.yaml")["system"]
        standard_entry = (
            PROJECT_ROOT / expected["dss_folder"] / expected["dss_file"]
        ).resolve()
        reference = document.get("system_ref", overrides.get("system_ref", feeder))
        if (
            reference != feeder
            or system_name != expected["name"]
            or entry != standard_entry
            or overlays
        ):
            raise ConfigError(
                f"YAML 与标准馈线 {feeder} 不一致。请调整系统引用、DSS 路径或附加脚本，"
                f"也可选择「{YAML_FEEDER}」。\n{summary}"
            )
    if not entry.is_file():
        raise ConfigError(f"找不到 DSS 入口文件。\n{summary}")
    if overlays:
        summary += f" | 附加 DSS 脚本 {', '.join(overlays)}"
    return summary
