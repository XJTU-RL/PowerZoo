"""配置边界：复用消费方 dataclass，检查类型与旧字段，不读取任何配置文件。"""

import copy
import dataclasses
import importlib
import math
import types
from typing import Any, Union, get_args, get_origin, get_type_hints


_VVC_LEGACY_VARIANTS = {
    # 13Bus variants
    "13Bus_cbat": {"system_ref": "13Bus", "bat_act_num": float("inf")},
    "13Bus_soc": {"system_ref": "13Bus", "soc_w": 20.0 / 33},
    "13Bus_cbat_soc": {
        "system_ref": "13Bus",
        "bat_act_num": float("inf"),
        "soc_w": 20.0 / 33,
    },
    # 34Bus variants
    "34Bus_pv": {
        "system_ref": "34Bus",
        "pv_control_enabled": True,
        "irrad_dss": "irrad_up_down.dss",
    },
    "34Bus_cbat": {"system_ref": "34Bus", "bat_act_num": float("inf"), "power_w": 1.0},
    "34Bus_soc": {
        "system_ref": "34Bus",
        "power_w": 1.0,
        "soc_w": 500.0 / 33,
        "dis_w": 4.0 / 33,
    },
    "34Bus_cbat_soc": {
        "system_ref": "34Bus",
        "bat_act_num": float("inf"),
        "power_w": 1.0,
        "soc_w": 500.0 / 33,
        "dis_w": 4.0 / 33,
    },
    # 123Bus variants
    "123Bus_cbat": {"system_ref": "123Bus", "bat_act_num": float("inf")},
    "123Bus_soc": {"system_ref": "123Bus", "soc_w": 500.0 / 33, "dis_w": 5.0 / 33},
    "123Bus_cbat_soc": {
        "system_ref": "123Bus",
        "bat_act_num": float("inf"),
        "soc_w": 500.0 / 33,
        "dis_w": 5.0 / 33,
    },
    # 8500Node variants
    "8500Node_cbat": {"system_ref": "8500Node", "bat_act_num": float("inf")},
    "8500Node_soc": {
        "system_ref": "8500Node",
        "soc_w": 10000.0 / 33,
        "dis_w": 100.0 / 33,
    },
    "8500Node_cbat_soc": {
        "system_ref": "8500Node",
        "bat_act_num": float("inf"),
        "soc_w": 10000.0 / 33,
        "dis_w": 100.0 / 33,
    },
}


def expand_legacy_variant(env_name: str) -> dict | None:
    """Expand a legacy VVC variant name to config parameters.

    Args:
        env_name: Legacy environment name (e.g., '13Bus_cbat')

    Returns:
        Dict of {system_ref + parameter overrides}, or None if not a legacy variant
    """
    return _VVC_LEGACY_VARIANTS.get(env_name)


class ConfigError(ValueError):
    """包含参数路径的可操作配置错误。"""


from utils.environment_registry import ENV_CONFIG_CLASSES


def environment_family(name):
    if not isinstance(name, str):
        raise ConfigError("run.env 必须是字符串")
    if name == "powerzoo":
        return "vvc"
    for family in ENV_CONFIG_CLASSES:
        if name == family or name.startswith(family + "_"):
            return family
    raise ConfigError(f"未知环境 {name!r}；请在 ENV_CONFIG_CLASSES 注册其配置类")


def environment_class(name):
    module, symbol = ENV_CONFIG_CLASSES[environment_family(name)]
    return getattr(importlib.import_module(module), symbol)


def deep_merge(base, override):
    """字典逐键合并；列表和标量整体替换，null 不会被忽略。"""
    if not isinstance(base, dict) or not isinstance(override, dict):
        raise ConfigError("配置合并的两侧都必须是映射；不能用 null/列表替代参数节")
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def typed_value(value, annotation, path):
    """只做明确的数值转换，避免 bool('false')、int(1.5) 等静默改变。"""
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is Any:
        return copy.deepcopy(value)
    if origin in (Union, types.UnionType):
        for option in args:
            try:
                return typed_value(value, option, path)
            except ConfigError:
                pass
        raise ConfigError(f"{path}: {value!r} 不符合 {annotation}")
    if annotation is type(None):
        if value is None:
            return None
    elif dataclasses.is_dataclass(annotation):
        if isinstance(value, annotation):
            return copy.deepcopy(value)
        return build_dataclass(annotation, value, path)
    elif annotation is bool:
        if type(value) is bool:
            return value
    elif annotation in (int, float):
        if type(value) in (int, float, str):
            try:
                number = float(value)
                if annotation is float and not math.isnan(number):
                    return number
                if annotation is int and math.isfinite(number) and number.is_integer():
                    return int(number)
            except (ValueError, OverflowError):
                pass
    elif annotation is str:
        if isinstance(value, str):
            return value
    elif origin in (list, tuple) or annotation in (list, tuple):
        if isinstance(value, (list, tuple)):
            item_type = args[0] if args else Any
            converted = [
                typed_value(item, item_type, f"{path}[{i}]")
                for i, item in enumerate(value)
            ]
            return (
                tuple(converted)
                if origin is tuple or annotation is tuple
                else converted
            )
    elif origin is dict or annotation is dict:
        if isinstance(value, dict):
            key_type, value_type = args or (Any, Any)
            return {
                typed_value(key, key_type, path): typed_value(
                    item, value_type, f"{path}.{key}"
                )
                for key, item in value.items()
            }
    raise ConfigError(f"{path}: 期望 {annotation}，实际为 {value!r}")


def build_dataclass(cls, values, path="env", strict=True):
    if not isinstance(values, dict):
        raise ConfigError(f"{path}: 必须是映射，实际为 {values!r}")
    fields = {field.name for field in dataclasses.fields(cls)}
    unknown = set(values) - fields
    if strict and unknown:
        raise ConfigError(
            f"{path}: 未知参数 {', '.join(map(str, sorted(unknown)))}；不会静默忽略"
        )
    hints = get_type_hints(cls)
    converted = {
        key: typed_value(value, hints[key], f"{path}.{key}")
        for key, value in values.items()
        if key in fields
    }
    try:
        config = cls(**converted)
        if hasattr(config, "validate") and cls.__name__ != "DeviceConfig":
            config.validate()
        return config
    except (AssertionError, TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


REWARD_ALIASES = {
    "power_loss": "power_w",
    "capacitor": "cap_w",
    "regulator": "reg_w",
    "battery_soc": "soc_w",
    "battery_discharge": "dis_w",
    "pv_control": "pv_w",
}


def normalize_env(values, family):
    """旧格式只在此转换；返回的键均指向实际消费方字段。"""
    if not isinstance(values, dict):
        raise ConfigError("env: 必须是映射")
    result = copy.deepcopy(values)
    aliases = {
        "episode_length": "max_episode_steps",
        "num_steps": "max_episode_steps",
        "show_labels": "show_node_labels",
    }
    if family == "vvc":
        aliases.update(pv_control="pv_control_enabled", pv="pv_control_enabled")
    if family == "stackelberg" and "system_name" not in result:
        suffix = str(result.get("env_name", "")).lower().removeprefix("stackelberg_")
        if suffix in ("13bus", "34bus", "123bus"):
            result["system_name"] = suffix[:-3] + "Bus"
    if family in ("dsr", "district_dispatch"):
        aliases.update(voltage_min="v_min", voltage_max="v_max")
    for old, new in aliases.items():
        if old in result:
            value = result.pop(old)
            if new in result and result[new] != value:
                raise ConfigError(f"env.{old} 与 env.{new} 冲突，请只保留 {new}")
            result[new] = value
    if "environment_specific" in result and "env_specific_config" in result:
        raise ConfigError(
            "env: 不要同时提供 environment_specific 和 env_specific_config"
        )
    legacy = result.pop("environment_specific", result.pop("env_specific_config", {}))
    if not isinstance(legacy, dict):
        raise ConfigError("env.environment_specific 必须是映射")
    for key in ("system_name", "dss_file"):
        if key in legacy and key not in result:
            result[key] = legacy[key]
    constraints = legacy.get("constraints", {})
    for key, value in constraints.items():
        target = aliases.get(key, key)
        result.setdefault(target, value)
    weights = legacy.get("reward_weights", {})
    if "reward_overrides" in result:
        weights = deep_merge(weights, result.pop("reward_overrides") or {})
    if family == "vvc":
        for key, value in weights.items():
            result.setdefault(REWARD_ALIASES.get(key, key), value)
    elif family == "smartgrid":
        result["reward_weights"] = deep_merge(weights, result.get("reward_weights", {}))
        for name, short in REWARD_ALIASES.items():
            if short in result:
                result["reward_weights"][name] = result.pop(short)
    devices = legacy.get("devices", {})
    for group, name, short in (
        ("regulators", "regulator", "reg_act_num"),
        ("batteries", "battery", "bat_act_num"),
        ("pv_systems", "pv", "pv_act_num"),
        ("capacitors", "capacitor", None),
    ):
        device = devices.get(group, {})
        if not device:
            continue
        action_num = (
            float("inf")
            if device.get("action_space") == "continuous"
            else device.get("action_num")
        )
        if family == "smartgrid":
            mapped = {key: device[key] for key in ("control_enabled",) if key in device}
            if action_num is not None:
                mapped["action_num"] = action_num
            result[name] = deep_merge(mapped, result.get(name, {}))
        elif family == "vvc":
            if short and action_num is not None:
                result.setdefault(short, action_num)
            if group == "pv_systems" and "control_enabled" in device:
                result.setdefault("pv_control_enabled", device["control_enabled"])
    if family == "smartgrid":
        for short, name in (
            ("reg_act_num", "regulator"),
            ("bat_act_num", "battery"),
            ("pv_act_num", "pv"),
        ):
            if short in result:
                result.setdefault(name, {})["action_num"] = result.pop(short)
        for old in ("pv_control", "pv_control_enabled"):
            if old in result:
                result.setdefault("pv", {})["control_enabled"] = result.pop(old)
    for key, value in (result.pop("constraint_overrides", None) or {}).items():
        result[aliases.get(key, key)] = value
    if family == "district_dispatch" and "district_configs" in result:
        result["district_devices"] = {
            int(str(key).replace("zone_", "").replace("district_", "")): value
            for key, value in result.pop("district_configs").items()
        }
    if family == "district_dispatch" and "tie_lines" in result:
        for tie in result["tie_lines"]:
            if "name" in tie:
                tie["line_name"] = tie.pop("name")
    if family == "stackelberg" and "ess_config" in result:
        ess = result["ess_config"]
        if not isinstance(ess, dict):
            raise ConfigError("env.ess_config 必须是映射")
        if "self_discharge_rate" in ess:
            rate = typed_value(
                ess.pop("self_discharge_rate"),
                float,
                "env.ess_config.self_discharge_rate",
            )
            if not 0 <= rate <= 1:
                raise ConfigError(
                    "env.ess_config.self_discharge_rate 必须在 0 到 1 之间"
                )
            if "retention_factor" in ess and ess["retention_factor"] != 1 - rate:
                raise ConfigError("ESS 自放电损失率与 retention_factor 冲突")
            ess["retention_factor"] = 1 - rate
    return result


def legacy_env_config(cls, values):
    """旧 Python 接口保留兼容；正式训练在 ConfigLoader 处严格校验。"""
    families = {
        "VVCConfig": "vvc",
        "SmartGridConfig": "smartgrid",
        "DSRConfig": "dsr",
        "StackelbergConfig": "stackelberg",
        "DistrictDispatchConfig": "district_dispatch",
    }
    normalized = normalize_env(values, families[cls.__name__])
    return build_dataclass(
        cls,
        {key: value for key, value in normalized.items() if value is not None},
        strict=False,
    )
