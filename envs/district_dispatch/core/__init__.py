"""公共 API 按需导入，配置预览不加载训练或 OpenDSS 运行时。"""

from importlib import import_module

_EXPORTS = {
    "DistrictCircuitAdapter": (
        "envs.district_dispatch.core.circuit_adapter",
        "DistrictCircuitAdapter",
    ),
    "DistrictDispatchConfig": (
        "envs.district_dispatch.core.config",
        "DistrictDispatchConfig",
    ),
    "DistrictDispatchCore": (
        "envs.district_dispatch.core.dispatch_core",
        "DistrictDispatchCore",
    ),
    "District": ("envs.district_dispatch.core.district", "District"),
    "EVCharger": ("envs.district_dispatch.core.district", "EVCharger"),
    "PVUnit": ("envs.district_dispatch.core.district", "PVUnit"),
    "StorageUnit": ("envs.district_dispatch.core.district", "StorageUnit"),
    "DistrictLoadProfile": (
        "envs.district_dispatch.core.loadprofile",
        "DistrictLoadProfile",
    ),
    "PowerExchangeManager": (
        "envs.district_dispatch.core.power_exchange",
        "PowerExchangeManager",
    ),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
