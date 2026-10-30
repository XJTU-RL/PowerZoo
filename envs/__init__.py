"""环境清单只投影公共注册表；同步任务的logger按需导入。"""


def __getattr__(name):
    if name == "ENV_REGISTRY":
        from utils.environment_registry import ENVIRONMENTS

        return ENVIRONMENTS
    if name != "LOGGER_REGISTRY":
        raise AttributeError(name)
    from envs.vvc.vvc_logger import VVCLogger
    from envs.smartgrid.logging.smartgrid_logger import SmartGridLogger
    from envs.dsr.dsr_logger import DSRLogger
    from envs.district_dispatch.district_dispatch_logger import DistrictDispatchLogger

    registry = {
        "vvc": VVCLogger,
        "powerzoo": VVCLogger,
        "smartgrid": SmartGridLogger,
        "dsr": DSRLogger,
        "district_dispatch": DistrictDispatchLogger,
    }
    globals()[name] = registry
    return registry
