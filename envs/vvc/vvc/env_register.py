"""VVC工厂；配置只由ConfigLoader解析，不再维护另一份系统/环境默认表。"""

import os
import re
from envs.vvc.vvc.env import Env
from utils.path_utils import get_node_systems_dir


def get_info_and_folder(env_name):
    """兼容旧名称与_s倍率，但所有默认值复用正式解析器。"""
    from utils.config_pipeline import ConfigLoader
    from utils.config_validation import build_dataclass, expand_legacy_variant
    from envs.vvc.vvc.vvc_config import VVCConfig

    matched = re.fullmatch(r"(.+)_s([0-9]+(?:\.[0-9]+)?)", env_name)
    scale = float(matched[2]) if matched else None
    name = matched[1] if matched else env_name
    variant = dict(expand_legacy_variant(name) or {})
    system = variant.pop("system_ref", name)
    loaded = ConfigLoader().load(
        "happo", "vvc", system_ref=system, values={"env": variant}
    )
    config = build_dataclass(VVCConfig, loaded.env_args)
    if scale is not None:
        config.scale = scale
        config.soc_w *= scale**2
    info = {**config.to_env_info(), **config.to_sys_info()}
    return info, str(get_node_systems_dir())


def make_base_env(config_or_name, dss_act=False, worker_idx=None):
    """创建 VVC 基础环境

    支持两种调用方式:
    1. make_base_env(config: VVCConfig, worker_idx=rank)  # 新路径
    2. make_base_env(env_name: str, dss_act, worker_idx)  # legacy 路径
    """
    from envs.vvc.vvc.vvc_config import VVCConfig

    if isinstance(config_or_name, VVCConfig):
        config = config_or_name
        # 旧核心需要一个明确字典视图；不重新解析默认配置
        base_info = config.to_env_info()
        base_info.update(config.to_sys_info())

        # 通过 resolve_system_path 解析 folder_path
        from utils.path_utils import resolve_system_path
        system_dir = resolve_system_path(config.dss_folder or config.system_name)
        folder_path = str(system_dir.parent)
        base_info['system_name'] = system_dir.name

        dss_act = config.dss_act
    else:
        # Legacy 路径: 字符串 env_name
        env_name = config_or_name
        base_info, folder_path = get_info_and_folder(env_name)

    if worker_idx is None:
        return Env(folder_path, base_info, dss_act)
    else:
        base_file = os.path.join(folder_path, base_info['system_name'], base_info['dss_file'])
        assert os.path.exists(base_file), base_file + ' 不存在'
        # 用上下文管理器及时关闭输入文件，避免 Windows 异常退出时文件仍被占用。
        with open(base_file, 'r', encoding='utf-8') as fin, open(
            base_file[:-4] + '_' + str(worker_idx) + '.dss', 'w', encoding='utf-8'
        ) as fout:
            for line in fin:
                if line.strip() == 'redirect loadshape.dss':
                    fout.write('redirect loadshape_' + str(worker_idx) + '.dss\n')
                else:
                    fout.write(line)
        info = base_info.copy()
        info['dss_file'] = info['dss_file'][:-4] + '_' + str(worker_idx) + '.dss'
        info['worker_idx'] = worker_idx
        return Env(folder_path, info, dss_act)

def remove_parallel_dss(env_name, num_workers):
    from envs.vvc.vvc.vvc_config import VVCConfig
    if isinstance(env_name, VVCConfig):
        from utils.path_utils import resolve_system_path
        system_dir = resolve_system_path(env_name.dss_folder or env_name.system_name)
        base_info = dict(system_name=system_dir.name, dss_file=env_name.dss_file)
        folder_path = str(system_dir.parent)
    else:
        base_info, folder_path = get_info_and_folder(env_name)
    base_main = os.path.join(folder_path, base_info['system_name'], base_info['dss_file'])
    base_loadshape = os.path.join(folder_path, base_info['system_name'], 'loadshape.dss')

    bases = [base_main, base_loadshape]

    for base in bases:
        fname = base[:-4] + '_' + str(num_workers) + '.dss'
        if os.path.exists(fname):
            os.remove(fname)
