#!/usr/bin/env python3
"""
PowerZooEnv 包装器，提供更简单的初始化接口, 在多智能体算法汇总并没有使用该wrapper
集成优化功能并保持向后兼容性
"""
import os
from envs.power_envs.powerzoo_llm.powerzoo_env import PowerZooEnv, OptimizedPowerZooEnv
from envs.power_envs.powerzoo_llm.env import Env

# 向后兼容性导入
try:
    from llm_core.utils import get_logger
except ImportError:
    import logging
    def get_logger(name):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)

logger = get_logger(__name__)

class PowerZooEnvWrapper:
    """PowerZooEnv 的简化包装器"""
    
    @staticmethod
    def create_base_env(config, env_name: str = "13Bus", scenario=None, **kwargs):
        """创建基础环境（不包装）
        
        Args:
            config: 配置对象
            env_name: 环境名称
            scenario: 场景参数
            **kwargs: 其他参数
            
        Returns:
            PowerZooEnv实例
        """
        from envs.power_envs.powerzoo_llm.env_register import get_info_and_folder
        
        logger.info(f"创建PowerZoo环境: {env_name}")
        
        # 获取环境信息
        base_info, folder_path = get_info_and_folder(env_name)

        # 创建环境配置，使用正确的DSS文件
        env_info = base_info.copy()
        
        # 更新特定配置
        # 处理配置对象的不同类型 用另外一个config函数覆盖yaml文件
        if hasattr(config, 'source_bus'):
            source_bus = config.source_bus
            max_episode_steps = config.max_episode_steps
            action_space_config = config.action_space_config
        elif isinstance(config, dict):
            source_bus = config.get('source_bus', '650')
            max_episode_steps = config.get('max_episode_steps', 360)
            action_space_config = config.get('action_space_config', {})
        else:
            # 默认值
            source_bus = '650'
            max_episode_steps = 96
            action_space_config = {}
            
        env_info.update({
            'source_bus': source_bus,
            'max_episode_steps': max_episode_steps,
            'reg_act_num': action_space_config.get('reg_act_num', 33) if isinstance(action_space_config, dict) else 33,
            'bat_act_num': action_space_config.get('bat_act_num', 33) if isinstance(action_space_config, dict) else 33
        })
        
        # 如果原配置中没有irrad_dss，确保for_LLM为False
        if 'irrad_dss' not in env_info:
            env_info['for_LLM'] = False

        
        # 创建环境
        # 获取rank参数，默认为0
        rank = kwargs.get('rank', 0)
        env = make_base_env(env_name, env_info, str(folder_path), 
                      kwargs.get('dss_act', False), worker_idx=rank)
        powerzoo_env = PowerZooEnv(env=env, config=config, rank=rank)
        
        logger.info(f"PowerZoo环境创建成功: {env_name}")
        return powerzoo_env

def make_base_env(env_name, base_info, folder_path, dss_act=False, worker_idx=None):
    """创建环境实例"""
    
    if worker_idx is None:
        return Env(folder_path, base_info, dss_act)
    else:
        base_file = os.path.join(folder_path, base_info['system_name'], base_info['dss_file'])
        assert os.path.exists(base_file), base_file + ' does not exist'
        fin = open(base_file, 'r')
        
        # 创建主DSS文件
        with open(base_file[:-4] + '_' + str(worker_idx) + '.dss', 'w') as fout:
            for line in fin:
                if line.strip() == 'redirect loadshape.dss':
                    fout.write('redirect loadshape_' + str(worker_idx) + '.dss\n')
                elif line.strip() == 'redirect pv_data.dss':
                    fout.write('redirect pv_data_' + str(worker_idx) + '.dss\n')
                else:
                    fout.write(line)
        
        # 创建对应的loadshape文件和PV数据文件
        _create_loadshape_file(folder_path, base_info['system_name'], worker_idx)
        _create_pv_data_file(folder_path, base_info['system_name'], worker_idx)
        
        info = base_info.copy()
        info['dss_file'] = info['dss_file'][:-4] + '_' + str(worker_idx) + '.dss'
        info['worker_idx'] = worker_idx
        return Env(folder_path, info, dss_act)

def _create_loadshape_file(folder_path, system_name, worker_idx):
    """创建对应worker_idx的loadshape文件"""
    base_loadshape_file = os.path.join(folder_path, system_name, 'loadshape.dss')
    target_loadshape_file = os.path.join(folder_path, system_name, f'loadshape_{worker_idx}.dss')
    
    if os.path.exists(base_loadshape_file):
        with open(base_loadshape_file, 'r') as fin:
            with open(target_loadshape_file, 'w') as fout:
                for line in fin:
                    # 将路径中的 000 替换为对应的 worker_idx 格式
                    # 修复：数据文件实际在不同的worker目录中（000, 001, 002...）
                    if './loadshape/000/' in line:
                        new_line = line.replace('./loadshape/000/', f'./loadshape/{worker_idx:03d}/')
                        fout.write(new_line)
                    else:
                        fout.write(line)

def _create_pv_data_file(folder_path, system_name, worker_idx):
    """创建对应worker_idx的PV数据文件"""
    base_pv_data_file = os.path.join(folder_path, system_name, 'pv_data.dss')
    target_pv_data_file = os.path.join(folder_path, system_name, f'pv_data_{worker_idx}.dss')
    
    if os.path.exists(base_pv_data_file):
        with open(base_pv_data_file, 'r') as fin:
            with open(target_pv_data_file, 'w') as fout:
                for line in fin:
                    # 将路径中的 000 替换为对应的 worker_idx 格式
                    if './irradiation/000/' in line:
                        new_line = line.replace('./irradiation/000/', f'./irradiation/{worker_idx:03d}/')
                        fout.write(new_line)
                    elif './temperature/000/' in line:
                        new_line = line.replace('./temperature/000/', f'./temperature/{worker_idx:03d}/')
                        fout.write(new_line)
                    else:
                        fout.write(line)


# === 向后兼容性别名和工厂函数 ===

def create_optimized_env(config, env_name: str = "13Bus", **kwargs):
    """创建优化环境的便捷函数（向后兼容）"""
    return PowerZooEnvWrapper.create_base_env(
        config, env_name, **kwargs
    )

def create_standard_env(config, env_name: str = "13Bus", **kwargs):
    """创建标准环境的便捷函数（向后兼容）"""
    return PowerZooEnvWrapper.create_base_env(
        config, env_name, **kwargs
    )

# 别名支持
PowerZooEnvOptimized = PowerZooEnv  # 向后兼容