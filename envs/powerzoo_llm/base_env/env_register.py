# -*- coding: utf-8 -*-
import os
import json
from pathlib import Path
import re
try:
    from envs.powerzoo_llm.base_env.env import Env
except ImportError:
    # 相对导入用于测试
    from .env import Env

# 从JSON文件加载系统信息
def load_system_info():
    """从JSON配置文件加载系统信息
    
    Returns:
        dict: 系统信息字典
    """
    config_path = Path(__file__).resolve().parent.parent.parent.parent / 'configs' / 'sys_cfgs' / 'system_info.json'
    
    # 如果JSON文件存在，从文件加载
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('system_info', {})
        except Exception as e:
            print(f"Warning: Failed to load system_info.json: {e}")
            print("Falling back to default system info...")
    
    # 如果文件不存在或加载失败，使用默认值
    return {
        '13Bus': {
            'source_bus': 'sourcebus',
            'node_size': 500,
            'shift': 10,
            'show_node_labels': True
        },
        '34Bus': {
            'source_bus': 'sourcebus',
            'node_size': 500,
            'shift': 80,
            'show_node_labels': True
        },
        '34Bus_PV': {
            'source_bus': 'sourcebus',
            'node_size': 500,
            'shift': 80,
            'show_node_labels': True
        },
        '123Bus': {
            'source_bus': '150',
            'node_size': 400,
            'shift': 80,
            'show_node_labels': True
        },
        '8500-Node': {
            'source_bus': 'e192860',
            'node_size': 10,
            'shift': 0,
            'show_node_labels': False
        }
    }

# 加载系统信息
_SYS_INFO = load_system_info()

# 从JSON文件加载环境信息
def load_environments_info():
    """从JSON配置文件加载环境信息
    
    Returns:
        dict: 环境信息字典
    """
    config_path = Path(__file__).resolve().parent.parent.parent.parent / 'configs' / 'sys_cfgs'/'environments_info.json'
    
    # 如果JSON文件存在，从文件加载
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                env_info = data.get('environments', {})
                
                # 处理特殊值（如 "inf" 转换为 float('inf')）
                for _, env_config in env_info.items():
                    if 'pv_act_num' in env_config and env_config['pv_act_num'] == 'inf':
                        env_config['pv_act_num'] = float('inf')
                    if 'bat_act_num' in env_config and env_config['bat_act_num'] == 'inf':
                        env_config['bat_act_num'] = float('inf')
                
                return env_info
        except Exception as e:
            print(f"Warning: Failed to load environments_info.json: {e}")
            print("Falling back to default environment info...")
    
    # 如果文件不存在或加载失败，返回空字典（将使用下面的默认值）
    return {}

# 加载环境信息
_ENV_INFO_FROM_JSON = load_environments_info()

# 默认环境信息（作为后备）
_ENV_INFO_DEFAULT = {
    '13Bus': {
        'system_name': '13Bus',             # 电力系统的名称
        'dss_file': 'IEEE13Nodeckt_daily.dss',  # 使用duty版本的DSS文件
        'for_LLM': False,
        'max_episode_steps': 24,            # 每个仿真episode的最大步数
        'reg_act_num': 33,                  # 可用的调节动作数量
        'bat_act_num': 33,                  # 可用的电池动作数量
        'pv_control': False,                # 默认禁用PV控制
        'pv_act_num': float('inf'),         # 连续PV控制
        'power_w': 10.0,                   # 与功率相关的奖励权重
        'cap_w': 1.0/33,                   # 与电容相关的奖励权重
        'reg_w': 1.0/33,                   # 与调节动作相关的奖励权重
        'soc_w': 0.0/33,                   # 与电池状态相关的奖励权重
        'dis_w': 6.0/33                    # 与电池放电动作相关的奖励权重
    },
        '13Bus_with_irrads': {
        'system_name': '13Bus',             # 电力系统的名称
        'dss_file': 'IEEE13Nodeckt_duty.dss',  # 使用duty版本的DSS文件
        'for_LLM': False,
        'max_episode_steps': 24,            # 每个仿真episode的最大步数
        'reg_act_num': 33,                  # 可用的调节动作数量
        'bat_act_num': 33,                  # 可用的电池动作数量
        'power_w': 10.0,                   # 与功率相关的奖励权重
        'cap_w': 1.0/33,                   # 与电容相关的奖励权重
        'reg_w': 1.0/33,                   # 与调节动作相关的奖励权重
        'soc_w': 0.0/33,                   # 与电池状态相关的奖励权重
        'dis_w': 6.0/33                    # 与电池放电动作相关的奖励权重
    },
   
    '13Bus_cbat': {
        'system_name': '13Bus',
        'dss_file': 'IEEE13Nodeckt_duty.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': float('inf'),
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 6.0/33,
    },
   
    '13Bus_soc': {
        'system_name': '13Bus',
        'dss_file': 'IEEE13Nodeckt_duty.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': 33,
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 20.0/33,
        'dis_w': 1.0/33
    },

    '13Bus_cbat_soc': {
        'system_name': '13Bus',
        'dss_file': 'IEEE13Nodeckt_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': float('inf'),
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 20.0/33,
        'dis_w': 1.0/33,
    },

    '34Bus': {
        'system_name': '34Bus',
        'dss_file': 'ieee34Mod1_duty.dss',
        'for_LLM': False,
        'max_episode_steps': 360,           # 匹配loadshape数据点数
        'reg_act_num': 33,
        'bat_act_num': 33,
        'pv_control': False,                # 默认禁用PV控制
        'pv_act_num': float('inf'),         # 连续PV控制
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 10.0/33,
    },
    '34Bus_pv': {
        'system_name': '34Bus_PV',
        'dss_file': 'ieee34Mod1_duty.dss',
        'for_LLM': False,
        'max_episode_steps': 360,           # 匹配loadshape数据点数和配置文件episode_length
        'reg_act_num': 33,
        'bat_act_num': 33,
        'pv_control': True,                 # 启用PV控制
        'pv_act_num': float('inf'),         # 启用连续PV控制，如果是离散则填入整数值
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 10.0/33,
        'pv_w': 2.0/33,                    # PV控制奖励权重
    },
    
    '34Bus_pv_discrete': {
        'system_name': '34Bus_PV',
        'dss_file': 'ieee34Mod1_duty.dss',
        'for_LLM': False,
        'max_episode_steps': 360,
        'reg_act_num': 33,
        'bat_act_num': 33,
        'pv_control': True,                 # 启用PV控制
        'pv_act_num': 21,                   # 离散PV控制 (21个等级)
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 10.0/33,
        'pv_w': 2.0/33,                    # PV控制奖励权重
    },

    '34Bus_cbat': {
        'system_name': '34Bus',
        'dss_file': 'ieee34Mod1_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': float('inf'),
        'power_w': 1.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 10.0/33,
    },

    '34Bus_soc': {
        'system_name': '34Bus',
        'dss_file': 'ieee34Mod1_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': 33,
        'power_w': 1.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 500.0/33,
        'dis_w': 4.0/33,
    },

    '34Bus_cbat_soc': {
        'system_name': '34Bus',
        'dss_file': 'ieee34Mod1_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': float('inf'),
        'power_w': 1.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 500.0/33,
        'dis_w': 4.0/33,
    },

    '123Bus': {
        'system_name': '123Bus',
        'dss_file': 'IEEE123Master_daily.dss',
        'for_LLM': False,
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': 33,
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 7.0/33,
    },

    '123Bus_cbat': {
        'system_name': '123Bus',
        'dss_file': 'IEEE123Master_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': float('inf'),
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 7.0/33,
    },

    '123Bus_soc': {
        'system_name': '123Bus',
        'dss_file': 'IEEE123Master_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': 33,
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 500.0/33,
        'dis_w': 5.0/33,
    },

    '123Bus_cbat_soc': {
        'system_name': '123Bus',
        'dss_file': 'IEEE123Master_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': float('inf'),
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 500.0/33,
        'dis_w': 5.0/33,
    },

    '8500Node': {
        'system_name': '8500-Node',
        'dss_file': 'Master_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': 33,
        'power_w': 1.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 200.0/33,
    },
    
    '8500Node_cbat': {
        'system_name': '8500-Node',
        'dss_file': 'Master_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': float('inf'),
        'power_w': 1.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 200.0/33,
    },
}

# 使用JSON文件中的信息，如果不存在则使用默认值
_ENV_INFO = _ENV_INFO_FROM_JSON if _ENV_INFO_FROM_JSON else _ENV_INFO_DEFAULT

# add system information to environment
for env in _ENV_INFO.keys():
    sys = _ENV_INFO[env]['system_name']
    if sys in _SYS_INFO:
        _ENV_INFO[env].update(_SYS_INFO[sys])



####################### functions ########################


def get_data_root():
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent  # 项目根目录位置（需要再往上一级）
    return ROOT_DIR  # 返回项目根目录，system_name中已包含node_systems路径

def get_info_from_config(env_name, config_dict=None):
    """从配置文件获取环境信息（使用新的配置加载器）
    
    Args:
        env_name: 环境名称
        config_dict: 完整的配置字典（可选）
        
    Returns:
        base_info: 环境信息字典
    """
    from envs.powerzoo_llm.base_env.config_loader import get_env_config
    
    # 提取 env_args
    env_args = {}
    
    if config_dict:
        # 从 config_dict 提取参数
        env_args = config_dict.get('env_args', {}).copy()
        
        # 提取关键参数
        if 'dss_file' in config_dict:
            env_args['dss_file'] = config_dict['dss_file']
        
        if 'train' in config_dict:
            env_args['episode_length'] = config_dict['train'].get('episode_length', 360)
        
        # 从 environment_specific 提取参数（如果存在）
        if 'environment_specific' in config_dict:
            env_config = config_dict['environment_specific']
            
            # 设备配置
            devices = env_config.get('devices', {})
            if 'regulators' in devices:
                env_args['reg_act_num'] = devices['regulators'].get('action_num', 33)
            if 'batteries' in devices:
                env_args['bat_act_num'] = devices['batteries'].get('action_num', 33)
            if 'pv_systems' in devices:
                pv_config = devices['pv_systems']
                env_args['pv_control'] = pv_config.get('control_enabled', False)
                if pv_config.get('action_space') == 'continuous':
                    env_args['pv_act_num'] = float('inf')
                else:
                    env_args['pv_act_num'] = pv_config.get('action_num', 21)
    
    # 使用配置加载器获取配置
    return get_env_config(env_name, env_args)

def get_info_and_folder(env_name, config_dict=None):
    # check env scale and env name
    is_scaled = re.match('.*(_s)([0-9]*[.])?[0-9]+?', env_name)
    if is_scaled:
        matched_str = is_scaled.group(0)
        idx = matched_str.rfind('_s')
        env_name = matched_str[:idx]
        scale = float(matched_str[idx+2:])
    
    # 优先从配置文件获取信息
    if config_dict:
        base_info = get_info_from_config(env_name, config_dict)
    else:
        # 处理路径形式的环境名（如 /home/xxx/node_systems/34Bus_PV_Aggressive）
        if env_name.startswith('/') and 'node_systems' in env_name:
            # 从路径中提取系统名，映射到已知的环境配置
            if '34Bus_PV_Aggressive' in env_name:
                mapped_env_name = '34Bus_pv'
            elif '34Bus_PV_Conservative' in env_name:
                mapped_env_name = '34Bus_pv'
            elif '34Bus_PV_Optimized' in env_name:
                mapped_env_name = '34Bus_pv'
            elif '34Bus_PV' in env_name:
                mapped_env_name = '34Bus_pv'
            elif '34Bus' in env_name:
                mapped_env_name = '34Bus'
            elif '13Bus' in env_name:
                mapped_env_name = '13Bus'
            else:
                # 默认使用34Bus_pv配置
                mapped_env_name = '34Bus_pv'
            
            assert mapped_env_name in _ENV_INFO, f"Mapped environment {mapped_env_name} not implemented"
            base_info = _ENV_INFO[mapped_env_name].copy()
            # 更新system_name为实际路径
            base_info['system_name'] = env_name
        else:
            # 回退到_ENV_INFO（保持向后兼容）
            assert env_name in _ENV_INFO, env_name + ' not implemented'
            base_info = _ENV_INFO[env_name].copy()
    
    if is_scaled:
        base_info['scale'] = scale
        base_info['soc_w'] = base_info['soc_w'] * (scale**2)

    # get folder path
    folder_path = get_data_root()
    folder_path = os.path.abspath(folder_path)
    return base_info, folder_path

def make_base_env(env_name, dss_act=False, worker_idx=None, config_dict=None):
    """创建环境实例，同时创建对应的数据
    
    Args:
        env_name: 环境名称
        dss_act: 是否使用DSS控制器
        worker_idx: 工作进程索引
        config_dict: 完整的配置字典（可选）
        
    Returns:
        Env实例
    """
    base_info, folder_path = get_info_and_folder(env_name, config_dict)

    if worker_idx is None:
        return Env(folder_path, base_info, dss_act)
    else:
        # Construct the base file path
        # If system_name contains node_systems, it's a full path from project root
        if 'node_systems/' in base_info['system_name']:
            # Treat system_name as full path from project root
            base_file = os.path.join(folder_path, base_info['system_name'], base_info['dss_file'])
        else:
            # Legacy path - system_name is just the system folder name
            base_file = os.path.join(folder_path, 'node_systems', base_info['system_name'], base_info['dss_file'])
        assert os.path.exists(base_file), base_file + ' does not exist'
        fin = open(base_file, 'r')
        
        with open(base_file[:-4] + '_' + str(worker_idx) + '.dss', 'w') as fout:
            for line in fin:
                if line.strip() == 'redirect loadshape.dss':
                    fout.write('redirect loadshape_' + str(worker_idx) + '.dss\n')
                elif line.strip() == 'redirect pv_data.dss':
                    fout.write('redirect pv_data_' + str(worker_idx) + '.dss\n')
                else:
                    fout.write(line)
        
        # 创建worker特定的loadshape和pv_data文件
        _create_loadshape_file(folder_path, base_info['system_name'], worker_idx)
        _create_pv_data_file(folder_path, base_info['system_name'], worker_idx)
        info = base_info.copy()
        info['dss_file'] = info['dss_file'][:-4] + '_' + str(worker_idx) + '.dss'
        info['worker_idx'] = worker_idx
        return Env(folder_path, info, dss_act)

def _create_loadshape_file(folder_path, system_name, worker_idx):
    """创建对应worker_idx的loadshape文件"""
    # Handle system_name that includes node_systems path
    if 'node_systems/' in system_name:
        base_loadshape_file = os.path.join(folder_path, system_name, 'loadshape.dss')
        target_loadshape_file = os.path.join(folder_path, system_name, f'loadshape_{worker_idx}.dss')
    else:
        base_loadshape_file = os.path.join(folder_path, 'node_systems', system_name, 'loadshape.dss')
        target_loadshape_file = os.path.join(folder_path, 'node_systems', system_name, f'loadshape_{worker_idx}.dss')
    
    if os.path.exists(base_loadshape_file):
        # 确保对应的loadshape数据目录存在
        if 'node_systems/' in system_name:
            loadshape_data_dir = os.path.join(folder_path, system_name, 'loadshape', f'{worker_idx:03d}')
            base_data_dir = os.path.join(folder_path, system_name, 'loadshape', '000')
        else:
            loadshape_data_dir = os.path.join(folder_path, 'node_systems', system_name, 'loadshape', f'{worker_idx:03d}')
            base_data_dir = os.path.join(folder_path, 'node_systems', system_name, 'loadshape', '000')
        
        # 如果目标数据目录不存在，则从000目录复制
        if not os.path.exists(loadshape_data_dir) and os.path.exists(base_data_dir):
            import shutil
            os.makedirs(os.path.dirname(loadshape_data_dir), exist_ok=True)
            shutil.copytree(base_data_dir, loadshape_data_dir)
            print(f"创建loadshape数据目录: {loadshape_data_dir}")
        
        with open(base_loadshape_file, 'r') as fin:
            with open(target_loadshape_file, 'w') as fout:
                for line in fin:
                    # 将路径中的 000 替换为对应的 worker_idx 格式
                    if './loadshape/000/' in line:
                        new_line = line.replace('./loadshape/000/', f'./loadshape/{worker_idx:03d}/')
                        fout.write(new_line)
                    else:
                        fout.write(line)

def _create_pv_data_file(folder_path, system_name, worker_idx):
    """创建对应worker_idx的PV数据文件"""
    # Handle system_name that includes node_systems path
    if 'node_systems/' in system_name:
        base_pv_data_file = os.path.join(folder_path, system_name, 'pv_data.dss')
        target_pv_data_file = os.path.join(folder_path, system_name, f'pv_data_{worker_idx}.dss')
    else:
        base_pv_data_file = os.path.join(folder_path, 'node_systems', system_name, 'pv_data.dss')
        target_pv_data_file = os.path.join(folder_path, 'node_systems', system_name, f'pv_data_{worker_idx}.dss')
    
    # 检查目标文件是否已经由ConfigGenerator生成（优先级更高）
    if os.path.exists(target_pv_data_file):
        # 检查文件是否是自动生成的（包含特定标识）
        try:
            with open(target_pv_data_file, 'r') as f:
                content = f.read(200)  # 只读前200字符
                if "自动生成时间" in content:
                    # 这是ConfigGenerator生成的正确文件，不需要覆盖
                    return
        except:
            pass
    
    if os.path.exists(base_pv_data_file):
        # 确保对应的irradiation和temperature数据目录存在
        if 'node_systems/' in system_name:
            irrad_data_dir = os.path.join(folder_path, system_name, 'irradiation', f'{worker_idx:03d}')
            temp_data_dir = os.path.join(folder_path, system_name, 'temperature', f'{worker_idx:03d}')
            base_irrad_dir = os.path.join(folder_path, system_name, 'irradiation', '000')
            base_temp_dir = os.path.join(folder_path, system_name, 'temperature', '000')
        else:
            irrad_data_dir = os.path.join(folder_path, 'node_systems', system_name, 'irradiation', f'{worker_idx:03d}')
            temp_data_dir = os.path.join(folder_path, 'node_systems', system_name, 'temperature', f'{worker_idx:03d}')
            base_irrad_dir = os.path.join(folder_path, 'node_systems', system_name, 'irradiation', '000')
            base_temp_dir = os.path.join(folder_path, 'node_systems', system_name, 'temperature', '000')
        
        # 如果目标数据目录不存在，则从000目录复制
        if not os.path.exists(irrad_data_dir) and os.path.exists(base_irrad_dir):
            import shutil
            os.makedirs(os.path.dirname(irrad_data_dir), exist_ok=True)
            shutil.copytree(base_irrad_dir, irrad_data_dir)
            print(f"创建irradiation数据目录: {irrad_data_dir}")
            
        if not os.path.exists(temp_data_dir) and os.path.exists(base_temp_dir):
            import shutil
            os.makedirs(os.path.dirname(temp_data_dir), exist_ok=True)
            shutil.copytree(base_temp_dir, temp_data_dir)
            print(f"创建temperature数据目录: {temp_data_dir}")
        
        with open(base_pv_data_file, 'r') as fin:
            with open(target_pv_data_file, 'w') as fout:
                for line in fin:
                    # 将路径中的 000 替换为对应的 worker_idx 格式
                    new_line = line.replace('./irradiation/000/', f'./irradiation/{worker_idx:03d}/')
                    new_line = new_line.replace('./temperature/000/', f'./temperature/{worker_idx:03d}/')
                    fout.write(new_line)
        
def remove_parallel_dss(env_name, num_workers):
    """删除特定worker_idx的临时DSS文件"""
    base_info, folder_path = get_info_and_folder(env_name)
    base_main = os.path.join(folder_path, base_info['system_name'], base_info['dss_file'])
    base_loadshape = os.path.join(folder_path, base_info['system_name'], 'loadshape.dss')

    bases = [base_main, base_loadshape]

    for base in bases:
        fname = base[:-4] + '_' + str(num_workers) + '.dss'
        if os.path.exists(fname):
            os.remove(fname)


def cleanup_all_parallel_dss(node_systems_path=None):
    """
    批量清理所有节点系统目录下的临时DSS文件
    保留原始基础文件，删除所有带数字后缀的临时文件
    
    Args:
        node_systems_path: node_systems目录路径，默认使用get_data_root()
        
    Returns:
        tuple: (成功删除的文件数量, 失败的文件数量)
    """
    import glob
    
    if node_systems_path is None:
        node_systems_path = get_data_root()
    
    if not os.path.exists(node_systems_path):
        print(f"警告: 目录不存在 {node_systems_path}")
        return 0, 0
    
    # 系统目录列表
    system_dirs = ['13Bus', '34Bus', '123Bus', '8500-Node', '9500-Node']
    
    total_cleaned = 0
    total_failed = 0
    
    # 临时文件匹配模式：*_数字.dss
    temp_pattern = re.compile(r'^.+_\d+\.dss$')
    
    for system_name in system_dirs:
        system_dir = os.path.join(node_systems_path, system_name)
        if not os.path.exists(system_dir):
            continue
            
        # 查找所有DSS文件
        dss_files = glob.glob(os.path.join(system_dir, '*.dss'))
        temp_files = [f for f in dss_files if temp_pattern.match(os.path.basename(f))]
        
        if not temp_files:
            continue
            
        print(f"清理 {system_name}: 发现 {len(temp_files)} 个临时文件")
        
        # 删除临时文件
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                total_cleaned += 1
                print(f"  ✓ 已删除: {os.path.basename(temp_file)}")
            except Exception as e:
                total_failed += 1
                print(f"  ✗ 删除失败 {os.path.basename(temp_file)}: {e}")
    
    print(f"\n清理完成: 成功删除 {total_cleaned} 个文件, 失败 {total_failed} 个文件")
    return total_cleaned, total_failed