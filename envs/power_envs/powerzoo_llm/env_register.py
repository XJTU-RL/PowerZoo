# -*- coding: utf-8 -*-
import os
import inspect
from pathlib import Path
import yaml
import re
try:
    from envs.power_envs.powerzoo_llm.env import Env
except ImportError:
    # 相对导入用于测试
    from .env import Env

# map from system_name to fixed information of the system

# 电力系统的固定信息，以系统名称为键
_SYS_INFO = {
    '13Bus': {
        'source_bus': 'sourcebus',      # 电力系统中的源节点
        'node_size': 500,              # 节点的大小
        'shift': 10,                   # 节点位置的偏移
        'show_node_labels': True       # 是否显示节点标签
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


# map from env_name to the necessary information
_ENV_INFO = {
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


    '8500Node_soc': {
        'system_name': '8500-Node',
        'dss_file': 'Master_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': 33,
        'power_w': 1.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 10000/33,
        'dis_w': 100/33,
    },

    '8500Node_cbat_soc': {
        'system_name': '8500-Node',
        'dss_file': 'Master_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': float('inf'),
        'power_w': 1.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 10000/33,
        'dis_w': 100/33,
    }
}

# add system information to environment
for env in _ENV_INFO.keys():
    sys = _ENV_INFO[env]['system_name']
    _ENV_INFO[env].update(_SYS_INFO[sys])



####################### functions ########################


def get_data_root():
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # 项目根目录位置
    return ROOT_DIR / 'node_systems'

def get_info_and_folder(env_name):
    # check env scale and env name
    is_scaled = re.match('.*(_s)([0-9]*[.])?[0-9]+?', env_name)
    if is_scaled:
        matched_str = is_scaled.group(0)
        idx = matched_str.rfind('_s')
        env_name = matched_str[:idx]
        scale = float(matched_str[idx+2:])
    assert env_name in _ENV_INFO, env_name + ' not implemented' # 检查环境名称是否在已实现环境列表中

    # get base_info
    base_info = _ENV_INFO[env_name].copy()
    if is_scaled:
        base_info['scale'] = scale
        base_info['soc_w'] = base_info['soc_w'] * (scale**2)

    # get folder path
    folder_path = get_data_root()
    folder_path = os.path.abspath(folder_path)
    return base_info, folder_path

def make_base_env(env_name, dss_act=False, worker_idx=None):
    """创建环境实例，同时创建对应的数据
    
    Args:
        env_name: 环境名称
        dss_act: 是否使用DSS控制器
        worker_idx: 工作进程索引
        
    Returns:
        Env实例
    """
    base_info, folder_path = get_info_and_folder(env_name)

    if worker_idx is None:
        return Env(folder_path, base_info, dss_act)
    else:
        base_file = os.path.join(folder_path, base_info['system_name'], base_info['dss_file'])
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
    base_loadshape_file = os.path.join(folder_path, system_name, 'loadshape.dss')
    target_loadshape_file = os.path.join(folder_path, system_name, f'loadshape_{worker_idx}.dss')
    
    if os.path.exists(base_loadshape_file):
        # 确保对应的loadshape数据目录存在
        loadshape_data_dir = os.path.join(folder_path, system_name, 'loadshape', f'{worker_idx:03d}')
        base_data_dir = os.path.join(folder_path, system_name, 'loadshape', '000')
        
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
    base_pv_data_file = os.path.join(folder_path, system_name, 'pv_data.dss')
    target_pv_data_file = os.path.join(folder_path, system_name, f'pv_data_{worker_idx}.dss')
    
    if os.path.exists(base_pv_data_file):
        # 确保对应的irradiation和temperature数据目录存在
        irrad_data_dir = os.path.join(folder_path, system_name, 'irradiation', f'{worker_idx:03d}')
        temp_data_dir = os.path.join(folder_path, system_name, 'temperature', f'{worker_idx:03d}')
        base_irrad_dir = os.path.join(folder_path, system_name, 'irradiation', '000')
        base_temp_dir = os.path.join(folder_path, system_name, 'temperature', '000')
        
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