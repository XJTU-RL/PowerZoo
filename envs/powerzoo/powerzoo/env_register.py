# Copyright 2021 Siemens Corporation
# SPDX-License-Identifier: MIT

import os
import inspect
import re
from envs.powerzoo.powerzoo.env import Env

# map from system_name to fixed information of the system

# 电力系统的固定信息，以系统名称为键
_SYS_INFO = {
    '13Bus': {
        'source_bus': 'sourcebus',      # 电力系统中的源节点
        'node_size': 500,              # 节点的大小
        'shift': 10,                   # 节点位置的偏移
        'show_node_labels': True,       # 是否显示节点标签
        'load_noise':True               #是否使用噪声
    },

    '34Bus': {
        'source_bus': 'sourcebus',
        'node_size': 500,
        'shift': 80,
        'show_node_labels': True,
        'load_noise':False    
    },

    '123Bus': {
        'source_bus': '150',
        'node_size': 400,
        'shift': 80,
        'show_node_labels': True,
        'load_noise':False 
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
        'dss_file': 'IEEE13Nodeckt_daily.dss',  # 使用的OpenDSS DSS文件
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
        'dss_file': 'IEEE13Nodeckt_daily.dss',
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
        'dss_file': 'IEEE13Nodeckt_daily.dss',
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
        'dss_file': 'ieee34Mod1_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': 33,
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 10.0/33,
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
def get_info_and_folder(env_name):
    # check env scale and env name
    is_scaled = re.match('.*(_s)([0-9]*[.])?[0-9]+?', env_name)
    if is_scaled:
        matched_str = is_scaled.group(0)
        idx = matched_str.rfind('_s')
        env_name = matched_str[:idx]
        scale = float(matched_str[idx+2:])
    assert env_name in _ENV_INFO, env_name + ' not implemented'

    # get base_info
    base_info = _ENV_INFO[env_name].copy()
    if is_scaled:
        base_info['scale'] = scale
        base_info['soc_w'] = base_info['soc_w'] * (scale**2)

    # get folder path
    folder_path = os.path.join(os.path.dirname(os.path.abspath(inspect.getsourcefile(Env))), '..', 'systems')
    folder_path = os.path.abspath(folder_path)
    return base_info, folder_path

def make_env(env_name, dss_act=False, worker_idx=None):
    base_info, folder_path = get_info_and_folder(env_name)

    if worker_idx is None:
        return Env(folder_path, base_info, dss_act)
    else:
        base_file = os.path.join(folder_path, base_info['system_name'], base_info['dss_file'])
        assert os.path.exists(base_file), base_file + ' 不存在'
        fin = open(base_file, 'r')
        
        with open(base_file[:-4] + '_' + str(worker_idx) + '.dss', 'w') as fout:
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
    base_info, folder_path = get_info_and_folder(env_name)
    base_main = os.path.join(folder_path, base_info['system_name'], base_info['dss_file'])
    base_loadshape = os.path.join(folder_path, base_info['system_name'], 'loadshape.dss')

    bases = [base_main, base_loadshape]

    for base in bases:
        fname = base[:-4] + '_' + str(num_workers) + '.dss'
        if os.path.exists(fname):
            os.remove(fname)