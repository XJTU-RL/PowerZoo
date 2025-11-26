"""
SmartGrid 环境配置加载器
简洁统一的配置管理系统
"""
import json
import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """统一的配置加载器"""
    
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        self.configs_dir = self.project_root / 'configs'
        
        # 加载所有配置文件（一次性加载，避免重复IO）
        self.system_info = self._load_json('sys_cfgs/system_info.json', 'system_info')
        self.environments_info = self._load_json('sys_cfgs/environments_info.json', 'environments')
        self.pv_plans = self._load_pv_plans()
    
    def _load_json(self, relative_path: str, key: str) -> Dict[str, Any]:
        """加载JSON配置文件"""
        file_path = self.configs_dir / relative_path
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                result = data.get(key, {})
                # 处理特殊值
                self._process_special_values(result)
                return result
        return {}
    
    def _load_pv_plans(self) -> Dict[str, Any]:
        """加载所有PV方案配置"""
        pv_plans = {}
        pv_plans_dir = self.configs_dir / 'envs_cfgs' / 'smartgrid_pv_plans'

        for plan in ['aggressive', 'conservative', 'optimized']:
            plan_file = pv_plans_dir / f'smartgrid_{plan}.yaml'
            if plan_file.exists():
                with open(plan_file, 'r', encoding='utf-8') as f:
                    pv_plans[plan] = yaml.safe_load(f)
        
        return pv_plans
    
    def _process_special_values(self, config: Any) -> None:
        """处理配置中的特殊值（如 'inf'）"""
        if isinstance(config, dict):
            for key, value in config.items():
                if value == 'inf':
                    config[key] = float('inf')
                elif isinstance(value, dict):
                    self._process_special_values(value)
    
    def get_config(self, env_name: str, scenario: str = None) -> Dict[str, Any]:
        """获取环境配置
        
        Args:
            env_name: 环境名称（如 '34Bus_pv'）
            scenario: PV场景（'aggressive', 'conservative', 'optimized'）
        
        Returns:
            完整的环境配置
        """
        # 1. 基础配置
        if scenario:
            # 如果指定了场景，映射到对应的环境
            scenario_map = {
                'aggressive': '34Bus_PV_Aggressive',
                'conservative': '34Bus_PV_Conservative',
                'optimized': '34Bus_PV_Optimized'
            }
            base_env = scenario_map.get(scenario, env_name)
        else:
            base_env = env_name
        
        # 从environments_info获取基础配置
        config = self.environments_info.get(base_env, {}).copy()
        
        # 2. 添加系统信息
        system_name = config.get('system_name', '34Bus_PV')
        if system_name in self.system_info:
            config.update(self.system_info[system_name])
        
        # 3. 如果有PV场景，合并场景特定配置
        if scenario and scenario in self.pv_plans:
            plan_config = self.pv_plans[scenario]
            # 合并关键配置
            if 'reward_weights' in plan_config:
                config.update(plan_config['reward_weights'])
            if 'episode_length' in plan_config:
                config['max_episode_steps'] = plan_config['episode_length']
            if 'dss_file' in plan_config:
                # 解析DSS文件路径
                self._parse_dss_path(config, plan_config['dss_file'])
        
        return config
    
    def _parse_dss_path(self, config: Dict[str, Any], dss_path: str) -> None:
        """解析DSS文件路径并更新配置"""
        if dss_path.startswith('./node_systems/') and '/' in dss_path:
            parts = dss_path.split('/')
            if len(parts) >= 4:
                config['system_name'] = f"node_systems/{parts[2]}"
                config['dss_file'] = parts[3]


# 全局实例
_loader = ConfigLoader()


def get_env_config(env_name: str, env_args: Dict[str, Any] = None) -> Dict[str, Any]:
    """获取环境配置的简便接口
    
    Args:
        env_name: 环境名称
        env_args: 来自envs_tools.py的参数（可选）
    
    Returns:
        完整的环境配置
    """
    if env_args is None:
        env_args = {}
    
    # 提取场景信息
    scenario = env_args.get('pv_scenario', env_args.get('scenario'))
    
    # 获取基础配置
    config = _loader.get_config(env_name, scenario)
    
    # 覆盖env_args中的参数（优先级最高）
    override_keys = [
        'max_episode_steps', 'episode_length', 'num_steps',
        'reg_act_num', 'bat_act_num', 'pv_act_num', 'pv_control',
        'for_LLM', 'llm_enhanced', 'dss_file', 'system_name'
    ]
    
    for key in override_keys:
        if key in env_args:
            if key in ['num_steps', 'episode_length']:
                config['max_episode_steps'] = env_args[key]
            elif key == 'llm_enhanced':
                config['for_LLM'] = env_args[key]
            else:
                config[key] = env_args[key]
    
    # 如果提供了dss_file，解析路径
    if 'dss_file' in env_args:
        _loader._parse_dss_path(config, env_args['dss_file'])
    
    # 设置默认值
    config.setdefault('max_episode_steps', 360)
    config.setdefault('for_LLM', False)
    config.setdefault('reg_act_num', 33)
    config.setdefault('bat_act_num', 33)
    config.setdefault('pv_control', False)
    config.setdefault('pv_act_num', float('inf'))
    
    # 确保奖励权重存在
    config.setdefault('power_w', 1.0)
    config.setdefault('cap_w', 0.0303)
    config.setdefault('reg_w', 0.0303)
    config.setdefault('soc_w', 0.0)
    config.setdefault('dis_w', 0.303)
    if config.get('pv_control'):
        config.setdefault('pv_w', 0.0606)
    
    return config