# -*- coding: utf-8 -*-
"""VVC 环境配置 — 替代 env_register._ENV_INFO 硬编码字典

VVCConfig 是 VVC 环境的唯一配置数据源。
通过 from_env_args() 从 env_args dict 构建，修复了历史键名不匹配问题
(episode_length vs max_episode_steps, pv_control vs pv_control_enabled)，
消除了 _ENV_INFO 中的硬编码默认值。

_ENV_INFO 和 _SYS_INFO 保留在 env_register.py 作为过渡期兜底。
"""
import dataclasses
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class VVCConfig:
    """VVC 环境配置 — 唯一数据源

    字段与 _ENV_INFO 条目 + _SYS_INFO 条目一一对应。
    默认值取自 '13Bus' 配置。
    """

    # 系统配置
    system_name: str = '13Bus'
    dss_file: str = 'IEEE13Nodeckt_daily.dss'
    source_bus: str = 'sourcebus'
    max_episode_steps: int = 24
    seed: int = 123456

    # 设备动作维度
    reg_act_num: int = 33
    bat_act_num: Union[int, float] = 33
    pv_act_num: Union[int, float] = 33
    pv_control_enabled: bool = False
    irrad_dss: Optional[str] = None

    # 奖励权重 (与 _ENV_INFO 键一一对应)
    power_w: float = 10.0
    cap_w: float = 0.0303    # 1.0/33
    reg_w: float = 0.0303    # 1.0/33
    soc_w: float = 0.0
    dis_w: float = 0.1818    # 6.0/33

    # 显示配置 (来自 _SYS_INFO)
    node_size: int = 500
    shift: int = 10
    show_node_labels: bool = True
    load_noise: bool = True

    # 运行时
    scale: float = 1.0
    use_render: bool = False
    useS: bool = False
    record_node: bool = False
    dss_act: bool = False
    for_LLM: bool = False
    env_name: str = '13Bus'  # for backward compat (close() 需要)

    @classmethod
    def from_env_args(cls, env_args: dict) -> 'VVCConfig':
        """从 env_args 构建配置

        支持字段反射 + 键名别名映射。未知键安全忽略。

        Args:
            env_args: 训练脚本传入的环境参数 dict

        Returns:
            填充完毕的 VVCConfig 实例
        """
        valid_fields = {f.name for f in dataclasses.fields(cls)}

        # 提取已知字段 (跳过 None 值，保留默认)
        kwargs = {k: v for k, v in env_args.items() if k in valid_fields and v is not None}

        # 键名别名: 历史配置中的不同命名
        aliases = {
            'episode_length': 'max_episode_steps',
            'num_steps': 'max_episode_steps',
            'pv_control': 'pv_control_enabled',
        }
        for old_key, new_key in aliases.items():
            if old_key in env_args and new_key not in kwargs and env_args[old_key] is not None:
                kwargs[new_key] = env_args[old_key]

        # 显示相关别名
        display_aliases = {
            'show_labels': 'show_node_labels',
        }
        for old_key, new_key in display_aliases.items():
            if old_key in env_args and new_key not in kwargs:
                kwargs[new_key] = env_args[old_key]

        # PV 标志: _ENV_INFO 用 'pv': True 表示 pv_control_enabled
        if 'pv' in env_args and env_args['pv'] and 'pv_control_enabled' not in kwargs:
            kwargs['pv_control_enabled'] = True

        # env_specific_config 中的奖励权重 (system YAML 格式)
        if 'env_specific_config' in env_args:
            esc = env_args['env_specific_config']
            rw = esc.get('reward_weights', {})
            reward_map = {
                'power_loss': 'power_w',
                'capacitor': 'cap_w',
                'regulator': 'reg_w',
                'battery_soc': 'soc_w',
                'battery_discharge': 'dis_w',
            }
            for yaml_key, config_key in reward_map.items():
                if yaml_key in rw and config_key not in kwargs:
                    kwargs[config_key] = rw[yaml_key]

        return cls(**kwargs)

    def to_env_info(self) -> dict:
        """转为 env_register.make_base_env 期望的 dict 格式 (legacy compat)

        合并 _ENV_INFO + _SYS_INFO 的内容，供 Env.__init__() 消费。
        """
        info = {
            'system_name': self.system_name,
            'dss_file': self.dss_file,
            'max_episode_steps': self.max_episode_steps,
            'reg_act_num': self.reg_act_num,
            'bat_act_num': self.bat_act_num,
            'power_w': self.power_w,
            'cap_w': self.cap_w,
            'reg_w': self.reg_w,
            'soc_w': self.soc_w,
            'dis_w': self.dis_w,
            'scale': self.scale,
        }
        if self.pv_control_enabled:
            info['pv'] = True
            info['pv_act_num'] = self.pv_act_num
        if self.irrad_dss:
            info['irrad_dss'] = self.irrad_dss
        if self.for_LLM:
            info['for_LLM'] = True
        return info

    def to_sys_info(self) -> dict:
        """转为 _SYS_INFO 提供的 dict 格式 (legacy compat)"""
        return {
            'source_bus': self.source_bus,
            'node_size': self.node_size,
            'shift': self.shift,
            'show_node_labels': self.show_node_labels,
            'load_noise': self.load_noise,
        }
