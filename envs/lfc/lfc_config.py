# -*- coding: utf-8 -*-
"""LFC（负荷频率控制）任务配置。

字段与论文实现（papers/ohasac/code）的 ``lfc.yaml`` env 节一一对应；
仅 ``dss_folder_path`` 语义增强：为 None 时由包装器从包内
``IEEE9BusSystem/systems/<system>`` 推导（论文为作者机器绝对路径，P0-4 修复）。
``seed``/``worker_idx`` 由 PowerZoo 的 make_power_env 注入；
``evaluation`` 由评估分支注入（对应论文 make_eval_env 的 ranks=4 语义）。
"""

from dataclasses import dataclass, field


@dataclass
class LFCConfig:
    # Simulink 基础模型名（worker 副本自动加 _1/_2；评估用 *_eval）
    env_name: str = "IEEE9BusSystem"
    # systems/ 下的算例目录（13Bus / 34Bus）
    system: str = "34Bus"
    # 显式 DSS 目录（None=由包内 systems/<system> 推导）
    dss_folder_path: str | None = None
    # 训练用主 DSS 文件（评估文件自动取 <stem>_eval.dss——论文 P0-6 修复语义）
    dss_file: str = "ieee34Mod1_duty.dss"
    # 单回合最大环境步数（1 步 = 1 秒联合仿真）
    max_episode_steps: int = 30
    seed: int = 123456
    # 奖励权重（论文 f1~f7，含义见 LFC_gym 的 reward 拼装）
    reward_parameters: dict = field(
        default_factory=lambda: {
            "f1": 7.5,
            "f2": 0.0,
            "f3": 0.1,
            "f4": 0.2,
            "f5": 0.15,
            "f6": 0.1,
            "f7": 0.1,
        }
    )
    # 900s 长时程负荷扰动实验
    long_term_load_disturbance: bool = False
    # 辐照度相关（exp2 口径 use_irrad=True）
    use_irrad: bool = False
    use_render: bool = False
    irrad_style: str = "Irrad_constant"
    render_fault: int = 1
    # PowerZoo 注入字段
    worker_idx: int = 0
    evaluation: bool = False

    def __post_init__(self):
        if self.system not in ("13Bus", "34Bus"):
            raise ValueError(
                f"未知 LFC 算例 {self.system!r}；当前随包提供 13Bus / 34Bus"
            )
        for key in ("f1", "f2", "f3", "f4", "f5", "f6", "f7"):
            if key not in self.reward_parameters:
                raise ValueError(f"reward_parameters 缺少 {key}")
