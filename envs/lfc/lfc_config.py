# -*- coding: utf-8 -*-
"""LFC（负荷频率控制）任务配置。

f1~f7 为论文奖励权重（显式字段替代论文的嵌套 dict，获得白名单校验与
非负有限校验）；``reward_parameters`` 以属性形式提供论文内核期望的
dict 视图，内核代码零改动。
``seed``/``worker_idx`` 由 PowerZoo 的 make_power_env 注入；
``evaluation`` 由评估分支注入（对应论文 make_eval_env 的 ranks=4 语义）。
"""

from dataclasses import dataclass


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
    # ---- 论文奖励权重（含义见 LFC_gym 的 reward 拼装） ----
    f1: float = 7.5   # 电网频率偏差（比例项）
    f2: float = 0.0   # 电网频率越限
    f3: float = 0.1   # 均衡光伏出力
    f4: float = 0.2   # 储能均衡出力
    f5: float = 0.15  # 储能充放电诱导
    f6: float = 0.1   # 储能放电惩罚
    f7: float = 0.1   # 储能与光伏出力平滑
    # ---- 900s 长时程负荷扰动实验 ----
    long_term_load_disturbance: bool = False
    # ---- 辐照度相关（exp2 口径 use_irrad=True） ----
    use_irrad: bool = False
    use_render: bool = False
    irrad_style: str = "Irrad_constant"
    render_fault: int = 1
    # ---- PowerZoo 注入字段 ----
    worker_idx: int = 0
    evaluation: bool = False

    def __post_init__(self):
        import math

        if self.system not in ("13Bus", "34Bus"):
            raise ValueError(f"未知 LFC 算例 {self.system!r}；当前随包提供 13Bus / 34Bus")
        for key in ("f1", "f2", "f3", "f4", "f5", "f6", "f7"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} 必须为非负有限值")

    @property
    def reward_parameters(self):
        """论文内核期望的奖励权重 dict 视图。"""
        return {f"f{i}": getattr(self, f"f{i}") for i in range(1, 8)}
