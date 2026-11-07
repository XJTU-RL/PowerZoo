# -*- coding: utf-8 -*-
"""
VVC CMDP (Constrained Markov Decision Process) 奖励函数模块

该模块实现了基于约束马尔可夫决策过程范式的奖励函数，
将电压安全转换为约束成本，主奖励专注于经济性优化。

核心设计原则：
- CMDP范式：奖励-约束分离
- 电压安全 → 约束成本（平方铰链损失）
- 主奖励：网损最小化 + 控制平滑 + PV合理利用
- 所有组件规模归一化，移除不可导项

@File      : vvc_reward.py
@Time      : 2025-08-08
@Author    : Xiaodong Zheng (with Claude Code)
@Email     : zxd_xjtu@stu.xjtu.edu.cn
"""

import logging
import math
import numpy as np
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple
import time

# 导入常量
try:
    from envs.smartgrid.constants import VOLTAGE, REWARD, POWER_FACTOR
except ImportError:
    # 相对导入用于测试
    from ..constants import VOLTAGE, REWARD, POWER_FACTOR

# 获取logger
logger = logging.getLogger(__name__)


def performance_monitor(func):
    """性能监控装饰器，记录耗时超过100ms的操作"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            elapsed = time.time() - start_time
            if elapsed > 0.1:
                logger.debug(f"{func.__name__} 耗时: {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} 执行失败 (耗时{elapsed:.3f}s): {e}")
            raise

    return wrapper


class VVCReward:
    """
    VVC环境的CMDP奖励函数类

    基于约束马尔可夫决策过程范式设计，将电压安全转换为约束成本，
    主奖励聚焦于经济性目标（网损、控制平滑、PV利用）。

    核心组件：
    - voltage_cost(): 电压约束成本，平方铰链损失
    - powerloss_reward(): 网损奖励，光滑单调惩罚
    - control_reward(): 控制奖励，设备数归一+动作平滑
    - pv_reward(): PV奖励，基于本地电压方向性支撑
    - reward_main(): 主奖励组合（不含电压成本）

    Attributes:
            env: 环境实例引用
            voltage_threshold: 电压约束死区 (±ε)
            power_base: 网损基准值归一化
            weights: 各组件权重配置
            use_action_smoothing: 是否启用动作平滑惩罚
            voltage_range: 电压安全范围
            previous_actions: 上一步动作缓存（用于平滑惩罚）
    """

    def __init__(self, env, info: Dict[str, Any]):
        """
        初始化CMDP奖励函数

        Args:
                env: 环境实例，需提供obs, cap_num, reg_num, pv_num等属性
                info: 配置信息字典
                        - voltage_threshold: 电压死区，默认0.02 (±2%)
                        - power_base: 网损基准值，默认0.02 (系统额定功率2%)
                        - powerloss_weight: 网损奖励权重，默认1.0
                        - control_weight: 控制奖励权重，默认0.5
                        - pv_weight: PV奖励权重，默认0.8
                        - action_smoothing_weight: 动作平滑权重，默认0.3
                        - voltage_range: 电压安全范围，默认(0.95, 1.05)
                        - use_action_smoothing: 是否启用动作平滑，默认True
        """
        self.env = env

        from envs.smartgrid.base_env.env_config import SmartGridConfig

        if isinstance(info, SmartGridConfig):
            # 正式运行直接消费统一配置，不经过旧 to_info_dict/别名映射。
            self.voltage_threshold = info.voltage_threshold
            self.voltage_range = (info.voltage_min, info.voltage_max)
            self.power_base = info.power_base
            w = info.reward_weights
            self.weights = dict(
                powerloss=w.power_loss,
                control=w.control,
                pv=w.pv,
                action_smoothing=w.action_smoothing,
            )
            self.use_action_smoothing = info.use_action_smoothing
            self.control_weights = {
                name: getattr(w, name)
                for name in (
                    "capacitor",
                    "regulator",
                    "battery_soc",
                    "battery_discharge",
                    "pv_control",
                )
            }
        else:
            # 约束参数 - 使用常量作为默认值
            self.voltage_threshold = info.get("voltage_threshold", VOLTAGE.DEADBAND)
            self.voltage_range = info.get(
                "voltage_range", (VOLTAGE.MIN_PU, VOLTAGE.MAX_PU)
            )

            # 归一化参数
            self.power_base = info.get("power_base", REWARD.POWER_LOSS_BASE)

            configured_weights = info.get("reward_weights", {})
            # 权重配置 - 使用常量作为默认值
            self.weights = {
                "powerloss": configured_weights.get(
                    "power_loss", info.get("powerloss_weight", REWARD.POWER_LOSS_WEIGHT)
                ),
                "control": configured_weights.get(
                    "control", info.get("control_weight", REWARD.CONTROL_WEIGHT)
                ),
                "pv": configured_weights.get(
                    "pv", info.get("pv_weight", REWARD.PV_WEIGHT)
                ),
                "action_smoothing": configured_weights.get(
                    "action_smoothing",
                    info.get("action_smoothing_weight", REWARD.ACTION_SMOOTHING_WEIGHT),
                ),
            }

            # 动作平滑配置
            self.use_action_smoothing = info.get("use_action_smoothing", True)
            self.control_weights = {
                name: configured_weights.get(name, 1.0)
                for name in (
                    "capacitor",
                    "regulator",
                    "battery_soc",
                    "battery_discharge",
                    "pv_control",
                )
            }
        self.previous_actions = None  # 缓存上一步动作

        # 电压目标点（范围中心）
        v_min, v_max = self.voltage_range
        self.voltage_target = (v_min + v_max) / 2.0  # 1.0pu

        logger.info(
            f"CMDP奖励函数初始化完成 - 电压死区: ±{self.voltage_threshold:.3f}, "
            f"网损基准: {self.power_base:.3f}, 动作平滑: {self.use_action_smoothing}"
        )

    @performance_monitor
    def voltage_cost(self) -> float:
        """
        电压约束成本函数（CMDP约束项）

        使用平方铰链损失计算电压违规成本：
        cost = mean_over_buses(max(0, |V - V_ref| - ε)²)

        其中：
        - V: 实际电压
        - V_ref: 目标电压（1.0 p.u.）
        - ε: 电压死区阈值
        - 平方项确保函数光滑可导
        - 按测点平均避免单点异常影响

        Returns:
                float: 电压约束成本 (≥0)，0表示无违规，>0表示有违规
        """
        bus_voltages = self.env.obs.get("bus_voltages", {})
        if not bus_voltages:
            return 0.0

        total_cost = 0.0
        total_measurements = 0

        for _, voltages in bus_voltages.items():
            if not voltages:
                continue

            for v in voltages:
                # 计算电压偏差
                voltage_deviation = abs(v - self.voltage_target)

                # 平方铰链损失：max(0, |deviation| - threshold)²
                if voltage_deviation > self.voltage_threshold:
                    excess_deviation = voltage_deviation - self.voltage_threshold
                    cost = excess_deviation**2
                    total_cost += cost

                total_measurements += 1

        # 按测点平均
        if total_measurements > 0:
            average_cost = total_cost / total_measurements
        else:
            average_cost = 0.0

        return average_cost

    @performance_monitor
    def powerloss_reward(self) -> float:
        """
        网损奖励函数（主奖励组件）

        使用光滑单调惩罚替代原有阶跃函数：
        reward = -log(1 + power_loss / power_base)

        特点：
        - 光滑可导，适合梯度优化
        - 单调递减，网损越高奖励越低
        - 归一化处理，适用不同规模系统
        - 移除功率平衡叠加，避免重复计算

        Returns:
                float: 网损奖励（负值），网损越低奖励越高
        """
        current_loss = self.env.obs.get("power_loss_ratio", 0)

        # 处理标量和数组类型
        if hasattr(current_loss, "__len__"):
            current_loss = float(current_loss[0]) if len(current_loss) > 0 else 0.0
        else:
            current_loss = float(current_loss)

        # 确保损失率为正值
        loss_ratio = max(0.0, current_loss)

        # 光滑单调惩罚
        reward = -math.log(1.0 + loss_ratio / self.power_base)

        return reward

    @performance_monitor
    def control_reward(
        self,
        capdiff: List[float],
        regdiff: List[float],
        soc_err: List[float],
        discharge_err: List[float],
        pv_diff: Optional[List[float]] = None,
    ) -> float:
        """
        控制动作奖励函数（主奖励组件）

        包含两部分：
        1. 设备数归一化的控制成本
        2. 动作平滑惩罚 |a_t - a_{t-1}|

        Args:
                capdiff: 电容器状态变化
                regdiff: 调压器抽头变化
                soc_err: 电池SOC误差
                discharge_err: 电池放电误差
                pv_diff: PV控制变化

        Returns:
                float: 控制奖励（负值），动作越剧烈奖励越低
        """
        # 处理pv_diff的None值（使用is None而不是or）
        if pv_diff is None:
            pv_diff = []

        # 1. 基础控制成本（按设备数归一化）
        # 使用len()检查是否为空，避免numpy数组的ambiguous truth value
        # 转换为列表以确保兼容性
        capdiff_list = list(capdiff) if hasattr(capdiff, "__iter__") else []
        regdiff_list = list(regdiff) if hasattr(regdiff, "__iter__") else []

        cap_cost = (
            sum(capdiff_list) / max(self.env.cap_num, 1)
            if len(capdiff_list) > 0
            else 0.0
        )
        reg_cost = (
            sum(regdiff_list) / max(self.env.reg_num, 1)
            if len(regdiff_list) > 0
            else 0.0
        )

        # 电池SOC误差仅在episode结束时计算
        soc_err_list = list(soc_err) if hasattr(soc_err, "__iter__") else []
        soc_cost = 0.0
        if hasattr(self.env, "t") and hasattr(self.env, "horizon"):
            if self.env.t == self.env.horizon and len(soc_err_list) > 0:
                bat_num = getattr(self.env, "bat_num", len(soc_err_list))
                soc_cost = sum(soc_err_list) / max(bat_num, 1)

        # 电池放电误差
        discharge_err_list = (
            list(discharge_err) if hasattr(discharge_err, "__iter__") else []
        )
        discharge_cost = (
            sum(discharge_err_list) / max(len(discharge_err_list), 1)
            if len(discharge_err_list) > 0
            else 0.0
        )

        # PV控制成本
        pv_diff_list = list(pv_diff) if hasattr(pv_diff, "__iter__") else []
        pv_cost = (
            sum(pv_diff_list) / max(self.env.pv_num, 1)
            if len(pv_diff_list) > 0
            else 0.0
        )

        # 基础控制成本
        base_control_cost = (
            self.control_weights["capacitor"] * cap_cost
            + self.control_weights["regulator"] * reg_cost
            + self.control_weights["battery_soc"] * soc_cost
            + self.control_weights["battery_discharge"] * discharge_cost
            + self.control_weights["pv_control"] * pv_cost
        )

        # 2. 动作平滑惩罚
        smoothing_penalty = 0.0
        if self.use_action_smoothing and self.previous_actions is not None:
            # 使用前面已经转换的列表（避免重复转换）
            current_actions = getattr(
                self.env,
                "applied_control_vector",
                capdiff_list + regdiff_list + pv_diff_list,
            )

            if len(current_actions) == len(self.previous_actions):
                # 计算动作差异的L1范数
                action_diff = sum(
                    abs(curr - prev)
                    for curr, prev in zip(current_actions, self.previous_actions)
                )

                # 按动作数归一化
                smoothing_penalty = action_diff / max(len(current_actions), 1)

        # 更新动作缓存
        if self.use_action_smoothing:
            # 使用前面已经转换的列表（避免重复转换）
            self.previous_actions = list(
                getattr(
                    self.env,
                    "applied_control_vector",
                    capdiff_list + regdiff_list + pv_diff_list,
                )
            )

        # 组合控制奖励
        total_control_cost = (
            base_control_cost + self.weights["action_smoothing"] * smoothing_penalty
        )

        return -total_control_cost

    def pv_reward(self):
        """按原始 DSS Pmpp 归一化的实际发电；不从请求动作伪造出力。"""
        if not getattr(self.env, "pv_control_enabled", False):
            return 0.0
        states = self.env.obs.get("pv_statuses", {})
        return float(np.mean([v[0] for v in states.values()])) * 0.5 if states else 0.0

    def reward_main(
        self,
        capdiff: List[float],
        regdiff: List[float],
        soc_err: List[float],
        discharge_err: List[float],
        pv_diff: Optional[List[float]] = None,
    ) -> float:
        """
        主奖励函数（CMDP主目标，不包含电压约束成本）

        组合三个经济性目标：
        1. 网损最小化奖励
        2. 控制动作平滑奖励
        3. PV合理利用奖励

        Args:
                capdiff: 电容器状态变化
                regdiff: 调压器抽头变化
                soc_err: 电池SOC误差
                discharge_err: 电池放电误差
                pv_diff: PV控制变化

        Returns:
                float: 主奖励值，越高表示经济性越好
        """
        # 计算各组件奖励
        powerloss_r = self.powerloss_reward()
        control_r = self.control_reward(
            capdiff, regdiff, soc_err, discharge_err, pv_diff
        )
        pv_r = self.pv_reward()

        # 加权组合主奖励
        main_reward = (
            self.weights["powerloss"] * powerloss_r
            + self.weights["control"] * control_r
            + self.weights["pv"] * pv_r
        )

        return main_reward

    def get_reward_info(
        self,
        capdiff: List[float],
        regdiff: List[float],
        soc_err: List[float],
        discharge_err: List[float],
        pv_diff: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        生成详细的奖励信息字典

        Args:
                capdiff: 电容器状态变化
                regdiff: 调压器抽头变化
                soc_err: 电池SOC误差
                discharge_err: 电池放电误差
                pv_diff: PV控制变化

        Returns:
                Dict[str, Any]: 包含各组件奖励和约束成本的信息字典
        """
        # 计算所有组件
        powerloss_r = self.powerloss_reward()
        control_r = self.control_reward(
            capdiff, regdiff, soc_err, discharge_err, pv_diff
        )
        pv_r = self.pv_reward()
        voltage_c = self.voltage_cost()
        main_r = (
            self.weights["powerloss"] * powerloss_r
            + self.weights["control"] * control_r
            + self.weights["pv"] * pv_r
        )

        # 电压合规率计算
        voltage_compliance_rate = self._calculate_voltage_compliance()

        info = {
            # 主奖励组件
            "powerloss_reward": powerloss_r,
            "control_reward": control_r,
            "pv_reward": pv_r,
            "reward_main": main_r,
            # 约束成本（CMDP）
            "cost_voltage": voltage_c,
            # 辅助信息
            "voltage_compliance_rate": voltage_compliance_rate,
            "av_cap_err": sum(list(capdiff) if hasattr(capdiff, "__iter__") else [])
            / max(self.env.cap_num, 1),
            "av_reg_err": sum(list(regdiff) if hasattr(regdiff, "__iter__") else [])
            / max(self.env.reg_num, 1),
            "av_soc_err": sum(list(soc_err) if hasattr(soc_err, "__iter__") else [])
            / max(getattr(self.env, "bat_num", 1), 1),
            "av_dis_err": sum(
                list(discharge_err) if hasattr(discharge_err, "__iter__") else []
            )
            / max(getattr(self.env, "bat_num", 1), 1),
            # 权重信息
            "weights": self.weights.copy(),
            "voltage_threshold": self.voltage_threshold,
            "use_action_smoothing": self.use_action_smoothing,
        }

        return info

    def _calculate_voltage_compliance(self) -> float:
        """
        计算电压合规率

        Returns:
                float: 电压合规率（0-1），1表示全部合规
        """
        bus_voltages = self.env.obs.get("bus_voltages", {})
        if not bus_voltages:
            return 1.0  # 无电压数据时默认合规

        total_measurements = 0
        compliant_measurements = 0
        v_min, v_max = self.voltage_range

        for voltages in bus_voltages.values():
            if not voltages:
                continue
            for v in voltages:
                total_measurements += 1
                if v_min <= v <= v_max:
                    compliant_measurements += 1

        return compliant_measurements / max(total_measurements, 1)

    def composite_reward(
        self,
        cd: List[float],
        rd: List[float],
        soc: List[float],
        dis: List[float],
        pv_diff: Optional[List[float]] = None,
        full: bool = True,
        record_node: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        复合奖励计算函数 - 兼容现有环境接口

        在CMDP范式下，返回主奖励作为主要信号，
        约束成本通过info字典传递给约束优化算法。

        Args:
                cd: 电容器状态变化
                rd: 调压器抽头变化
                soc: 电池SOC误差
                dis: 电池放电误差
                pv_diff: PV控制变化
                full: 是否返回完整信息（向后兼容，CMDP版本总是返回完整信息）
                record_node: 是否记录违规节点

        Returns:
                tuple: (主奖励值, 信息字典)
                        - 主奖励：经济性目标，用于策略优化
                        - 信息字典：包含约束成本cost_voltage等详细信息
        """
        # 计算主奖励
        main_reward = None  # 在 get_reward_info 中只计算一次，避免更新两次平滑历史

        # 生成详细信息
        info = self.get_reward_info(cd, rd, soc, dis, pv_diff)
        main_reward = info["reward_main"]

        # 兼容性：记录违规节点（如果需要）
        if record_node:
            violated_nodes = []
            bus_voltages = self.env.obs.get("bus_voltages", {})
            v_min, v_max = self.voltage_range

            for bus_name, voltages in bus_voltages.items():
                if not voltages:
                    continue
                for v in voltages:
                    if not (v_min <= v <= v_max):
                        if bus_name not in violated_nodes:
                            violated_nodes.append(bus_name)

            info["violated_nodes"] = violated_nodes

        # 记录到日志（调试模式）
        if logger.isEnabledFor(logging.DEBUG):
            self._log_reward_components(info)

        return main_reward, info

    def _log_reward_components(self, components: Dict[str, Any]) -> None:
        """
        记录奖励组件到日志

        Args:
                components: 奖励组件字典
        """
        log_parts = []

        # 记录主要组件
        main_components = [
            "reward_main",
            "powerloss_reward",
            "control_reward",
            "pv_reward",
            "cost_voltage",
        ]
        for key in main_components:
            if key in components and isinstance(components[key], (int, float)):
                log_parts.append(f"{key}={components[key]:.3f}")

        if log_parts:
            logger.debug(f"CMDP奖励组件: {', '.join(log_parts)}")

    def reset_episode(self) -> None:
        """
        重置episode状态

        在新episode开始时调用，清理episode相关的状态。
        """
        self.previous_actions = None
        logger.debug("CMDP奖励函数状态已重置")
