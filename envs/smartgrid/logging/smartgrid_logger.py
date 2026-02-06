# -*- coding: utf-8 -*-
"""
@File      : smartgrid_logger.py
@Time      : 2025-08-05
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: PowerZooLLM专属Logger类，用于记录和管理PowerZoo LLM环境的训练与评估过程。
             相比于原始VVCLogger，增加了对PV系统、电池系统的详细监控。

主要功能：
- 继承BaseLogger类，提供基础日志功能
- 详细记录各种奖励组成（电压、功率损耗、控制成本等）
- 实时监控物理量（有功/无功功率、电压、损耗等）
- 专门记录PV系统状态（输出功率、功率因数、利用率等）
- 专门记录电池系统状态（充放电功率、SOC、效率等）
- 支持多线程环境的数据收集和统计
- 集成TensorBoard可视化
"""

import numpy as np
import time
from pathlib import Path
from textwrap import dedent

from common.base_logger import BaseLogger
from envs.smartgrid.logging.base_logger import get_logger
from envs.smartgrid.logging.unified_logger import UnifiedLogManager, get_unified_log_manager
from envs.smartgrid.logging.visualization_manager import VisualizationManager

logger = get_logger(__name__)


class SmartGridLogger(BaseLogger):
    """PowerZoo LLM环境专用Logger类"""
    
    def __init__(self, args, algo_args, env_args, num_agents, writer, run_dir):
        """初始化PowerZooLLM Logger
        
        Args:
            args: 主要参数
            algo_args: 算法参数
            env_args: 环境参数
            num_agents: 智能体数量
            writer: TensorBoard writer
            run_dir: 运行目录
        """
        # 创建统一日志管理器，传入现有的run_dir
        self.log_manager = get_unified_log_manager(args, algo_args, env_args, existing_run_dir=run_dir)
        
        # 使用统一管理器的路径
        unified_run_dir = str(self.log_manager.run_dir)
        
        super(SmartGridLogger, self).__init__(
            args, algo_args, env_args, num_agents, writer, unified_run_dir
        )
        
        # 初始化可视化管理器
        self.viz_manager = None
        if env_args.get('enable_visualization', True):
            plots_dir = self.log_manager.get_path('plots')
            self.viz_manager = VisualizationManager(
                save_dir=str(plots_dir),
                plot_interval=env_args.get('plot_interval', 100),
                buffer_size=env_args.get('viz_buffer_size', 10000),
                enable_plotting=True
            )
            logger.info(f"可视化管理器已初始化: {plots_dir}")
        
        # CMDP相关追踪
        self.train_episode_cost_voltage = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_cost_voltage = []
        
        self.train_episode_lambda = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_lambda = []
        
        logger.info(f"SmartGridLogger 使用统一日志路径: {unified_run_dir}")
        
    def get_task_name(self):
        """获取任务名称"""
        return self.env_args["env_name"]
    
    def init(self, episodes):
        """初始化记录器
        
        Args:
            episodes (int): 总episode数量
        """
        self.start = time.time()
        self.episodes = episodes
        
        # 基础奖励追踪
        self.train_episode_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_rewards = []
        
        # 分离的奖励组成追踪
        self.train_episode_powerloss_reward = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_powerloss_reward = []
        
        self.train_episode_voltage_reward = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_voltage_reward = []
        
        self.train_episode_ctrl_reward = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_ctrl_reward = []
        
        # 新增：PV相关奖励
        self.train_episode_pv_utilization_reward = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_pv_utilization_reward = []
        
        # 物理量追踪
        self.train_episode_power_loss_kw = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_power_loss_kw = []
        
        self.train_episode_power_loss_kvar = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_power_loss_kvar = []
        
        self.train_episode_total_power_kw = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_total_power_kw = []

        self.train_episode_total_power_kvar = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_total_power_kvar = []

        # 新增：记录电网总负荷
        self.train_episode_total_load_kw = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_total_load_kw = []
        
        # 控制设备追踪
        self.train_episode_capacitor_control = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_capacitor_control = []
        
        self.train_episode_regulator_control = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_regulator_control = []
        
        # 电池系统追踪
        self.train_episode_battery_charge = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_battery_charge = []
        
        self.train_episode_battery_discharge = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_battery_discharge = []
        
        self.train_episode_battery_soc = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_battery_soc = []
        
        # PV系统追踪
        self.train_episode_pv_output_kw = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_pv_output_kw = []
        
        self.train_episode_pv_power_factor = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_pv_power_factor = []
        
        self.train_episode_pv_utilization = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_pv_utilization = []
        
        # 电压违规追踪
        self.train_episode_voltage_violations = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_voltage_violations = []
        
        # 新增：单步单bus平均电压违规追踪
        self.train_episode_voltage_violations_per_bus = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_voltage_violations_per_bus = []
        
        # 新增：单步电压违规率追踪
        self.train_episode_voltage_violation_rate = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_voltage_violation_rate = []
        
    def episode_init(self, episode):
        """初始化每个episode的记录器"""
        self.episode = episode
        
    def per_step(self, data):
        """处理每步的数据
        
        Args:
            data (tuple): 包含obs, share_obs, rewards, dones, infos等数据
        """
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data
        
        dones_env = np.all(dones, axis=1)
        reward_env = np.mean(rewards, axis=1).flatten()
        
        # 解析各种奖励组成
        powerloss_reward = self._extract_info_value(infos, 'power_loss_ratio', 0)
        voltage_reward = self._extract_info_value(infos, 'vol_reward', 0)
        ctrl_reward = self._extract_info_value(infos, 'ctrl_reward', 0)
        pv_utilization_reward = self._extract_info_value(infos, 'pv_utilization_reward', 0)
        
        # 解析物理量
        power_loss_kw = self._extract_info_value(infos, 'power_loss_kw', 0)
        power_loss_kvar = self._extract_info_value(infos, 'power_loss_kvar', 0)
        total_power_kw = self._extract_info_value(infos, 'total_power_kw', 0)
        total_power_kvar = self._extract_info_value(infos, 'total_power_kvar', 0)
        total_load_kw = self._extract_info_value(infos, 'total_load_kw', 0)
        
        # 解析控制量
        capacitor_ctrl = self._extract_info_value(infos, 'capacitor_ctrl', 0)
        regulator_ctrl = self._extract_info_value(infos, 'regulator_ctrl', 0)
        
        # 解析电池系统信息
        battery_charge = self._extract_info_value(infos, 'battery_charge_kw', 0)
        battery_discharge = self._extract_info_value(infos, 'battery_discharge_kw', 0)
        battery_soc = self._extract_info_value(infos, 'battery_avg_soc', 0.0)
        
        # 解析PV系统信息
        pv_output_kw = self._extract_info_value(infos, 'pv_output_kw', 0)
        pv_power_factor = self._extract_info_value(infos, 'pv_avg_power_factor', 1.0)
        pv_utilization = self._extract_info_value(infos, 'pv_utilization', 0)
        
        # 解析电压违规
        voltage_violations = self._extract_info_value(infos, 'voltage_violation_count', 0)
        voltage_violations_per_bus = self._extract_info_value(infos, 'voltage_violations_per_bus', 0)
        voltage_violation_rate = self._extract_info_value(infos, 'voltage_violation_rate', 0)
        
        # 解析CMDP相关信息
        cost_voltage = self._extract_info_value(infos, 'cost_voltage', 0)
        lambda_value = self._extract_info_value(infos, 'lambda', 0)
        
        # 更新累计值
        self.train_episode_rewards += reward_env
        
        # 处理奖励组成 - 确保正确的除法操作
        episode_length = float(self.algo_args["train"]["episode_length"])
        n_threads = self.algo_args["train"]["n_rollout_threads"]
        
        # 确保所有值都是一维数组且长度正确
        def ensure_correct_shape(arr, expected_len=n_threads):
            """确保数组有正确的形状用于累加"""
            arr = np.atleast_1d(arr).flatten()
            if len(arr) != expected_len:
                # 如果长度不匹配，取前n_threads个元素或填充
                if len(arr) > expected_len:
                    arr = arr[:expected_len]
                else:
                    # 如果元素不足，用最后一个值填充
                    arr = np.pad(arr, (0, expected_len - len(arr)), mode='edge')
            return arr
        
        powerloss_reward = ensure_correct_shape(powerloss_reward)
        voltage_reward = ensure_correct_shape(voltage_reward)
        ctrl_reward = ensure_correct_shape(ctrl_reward)
        pv_utilization_reward = ensure_correct_shape(pv_utilization_reward)
        
        self.train_episode_powerloss_reward += powerloss_reward / episode_length
        self.train_episode_voltage_reward += voltage_reward
        self.train_episode_ctrl_reward += ctrl_reward
        self.train_episode_pv_utilization_reward += pv_utilization_reward
        
        # 更新物理量 - 功率是瞬时值，应该累加后求平均，而不是每步除以episode_length
        power_loss_kw = ensure_correct_shape(power_loss_kw)
        power_loss_kvar = ensure_correct_shape(power_loss_kvar)
        total_power_kw = ensure_correct_shape(total_power_kw)
        total_power_kvar = ensure_correct_shape(total_power_kvar)
        total_load_kw = ensure_correct_shape(total_load_kw)
        
        # 累加瞬时功率值，最后会在episode结束时求平均
        self.train_episode_power_loss_kw += power_loss_kw
        self.train_episode_power_loss_kvar += power_loss_kvar
        self.train_episode_total_power_kw += total_power_kw
        self.train_episode_total_power_kvar += total_power_kvar
        self.train_episode_total_load_kw += total_load_kw
        
        # 更新控制量
        capacitor_ctrl = ensure_correct_shape(capacitor_ctrl)
        regulator_ctrl = ensure_correct_shape(regulator_ctrl)
        
        self.train_episode_capacitor_control += capacitor_ctrl
        self.train_episode_regulator_control += regulator_ctrl
        
        # 更新电池系统 - 功率是瞬时值，SOC是状态值
        battery_charge = ensure_correct_shape(battery_charge)
        battery_discharge = ensure_correct_shape(battery_discharge)
        battery_soc = ensure_correct_shape(battery_soc)
        
        self.train_episode_battery_charge += battery_charge  # 瞬时充电功率
        self.train_episode_battery_discharge += battery_discharge  # 瞬时放电功率
        self.train_episode_battery_soc += battery_soc  # SOC状态值
        
        # 更新PV系统 - 功率是瞬时值，利用率和功率因数是比率
        pv_output_kw = ensure_correct_shape(pv_output_kw)
        pv_power_factor = ensure_correct_shape(pv_power_factor)
        pv_utilization = ensure_correct_shape(pv_utilization)
        
        self.train_episode_pv_output_kw += pv_output_kw  # 瞬时输出功率
        self.train_episode_pv_power_factor += pv_power_factor  # 功率因数
        self.train_episode_pv_utilization += pv_utilization  # 利用率
        
        # 更新电压违规
        voltage_violations = ensure_correct_shape(voltage_violations)
        voltage_violations_per_bus = ensure_correct_shape(voltage_violations_per_bus)
        voltage_violation_rate = ensure_correct_shape(voltage_violation_rate)
        
        self.train_episode_voltage_violations += voltage_violations
        self.train_episode_voltage_violations_per_bus += voltage_violations_per_bus
        # 修复：违规率不应该累加，应该记录最新的值或者计算平均值
        # voltage_violation_rate 是当前步的违规率，不应该累加
        self.train_episode_voltage_violation_rate = voltage_violation_rate  # 记录最新值而不是累加
        
        # 更新CMDP追踪
        cost_voltage = ensure_correct_shape(cost_voltage)
        lambda_value = ensure_correct_shape(lambda_value)
        self.train_episode_cost_voltage += cost_voltage
        self.train_episode_lambda = lambda_value  # Lambda是瞬时值，不累加
        
        # 更新可视化管理器
        if self.viz_manager and len(infos) > 0 and isinstance(infos[0], list) and len(infos[0]) > 0:
            # 提取第一个环境的info用于可视化
            viz_info = infos[0][0] if isinstance(infos[0][0], dict) else {}
            
            # 添加更多可视化需要的信息
            viz_info.update({
                'reward_main': reward_env[0] if len(reward_env) > 0 else 0,
                'powerloss_reward': powerloss_reward[0] if len(powerloss_reward) > 0 else 0,
                'control_reward': ctrl_reward[0] if len(ctrl_reward) > 0 else 0,
                'pv_reward': pv_utilization_reward[0] if len(pv_utilization_reward) > 0 else 0,
                'cost_voltage': cost_voltage[0] if len(cost_voltage) > 0 else 0,
                'lambda': lambda_value[0] if len(lambda_value) > 0 else 0,
                'voltage_violation_rate': voltage_violation_rate[0] if len(voltage_violation_rate) > 0 else 0
            })
            
            self.viz_manager.update(viz_info)
        
        # 处理完成的episode
        for t in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[t]:
                self._record_episode_done(t)
                
    def _extract_info_value(self, infos, key, default):
        """从infos中提取指定key的值
        
        Args:
            infos: 信息列表 - 期望格式: [[{key: value}], [{key: value}], ...]
            key: 要提取的键
            default: 默认值
            
        Returns:
            提取的值数组
        """
        try:
            if not infos:
                return np.array([default])
            
            # 根据infos的实际结构进行提取
            values = []
            for info_item in infos:
                if isinstance(info_item, list) and len(info_item) > 0:
                    # infos结构: [[{...}], [{...}], ...]
                    info_dict = info_item[0]
                    if isinstance(info_dict, dict):
                        values.append(info_dict.get(key, default))
                    else:
                        values.append(default)
                elif isinstance(info_item, dict):
                    # infos结构: [{...}, {...}, ...]
                    values.append(info_item.get(key, default))
                else:
                    values.append(default)
            
            if not values:
                return np.array([default])
                
            # 转换为numpy数组并确保是数值类型
            return np.array(values, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"提取信息 '{key}' 时出错: {e}")
            return np.array([default], dtype=np.float32)
    
    def _calculate_power_loss_percentage(self):
        """计算功率损耗百分比
        
        功率损耗百分比 = 功率损耗 / (电网总负荷 + PV总发电功率) * 100
        
        Returns:
            float: 功率损耗百分比
        """
        if not self.done_episodes_power_loss_kw or not self.done_episodes_total_load_kw:
            return 0.0

        avg_power_loss = np.mean(self.done_episodes_power_loss_kw)
        avg_total_load = abs(np.mean(self.done_episodes_total_load_kw))  # 电网总负荷（取绝对值）
        avg_pv_generation = (
            np.mean(self.done_episodes_pv_output_kw)
            if self.done_episodes_pv_output_kw
            else 0.0
        )  # PV发电功率

        total_input_power = avg_total_load + avg_pv_generation
        if total_input_power <= 0:
            return 0.0

        power_loss_percentage = (avg_power_loss / total_input_power) * 100

        # 限制在合理范围内（通常不会超过50%）
        power_loss_percentage = min(power_loss_percentage, 50.0)

        return power_loss_percentage
    
    def _record_episode_done(self, thread_id):
        """记录完成的episode数据
        
        Args:
            thread_id: 线程ID
        """
        # 记录总奖励
        self.done_episodes_rewards.append(self.train_episode_rewards[thread_id])
        self.train_episode_rewards[thread_id] = 0
        
        # 记录奖励组成
        self.done_episodes_powerloss_reward.append(self.train_episode_powerloss_reward[thread_id])
        self.train_episode_powerloss_reward[thread_id] = 0
        
        self.done_episodes_voltage_reward.append(self.train_episode_voltage_reward[thread_id])
        self.train_episode_voltage_reward[thread_id] = 0
        
        self.done_episodes_ctrl_reward.append(self.train_episode_ctrl_reward[thread_id])
        self.train_episode_ctrl_reward[thread_id] = 0
        
        self.done_episodes_pv_utilization_reward.append(self.train_episode_pv_utilization_reward[thread_id])
        self.train_episode_pv_utilization_reward[thread_id] = 0
        
        # 记录物理量 - 计算平均功率
        episode_length = float(self.algo_args["train"]["episode_length"])
        self.done_episodes_power_loss_kw.append(self.train_episode_power_loss_kw[thread_id] / episode_length)
        self.train_episode_power_loss_kw[thread_id] = 0
        
        self.done_episodes_power_loss_kvar.append(self.train_episode_power_loss_kvar[thread_id] / episode_length)
        self.train_episode_power_loss_kvar[thread_id] = 0
        
        self.done_episodes_total_power_kw.append(self.train_episode_total_power_kw[thread_id] / episode_length)
        self.train_episode_total_power_kw[thread_id] = 0

        self.done_episodes_total_power_kvar.append(self.train_episode_total_power_kvar[thread_id] / episode_length)
        self.train_episode_total_power_kvar[thread_id] = 0

        self.done_episodes_total_load_kw.append(self.train_episode_total_load_kw[thread_id] / episode_length)
        self.train_episode_total_load_kw[thread_id] = 0
        
        # 记录控制量
        self.done_episodes_capacitor_control.append(self.train_episode_capacitor_control[thread_id])
        self.train_episode_capacitor_control[thread_id] = 0
        
        self.done_episodes_regulator_control.append(self.train_episode_regulator_control[thread_id])
        self.train_episode_regulator_control[thread_id] = 0
        
        # 记录电池系统 - 计算平均值
        self.done_episodes_battery_charge.append(self.train_episode_battery_charge[thread_id] / episode_length)
        self.train_episode_battery_charge[thread_id] = 0
        
        self.done_episodes_battery_discharge.append(self.train_episode_battery_discharge[thread_id] / episode_length)
        self.train_episode_battery_discharge[thread_id] = 0
        
        self.done_episodes_battery_soc.append(self.train_episode_battery_soc[thread_id] / episode_length)
        self.train_episode_battery_soc[thread_id] = 0
        
        # 记录PV系统 - 计算平均值
        self.done_episodes_pv_output_kw.append(self.train_episode_pv_output_kw[thread_id] / episode_length)
        self.train_episode_pv_output_kw[thread_id] = 0
        
        self.done_episodes_pv_power_factor.append(self.train_episode_pv_power_factor[thread_id] / episode_length)
        self.train_episode_pv_power_factor[thread_id] = 0
        
        self.done_episodes_pv_utilization.append(self.train_episode_pv_utilization[thread_id] / episode_length)
        self.train_episode_pv_utilization[thread_id] = 0
        
        # 记录电压违规
        self.done_episodes_voltage_violations.append(self.train_episode_voltage_violations[thread_id])
        self.train_episode_voltage_violations[thread_id] = 0
        
        # 记录单步单bus平均电压违规
        self.done_episodes_voltage_violations_per_bus.append(
            self.train_episode_voltage_violations_per_bus[thread_id] / episode_length
        )
        self.train_episode_voltage_violations_per_bus[thread_id] = 0
        
        # 记录单步电压违规率
        # 修复：违规率已经是比率了，不需要除以episode_length
        self.done_episodes_voltage_violation_rate.append(
            self.train_episode_voltage_violation_rate[thread_id]
        )
        self.train_episode_voltage_violation_rate[thread_id] = 0
        
        # 记录CMDP相关
        avg_cost = self.train_episode_cost_voltage[thread_id] / episode_length
        self.done_episodes_cost_voltage.append(avg_cost)
        self.train_episode_cost_voltage[thread_id] = 0
        
        # Lambda是瞬时值，取最后一个
        self.done_episodes_lambda.append(self.train_episode_lambda[thread_id])
        
        # 更新可视化管理器的episode结束信息
        if self.viz_manager:
            episode_reward = self.done_episodes_rewards[-1] if self.done_episodes_rewards else 0
            self.viz_manager.update_episode_end(
                episode_reward=episode_reward,
                episode_cost=avg_cost,
                lambda_value=self.train_episode_lambda[thread_id]
            )
        
    def episode_log(self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer):
        """记录episode的日志信息
        
        Args:
            actor_train_infos: actor训练信息
            critic_train_info: critic训练信息
            actor_buffer: actor缓冲区
            critic_buffer: critic缓冲区
        """
        # 计算总步数
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        
        self.end = time.time()
        
        # 计算平均步数奖励
        critic_train_info["average_step_rewards"] = critic_buffer.get_mean_rewards()
        self.log_train(actor_train_infos, critic_train_info)
        
        training_info = dedent(f"""
            环境：{self.args["env"]} 任务: {self.task_name} 算法: {self.args["algo"]} 实验名称: {self.args["exp_name"]} 
            更新次数 {self.episode}/{self.episodes} episodes, 总时间步数 {self.total_num_steps}/{self.algo_args["train"]["num_env_steps"]}, FPS {int(self.total_num_steps / (self.end - self.start))}. 
            平均步数奖励为 {critic_train_info["average_step_rewards"]:.4f}.\n""")
        
        print(training_info)
        self.log_training_info.write(training_info)
        self.log_training_info.flush()
        
        # 记录详细的训练指标
        if len(self.done_episodes_rewards) > 0:
            self._log_episode_metrics()
            
    def _log_episode_metrics(self):
        """记录episode级别的指标到tensorboard和日志文件"""
        
        # 辅助函数：安全计算平均值，处理NaN和Inf
        def safe_mean(values, default=0.0):
            if not values:
                return default
            arr = np.array(values)
            # 过滤掉NaN和Inf
            valid_mask = np.isfinite(arr)
            if np.any(valid_mask):
                return np.mean(arr[valid_mask])
            else:
                logger.warning(f"所有值都是NaN或Inf，返回默认值: {default}")
                return default
        
        # 计算平均值
        metrics = {
            # 总奖励
            "total_reward": safe_mean(self.done_episodes_rewards),
            
            # 奖励组成
            "powerloss_reward": safe_mean(self.done_episodes_powerloss_reward),
            "voltage_reward": safe_mean(self.done_episodes_voltage_reward),
            "control_reward": safe_mean(self.done_episodes_ctrl_reward),
            "pv_utilization_reward": safe_mean(self.done_episodes_pv_utilization_reward),
            
            # 功率损耗
            "power_loss_kw": safe_mean(self.done_episodes_power_loss_kw),
            "power_loss_kvar": safe_mean(self.done_episodes_power_loss_kvar),
            # 修复：计算功率损耗百分比，应该考虑电网总负荷+PV总发电功率作为基准
            "power_loss_percentage": self._calculate_power_loss_percentage(),
            
            # 总功率
            "total_power_kw": safe_mean(self.done_episodes_total_power_kw),
            "total_power_kvar": safe_mean(self.done_episodes_total_power_kvar),
            "total_load_kw": safe_mean(self.done_episodes_total_load_kw),
            
            # 控制动作
            "capacitor_switches": safe_mean(self.done_episodes_capacitor_control),
            "regulator_changes": safe_mean(self.done_episodes_regulator_control),
            
            # 电池系统
            "battery_charge_kw": safe_mean(self.done_episodes_battery_charge),
            "battery_discharge_kw": safe_mean(self.done_episodes_battery_discharge),
            "battery_avg_soc": safe_mean(self.done_episodes_battery_soc, 0.0),  # 保持为0-1范围，格式化时转换
            
            # PV系统
            "pv_output_kw": safe_mean(self.done_episodes_pv_output_kw),
            "pv_avg_power_factor": safe_mean(self.done_episodes_pv_power_factor, 1.0),
            "pv_utilization": safe_mean(self.done_episodes_pv_utilization),  # 保持为0-1范围，格式化时转换
            
            # 电压质量
            "voltage_violations": safe_mean(self.done_episodes_voltage_violations),
            "voltage_violations_per_bus": safe_mean(self.done_episodes_voltage_violations_per_bus),
            "voltage_violation_rate": safe_mean(self.done_episodes_voltage_violation_rate),  # 保持为0-1范围，格式化时转换
            
            # CMDP相关
            "cost_voltage": safe_mean(self.done_episodes_cost_voltage) if self.done_episodes_cost_voltage else 0,
            "lambda": safe_mean(self.done_episodes_lambda) if self.done_episodes_lambda else 0,
        }
        
        # 打印关键指标
        log_info = dedent(f"""
            ========== Episode {self.episode} 训练指标 ==========
            平均总奖励: {metrics['total_reward']:.4f}
            功率损耗: {metrics['power_loss_kw']:.2f} kW ({metrics['power_loss_percentage']:.2f}%)
            电压违规次数: {metrics['voltage_violations']:.1f}
            单步单bus电压违规: {metrics['voltage_violations_per_bus']:.3f}
            电压违规率: {metrics['voltage_violation_rate']:.2%}
            CMDP电压成本: {metrics['cost_voltage']:.4f}
            Lagrangian Lambda: {metrics['lambda']:.4f}
            PV利用率: {metrics['pv_utilization']:.2%}
            电池平均SOC: {metrics['battery_avg_soc']:.2%}
            ================================================
        """)
        
        print(log_info)
        self.log_training_info.write(log_info)
        self.log_training_info.flush()
        
        # 记录到tensorboard
        for key, value in metrics.items():
            self.writer.add_scalar(f"train/{key}", value, self.total_num_steps)
            
        # 记录奖励分解到tensorboard（用于分析奖励函数）
        self.writer.add_scalars(
            "train/reward_breakdown",
            {
                "total": metrics['total_reward'],
                "powerloss": metrics['powerloss_reward'],
                "voltage": metrics['voltage_reward'],
                "control": metrics['control_reward'],
                "pv_utilization": metrics['pv_utilization_reward'],
            },
            self.total_num_steps
        )
        
        # 记录功率相关指标
        self.writer.add_scalars(
            "train/power_metrics",
            {
                "loss_kw": metrics['power_loss_kw'],
                "loss_kvar": metrics['power_loss_kvar'],
                "total_kw": metrics['total_power_kw'],
                "total_kvar": metrics['total_power_kvar'],
                "load_kw": metrics['total_load_kw'],
                "loss_percentage": metrics['power_loss_percentage'],
            },
            self.total_num_steps
        )
        
        # 记录电压质量指标
        self.writer.add_scalars(
            "train/voltage_quality",
            {
                "violations_total": metrics['voltage_violations'],
                "violations_per_bus": metrics['voltage_violations_per_bus'],
                "violation_rate": metrics['voltage_violation_rate'],
            },
            self.total_num_steps
        )
        
        # 记录可再生能源指标
        self.writer.add_scalars(
            "train/renewable_metrics",
            {
                "pv_output_kw": metrics['pv_output_kw'],
                "pv_power_factor": metrics['pv_avg_power_factor'],
                "pv_utilization": metrics['pv_utilization'],
                "battery_charge": metrics['battery_charge_kw'],
                "battery_discharge": metrics['battery_discharge_kw'],
                "battery_soc": metrics['battery_avg_soc'],
            },
            self.total_num_steps
        )
        
        # 清空已完成的episodes列表
        self._clear_done_episodes()
        
    def _clear_done_episodes(self):
        """清空已完成episodes的数据列表"""
        self.done_episodes_rewards = []
        self.done_episodes_powerloss_reward = []
        self.done_episodes_voltage_reward = []
        self.done_episodes_ctrl_reward = []
        self.done_episodes_pv_utilization_reward = []
        self.done_episodes_power_loss_kw = []
        self.done_episodes_power_loss_kvar = []
        self.done_episodes_total_power_kw = []
        self.done_episodes_total_power_kvar = []
        self.done_episodes_total_load_kw = []
        self.done_episodes_capacitor_control = []
        self.done_episodes_regulator_control = []
        self.done_episodes_battery_charge = []
        self.done_episodes_battery_discharge = []
        self.done_episodes_battery_soc = []
        self.done_episodes_pv_output_kw = []
        self.done_episodes_pv_power_factor = []
        self.done_episodes_pv_utilization = []
        self.done_episodes_voltage_violations = []
        self.done_episodes_voltage_violations_per_bus = []
        self.done_episodes_voltage_violation_rate = []
        
    def eval_init(self):
        """初始化评估过程"""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        
        # 初始化评估数据收集器
        self.eval_env_reward_infos = {}
        self._init_eval_collectors()
        
    def _init_eval_collectors(self):
        """初始化评估数据收集器"""
        n_eval_threads = self.algo_args["eval"]["n_eval_rollout_threads"]
        
        # 奖励收集器
        self.eval_episode_rewards = []
        self.one_episode_rewards = []
        
        # 各项指标收集器
        collectors = [
            'powerloss', 'voltage', 'ctrl', 'pv_utilization',
            'power_loss_kw', 'power_loss_kvar', 'total_power_kw', 'total_power_kvar',
            'capacitor_control', 'regulator_control',
            'battery_charge', 'battery_discharge', 'battery_soc',
            'pv_output_kw', 'pv_power_factor', 'pv_utilization_eval',
            'voltage_violations', 'voltage_violations_per_bus', 'voltage_violation_rate'
        ]
        
        for collector in collectors:
            setattr(self, f'eval_{collector}_episode_rewards', [])
            setattr(self, f'one_{collector}_episode_rewards', [])
            
        # 为每个评估线程初始化列表
        for eval_i in range(n_eval_threads):
            self.one_episode_rewards.append([])
            self.eval_episode_rewards.append([])
            
            for collector in collectors:
                getattr(self, f'one_{collector}_episode_rewards').append([])
                getattr(self, f'eval_{collector}_episode_rewards').append([])
                
        # 为新增的电压质量指标初始化
        self.one_voltage_violations_per_bus_episode_rewards = []
        self.eval_voltage_violations_per_bus_episode_rewards = []
        self.one_voltage_violation_rate_episode_rewards = []
        self.eval_voltage_violation_rate_episode_rewards = []
        
        for eval_i in range(n_eval_threads):
            self.one_voltage_violations_per_bus_episode_rewards.append([])
            self.eval_voltage_violations_per_bus_episode_rewards.append([])
            self.one_voltage_violation_rate_episode_rewards.append([])
            self.eval_voltage_violation_rate_episode_rewards.append([])
                
    def eval_per_step(self, eval_data):
        """记录评估过程中的每步数据
        
        Args:
            eval_data: 评估数据元组
        """
        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = eval_data
        
        # 提取各项指标（类似训练时的处理）
        metrics = self._extract_eval_metrics(eval_infos)
        
        # 记录到对应的收集器
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
            
            for key, values in metrics.items():
                collector_name = f'one_{key}_episode_rewards'
                if hasattr(self, collector_name):
                    getattr(self, collector_name)[eval_i].append(values[eval_i])
                    
        self.eval_infos = eval_infos
        
    def _extract_eval_metrics(self, eval_infos):
        """从评估信息中提取各项指标"""
        metrics = {}
        
        # 需要提取的指标及其默认值
        metric_keys = {
            'powerloss': ('power_loss_ratio', 0),
            'voltage': ('vol_reward', 0),
            'ctrl': ('ctrl_reward', 0),
            'pv_utilization': ('pv_utilization_reward', 0),
            'power_loss_kw': ('power_loss_kw', 0),
            'power_loss_kvar': ('power_loss_kvar', 0),
            'total_power_kw': ('total_power_kw', 0),
            'total_power_kvar': ('total_power_kvar', 0),
            'capacitor_control': ('capacitor_ctrl', 0),
            'regulator_control': ('regulator_ctrl', 0),
            'battery_charge': ('battery_charge_kw', 0),
            'battery_discharge': ('battery_discharge_kw', 0),
            'battery_soc': ('battery_avg_soc', 0.0),
            'pv_output_kw': ('pv_output_kw', 0),
            'pv_power_factor': ('pv_avg_power_factor', 1.0),
            'pv_utilization_eval': ('pv_utilization', 0),
            'voltage_violations': ('voltage_violation_count', 0),
            'voltage_violations_per_bus': ('voltage_violations_per_bus', 0),
            'voltage_violation_rate': ('voltage_violation_rate', 0),
        }
        
        for metric_name, (info_key, default_value) in metric_keys.items():
            values = self._extract_info_value(eval_infos, info_key, default_value)
            values_env = np.array(values).reshape((-1, 1, 1))  # 确保形状正确
            metrics[metric_name] = values_env
            
        return metrics
        
    def eval_thread_done(self, tid):
        """记录每个评估线程的完成信息
        
        Args:
            tid: 线程ID
        """
        # 计算episode总和
        self.eval_episode_rewards[tid].append(
            np.sum(self.one_episode_rewards[tid], axis=0)
        )
        
        # 计算各项指标的episode总和
        episode_length = len(self.one_episode_rewards[tid])
        
        # 需要求平均的指标
        avg_metrics = ['powerloss', 'power_loss_kw', 'power_loss_kvar', 
                      'total_power_kw', 'total_power_kvar', 'battery_soc',
                      'pv_output_kw', 'pv_power_factor', 'pv_utilization_eval']
        
        # 需要求和的指标
        sum_metrics = ['voltage', 'ctrl', 'pv_utilization', 'capacitor_control',
                      'regulator_control', 'battery_charge', 'battery_discharge',
                      'voltage_violations']
        
        # 为新的电压质量指标增加到平均指标列表
        avg_metrics.extend(['voltage_violations_per_bus', 'voltage_violation_rate'])
        
        for metric in avg_metrics:
            collector = f'eval_{metric}_episode_rewards'
            one_episode = f'one_{metric}_episode_rewards'
            if hasattr(self, collector):
                getattr(self, collector)[tid].append(
                    np.sum(getattr(self, one_episode)[tid], axis=0) / episode_length
                )
                
        for metric in sum_metrics:
            collector = f'eval_{metric}_episode_rewards'
            one_episode = f'one_{metric}_episode_rewards'
            if hasattr(self, collector):
                getattr(self, collector)[tid].append(
                    np.sum(getattr(self, one_episode)[tid], axis=0)
                )
        
        # 清空当前episode数据
        self.one_episode_rewards[tid] = []
        for metric in avg_metrics + sum_metrics:
            one_episode = f'one_{metric}_episode_rewards'
            if hasattr(self, one_episode):
                getattr(self, one_episode)[tid] = []
                
    def eval_log(self, eval_episode):
        """记录评估信息"""
        # 合并所有线程的数据
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        
        # 收集所有评估指标
        eval_metrics = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
        }
        
        # 添加其他指标
        metric_names = [
            'powerloss', 'voltage', 'ctrl', 'pv_utilization',
            'power_loss_kw', 'power_loss_kvar', 'total_power_kw', 'total_power_kvar',
            'capacitor_control', 'regulator_control',
            'battery_charge', 'battery_discharge', 'battery_soc',
            'pv_output_kw', 'pv_power_factor', 'pv_utilization_eval',
            'voltage_violations', 'voltage_violations_per_bus', 'voltage_violation_rate'
        ]
        
        for metric in metric_names:
            collector = f'eval_{metric}_episode_rewards'
            if hasattr(self, collector):
                data = getattr(self, collector)
                concatenated = np.concatenate([d for d in data if d]) if data else np.array([])
                eval_metrics[f'eval_{metric}_average'] = concatenated
                
        self.eval_env_reward_infos.update(eval_metrics)
        self.log_env(eval_metrics)
        
        # 打印评估摘要
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        eval_power_loss = np.mean(eval_metrics.get('eval_power_loss_kw_average', [0]))
        eval_voltage_violations = np.mean(eval_metrics.get('eval_voltage_violations_average', [0]))
        eval_voltage_violations_per_bus = np.mean(eval_metrics.get('eval_voltage_violations_per_bus_average', [0]))
        eval_voltage_violation_rate = np.mean(eval_metrics.get('eval_voltage_violation_rate_average', [0]))
        eval_pv_utilization = np.mean(eval_metrics.get('eval_pv_utilization_eval_average', [0]))
        
        eval_summary = dedent(f"""
            ========== 评估结果 (Episode {self.episode}) ==========
            平均奖励: {eval_avg_rew:.4f}
            平均功率损耗: {eval_power_loss:.2f} kW
            平均电压违规: {eval_voltage_violations:.1f}
            单步单bus电压违规: {eval_voltage_violations_per_bus:.3f}
            电压违规率: {eval_voltage_violation_rate:.2%}
            平均PV利用率: {eval_pv_utilization:.2%}
            ===================================================
        """)
        
        print(eval_summary)
        
        # 写入progress文件
        log_info = f"{self.total_num_steps},{eval_avg_rew:.4f}\n"
        self.log_file.write(log_info)
        self.log_file.flush()
        
    def get_result(self):
        """获取训练和评估的结果
        
        Returns:
            包含主要指标的字典
        """
        result = {
            "train_avg_reward": np.mean(self.done_episodes_rewards) if self.done_episodes_rewards else 0,
            "eval_avg_reward": np.mean(self.eval_env_reward_infos.get("eval_average_episode_rewards", [])) if "eval_average_episode_rewards" in self.eval_env_reward_infos else 0,
            "eval_power_loss_kw": np.mean(self.eval_env_reward_infos.get("eval_power_loss_kw_average", [])) if "eval_power_loss_kw_average" in self.eval_env_reward_infos else 0,
            "eval_voltage_violations": np.mean(self.eval_env_reward_infos.get("eval_voltage_violations_average", [])) if "eval_voltage_violations_average" in self.eval_env_reward_infos else 0,
            "eval_pv_utilization": np.mean(self.eval_env_reward_infos.get("eval_pv_utilization_eval_average", [])) if "eval_pv_utilization_eval_average" in self.eval_env_reward_infos else 0,
            "eval_battery_soc": np.mean(self.eval_env_reward_infos.get("eval_battery_soc_average", [])) if "eval_battery_soc_average" in self.eval_env_reward_infos else 0,
            "eval_voltage_violations_per_bus": np.mean(self.eval_env_reward_infos.get("eval_voltage_violations_per_bus_average", [])) if "eval_voltage_violations_per_bus_average" in self.eval_env_reward_infos else 0,
            "eval_voltage_violation_rate": np.mean(self.eval_env_reward_infos.get("eval_voltage_violation_rate_average", [])) if "eval_voltage_violation_rate_average" in self.eval_env_reward_infos else 0,
        }
        return result