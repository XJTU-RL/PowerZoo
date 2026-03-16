# -*- coding: utf-8 -*-
"""
DSR Environment
统一的配电网恢复环境，支持标准RL和DAN算法
"""

import copy
import numpy as np

# Conditional gym/gymnasium import
try:
	import gymnasium as gym
	from gymnasium.spaces import Discrete, Box
except ImportError:
	import gym
	from gym.spaces import Discrete, Box
from typing import List, Dict, Any, Tuple, Optional

from envs.dsr.core.dsr_core import DSRCoreEnv
from envs.dsr.core.config import DSRConfig


class DSREnv:
    """统一的配电网恢复环境，支持多种算法需求"""
    
    def __init__(self, config_or_args, rank: Optional[int] = None):
        """
        初始化DSR环境

        Args:
            config_or_args: DSRConfig 实例或环境参数字典（向后兼容）
            rank: 并行进程编号
        """
        if isinstance(config_or_args, DSRConfig):
            self.config = config_or_args
            self.args = {}
        else:
            self.args = copy.deepcopy(config_or_args)
            self.config = DSRConfig.from_env_args(self.args)

        self.rank = rank

        # 创建核心环境
        self.core_env = DSRCoreEnv(self.config, worker_idx=rank)
        self.dsr_core = self.core_env  # 兼容性别名

        # 设置智能体信息
        self.n_agents = self.core_env.n_agents

        # 复制必要属性
        self.agent_types = self.core_env.agent_types
        self.agent_bus_mapping = self.core_env.agent_bus_mapping

        # 定义观测和动作空间
        self._setup_spaces()

        # 环境状态
        self.current_step = 0

        # PowerZoo兼容性设置（优先从 config，回退到 args）
        self.env_name = self.config.env_name
        self.useS = self.args.get('useS', False)
        self.use_render = self.config.use_render
        self.record_node = self.config.record_node

        # DAN算法支持标志
        self.use_dan = self.args.get('use_dan', False)

        # 增强功能参数（主要用于DAN算法）
        if self.use_dan:
            self._setup_dan_features()
        else:
            # 使用基础奖励权重
            self.reward_restore = self.config.reward_restore
            self.reward_voltage = self.config.reward_voltage
            self.reward_overload = self.config.reward_overload
            self.reward_done = self.config.reward_done
            self.use_enhanced_action_mask = False
            self.use_progressive_penalty = False
            self.terminate_on_severe_overload = False

        # 设置ordered_agents_pairs和agents_bus
        if self.useS:
            agents_names = [f"agent_{i}" for i in range(self.n_agents)]
            update_orders = list(range(self.n_agents))
            self.ordered_agents_pairs = dict(zip(agents_names, update_orders))
            self.agents_bus = getattr(self.core_env, 'agents_bus', None)
        else:
            self.ordered_agents_pairs = None
            self.agents_bus = None

        # 动作是否为离散
        self.discrete = True
    
    def _setup_dan_features(self):
        """设置DAN算法相关的增强功能

        优先从 self.config（DSRConfig 字段）读取，
        DAN 专属参数（不在 DSRConfig 中的）从 self.args dict 回退读取。
        """
        # 优化的奖励函数权重（config 中有对应字段）
        self.reward_restore = self.config.reward_restore
        self.reward_voltage = self.args.get('reward_voltage', 2.0)  # DAN用更高权重
        self.reward_overload = self.args.get('reward_overload', 8.0)  # DAN用更高权重
        self.reward_severe_overload = self.args.get('reward_severe_overload', 50.0)  # 严重过载惩罚
        self.reward_done = self.config.reward_done

        # 安全约束参数（DAN专属）
        self.severe_overload_threshold = self.args.get('severe_overload_threshold', 2.0)
        self.max_overload_current = self.args.get('max_overload_current', 1000.0)
        self.terminate_on_severe_overload = self.args.get('terminate_on_severe_overload', True)

        # 渐进式惩罚参数（DAN专属）
        self.use_progressive_penalty = self.args.get('use_progressive_penalty', True)
        self.overload_penalty_levels = self.args.get('overload_penalty_levels', [1.2, 1.5, 2.0])
        self.overload_penalty_weights = self.args.get('overload_penalty_weights', [1.0, 3.0, 8.0, 20.0])

        # 动作掩码增强（DAN专属）
        self.use_enhanced_action_mask = self.args.get('use_enhanced_action_mask', True)
        self.action_mask_safety_margin = self.args.get('action_mask_safety_margin', 0.1)
    
    def _setup_spaces(self):
        """设置观测和动作空间"""
        # 观测空间维度
        obs_dim = self._calculate_obs_dim()
        
        # 每个智能体的观测空间
        self.observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        
        # 共享观测空间（与单智能体观测相同）
        self.share_observation_space = self.observation_space.copy()
        
        # 动作空间（初始设置，将在reset后更新）
        self.action_space = []
        self._setup_initial_action_spaces()
    
    def _setup_initial_action_spaces(self):
        """设置初始动作空间（在知道故障线路之前）"""
        self.action_space = []
        
        # 计算最大动作空间大小
        max_switch_actions = len(self.core_env.faultable_lines) + 1
        max_pv_actions = self.config.pv_power_levels
        max_load_actions = self.config.load_action_levels
        
        # 使用所有类型中的最大动作数作为统一的动作空间大小
        max_actions = max(max_switch_actions, max_pv_actions, max_load_actions)
        
        for i in range(self.n_agents):
            # 所有智能体使用相同的动作空间大小
            self.action_space.append(Discrete(max_actions))
    
    def _calculate_obs_dim(self) -> int:
        """计算观测维度"""
        # 基础观测维度包括：
        # - 时间步信息: 2 (current_step/max_steps, restored_ratio)
        # - 母线电压信息: n_bus
        # - 设备状态信息: n_switch + n_pv + n_load
        # - 智能体特定信息: 配置的预留维度
        
        n_bus = self.core_env.n_bus
        n_devices = len(self.core_env.faultable_lines) + self.config.n_pv + len(self.core_env.load_info)
        
        return 2 + n_bus + n_devices + self.config.obs_reserved_dim
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray], 
                                                List[List[float]], List[bool], 
                                                List[Dict], List[List[int]]]:
        """
        执行一步动作
        
        Returns:
            local_obs: 局部观测
            global_state: 全局状态
            rewards: 奖励列表
            dones: 结束标志列表
            infos: 信息字典列表
            available_actions: 可用动作
        """
        if self.use_dan:
            return self._step_dan(actions)
        else:
            return self._step_standard(actions)
    
    def _step_standard(self, actions):
        """标准环境步进（用于非DAN算法）"""
        # 执行动作
        obs_dict, state_dict, rewards, done, info = self.core_env.step(actions)
        
        # 转换观测格式
        local_obs = self._convert_observations(obs_dict)
        global_state = self._convert_observations(state_dict)
        
        # 转换奖励格式 - HAPPO兼容的numpy数组 (n_agents, 1)
        rewards_array = np.array([[float(reward)] for reward in rewards], dtype=np.float32)

        # 转换结束标志格式 - HAPPO兼容的numpy布尔数组 (n_agents,)
        dones_array = np.array([bool(done)] * self.n_agents, dtype=bool)

        # 转换信息格式
        infos_list = [info] * self.n_agents

        # 获取可用动作
        available_actions = self.get_avail_actions()

        # 处理时间截断
        if done and self.core_env.current_step >= self.config.max_episode_steps:
            for info_dict in infos_list:
                info_dict["TimeLimit.truncated"] = True
                info_dict["bad_transition"] = True

        return local_obs, global_state, rewards_array, dones_array, infos_list, available_actions
    
    def _step_dan(self, actions):
        """DAN算法增强的环境步进

        使用 dsr_core.step() 推进仿真（动作执行 + 潮流计算），
        但用 DAN 特有的奖励函数替代标准奖励。
        """
        # 执行动作并获取仿真结果（利用 core 的返回值避免重复计算）
        obs_result, _, _, _, _ = self.dsr_core.step(actions)
        self.current_step += 1

        # 使用 core.step() 返回的观测（避免冗余调用 _get_observations）
        observations = obs_result
        
        # 计算奖励
        rewards = []
        infos = []
        for agent_id in range(self.n_agents):
            reward, info = self._calculate_dan_reward(agent_id)
            rewards.append(reward)
            infos.append(info)
            
        # 检查终止条件
        should_terminate, termination_reason = self._check_termination_conditions()

        # 添加终止原因到info
        if should_terminate:
            for info in infos:
                info['termination_reason'] = termination_reason

        # 转换为标准格式
        local_obs = self._convert_observations(observations)
        global_state = self._convert_observations(observations)

        # HAPPO兼容的numpy数组格式
        rewards_array = np.array([[float(reward)] for reward in rewards], dtype=np.float32)
        dones_array = np.array([bool(should_terminate)] * self.n_agents, dtype=bool)

        available_actions = self.get_avail_actions()

        return local_obs, global_state, rewards_array, dones_array, infos, available_actions
    
    def _calculate_dan_reward(self, agent_id):
        """计算DAN算法的优化奖励函数"""
        info = {}
        
        # 1. 负载恢复奖励
        restored_load_ratio = self.dsr_core._get_restored_load_ratio()
        restore_reward = self.reward_restore * restored_load_ratio
        info['restore_reward'] = restore_reward
        
        # 2. 电压违规惩罚
        voltage_violations = self.dsr_core._get_voltage_violations()
        voltage_penalty = -self.reward_voltage * voltage_violations
        info['voltage_penalty'] = voltage_penalty
        info['voltage_violations'] = voltage_violations
        
        # 3. 优化的过载惩罚
        overload_penalty, overload_info = self._calculate_overload_penalty()
        info.update(overload_info)
        
        # 4. 完成奖励
        done_reward = 0
        if self.dsr_core._check_restoration_complete():
            done_reward = self.reward_done
        info['done_reward'] = done_reward
        
        # 总奖励
        total_reward = restore_reward + voltage_penalty + overload_penalty + done_reward
        info['total_reward'] = total_reward
        
        return total_reward, info
    
    def _calculate_overload_penalty(self):
        """计算优化的过载惩罚"""
        # 获取线路过载数量
        overload_count = self.dsr_core._get_line_overloads()
        
        # 获取过载详情
        overload_details = []
        if hasattr(self.dsr_core, 'overload_details'):
            overload_details = self.dsr_core.overload_details
        
        info = {
            'overload_penalty': 0,
            'overload_count': overload_count,
            'severe_overload_count': 0,
            'max_overload_ratio': 0,
            'overload_details': overload_details
        }
        
        if overload_count == 0:
            return 0, info
            
        total_penalty = 0
        severe_overload_count = 0
        max_overload_ratio = 0
        
        for detail in overload_details:
            current = detail['current']
            rating = detail['rating']
            ratio = detail['ratio']
            
            max_overload_ratio = max(max_overload_ratio, ratio)
            
            # 检查异常过载电流
            if current > self.max_overload_current:
                # 异常过载，给予极大惩罚
                total_penalty -= 100.0
                severe_overload_count += 1
                continue
                
            if self.use_progressive_penalty:
                # 渐进式惩罚
                penalty_weight = self._get_progressive_penalty_weight(ratio)
                # Apply exponential penalty for severe overloads
                severity_multiplier = min(ratio, 10.0)  # Cap at 10x
                total_penalty -= penalty_weight * severity_multiplier
            else:
                # 基础过载惩罚 (scaled by severity)
                base_penalty = self.reward_overload
                if ratio > 1.0:
                    # Exponential scaling for overloads
                    severity_factor = min(ratio ** 2, 100.0)  # Cap at 100x
                    base_penalty *= severity_factor
                total_penalty -= base_penalty
                
            # 严重过载检查 with exponential scaling
            if ratio >= self.severe_overload_threshold:
                severe_overload_count += 1
                severe_factor = min((ratio / self.severe_overload_threshold) ** 3, 1000.0)  # Cap at 1000x
                total_penalty -= self.reward_severe_overload * severe_factor
                
        info['overload_penalty'] = total_penalty
        info['severe_overload_count'] = severe_overload_count
        info['max_overload_ratio'] = max_overload_ratio
        
        return total_penalty, info
    
    def _get_progressive_penalty_weight(self, overload_ratio):
        """根据过载比例获取渐进式惩罚权重"""
        for i, threshold in enumerate(self.overload_penalty_levels):
            if overload_ratio <= threshold:
                return self.overload_penalty_weights[i]
        # 超过最高等级
        return self.overload_penalty_weights[-1]
    
    def _check_termination_conditions(self):
        """检查终止条件"""
        # 检查原有终止条件
        if self.dsr_core._check_restoration_complete():
            return True, "restoration_complete"
            
        if self.current_step >= self.dsr_core.config.max_episode_steps:
            return True, "max_steps_reached"
            
        # 检查严重过载终止条件
        if self.terminate_on_severe_overload:
            # 获取过载详情
            self.dsr_core._get_line_overloads()  # 调用以更新overload_details
            if hasattr(self.dsr_core, 'overload_details'):
                for detail in self.dsr_core.overload_details:
                    current = detail['current']
                    ratio = detail['ratio']

                    # 异常过载电流终止 - 在循环内检查每条线路
                    if current > self.max_overload_current:
                        return True, "abnormal_overload_current"

                    # 严重过载终止 - 在循环内检查每条线路
                    if ratio >= self.severe_overload_threshold:
                        return True, "severe_overload"

        return False, None
    
    def reset(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[int]]]:
        """
        重置环境
        
        Returns:
            observations: 初始观测
            states: 初始状态
            available_actions: 可用动作
        """
        # 重置核心环境
        obs_dict, state_dict = self.core_env.reset()
        
        # 重置步数
        self.current_step = 0
        
        # 转换观测格式
        observations = self._convert_observations(obs_dict)
        states = self._convert_observations(state_dict)
        
        # 获取可用动作
        available_actions = self.get_avail_actions()
        
        return observations, states, available_actions
    
    def _convert_observations(self, obs_dict: Dict[str, Any]) -> List[np.ndarray]:
        """将观测字典转换为智能体观测列表"""
        obs_list = []
        
        for i in range(self.n_agents):
            # 构造智能体i的观测
            obs = self._build_agent_observation(i, obs_dict)
            obs_list.append(obs)
        
        return obs_list
    
    def _build_agent_observation(self, agent_id: int, obs_dict: Dict[str, Any]) -> np.ndarray:
        """构建单个智能体的观测"""
        obs_components = []
        
        # 基础时间信息
        obs_components.extend([
            obs_dict['current_step'] / obs_dict['max_steps'],  # 归一化时间步
            len(obs_dict['energized_buses']) / self.core_env.n_bus,  # 通电率
        ])
        
        # 母线电压信息（简化：每个母线取最小相电压）
        for bus_name in self.core_env.all_bus_names:
            if bus_name in obs_dict['bus_voltages']:
                voltages = obs_dict['bus_voltages'][bus_name]
                if voltages:
                    min_voltage = min(voltages)
                    # 检查并处理无效值
                    if np.isnan(min_voltage) or np.isinf(min_voltage):
                        min_voltage = self.config.default_voltage
                    # 限制电压范围在合理区间内
                    min_voltage = np.clip(min_voltage, 0.5, 1.5)
                else:
                    min_voltage = self.config.default_voltage
                obs_components.append(min_voltage)
            else:
                obs_components.append(self.config.default_voltage)  # 配置的默认电压
        
        # 根据智能体类型添加特定观测
        agent_type = self.core_env.agent_types[agent_id]
        
        if agent_type == 'switch':
            # 开关智能体：线路状态
            for line_name in self.core_env.faultable_lines:
                if line_name in obs_dict['line_states']:
                    obs_components.append(float(obs_dict['line_states'][line_name]))
                else:
                    obs_components.append(0.0)
        
        elif agent_type == 'pv':
            # PV智能体：PV状态和附近负荷状态
            agent_idx = self.core_env.agent_indices[agent_id]
            if agent_idx < len(self.core_env.pv_agents):
                pv_agent = self.core_env.pv_agents[agent_idx]
                obs_components.extend([
                    pv_agent['current_power'] / pv_agent['max_power'],  # 当前功率比
                    float(pv_agent['bus'] in obs_dict['energized_buses']),  # 母线通电状态
                ])
        
        elif agent_type == 'load':
            # 负荷智能体：负荷状态和优先级信息
            agent_idx = self.core_env.agent_indices[agent_id]
            if agent_idx < len(self.core_env.load_agents):
                load_agent = self.core_env.load_agents[agent_idx]
                
                # 处理聚合负荷智能体
                managed_loads = load_agent.get('managed_loads', [])
                if managed_loads:
                    # 聚合负荷智能体的观测
                    # 1. 管理的负荷总体状态
                    enabled_count = sum(1 for load in managed_loads 
                                      if obs_dict['load_states'].get(load, False))
                    total_count = len(managed_loads)
                    restoration_ratio = enabled_count / max(total_count, 1)
                    
                    # 2. 平均优先级和总功率信息
                    avg_priority = load_agent['priority'] / self.config.max_priority_level
                    
                    # 3. 可恢复负荷比例（母线通电的负荷）
                    restorable_count = sum(1 for load in managed_loads
                                         if self.core_env.load_info[load]['bus'] in obs_dict['energized_buses'])
                    restorable_ratio = restorable_count / max(total_count, 1)
                    
                    obs_components.extend([
                        restoration_ratio,  # 已恢复负荷比例
                        avg_priority,      # 平均优先级
                        restorable_ratio,  # 可恢复负荷比例
                        float(total_count),  # 管理的负荷数量（归一化）
                    ])
                else:
                    # 单个负荷智能体的观测
                    load_name = load_agent['load_name']
                    obs_components.extend([
                        float(obs_dict['load_states'].get(load_name, False)),  # 负荷状态
                        load_agent['priority'] / self.config.max_priority_level,  # 归一化优先级
                        float(load_agent['bus'] in obs_dict['energized_buses']),  # 母线通电状态
                        1.0,  # 管理1个负荷
                    ])
        
        # 统一的维度处理：确保观测维度正确
        target_dim = self._calculate_obs_dim()
        
        # 调试信息：检查obs_components的结构
        try:
            # 展平嵌套列表
            flattened_obs = []
            for item in obs_components:
                if isinstance(item, (list, np.ndarray)):
                    if isinstance(item, np.ndarray):
                        flattened_obs.extend(item.flatten())
                    else:
                        # 递归展平嵌套列表
                        def flatten_list(lst):
                            result = []
                            for element in lst:
                                if isinstance(element, (list, np.ndarray)):
                                    if isinstance(element, np.ndarray):
                                        result.extend(element.flatten())
                                    else:
                                        result.extend(flatten_list(element))
                                else:
                                    result.append(element)
                            return result
                        flattened_obs.extend(flatten_list(item))
                else:
                    flattened_obs.append(item)
            
            # 确保所有元素都是数值类型
            flattened_obs = [float(x) if not np.isnan(float(x)) else 0.0 for x in flattened_obs]
            
            # 调整到目标维度
            if len(flattened_obs) > target_dim:
                # 截断
                flattened_obs = flattened_obs[:target_dim]
            elif len(flattened_obs) < target_dim:
                # 填充
                padding_needed = target_dim - len(flattened_obs)
                flattened_obs.extend([0.0] * padding_needed)
            
            # 转换为numpy数组
            obs_array = np.array(flattened_obs, dtype=np.float32)
            
        except Exception as e:
            print(f"错误: 智能体 {agent_id} 观测构建失败: {e}")
            # 创建默认观测
            obs_array = np.zeros(target_dim, dtype=np.float32)
        
        # 检查并处理NaN和无穷大值
        if np.any(np.isnan(obs_array)) or np.any(np.isinf(obs_array)):
            # 将NaN和无穷大值替换为0
            obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs_array
    
    def get_avail_actions(self) -> List[List[int]]:
        """获取所有智能体的可用动作"""
        if self.use_dan and self.use_enhanced_action_mask:
            # 使用增强的动作掩码
            avail_actions = []
            for agent_id in range(self.n_agents):
                mask = self._get_enhanced_action_mask(agent_id)
                avail_actions.append(mask.tolist())
            return avail_actions
        else:
            # 使用标准动作掩码
            avail_actions = self.core_env.get_available_actions()
            # 确保返回numpy兼容的格式
            return np.array(avail_actions, dtype=object).tolist()
    
    def _get_enhanced_action_mask(self, agent_id):
        """Get enhanced action mask based on safety predictions"""
        # Get basic action mask
        basic_mask = self._get_action_mask(agent_id)
        
        if not self.use_enhanced_action_mask:
            return basic_mask
            
        # Enhanced safety-based masking
        enhanced_mask = basic_mask.copy()
        
        # Get current system state
        try:
            # Check current overload status
            overload_count = self.dsr_core._get_line_overloads()
            
            # If system is already severely overloaded, be more conservative
            if overload_count > 0 and hasattr(self.dsr_core, 'overload_details'):
                overload_details = self.dsr_core.overload_details
                if len(overload_details) > 0:
                    max_overload_ratio = max([detail['ratio'] for detail in overload_details])
                    if max_overload_ratio >= self.severe_overload_threshold:
                        # Mask actions that might worsen the situation
                        # For switch agents, prefer opening switches to reduce load
                        if self._would_action_increase_overload(agent_id, enhanced_mask):
                            # Apply conservative masking
                            pass
                        
        except Exception as e:
            # If enhanced masking fails, fall back to basic mask
            return basic_mask
            
        # Ensure at least one action is available
        if not any(enhanced_mask):
            # If all actions are masked, allow the safest action (usually 'do nothing')
            enhanced_mask[0] = True  # Assuming action 0 is 'do nothing'
            
        return enhanced_mask
    
    def _get_action_mask(self, agent_id):
        """Get basic action mask"""
        # Get available actions from dsr_core
        avail_actions = self.dsr_core.get_available_actions()
        if agent_id < len(avail_actions):
            return np.array(avail_actions[agent_id], dtype=bool)
        else:
            # Return default mask
            return np.ones(self.action_space[agent_id].n, dtype=bool)
    
    def _would_action_increase_overload(self, agent_id, action_mask):
        """Predict if actions would increase system overloads"""
        # Simplified heuristic - can be improved with more sophisticated prediction
        try:
            # Basic safety check based on current system state
            overload_count = self.dsr_core._get_line_overloads()
            return overload_count > 0
        except:
            pass
        return False
    
    # DAN算法专用方法
    def get_neighbor_observations(self, agent_id):
        """获取邻居智能体的观测（用于DAN）"""
        if not self.use_dan:
            raise NotImplementedError("get_neighbor_observations仅在use_dan=True时可用")
        
        max_neighbors = self.args.get('max_neighbors', 5) if self.args else 5
        
        # NOTE: 根据论文，邻居定义为同一微电网内的其他智能体
        # 在DSR环境中，我们根据电气连接关系确定邻居
        
        # 获取当前智能体的微电网ID
        if hasattr(self, 'get_agent_microgrid'):
            agent_mg = self.get_agent_microgrid(agent_id)
            neighbor_ids = []
            
            # 找到同一微电网内的其他智能体
            for i in range(self.n_agents):
                if i != agent_id and self.get_agent_microgrid(i) == agent_mg:
                    neighbor_ids.append(i)
        else:
            # 如果没有微电网信息，使用距离最近的智能体作为邻居
            # 这是一个简化实现，实际应根据电网拓扑确定
            neighbor_ids = [i for i in range(self.n_agents) if i != agent_id]
            # 限制邻居数量
            if len(neighbor_ids) > max_neighbors:
                # TODO: 根据电气距离或其他度量选择最相关的邻居
                neighbor_ids = neighbor_ids[:max_neighbors]
        
        # 准备邻居观测和掩码
        neighbor_obs = []
        agent_mask = []
        
        # 填充邻居观测
        for i in range(max_neighbors):
            if i < len(neighbor_ids):
                obs = self._get_observation(neighbor_ids[i])
                neighbor_obs.append(obs)
                agent_mask.append(1.0)
            else:
                # 填充零观测
                obs = np.zeros(self.observation_space[0].shape[0])
                neighbor_obs.append(obs)
                agent_mask.append(0.0)
        
        return np.array(neighbor_obs), np.array(agent_mask)
    
    def get_all_neighbor_observations(self):
        """获取所有智能体的邻居观测"""
        if not self.use_dan:
            raise NotImplementedError("get_all_neighbor_observations仅在use_dan=True时可用")
        
        all_neighbor_obs = []
        all_agent_masks = []
        
        for agent_id in range(self.n_agents):
            neighbor_obs, agent_mask = self.get_neighbor_observations(agent_id)
            all_neighbor_obs.append(neighbor_obs)
            all_agent_masks.append(agent_mask)
        
        return np.array(all_neighbor_obs), np.array(all_agent_masks)
    
    def _get_observation(self, agent_id):
        """获取单个智能体的观测"""
        all_obs = self.dsr_core._get_observations()
        # 根据智能体类型构建观测
        agent_type = self.agent_types[agent_id] if agent_id < len(self.agent_types) else 'load'
        
        # 构建基础观测向量
        obs = []
        
        # 添加时间步信息
        obs.append(self.current_step / self.dsr_core.config.max_episode_steps)
        
        # 添加全局信息
        obs.append(len(all_obs['energized_buses']) / self.dsr_core.n_bus if self.dsr_core.n_bus > 0 else 0)
        obs.append(self.dsr_core._get_restored_load_ratio())
        
        # 添加智能体特定信息
        if agent_type == 'switch':
            # 开关智能体：线路状态信息
            for line_name in self.dsr_core.faultable_lines[:10]:  # 限制数量
                if line_name in all_obs['line_states']:
                    obs.append(float(all_obs['line_states'][line_name]))
                else:
                    obs.append(0.0)
        elif agent_type == 'pv':
            # PV智能体：当前功率输出
            if agent_id - 1 < len(self.dsr_core.pv_agents):
                pv_agent = self.dsr_core.pv_agents[agent_id - 1]
                obs.append(pv_agent['current_power'] / pv_agent['max_power'])
            else:
                obs.append(0.0)
        elif agent_type == 'load':
            # 负荷智能体：母线电压和负荷状态
            if agent_id < len(self.agent_bus_mapping):
                bus = self.agent_bus_mapping.get(agent_id, '')
                if bus in all_obs['bus_voltages']:
                    voltage = all_obs['bus_voltages'][bus][0]
                    obs.append(voltage)
                else:
                    obs.append(1.0)
            else:
                obs.append(1.0)
        
        # 填充到固定维度
        obs_dim = getattr(self.dsr_core.config, 'obs_reserved_dim', 10)
        while len(obs) < obs_dim:
            obs.append(0.0)
        
        return np.array(obs[:obs_dim])
    
    def get_avail_agent_actions(self, agent_id: int) -> List[int]:
        """获取单个智能体的可用动作"""
        avail_actions = self.get_avail_actions()
        return avail_actions[agent_id]
    
    def render(self):
        """渲染环境（预留接口）"""
        pass
    
    def close(self):
        """关闭环境"""
        self.core_env.close()
        print("DSR环境已关闭")
    
    def seed(self, seed: int):
        """设置随机种子"""
        self.core_env.seed(seed)
    
    @property
    def agents(self) -> List[int]:
        """智能体ID列表"""
        return list(range(self.n_agents))


# 兼容性别名
DSREnvDAN = DSREnv  # 向后兼容

# 环境创建函数
def make_dsr_env(config_or_args, rank: Optional[int] = None) -> DSREnv:
    """创建DSR环境"""
    return DSREnv(config_or_args, rank)