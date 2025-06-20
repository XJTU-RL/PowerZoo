"""
@File    : dsr_env_optimized.py
@Time    : 2024/05/24
@Author  : Xiaodong Zheng
@Description: Optimized DSR environment with improved reward function and safety constraints.
"""

import numpy as np
import torch
from envs.dsr.dsr_env import DSREnv
from envs.dsr.core.dsr_core import DSRCoreEnv

class DSREnvOptimized(DSREnv):
    """优化的DSR环境，改进奖励函数和安全约束"""
    
    def __init__(self, args):
        super(DSREnvOptimized, self).__init__(args)
        
        # 优化的奖励函数权重
        self.reward_restore = getattr(args, 'reward_restore', 20.0)
        self.reward_voltage = getattr(args, 'reward_voltage', 2.0)  # 增加电压惩罚权重
        self.reward_overload = getattr(args, 'reward_overload', 8.0)  # 大幅增加过载惩罚权重
        self.reward_severe_overload = getattr(args, 'reward_severe_overload', 50.0)  # 严重过载惩罚
        self.reward_done = getattr(args, 'reward_done', -5.0)
        
        # 安全约束参数
        self.severe_overload_threshold = getattr(args, 'severe_overload_threshold', 2.0)  # 严重过载阈值（倍数）
        self.max_overload_current = getattr(args, 'max_overload_current', 1000.0)  # 最大允许过载电流（安培）
        self.terminate_on_severe_overload = getattr(args, 'terminate_on_severe_overload', True)
        
        # 渐进式惩罚参数
        self.use_progressive_penalty = getattr(args, 'use_progressive_penalty', True)
        self.overload_penalty_levels = getattr(args, 'overload_penalty_levels', [1.2, 1.5, 2.0])  # 过载等级阈值
        self.overload_penalty_weights = getattr(args, 'overload_penalty_weights', [1.0, 3.0, 8.0, 20.0])  # 对应惩罚权重
        
        # 动作掩码增强
        self.use_enhanced_action_mask = getattr(args, 'use_enhanced_action_mask', True)
        self.action_mask_safety_margin = getattr(args, 'action_mask_safety_margin', 0.1)  # 安全边际
        
    def _calculate_reward(self, agent_id):
        """计算优化的奖励函数
        Args:
            agent_id: 智能体ID
        Returns:
            reward: 奖励值
            info: 奖励信息字典
        """
        info = {}
        
        # 1. 负载恢复奖励
        restored_load_ratio = self.dsr_core.get_restored_load_ratio()
        restore_reward = self.reward_restore * restored_load_ratio
        info['restore_reward'] = restore_reward
        
        # 2. 电压违规惩罚
        voltage_violations = self.dsr_core.get_voltage_violations()
        voltage_penalty = -self.reward_voltage * voltage_violations
        info['voltage_penalty'] = voltage_penalty
        info['voltage_violations'] = voltage_violations
        
        # 3. 优化的过载惩罚
        overload_penalty, overload_info = self._calculate_overload_penalty()
        info.update(overload_info)
        
        # 4. 完成奖励
        done_reward = 0
        if self.dsr_core.check_restoration_complete():
            done_reward = self.reward_done
        info['done_reward'] = done_reward
        
        # 总奖励
        total_reward = restore_reward + voltage_penalty + overload_penalty + done_reward
        info['total_reward'] = total_reward
        
        return total_reward, info
        
    def _calculate_overload_penalty(self):
        """计算优化的过载惩罚
        Returns:
            penalty: 惩罚值
            info: 过载信息字典
        """
        overload_info = self.dsr_core.get_line_overloads()
        overload_count = overload_info['count']
        overload_details = overload_info['details']
        
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
        """根据过载比例获取渐进式惩罚权重
        Args:
            overload_ratio: 过载比例
        Returns:
            weight: 惩罚权重
        """
        for i, threshold in enumerate(self.overload_penalty_levels):
            if overload_ratio <= threshold:
                return self.overload_penalty_weights[i]
        # 超过最高等级
        return self.overload_penalty_weights[-1]
        
    def _check_termination_conditions(self):
        """检查终止条件
        Returns:
            should_terminate: 是否应该终止
            termination_reason: 终止原因
        """
        # 检查原有终止条件
        if self.dsr_core.check_restoration_complete():
            return True, "restoration_complete"
            
        if self.current_step >= self.max_episode_steps:
            return True, "max_steps_reached"
            
        # 检查严重过载终止条件
        if self.terminate_on_severe_overload:
            overload_info = self.dsr_core.get_line_overloads()
            for detail in overload_info['details']:
                current = detail['current']
                ratio = detail['ratio']
                
                # 异常过载电流终止
                if current > self.max_overload_current:
                    return True, "abnormal_overload_current"
                    
                # 严重过载终止
                if ratio >= self.severe_overload_threshold:
                    return True, "severe_overload"
                    
        return False, None
        
    def _get_enhanced_action_mask(self, agent_id):
        """Get enhanced action mask based on safety predictions
        Args:
            agent_id: Agent ID
        Returns:
            action_mask: Enhanced action mask
        """
        # Get basic action mask
        basic_mask = self._get_action_mask(agent_id)
        
        if not self.use_enhanced_action_mask:
            return basic_mask
            
        # Enhanced safety-based masking
        enhanced_mask = basic_mask.copy()
        
        # Get current system state
        try:
            # Check current overload status
            overload_info = self.dsr_core.get_line_overloads()
            
            # If system is already severely overloaded, be more conservative
            if len(overload_info['details']) > 0:
                max_overload_ratio = max([detail['ratio'] for detail in overload_info['details']])
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
        
    def _would_action_increase_overload(self, agent_id, action_mask):
        """Predict if actions would increase system overloads
        Args:
            agent_id: Agent ID
            action_mask: Current action mask
        Returns:
            bool: True if actions would likely increase overloads
        """
        # Simplified heuristic - can be improved with more sophisticated prediction
        try:
            # Basic safety check based on current system state
            overload_info = self.dsr_core.get_line_overloads()
            return overload_info['count'] > 0
        except:
            pass
        return False
        
    def step(self, actions):
        """环境步进
        Args:
            actions: 动作列表
        Returns:
            observations: 观测
            rewards: 奖励
            dones: 完成标志
            infos: 信息字典
        """
        # 执行动作
        self.dsr_core.step(actions)
        self.current_step += 1
        
        # 获取观测
        observations = self._get_observations()
        
        # 计算奖励
        rewards = []
        infos = []
        for agent_id in range(self.num_agents):
            reward, info = self._calculate_reward(agent_id)
            rewards.append(reward)
            infos.append(info)
            
        # 检查终止条件
        should_terminate, termination_reason = self._check_termination_conditions()
        dones = [should_terminate] * self.num_agents
        
        # 添加终止原因到info
        if should_terminate:
            for info in infos:
                info['termination_reason'] = termination_reason
                
        return observations, rewards, dones, infos
        
    def get_neighbor_observations(self, agent_id):
        """获取邻居智能体的观测（用于DAN）
        Args:
            agent_id: 当前智能体ID
        Returns:
            neighbor_obs: 邻居观测列表 [max_neighbors, obs_dim]
            agent_mask: 智能体掩码 [max_neighbors]
        """
        max_neighbors = getattr(self.args, 'max_neighbors', 5)
        
        # NOTE: 根据论文，邻居定义为同一微电网内的其他智能体
        # 在DSR环境中，我们根据电气连接关系确定邻居
        
        # 获取当前智能体的微电网ID
        if hasattr(self, 'get_agent_microgrid'):
            agent_mg = self.get_agent_microgrid(agent_id)
            neighbor_ids = []
            
            # 找到同一微电网内的其他智能体
            for i in range(self.num_agents):
                if i != agent_id and self.get_agent_microgrid(i) == agent_mg:
                    neighbor_ids.append(i)
        else:
            # 如果没有微电网信息，使用距离最近的智能体作为邻居
            # 这是一个简化实现，实际应根据电网拓扑确定
            neighbor_ids = [i for i in range(self.num_agents) if i != agent_id]
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
        """获取所有智能体的邻居观测
        Returns:
            all_neighbor_obs: 所有智能体的邻居观测 [num_agents, max_neighbors, obs_dim]
            all_agent_masks: 所有智能体的掩码 [num_agents, max_neighbors]
        """
        all_neighbor_obs = []
        all_agent_masks = []
        
        for agent_id in range(self.num_agents):
            neighbor_obs, agent_mask = self.get_neighbor_observations(agent_id)
            all_neighbor_obs.append(neighbor_obs)
            all_agent_masks.append(agent_mask)
        
        return np.array(all_neighbor_obs), np.array(all_agent_masks)