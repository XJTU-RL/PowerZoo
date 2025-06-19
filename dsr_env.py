"""
配电网恢复多智能体强化学习环境
基于论文: A multi-agent reinforcement learning method for distribution system restoration
"""

import numpy as np
import gym
from gym import spaces
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import pandapower as pp
import pandapower.networks as pn
from dataclasses import dataclass
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DSRConfig:
    """配电网恢复环境配置"""
    # 网络配置
    network_name: str = "ieee123"  # 电网名称
    max_steps: int = 15  # 最大恢复步数
    
    # 设备数量
    n_dg: int = 7  # 柴油发电机数量
    n_pv: int = 9  # 光伏数量
    n_switch: int = 20  # 开关数量
    n_load_levels: int = 3  # 负荷优先级等级
    
    # 约束参数
    v_min: float = 0.95  # 最小电压标幺值
    v_max: float = 1.05  # 最大电压标幺值
    max_load_per_step: float = 500.0  # 每步最大恢复负荷(kW)
    
    # 奖励权重
    reward_restore: float = 20.0  # 负荷恢复奖励权重
    reward_voltage: float = 1.0  # 电压越限惩罚权重
    reward_overload: float = 1.0  # 线路过载惩罚权重
    reward_done: float = -5.0  # 失败惩罚
    
    # 算法配置
    use_action_mask: bool = True  # 是否使用动作掩码
    use_dynamic_network: bool = True  # 是否使用动态智能体网络


class DSREnv(gym.Env):
    """配电网恢复环境"""
    
    def __init__(self, config: DSRConfig):
        super().__init__()
        self.config = config
        
        # 初始化电网
        self._init_network()
        
        # 初始化智能体
        self._init_agents()
        
        # 定义动作和观测空间
        self._init_spaces()
        
        # 环境状态
        self.current_step = 0
        self.done = False
        
        logger.info(f"DSR环境初始化完成: {self.n_agents}个智能体")
    
    def _init_network(self):
        """初始化配电网络"""
        # 使用pandapower创建IEEE 123节点系统
        # 这里简化处理，实际应该加载完整的网络数据
        self.net = pp.create_empty_network()
        
        # 添加外部电网（故障后断开）
        pp.create_ext_grid(self.net, bus=0, vm_pu=1.0)
        
        # 创建节点
        self.n_bus = 123
        for i in range(self.n_bus):
            pp.create_bus(self.net, vn_kv=4.16, index=i)
        
        # 创建线路（简化）
        self.n_line = 100
        # 实际应该根据IEEE 123拓扑创建
        
        # 创建开关
        self.switches = []
        for i in range(self.config.n_switch):
            # 简化：随机连接开关
            bus_i = np.random.randint(0, self.n_bus)
            bus_j = np.random.randint(0, self.n_bus)
            if bus_i != bus_j:
                sw_idx = pp.create_switch(self.net, bus_i, bus_j, et='b', closed=False)
                self.switches.append(sw_idx)
        
        # 创建DG（黑启动电源）
        self.dg_buses = np.random.choice(range(self.n_bus), self.config.n_dg, replace=False)
        for i, bus in enumerate(self.dg_buses):
            pp.create_gen(self.net, bus, p_mw=0.15, vm_pu=1.0, name=f"DG_{i}")
        
        # 创建PV
        self.pv_buses = np.random.choice(range(self.n_bus), self.config.n_pv, replace=False)
        for i, bus in enumerate(self.pv_buses):
            pp.create_sgen(self.net, bus, p_mw=0.0, name=f"PV_{i}")
        
        # 创建负荷
        self.loads = []
        load_priorities = [1, 2, 3]  # 优先级
        for i in range(85):  # 85个负荷
            bus = np.random.randint(0, self.n_bus)
            priority = np.random.choice(load_priorities, p=[0.15, 0.15, 0.7])
            load_idx = pp.create_load(self.net, bus, p_mw=0.05, name=f"Load_{i}")
            self.loads.append({
                'idx': load_idx,
                'priority': priority,
                'bus': bus,
                'p_mw': 0.05
            })
    
    def _init_agents(self):
        """初始化智能体"""
        # 智能体类型：1个开关智能体 + PV智能体 + 负荷智能体
        self.n_agents = 1 + self.config.n_pv + len(self.loads)
        
        # 智能体映射
        self.agent_types = []
        self.agent_indices = []
        
        # 开关智能体
        self.agent_types.append('switch')
        self.agent_indices.append(None)
        
        # PV智能体
        for i in range(self.config.n_pv):
            self.agent_types.append('pv')
            self.agent_indices.append(i)
        
        # 负荷智能体
        for i in range(len(self.loads)):
            self.agent_types.append('load')
            self.agent_indices.append(i)
    
    def _init_spaces(self):
        """定义动作和观测空间"""
        # 观测空间（简化版本）
        obs_dim = 100  # 包含电压、负荷、发电等信息
        self.observation_space = [spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        ) for _ in range(self.n_agents)]
        
        # 共享观测空间
        self.share_observation_space = self.observation_space.copy()
        
        # 动作空间
        self.action_space = []
        
        # 开关智能体：选择操作哪个开关（离散）
        self.action_space.append(spaces.Discrete(self.config.n_switch + 1))
        
        # PV智能体：输出功率等级（离散化为11个等级）
        for _ in range(self.config.n_pv):
            self.action_space.append(spaces.Discrete(11))
        
        # 负荷智能体：是否恢复（二元）
        for _ in range(len(self.loads)):
            self.action_space.append(spaces.Discrete(2))
    
    def reset(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[int]]]:
        """重置环境"""
        self.current_step = 0
        self.done = False
        
        # 重置电网状态
        # 断开所有开关
        for sw_idx in self.switches:
            self.net.switch.at[sw_idx, 'closed'] = False
        
        # 断开外部电网
        self.net.ext_grid.in_service = False
        
        # 重置PV输出
        self.net.sgen.p_mw = 0.0
        
        # 断开所有负荷
        self.net.load.in_service = False
        
        # 生成随机故障
        self._generate_random_faults()
        
        # 获取初始观测
        obs = self._get_obs()
        state = self._get_state()
        avail_actions = self._get_available_actions()
        
        return obs, state, avail_actions
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray], 
                                                 List[List[float]], List[bool], 
                                                 List[Dict], List[List[int]]]:
        """执行动作"""
        self.current_step += 1
        
        # 执行动作
        self._execute_actions(actions)
        
        # 运行潮流计算
        try:
            pp.runpp(self.net, numba=False)
            converged = self.net.converged
        except:
            converged = False
        
        # 计算奖励
        reward = self._calculate_reward(converged)
        
        # 检查终止条件
        if not converged or self.current_step >= self.config.max_steps:
            self.done = True
        
        # 获取新观测
        obs = self._get_obs()
        state = self._get_state()
        avail_actions = self._get_available_actions()
        
        # 信息字典
        info = [{
            'converged': converged,
            'restored_load': self._get_restored_load_ratio()
        }]
        
        return (obs, state, [[reward]], [self.done], info, avail_actions)
    
    def _execute_actions(self, actions: List[int]):
        """执行智能体动作"""
        agent_idx = 0
        
        # 开关智能体动作
        switch_action = actions[agent_idx]
        if switch_action < self.config.n_switch:
            # 切换开关状态
            sw_idx = self.switches[switch_action]
            self.net.switch.at[sw_idx, 'closed'] = not self.net.switch.at[sw_idx, 'closed']
        agent_idx += 1
        
        # PV智能体动作
        for i in range(self.config.n_pv):
            pv_action = actions[agent_idx]
            # 设置PV输出（0-1之间，11个等级）
            output_ratio = pv_action / 10.0
            max_pv_output = 0.15  # 150kW
            self.net.sgen.at[i, 'p_mw'] = output_ratio * max_pv_output
            agent_idx += 1
        
        # 负荷智能体动作
        for i in range(len(self.loads)):
            load_action = actions[agent_idx]
            if load_action == 1:  # 恢复负荷
                load_idx = self.loads[i]['idx']
                # 检查负荷连接的节点是否已通电
                bus = self.loads[i]['bus']
                if self._is_bus_energized(bus):
                    self.net.load.at[load_idx, 'in_service'] = True
            agent_idx += 1
    
    def _get_obs(self) -> List[np.ndarray]:
        """获取观测"""
        obs_list = []
        
        for i in range(self.n_agents):
            # 简化的观测（实际应该根据智能体类型定制）
            obs = np.zeros(100)
            
            # 添加一些基本信息
            obs[0] = self.current_step / self.config.max_steps
            obs[1] = self._get_restored_load_ratio()
            
            # 根据智能体类型添加特定观测
            if self.agent_types[i] == 'switch':
                # 添加开关状态
                obs[10:10+self.config.n_switch] = self.net.switch.closed.values.astype(float)
            elif self.agent_types[i] == 'pv':
                # 添加PV相关信息
                pv_idx = self.agent_indices[i]
                obs[30] = self.net.sgen.at[pv_idx, 'p_mw']
            elif self.agent_types[i] == 'load':
                # 添加负荷相关信息
                load_idx = self.agent_indices[i]
                obs[40] = float(self.net.load.at[self.loads[load_idx]['idx'], 'in_service'])
            
            obs_list.append(obs)
        
        return obs_list
    
    def _get_state(self) -> List[np.ndarray]:
        """获取全局状态"""
        # 简化处理：与观测相同
        return self._get_obs()
    
    def _get_available_actions(self) -> List[List[int]]:
        """获取可用动作（动作掩码）"""
        avail_actions = []
        
        for i in range(self.n_agents):
            if self.agent_types[i] == 'switch':
                # 开关智能体：检查哪些开关可以操作
                avail = np.ones(self.config.n_switch + 1)
                # 可以添加约束检查
                avail_actions.append(avail.tolist())
            
            elif self.agent_types[i] == 'pv':
                # PV智能体：所有输出等级都可用
                avail_actions.append([1] * 11)
            
            elif self.agent_types[i] == 'load':
                # 负荷智能体：检查是否可以恢复
                avail = [1, 1]  # [不恢复, 恢复]
                load_idx = self.agent_indices[i]
                bus = self.loads[load_idx]['bus']
                
                # 如果节点未通电，不能恢复
                if not self._is_bus_energized(bus):
                    avail[1] = 0
                
                # 如果已经恢复，不能再恢复
                if self.net.load.at[self.loads[load_idx]['idx'], 'in_service']:
                    avail[1] = 0
                
                avail_actions.append(avail)
        
        return avail_actions
    
    def _calculate_reward(self, converged: bool) -> float:
        """计算奖励"""
        if not converged:
            return self.config.reward_done
        
        # 负荷恢复奖励
        restored_ratio = self._get_restored_load_ratio()
        r_restore = self.config.reward_restore * restored_ratio
        
        # 电压越限惩罚
        v_violations = np.sum((self.net.res_bus.vm_pu < self.config.v_min) | 
                             (self.net.res_bus.vm_pu > self.config.v_max))
        r_voltage = -self.config.reward_voltage * v_violations
        
        # 线路过载惩罚
        line_overloads = np.sum(self.net.res_line.loading_percent > 100)
        r_overload = -self.config.reward_overload * line_overloads
        
        return r_restore + r_voltage + r_overload
    
    def _get_restored_load_ratio(self) -> float:
        """计算加权负荷恢复率"""
        total_weighted_load = 0
        restored_weighted_load = 0
        
        for load in self.loads:
            weight = load['priority']
            power = load['p_mw']
            total_weighted_load += weight * power
            
            if self.net.load.at[load['idx'], 'in_service']:
                restored_weighted_load += weight * power
        
        if total_weighted_load == 0:
            return 0.0
        
        return restored_weighted_load / total_weighted_load
    
    def _is_bus_energized(self, bus: int) -> bool:
        """检查节点是否通电"""
        # 使用图论方法检查节点是否与DG连通
        G = nx.Graph()
        
        # 添加闭合开关和线路作为边
        for idx, row in self.net.switch.iterrows():
            if row['closed']:
                G.add_edge(row['bus'], row['element'])
        
        # 检查是否与任意DG连通
        for dg_bus in self.dg_buses:
            if nx.has_path(G, bus, dg_bus):
                return True
        
        return False
    
    def _generate_random_faults(self):
        """生成随机故障"""
        # 随机选择3-4条线路故障
        n_faults = np.random.randint(3, 5)
        # 实际实现需要处理线路故障
        pass
    
    def seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
    
    @property
    def agents(self) -> List[int]:
        """智能体ID列表"""
        return list(range(self.n_agents))
    
    def close(self):
        """关闭环境"""
        pass


class DSREnvWithDAN(DSREnv):
    """带动态智能体网络的配电网恢复环境"""
    
    def __init__(self, config: DSRConfig):
        super().__init__(config)
        self.microgrid_mapping = {}  # 智能体到微电网的映射
    
    def _get_obs(self) -> List[np.ndarray]:
        """获取观测（支持动态邻居）"""
        # 首先更新微电网映射
        self._update_microgrid_mapping()
        
        obs_list = []
        
        for i in range(self.n_agents):
            # 基础观测
            obs_base = super()._get_obs()[i]
            
            # 获取同一微电网内的邻居智能体
            neighbors = self._get_neighbors(i)
            
            # 动态聚合邻居信息（这里简化处理）
            neighbor_features = []
            for neighbor_id in neighbors:
                if neighbor_id != i:
                    neighbor_obs = super()._get_obs()[neighbor_id]
                    neighbor_features.append(neighbor_obs[:10])  # 取前10维特征
            
            # 使用注意力机制聚合（简化版本）
            if neighbor_features:
                aggregated = np.mean(neighbor_features, axis=0)
                obs = np.concatenate([obs_base, aggregated])
            else:
                obs = np.concatenate([obs_base, np.zeros(10)])
            
            obs_list.append(obs)
        
        return obs_list
    
    def _update_microgrid_mapping(self):
        """更新微电网映射"""
        # 使用图论方法识别微电网
        G = nx.Graph()
        
        # 添加所有节点
        G.add_nodes_from(range(self.n_bus))
        
        # 添加闭合开关和通电线路
        for idx, row in self.net.switch.iterrows():
            if row['closed']:
                G.add_edge(row['bus'], row['element'])
        
        # 识别连通分量（微电网）
        microgrids = list(nx.connected_components(G))
        
        # 更新智能体到微电网的映射
        self.microgrid_mapping = {}
        for mg_id, mg_buses in enumerate(microgrids):
            # 检查微电网是否有DG
            has_dg = any(bus in self.dg_buses for bus in mg_buses)
            if has_dg:
                # 将该微电网内的所有智能体映射到同一ID
                for agent_id in range(self.n_agents):
                    if self._is_agent_in_microgrid(agent_id, mg_buses):
                        self.microgrid_mapping[agent_id] = mg_id
    
    def _is_agent_in_microgrid(self, agent_id: int, mg_buses: set) -> bool:
        """检查智能体是否在指定微电网内"""
        if self.agent_types[agent_id] == 'switch':
            return True  # 开关智能体观察所有微电网
        elif self.agent_types[agent_id] == 'pv':
            pv_idx = self.agent_indices[agent_id]
            return self.pv_buses[pv_idx] in mg_buses
        elif self.agent_types[agent_id] == 'load':
            load_idx = self.agent_indices[agent_id]
            return self.loads[load_idx]['bus'] in mg_buses
        return False
    
    def _get_neighbors(self, agent_id: int) -> List[int]:
        """获取同一微电网内的邻居智能体"""
        if agent_id not in self.microgrid_mapping:
            return []
        
        mg_id = self.microgrid_mapping[agent_id]
        neighbors = []
        
        for other_id, other_mg_id in self.microgrid_mapping.items():
            if other_mg_id == mg_id:
                neighbors.append(other_id)
        
        return neighbors


# 创建环境的辅助函数
def make_dsr_env(config: Optional[DSRConfig] = None, use_dan: bool = True) -> DSREnv:
    """创建配电网恢复环境"""
    if config is None:
        config = DSRConfig()
    
    if use_dan and config.use_dynamic_network:
        return DSREnvWithDAN(config)
    else:
        return DSREnv(config)


# 测试代码
if __name__ == "__main__":
    # 创建环境
    config = DSRConfig(
        n_dg=7,
        n_pv=9,
        n_switch=20,
        max_steps=15,
        use_action_mask=True,
        use_dynamic_network=True
    )
    
    env = make_dsr_env(config)
    
    # 测试环境
    obs, state, avail_actions = env.reset()
    print(f"环境创建成功！智能体数量: {env.n_agents}")
    print(f"观测维度: {[o.shape for o in obs]}")
    print(f"动作空间: {[a.n for a in env.action_space]}")
    
    # 测试一步
    actions = [env.action_space[i].sample() for i in range(env.n_agents)]
    obs, state, rewards, dones, infos, avail_actions = env.step(actions)
    print(f"奖励: {rewards[0][0]:.3f}")
    print(f"恢复率: {infos[0]['restored_load']:.3f}")