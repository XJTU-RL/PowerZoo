
import os
import gym
import numpy as np
from envs.smartgrid.circuit_system import Circuits
from envs.smartgrid.data_process.loadprofile import LoadProfile
from envs.smartgrid.rewards.powerzoo_reward import PowerZooReward
from envs.smartgrid.rewards.lagrangian import LagrangianUpdater
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from functools import lru_cache, wraps
import logging
import time
from envs.smartgrid.utils import get_logger, log_reward_components, log_device_actions

# 导入自定义异常类和常量
from envs.smartgrid.exceptions import (
	DSSSimulationError,
	DSSConvergenceError,
	ActionValidationError,
	ConfigurationError
)
from envs.smartgrid.constants import VOLTAGE, SIMULATION

# 导入新的配置系统
from envs.smartgrid.base_env.env_config import SmartGridConfig


logger = get_logger(__name__)


def performance_monitor(func):
    """性能监控装饰器"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            elapsed = time.time() - start_time
            if elapsed > 0.1:  # 记录耗时超过100ms的操作
                logger.debug(f"{func.__name__} 耗时: {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} 执行失败 (耗时{elapsed:.3f}s): {e}")
            raise
    return wrapper

#### action space class ####
class ActionSpace:
    '''电容器、调压器、电池和光伏系统的动作空间封装

    属性:
        cap_num, reg_num, bat_num, pv_num (int): 电容器、调压器、电池和光伏系统的数量
        reg_act_num, bat_act_num, pv_act_num: 调压器、电池和光伏系统的动作数量
        space (gym.spaces): 来自gym的空间对象
        pv_control_enabled (bool): 是否启用光伏控制

    注意:
        当所有系统都使用离散动作时,space为MultiDiscrete类型;
        否则,space为MultiDiscrete和Box的元组,用于连续动作
    '''
    def __init__(self, CRBP_num: Tuple[int, int, int, int], RBP_act_num: Tuple[int, int, int], pv_control_enabled: bool = False):
        self.cap_num, self.reg_num, self.bat_num, self.pv_num = CRBP_num
        self.reg_act_num, self.bat_act_num, self.pv_act_num = RBP_act_num
        self.pv_control_enabled = pv_control_enabled
        
        # 缓存空间对象避免重复创建
        self._space = None
        self._initialize_space()
    
    def _initialize_space(self):
        """初始化动作空间 - 支持CRBP架构"""
        discrete_actions = ([2] * self.cap_num +  # 电容器动作 (0/1)
                          [self.reg_act_num] * self.reg_num)  # 调压器动作 (tap position)
        
        continuous_actions = []
        continuous_shape = 0
        
        # 处理电池动作空间
        if self.bat_num > 0:
            if isinstance(self.bat_act_num, (int, float)) and self.bat_act_num < float('inf'):
                discrete_actions.extend([int(self.bat_act_num)] * self.bat_num)  # 离散电池动作
            else:
                continuous_shape += self.bat_num  # 连续电池动作
        
        # 处理光伏动作空间（如果启用）
        if self.pv_control_enabled and self.pv_num > 0:
            if isinstance(self.pv_act_num, (int, float)) and self.pv_act_num < float('inf'):
                discrete_actions.extend([int(self.pv_act_num)] * self.pv_num)  # 离散PV动作
            else:
                continuous_shape += self.pv_num * 2  # 连续PV动作 (有功功率 + 功率因数)
        
        # 构建最终动作空间
        if continuous_shape > 0:
            # 混合动作空间：离散 + 连续
            discrete_space = gym.spaces.MultiDiscrete(discrete_actions) if discrete_actions else None
            continuous_space = gym.spaces.Box(low=-1, high=1, shape=(continuous_shape,), dtype=np.float32) #PV的两个控制量都是(-1,1)
            
            if discrete_space is not None:
                self._space = gym.spaces.Tuple((discrete_space, continuous_space))#(discrete * num_crb, continuous * num_pv)
            else:
                self._space = continuous_space
        else:
            # 纯离散动作空间
            self._space = gym.spaces.MultiDiscrete(discrete_actions)
    
    @property
    def space(self):
        return self._space

    def sample(self):
        """HAPPO兼容的采样方法"""
        ss = self._space.sample()
        
        # 处理混合动作空间：保持语义分离，而非强制转换
        if isinstance(self._space, gym.spaces.Tuple):
            if len(ss) == 2:  # 离散 + 连续
                discrete_part = ss[0]  # 保持离散动作为整数
                continuous_part = ss[1].astype(np.float32)  # 连续动作保持浮点数
                # 返回混合动作列表：[离散动作数组, 连续动作数组]
                return [discrete_part, continuous_part]
            else:
                # 处理其他Tuple情况，保持原始结构
                return list(ss)
        
        # 纯离散或纯连续动作空间
        return ss

    def seed(self, seed: int):
        self._space.seed(seed)

    def dim(self) -> int:
        """返回动作空间维度 - 支持CRBP混合动作空间"""
        if isinstance(self._space, gym.spaces.Tuple):
            total_dim = 0
            for subspace in self._space.spaces:
                if hasattr(subspace, 'shape'):
                    total_dim += subspace.shape[0] if subspace.shape else subspace.n
                elif hasattr(subspace, 'nvec'):
                    total_dim += len(subspace.nvec)
                else:
                    total_dim += subspace.n
            return total_dim
        elif hasattr(self._space, 'nvec'):
            return len(self._space.nvec)
        else:
            return self._space.shape[0] if self._space.shape else self._space.n

    def CRBP_num(self) -> Tuple[int, int, int, int]:
        """返回各设备数量"""
        return self.cap_num, self.reg_num, self.bat_num, self.pv_num

    def RBP_act_num(self) -> Tuple[int, int, int]:
        """返回调压器、电池、光伏动作数量"""
        return self.reg_act_num, self.bat_act_num, self.pv_act_num
    
    def get_action_dims(self) -> Dict[str, int]:
        """返回各设备类型的动作维度"""
        return {
            'capacitor': self.cap_num,
            'regulator': self.reg_num, 
            'battery': self.bat_num,
            'pv': self.pv_num if self.pv_control_enabled else 0
        }

#### environment class ####
class Env(gym.Env):
    """SmartGrid 强化学习环境

    支持两种初始化方式:
    1. 新方式（推荐）: Env(folder_path, config: SmartGridConfig)
    2. 旧方式（兼容）: Env(folder_path, info: Dict, dss_act: bool)

    Attributes:
        config (SmartGridConfig): 统一配置对象
        obs (dict): 观测/状态字典
        circuit (Circuits): 电路仿真对象
        ActionSpace: 动作空间封装
        ...
    """

    def __init__(
        self,
        folder_path: str,
        config_or_info: Union[SmartGridConfig, Dict[str, Any]],
        dss_act: bool = False
    ):
        super().__init__()

        # 统一配置处理 - 支持新旧两种接口
        if isinstance(config_or_info, SmartGridConfig):
            self.config = config_or_info
            info = config_or_info.to_info_dict()
            dss_act = config_or_info.dss_act
        else:
            # 兼容旧的 info 字典接口
            info = config_or_info
            self.config = SmartGridConfig.from_dict(info)
            self.config.dss_act = dss_act

        # === 基础配置 ===
        self.obs = {}
        self.dss_folder_path = self._resolve_system_path(folder_path, info['system_name'])
        self.dss_file = info['dss_file']
        self.source_bus = info.get('source_bus', 'sourcebus')
        self.node_size = info.get('node_size', 300)
        self.shift = info.get('shift', 50)
        self.show_node_labels = info.get('show_node_labels', False)
        self.scale = info.get('scale', 1.0)
        self.wrap_observation = True
        self.observe_load = False
        self.LLM = info.get('for_LLM', False)
        self.agents_bus = dict()

        # === 负载配置 ===
        worker_idx = info.get('worker_idx', None)
        self.load_profile = LoadProfile(
            info['max_episode_steps'],
            self.dss_folder_path,
            self.dss_file,
            worker_idx=worker_idx,
            pv_data_source=info.get('pv_data_source'),
            temperature_data_source=info.get('temperature_data_source')
        )

        # 生成负载数据
        self.num_profiles = self.load_profile.generate_episodes_from_existing_files(scale=self.scale)
        self.load_profile.select_load_profile(0)

        # === 时间和动作配置 ===
        self.horizon = info['max_episode_steps']
        self.reg_act_num = info['reg_act_num']
        self.bat_act_num = info['bat_act_num']
        self.pv_act_num = info.get('pv_act_num', float('inf'))
        self.pv_control_enabled = info.get('pv_control', False)

        # 参数验证
        self._validate_config()

        # === 创建电路对象 - 关键修复：传递完整的 RBP_act_num ===
        self.circuit = Circuits(
            os.path.join(self.dss_folder_path, self.dss_file),
            RBP_act_num=(self.reg_act_num, self.bat_act_num, self.pv_act_num),
            dss_act=dss_act,
            worker_idx=worker_idx
        )
        self.all_bus_names = self.circuit.dss.ActiveCircuit.AllBusNames
        self.cap_names = list(self.circuit.capacitors.keys())
        self.reg_names = list(self.circuit.regulators.keys())
        self.bat_names = list(self.circuit.batteries.keys())
        self.pv_names = list(self.circuit.pvs.keys()) if hasattr(self.circuit, 'pvs') else []

        self.cap_num = len(self.cap_names)
        self.reg_num = len(self.reg_names)
        self.bat_num = len(self.bat_names)
        self.pv_num = len(self.pv_names)

        assert self.cap_num >= 0 and self.reg_num >= 0 and self.bat_num >= 0 and self.pv_num >= 0 and \
               self.cap_num + self.reg_num + self.bat_num + self.pv_num >= 1, 'invalid CRBP_num'

        self.topology = self.build_graph()
        self.reward_func = PowerZooReward(self, info)
        self.t = 0

        # CMDP和Lagrangian配置
        self.use_cmdp = info.get('use_cmdp', True)
        self.lagrangian_updater = None
        self.lambda_history = []
        self.cost_history = []
        self.episode_costs = []

        if self.use_cmdp:
            self.lagrangian_updater = LagrangianUpdater(
                init_lambda=info.get('lambda_init', 1.0),
                lr=info.get('lambda_lr', 1e-3),
                target_cost=info.get('target_cost', 0.01),
                lambda_max=info.get('lambda_max', 100.0),
                lambda_min=info.get('lambda_min', 0.0),
                update_strategy=info.get('lambda_update_strategy', 'standard'),
                momentum=info.get('lambda_momentum', 0.0),
                adaptive_lr=info.get('lambda_adaptive_lr', False)
            )
            logger.info(f"CMDP模式启用 - 目标成本: {info.get('target_cost', 0.01)}, "
                       f"初始λ: {info.get('lambda_init', 1.0)}")

        # 创建动作空间和观测空间
        self.ActionSpace = ActionSpace(
            (self.cap_num, self.reg_num, self.bat_num, self.pv_num),
            (self.reg_act_num, self.bat_act_num, self.pv_act_num),
            pv_control_enabled=self.pv_control_enabled
        )
        self.action_space = self.ActionSpace.space
        self.reset_obs_space()
        self.useS = False
        self.use_render = False
        self.agents_bus = self.circuit.get_agent_bus_dict()

    def _resolve_system_path(self, folder_path: str, system_name: str) -> str:
        """解析系统目录路径

        支持多种路径格式:
        - 相对名称: '34Bus_PV' -> folder_path/node_systems/34Bus_PV
        - 绝对路径: '/home/.../34Bus_PV'
        - 带前缀: 'node_systems/34Bus_PV'
        """
        if os.path.isabs(system_name):
            return system_name
        if 'node_systems/' in system_name:
            return os.path.join(folder_path, system_name)
        return os.path.join(folder_path, 'node_systems', system_name)

    def _validate_config(self) -> None:
        """验证配置参数有效性"""
        if self.horizon < 1:
            raise ConfigurationError(f"invalid horizon: {self.horizon}")
        if self.reg_act_num < 2:
            raise ConfigurationError(f"reg_act_num must be >= 2, got {self.reg_act_num}")
        if self.bat_act_num != float('inf') and self.bat_act_num < 2:
            raise ConfigurationError(f"bat_act_num must be >= 2 or inf, got {self.bat_act_num}")
        if self.pv_act_num != float('inf') and self.pv_act_num < 2:
            raise ConfigurationError(f"pv_act_num must be >= 2 or inf, got {self.pv_act_num}")

    def reset_obs_space(self, wrap_observation=True, observe_load=False):
        '''
        根据包装和负载选项重置观测空间。
        
        建议通过此函数设置 wrap_observation 和 observe_load,
        而不是直接设置属性(例如 Env.wrap_observation)。
        
        '''
        self.wrap_observation = wrap_observation
        self.observe_load = observe_load
        
        self.reset(load_profile_idx=0)
        node_num = len(np.hstack( list(self.obs['bus_voltages'].values()) )) # 节点数量
        if observe_load: nload = len(self.obs['load_profile_t'])
        
        if self.wrap_observation:
            low, high = [0.8]*node_num, [1.2]*node_num  # add voltage bound
            low, high = low+[0]*self.cap_num, high+[1]*self.cap_num # add cap bound
            low, high = low+[0]*self.reg_num, high+[self.reg_act_num]*self.reg_num # add reg bound
            low, high = low+[0,-1]*self.bat_num, high+[1,1]*self.bat_num # add bat bound
            
            # 添加PV状态边界（如果启用PV控制）
            if self.pv_control_enabled and self.pv_num > 0:
                low, high = low+[0.0,-1.0]*self.pv_num, high+[1.0,1.0]*self.pv_num  # PV有功功率 + 功率因数
            
            if observe_load: low, high = low+[0.0]*nload, high+[1.0]*nload # add load bound
            low, high = np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)
            self.observation_space = gym.spaces.Box(low, high) 
        else:
            bat_dict = {bat: gym.spaces.Box(np.array([0,-1]), np.array([1,1]), dtype=np.float32) 
                        for bat in self.obs['bat_statuses'].keys()}
            
            obs_dict = {
                'bus_voltages': gym.spaces.Box(0.8, 1.2, shape=(node_num,)),
                'cap_statuses': gym.spaces.MultiDiscrete([2]*self.cap_num),
                'reg_statuses': gym.spaces.MultiDiscrete([self.reg_act_num]*self.reg_num), 
                'bat_statuses': gym.spaces.Dict(bat_dict)
            }
            
            # 添加PV状态空间（如果启用PV控制）
            if self.pv_control_enabled and self.pv_num > 0:
                pv_dict = {pv: gym.spaces.Box(np.array([0.0,-1.0]), np.array([1.0,1.0]), dtype=np.float32) 
                          for pv in self.pv_names}
                obs_dict['pv_statuses'] = gym.spaces.Dict(pv_dict)
            
            if observe_load: obs_dict['load_profile_t'] = gym.spaces.Box(0.0, 1.0, shape=(nload,))
            self.observation_space = gym.spaces.Dict(obs_dict)
            
        if self.wrap_observation:
            cur_dim = int(self.wrap_obs(self.obs).size)
            if hasattr(self, 'observation_space') and hasattr(self.observation_space, 'shape'):
                exp_dim = int(self.observation_space.shape[0])
                assert cur_dim == exp_dim, f"Obs dim mismatch after reset: {cur_dim} vs {exp_dim}"


    @performance_monitor
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """环境步进

        Args:
            action: Integer/Hybrid array of actions for capacitors, regulators, batteries, PV

        Returns:
            (observation, reward, done, info)
        """
        action_idx = 0
        self.str_action = ''  # 用于 plot_graph() 打印

        #### Capacitor ####
        if self.cap_num > 0:
            statuses = action[action_idx:action_idx + self.cap_num]
            capdiff = self.circuit.set_all_capacitor_statuses(statuses)
            cap_statuses = {cap: status for cap, status in zip(self.circuit.capacitors.keys(), statuses)}
            action_idx += self.cap_num
            self.str_action += 'Cap Status:' + str(statuses)
            for i, (cap_name, status) in enumerate(cap_statuses.items()):
                old_status = getattr(self.circuit.capacitors[cap_name], 'status', 0)
                log_device_actions(logger, "Capacitor", cap_name, old_status, status, capdiff[i])
        else:
            capdiff, cap_statuses = [], dict()

        #### Regulator ####
        if self.reg_num > 0:
            tapnums = action[action_idx:action_idx + self.reg_num]
            old_tapnums = [self.circuit.regulators[reg].tap for reg in self.reg_names]
            regdiff = self.circuit.set_all_regulator_tappings(tapnums)
            reg_statuses = {reg: self.circuit.regulators[reg].tap for reg in self.reg_names}
            action_idx += self.reg_num
            self.str_action += ' 调压器抽头状态' + str(tapnums)
            for i, reg_name in enumerate(self.reg_names):
                new_tap = reg_statuses[reg_name]
                log_device_actions(logger, "Regulator", reg_name, old_tapnums[i], new_tap, regdiff[i])
        else:
            regdiff, reg_statuses = [], dict()

        #### Battery ####
        if self.bat_num > 0:
            if isinstance(self.action_space, gym.spaces.Tuple) and isinstance(self.bat_act_num, (int, float)) and self.bat_act_num == float('inf'):
                # 连续电池控制，从连续段取
                continuous_start = len(action) - (self.bat_num + (self.pv_num * 2 if self.pv_control_enabled else 0))
                bat_actions = action[continuous_start:continuous_start + self.bat_num]
            else:
                bat_actions = action[action_idx:action_idx + self.bat_num]
                action_idx += self.bat_num

            old_bat_states = {}
            for bat_name in self.bat_names:
                if bat_name in self.circuit.batteries:
                    bat = self.circuit.batteries[bat_name]
                    if hasattr(bat, 'kw'):
                        old_bat_states[bat_name] = bat.kw
                    elif hasattr(bat, 'state'):
                        old_bat_states[bat_name] = bat.state
                    else:
                        old_bat_states[bat_name] = 0

            self.circuit.set_all_batteries_before_solve(bat_actions)
            self.str_action += ' Bat Status:' + str(bat_actions)

            for i, bat_name in enumerate(self.bat_names):
                if bat_name in self.circuit.batteries:
                    bat = self.circuit.batteries[bat_name]
                    new_state = getattr(bat, 'kw', getattr(bat, 'state', 0))
                    old_state = old_bat_states.get(bat_name, 0)
                    diff = abs(new_state - old_state) if isinstance(new_state, (int, float)) else 0
                    log_device_actions(logger, "Battery", bat_name, old_state, new_state, diff)

        #### PV ####
        if self.pv_control_enabled and self.pv_num > 0:
            if isinstance(self.action_space, gym.spaces.Tuple) and isinstance(self.pv_act_num, (int, float)) and self.pv_act_num == float('inf'):
                # 连续 PV 控制，假定二维：[P_ratio, pf] 或 [P_ratio, Q_ratio]
                continuous_start = len(action) - (self.pv_num * 2)
                pv_actions = action[continuous_start:]
                pv_actions = np.array(pv_actions).reshape(self.pv_num, 2)
            else:
                pv_actions = action[action_idx:action_idx + self.pv_num]
                action_idx += self.pv_num

            self.circuit.set_all_pvs_before_solve(pv_actions)
            self.str_action += ' PV Status:' + str(pv_actions)

        #### Solve DSS ####
        try:
            logger.debug(f"开始DSS求解 - 时步: {self.t}")
            self.circuit.dss.ActiveCircuit.Solution.Solve()
            converged = self.circuit.dss.ActiveCircuit.Solution.Converged
            if not converged:
                # 使用自定义异常类提供更详细的错误信息
                logger.warning(f"DSS求解未收敛 - 时步: {self.t}")
                # 尝试重新求解一次
                self.circuit.dss.ActiveCircuit.Solution.Solve()
                if not self.circuit.dss.ActiveCircuit.Solution.Converged:
                    raise DSSConvergenceError(f"DSS求解连续两次未收敛 - 时步: {self.t}")
            else:
                logger.debug(f"DSS求解成功收敛 - 时步: {self.t}")

            if self.t == 1:
                for pv_name in self.pv_names[:5]:
                    elem = self.circuit.dss.ActiveCircuit.CktElements(f"{pv_name}")
                    P = elem.TotalPowers[0]
                    Q = elem.TotalPowers[1]
                    logger.info(f"[PVCHK] {pv_name}: P={P:.1f} kW, Q={Q:.1f} kvar")

        except DSSConvergenceError:
            # 收敛失败时返回安全结果，但不中断训练
            logger.warning(f"DSS求解未收敛，使用安全默认值 - 时步: {self.t}")
            return self._get_safe_step_result()
        except Exception as e:
            logger.error(f"DSS求解失败: {e}", exc_info=True)
            raise DSSSimulationError(f"DSS求解失败: {e}") from e

        #### Battery after solve ####
        if self.bat_num > 0:
            soc_errs, dis_errs = self.circuit.set_all_batteries_after_solve()
            bat_statuses = {name: [bat.soc, -1 * bat.actual_power() / bat.max_kw] for name, bat in self.circuit.batteries.items()}
        else:
            soc_errs, dis_errs, bat_statuses = [], [], dict()

        #### Time step ####
        self.t += 1
        logger.debug(f"环境步骤完成 - 时步: {self.t}")

        #### Build obs: bus voltages (phase-only magnitudes) ####
        bus_voltages: Dict[str, List[float]] = {}
        violated_phases = 0
        total_phases = 0
        vmin, vmax = (0.95, 1.05)
        if hasattr(self, 'reward_func') and hasattr(self.reward_func, 'voltage_target_range'):
            vmin, vmax = self.reward_func.voltage_target_range

        for bus_name in self.all_bus_names:
            mags = self.circuit.bus_voltage(bus_name)  # 约定：已仅包含相导体幅值；若底层未过滤，请在该函数内过滤
            # 兜底：强制转为 list[float]
            mags = [float(x) for x in (mags or [])]
            bus_voltages[bus_name] = mags

            total_phases += len(mags)
            violated_phases += sum((v < vmin) or (v > vmax) for v in mags)

        if violated_phases > 0:
            logger.warning(f"电压违规相数: {violated_phases}/{total_phases} - 时步: {self.t}")

        self.obs['bus_voltages'] = bus_voltages
        self.obs['cap_statuses'] = cap_statuses
        self.obs['reg_statuses'] = reg_statuses
        self.obs['bat_statuses'] = bat_statuses

        # PV 状态（用于日志/奖励 shaping）
        if self.pv_control_enabled and self.pv_num > 0:
            pv_statuses = {}
            for pv_name in self.pv_names:
                if pv_name in self.circuit.pvs:
                    pv = self.circuit.pvs[pv_name]
                    if hasattr(pv, 'get_status'):
                        pv_statuses[pv_name] = pv.get_status()
                    else:
                        pv_statuses[pv_name] = [0.5, 1.0]  # 默认：50%功率，pf=1.0
                else:
                    pv_statuses[pv_name] = [0.0, 1.0]
            self.obs['pv_statuses'] = pv_statuses
        else:
            self.obs['pv_statuses'] = {}

        # 损耗（0~1 小数）
        self.obs['power_loss'] = self.circuit.calculate_loss_percentage() / 100.0
        self.obs['time'] = self.t
        if self.observe_load:
            self.obs['load_profile_t'] = self.all_load_profiles.iloc[self.t % self.horizon].to_dict()

        done = (self.t == self.horizon)

        #### PV diffs（预留）####
        pv_diffs = []
        if self.pv_control_enabled and self.pv_num > 0:
            pass

        #### Reward & info ####
        reward, info = self.reward_func.composite_reward(capdiff, regdiff, soc_errs, dis_errs, pv_diffs)

        # 追加：功率与损耗信息
        try:
            total_loss = self.circuit.total_loss()
            total_power = self.circuit.total_power()
            total_load = self.circuit.total_load_power()
            info['power_loss_kw'] = total_loss[0] if len(total_loss) > 0 else 0.0
            info['power_loss_kvar'] = total_loss[1] if len(total_loss) > 1 else 0.0
            info['total_power_kw'] = abs(total_power[0]) if len(total_power) > 0 else 100.0
            info['total_power_kvar'] = abs(total_power[1]) if len(total_power) > 1 else 50.0
            info['total_load_kw'] = total_load[0] if len(total_load) > 0 else 100.0
            info['total_load_kvar'] = total_load[1] if len(total_load) > 1 else 50.0
            info['power_loss_ratio'] = self.circuit.calculate_loss_percentage() / 100.0
            info['power_loss_percentage'] = self.circuit.calculate_loss_percentage()
        except Exception as e:
            logger.warning(f"获取功率信息时出错: {e}")
            info.update({
                'power_loss_kw': 0.0, 'power_loss_kvar': 0.0,
                'total_power_kw': 100.0, 'total_power_kvar': 50.0,
                'total_load_kw': 100.0, 'total_load_kvar': 50.0,
                'power_loss_ratio': 0.0, 'power_loss_percentage': 0.0
            })

        # 逐相与母线级违规统计（口径分离）
        total_buses = len(bus_voltages) if bus_voltages else 1
        violated_buses = sum(any((v < vmin) or (v > vmax) for v in mags) for mags in bus_voltages.values())

        # 兼容旧字段 + 新字段更清晰
        info['voltage_violation_count'] = violated_phases                     # 旧：逐相计数
        info['voltage_violations_per_bus'] = violated_buses / max(total_buses, 1)  # 旧名但语义改为“母线级占比”
        info['voltage_violation_rate'] = violated_buses / max(total_buses, 1)      # 同上（保留）
        # 新增更清晰的字段
        info['voltage_violation_count_phases'] = violated_phases
        info['voltage_violation_rate_phases'] = violated_phases / max(total_phases, 1)
        info['voltage_violation_rate_buses'] = violated_buses / max(total_buses, 1)

        # PV 利用率（0~1）
        info['pv_utilization'] = 0.0
        if self.pv_control_enabled and self.pv_num > 0 and 'pv_statuses' in self.obs:
            pv_powers = [status[0] for status in self.obs['pv_statuses'].values() if len(status) > 0]
            if pv_powers:
                valid_powers = []
                for power in pv_powers:
                    if isinstance(power, (int, float)) and not np.isnan(power):
                        corrected = min(max(power, 0.0), 1.0)
                        valid_powers.append(corrected)
                        if power != corrected:
                            logger.warning(f"PV功率比率异常值已修正: {power:.3f} -> {corrected:.3f}")
                    else:
                        logger.warning(f"PV功率比率无效值: {power}, 使用0.0")
                        valid_powers.append(0.0)
                if valid_powers:
                    info['pv_utilization'] = float(np.mean(valid_powers))

        # 电池 SOC
        if self.bat_num > 0:
            info['battery_avg_soc'] = 0.0
            if bat_statuses:
                soc_values = [soc for soc, _ in bat_statuses.values()]
                if soc_values:
                    info['battery_avg_soc'] = float(np.mean(soc_values))

        # 可选：无功电压敏感度
        if self.useS is True:
            self.Y = self.circuit.get_Y_matrix()
            self.agents_bus = self.circuit.get_agent_bus_dict()
            S = self.circuit.get_node_sensity(self.Y)
            filtered_S = {key: value for key, value in S.items() if any(key in values for values in self.agents_bus.values())}
            info['S'] = filtered_S

        # 渲染数据（可视化用）
        if self.use_render is True:
            info['bus_voltages'] = bus_voltages
            self.agents_bus = self.circuit.get_agent_bus_dict()
            info['agents_bus'] = self.agents_bus
            info['powerloss'] = self.circuit.total_loss()[0]
            info['powerloss_reward'] = -self.circuit.calculate_loss_percentage() / 100.0 * 10

        #### CMDP 部分（顺序修正：先记录 cost 再在 episode 结束时更新 λ）####
        if self.use_cmdp and 'cost_voltage' in info:
            cost_voltage = float(info['cost_voltage'])
            self.episode_costs.append(cost_voltage)
            if self.lagrangian_updater:
                info['lambda'] = self.lagrangian_updater.lmbda
                info['lambda_history_len'] = len(getattr(self, 'lambda_history', []))

            # 可选：在线拉格朗日惩罚到奖励（若你想环境就改 reward）
            if info.get('use_lagrangian_reward', False) and self.lagrangian_updater:
                lagrangian_penalty = self.lagrangian_updater.lmbda * cost_voltage
                info['reward_before_lagrangian'] = reward
                info['lagrangian_penalty'] = lagrangian_penalty
                reward = reward - lagrangian_penalty

        # 回合结束后再更新 λ（此时 episode_costs 已包含本步）
        if done and self.use_cmdp and self.lagrangian_updater and self.episode_costs:
            avg_episode_cost = float(np.mean(self.episode_costs))
            self.lagrangian_updater.update(avg_episode_cost)
            if hasattr(self, 'lambda_history'):
                self.lambda_history.append(self.lagrangian_updater.lmbda)
            if hasattr(self, 'cost_history'):
                self.cost_history.append(avg_episode_cost)
            logger.info(
                f"Episode结束 - Lagrangian更新: Lambda={self.lagrangian_updater.lmbda:.4f}, "
                f"平均成本={avg_episode_cost:.4f}, 目标成本={getattr(self.lagrangian_updater, 'target_cost', 0.0):.4f}"
            )

        #### Return ####
        if self.wrap_observation:
            return self.wrap_obs(self.obs), reward, done, info
        else:
            return self.obs, reward, done, info

    
    def _get_safe_step_result(self) -> Tuple:
        """获取安全的步进结果（错误时使用）"""
        if hasattr(self, 'observation_space') and hasattr(self.observation_space, 'shape'):
            safe_obs = np.zeros(self.observation_space.shape)
        else:
            safe_obs = np.zeros(100)  # 假设观测维度
        safe_info = {"error": True, "vol_reward": 0, "ctrl_reward": 0}
        return safe_obs, 0.0, True, safe_info
    
    def _get_safe_reset_result(self):
        """获取安全的重置结果（错误时使用）"""
        if hasattr(self, 'observation_space') and hasattr(self.observation_space, 'shape'):
            return np.zeros(self.observation_space.shape)
        else:
            return np.zeros(100)  # 假设观测维度

    @performance_monitor
    def reset(self, load_profile_idx: int = 0, irrad_train: int = 19, hours: str = '14to15/test'):
        """环境重置
        
        Args:
            load_profile_idx: ID number for load profile
            irrad_train: 光伏辐照度训练参数
            hours: 时间参数
        
        Returns:
            observation: wrapped observation
        """
        ###reset time
        self.t = 0
        
        # CMDP: 重置episode相关的成本记录
        if self.use_cmdp:
            self.episode_costs = []  # 清空当前episode的成本记录
            
            # 可选：重置Lagrangian（取决于训练策略）
            # 通常不重置Lambda，让其跨episode持续优化
            # 如果需要重置，取消下面的注释：
            # if self.lagrangian_updater:
            #     self.lagrangian_updater.reset()
            #     self.lambda_history = []
            #     self.cost_history = []
 
        ### choose load profile
        try:
            self.load_profile.select_load_profile(load_profile_idx)
            self.all_load_profiles = self.load_profile.get_load_profile_data(load_profile_idx)
        except Exception as e:
            logger.error(f"负载配置文件选择失败: {e}")
            return self._get_safe_reset_result()
            
        if self.LLM:#在此处引入光伏的可变性。
            try:
                logger.info(f"设置光伏参数: irrad_train={irrad_train}, episode_idx={load_profile_idx}")
                # 优先使用新的统一方法
                pv_success = self.load_profile.select_pv_temperature_profile(load_profile_idx)
                if not pv_success:
                    # 如果新方法失败，回退到原来的方法
                    logger.warning("使用原始光伏配置方法")
                    self.load_profile.select_irradiance_profile(style=irrad_train,hours=hours)
            except Exception as e:
                logger.warning(f"光伏配置失败: {e}")
                
        ### re-compile dss and reset batteries
        try:
            self.circuit.reset()
            # 确保时间步长设置正确（基于时间步进行LoadMult计算）
            self.circuit.dss.Text.Command = f"Set Hour={self.t}"
        except Exception as e:
            logger.error(f"电路重置失败: {e}")
            return self._get_safe_reset_result()

        ### node voltages
        bus_voltages = dict()
        for bus_name in self.all_bus_names:
            mags = self.circuit.bus_voltage(bus_name)
            bus_voltages[bus_name] = mags
        self.obs['bus_voltages'] = bus_voltages

        ### status of capacitor
        cap_statuses = {name:cap.status for name, cap in self.circuit.capacitors.items()}
        self.obs['cap_statuses'] = cap_statuses
        
        ### status of regulator
        reg_statuses = {name:reg.tap for name, reg in self.circuit.regulators.items()}
        self.obs['reg_statuses'] = reg_statuses

        ### status of battery
        bat_statuses = {name:[bat.soc, -1*bat.actual_power()/bat.max_kw] for name, bat in self.circuit.batteries.items()}
        self.obs['bat_statuses'] = bat_statuses
        
        ### status of PV systems (如果启用PV控制)
        if self.pv_control_enabled and self.pv_num > 0:
            pv_statuses = {}
            for pv_name in self.pv_names:
                if pv_name in self.circuit.pvs:
                    pv = self.circuit.pvs[pv_name]
                    # 初始化PV状态: [有功功率比例, 功率因数]
                    if hasattr(pv, 'get_status'):
                        pv_statuses[pv_name] = pv.get_status()
                    else:
                        pv_statuses[pv_name] = [0.5, 1.0]  # 初始值: 50%功率, 单位功率因数
                else:
                    pv_statuses[pv_name] = [0.0, 1.0]  # PV系统不存在时的默认值
            self.obs['pv_statuses'] = pv_statuses
        else:
            self.obs['pv_statuses'] = {}

        ### total power loss - 使用正确的计算方法
        self.obs['power_loss'] = self.circuit.calculate_loss_percentage() / 100.0  # 转换为小数形式
        
        ### time step tracker
        self.obs['time'] = self.t

        ### load for current timestep
        if self.observe_load:
            self.obs['load_profile_t'] = self.all_load_profiles.iloc[self.t].to_dict()

        ### Edge weight
        #self.obs['Y_matrix'] = self.circuit.edge_weight

        if self.wrap_observation:
            cur_dim = int(self.wrap_obs(self.obs).size)
            if hasattr(self, 'observation_space') and hasattr(self.observation_space, 'shape'):
                exp_dim = int(self.observation_space.shape[0])
                assert cur_dim == exp_dim, f"Obs dim mismatch after reset: {cur_dim} vs {exp_dim}"

        if self.wrap_observation:
            return self.wrap_obs(self.obs).astype(np.float32)
        else:
            return self.obs

    
    def dss_step(self):
        assert self.circuit.dss_act == True, 'Env.circuit.dss_act must be True'

        ### update time step
        prev_states = self.circuit.get_all_capacitor_statuses()
        prev_tapnums = self.circuit.get_all_regulator_tapnums()
        
        self.circuit.dss.ActiveCircuit.Solution.Solve()
        # 修复: 使用self.circuit.dss而不是未定义的dss
        self.circuit.dss.LoadShapes.Name = "MyIrrad"
        print("Irrad Npts=", self.circuit.dss.LoadShapes.Npts,
            "MinInterval(min)=", self.circuit.dss.LoadShapes.MinInterval,
            "UseActual=", self.circuit.dss.LoadShapes.UseActual)
        print("Irrad first10=", list(self.circuit.dss.LoadShapes.PMult)[:10])

        self.t += 1 
        cap_statuses = self.circuit.get_all_capacitor_statuses()
        reg_statuses = self.circuit.get_all_regulator_tapnums()
        capdiff = np.array([abs(prev_states[c]-cap_statuses[c]) for c in prev_states])
        regdiff = np.array([abs(prev_tapnums[r]-reg_statuses[r]) for r in prev_tapnums])

        # OpenDSS does not control batteries
        soc_errs, dis_errs, bat_statuses = [], [], dict()

        ### Update obs ###
        bus_voltages = dict()
        for bus_name in self.all_bus_names:
            mags = self.circuit.bus_voltage(bus_name)
            bus_voltages[bus_name] = mags

        self.obs['bus_voltages'] = bus_voltages
        self.obs['cap_statuses'] = cap_statuses
        self.obs['reg_statuses'] = reg_statuses
        self.obs['bat_statuses'] = bat_statuses
        # 使用新的正确方法计算功率损失百分比
        self.obs['power_loss'] = self.circuit.calculate_loss_percentage() / 100.0  # 转换为小数形式
        self.obs['time'] = self.t
        if self.observe_load:
            self.obs['load_profile_t'] = self.all_load_profiles.iloc[self.t%self.horizon].to_dict()

        done = (self.t == self.horizon)

        # 为dss_step方法添加空的PV差异列表（因为dss_step不处理PV控制）
        pv_diffs = []

        reward, info = self.reward_func.composite_reward(capdiff, regdiff,\
                                                         soc_errs, dis_errs, pv_diffs)
        # avoid dividing by zero
        info.update( {'av_cap_err': sum(capdiff)/(self.cap_num+1e-10),
                      'av_reg_err': sum(regdiff)/(self.reg_num+1e-10),
                      'av_dis_err': sum(dis_errs)/(self.bat_num+1e-10),
                      'av_soc_err': sum(soc_errs)/(self.bat_num+1e-10),
                      'av_soc': sum([soc for soc, _ in bat_statuses.values()])/ \
                                   (self.bat_num+1e-10)  })
        
        if self.wrap_observation:
            return self.wrap_obs(self.obs), reward, done, info
        else:
            return self.obs, reward, done, info

    def wrap_obs(self, obs):
        """ Wrap the observation dictionary (i.e., self.obs) to a numpy array - 支持CRBP架构
        
        Attribute:
            obs: the observation dictionary generated at self.reset() and self.step()
        
        Return:
            a numpy array of observation.
        
        """
        key_obs = ['bus_voltages', 'cap_statuses', 'reg_statuses', 'bat_statuses']
        
        # 添加PV状态（如果启用PV控制）
        if self.pv_control_enabled and self.pv_num > 0:
            key_obs.append('pv_statuses')
            
        if self.observe_load: key_obs.append('load_profile_t')

        mod_obs = []
        for var_dict in key_obs:
            # node voltage is a dict of dict, we only take minimum phase node voltage
            #if var_dict == 'bus_voltages': 
            #    for values in obs[var_dict].values():
            #        mod_obs.append(min(values))
            if var_dict in \
                ['bus_voltages','cap_statuses','reg_statuses', 'bat_statuses', 'pv_statuses', 'load_profile_t']:
                mod_obs = mod_obs + list(obs[var_dict].values())
            elif var_dict == 'power_loss_ratio':
                mod_obs.append(obs['power_loss_ratio'])
        return np.hstack(mod_obs)

    def get_obs(self, wrapped: bool = True):
        """
        返回当前环境的观测值。

        参数:
            wrapped (bool):
                - True:  返回扁平化后的 numpy 数组；
                - False: 返回原始的 obs 字典，包含 bus_voltages、cap_statuses、reg_statuses、bat_statuses 等所有字段。

        返回:
            numpy.ndarray 或 dict
        """
        if wrapped:
            # 与 reset()/step() 返回格式一致
            return self.wrap_obs(self.obs).astype(np.float32)
        else:
            # 原始观测字典
            return self.obs

    @performance_monitor
    def build_graph(self):
        """构建网络图
        
        Returns:
            Graph: Network graph
        """
        self.lines = dict()
        self.circuit.dss.ActiveCircuit.Lines.First
        while(True):
            bus1 = self.circuit.dss.ActiveCircuit.Lines.Bus1.split('.', 1)[0].lower()
            bus2 = self.circuit.dss.ActiveCircuit.Lines.Bus2.split('.', 1)[0].lower()
            line_name = self.circuit.dss.ActiveCircuit.Lines.Name.lower()
            self.lines[line_name] = (bus1, bus2)
            if self.circuit.dss.ActiveCircuit.Lines.Next==0:
                break

        transformer_names = self.circuit.dss.ActiveCircuit.Transformers.AllNames
        self.transformers = dict()
        for transformer_name in transformer_names:
            self.circuit.dss.ActiveCircuit.SetActiveElement('Transformer.' + transformer_name)
            buses = self.circuit.dss.ActiveCircuit.ActiveElement.BusNames
            #assert len(buses) == 2, 'Transformer {} has more than two terminals'.format(transformer_name)
            bus1 = buses[0].split('.', 1)[0].lower()
            bus2 = buses[1].split('.', 1)[0].lower()
            self.transformers[transformer_name] = (bus1, bus2)

        self.edges = [frozenset(edge) for _, edge in self.transformers.items()] + [frozenset(edge) for _, edge in self.lines.items()]
        if len(self.edges) != len(set(self.edges)):
            print('There are ' + str(len(self.edges)) + ' edges and ' + str(len(set(self.edges))) + ' unique edges. Overlapping transformer edges')

        self.circuit.topology.add_edges_from(self.edges)
        
        return self.circuit.topology

    def plot_graph(self, node_bound='minimum', 
                   vmin=0.95, vmax=1.05, 
                   cmap='jet', figsize=(18,12), 
                   text_loc_x=0, text_loc_y=400,
                   node_size=None, shift=None,
                   show_node_labels=None,
                   show_voltages=True,
                   show_controllers=True,
                   show_actions=False):
        """Function to plot system graph with voltage as node intensity
        
        Args:
            node_bound (str): Determine to plot max/min node voltage for nodes with more than one phase
            vmin (float): Min heatmap intensity
            vmax (float): Max heatmap intensity
            cmap (str): Colormap
            figsize (tuple): Figure size
            text_loc_x (int): x-coordinate for timestamp
            text_loc_y (int): y-coordinate for timestamp
            node_size (int): Node size. If None, initialize with environment setting
            shift (int): shift of node label. If None, initialize with environment setting
            show_node_labels (bool): show node label. If None, initialize with environment setting
            show_voltages (bool): show voltages
            show_controllers (bool): show controllers
            show_actions (bool): show actions
        
        Returns:
            fig: Matplotlib figure
            pos: dictionary of node positions
            
        """
        node_size = self.node_size if node_size is None else node_size
        shift = self.shift if shift is None else shift
        show_node_labels = self.show_node_labels if show_node_labels is None else show_node_labels
        
        #get normalized node voltages
        voltages, nodes = [], []
        pos = dict()

        assert node_bound in ['maximum', 'minimum'], 'invalid node_bound'
        for busname in self.all_bus_names:
            self.circuit.dss.Circuits.SetActiveBus(busname)
            if not self.circuit.dss.Circuits.Buses.Coorddefined: continue
            x = self.circuit.dss.Circuits.Buses.x
            y = self.circuit.dss.Circuits.Buses.y

            pos[busname] = (x,y)
            nodes.append(busname)
            bus_volts = [self.circuit.dss.Circuits.Buses.puVmagAngle[i] for i in range(len(self.circuit.dss.Circuits.Buses.puVmagAngle)) if i%2==0]
            if node_bound == 'minimum':
                voltages.append(min(bus_volts))
            elif node_bound == 'maximum':
                voltages.append(max(bus_volts))

        fig = plt.figure(figsize=figsize)
        graph = nx.Graph()

        # local lines, transformers and edges
        HasLocation = lambda p: (p[0] in pos and p[1] in pos)
        loc_lines = [pair for pair in self.lines.values() if HasLocation(pair)]
        loc_trans = [pair for pair in self.transformers.values() if HasLocation(pair)]

        graph.add_edges_from(loc_lines + loc_trans)
        nx.draw_networkx_edges(graph, pos, loc_lines, edge_color='k', width=3, label='lines')
        nx.draw_networkx_edges(graph, pos, loc_trans, edge_color='r', width=3, label='transformers')
        if show_voltages:
            nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=voltages, vmin=vmin, vmax=vmax, cmap=cmap, node_size=node_size)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            _ = plt.colorbar(sm)
        else:
            nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=np.ones(len(voltages)), vmin=vmin, vmax=vmax, cmap=cmap, node_size=node_size)

        if show_node_labels:
            node_labels = {node:node for node in pos}
            nx.draw_networkx_labels(graph, pos, labels= node_labels, font_size=15)

        # show source bus
        loc={self.source_bus:(pos[self.source_bus][0]+shift, pos[self.source_bus][1]-shift)}
        nx.draw_networkx_labels(graph, loc, labels={self.source_bus:'src'}, font_size=15)

        if show_controllers:
            if self.cap_num>0:
                labels = {self.circuit.capacitors[cap].bus1:'cap' for cap in self.cap_names}
                labels = {k:v for k,v in labels.items() if k in pos } # remove missing pos
                loc = {bus:(pos[bus][0]+shift,pos[bus][1]+shift) for bus in labels.keys()}
                nx.draw_networkx_labels(graph, loc, labels=labels, font_size=15, 
                                        font_color='darkorange')
            if self.bat_num>0:
                labels = {self.circuit.batteries[bat].bus1:'bat' for bat in self.bat_names}
                labels = {k:v for k,v in labels.items() if k in pos } # remove missing pos
                loc = {bus:(pos[bus][0]+shift,pos[bus][1]+shift) for bus in labels.keys()}
                nx.draw_networkx_labels(graph, loc, labels=labels, font_size=15, 
                                        font_color='darkviolet')
            if self.reg_num>0:
                regs = self.circuit.regulators
                labels = {(regs[r].bus1, regs[r].bus2):'reg' for r in self.reg_names}
                # accept if one of the edge's node is in pos
                labels = {k:v for k,v in labels.items() if (k[0] in pos or k[1] in pos) }
                
                loc = dict()
                for key in labels.keys():
                    _, _ = key  # 解包但不使用
                    lx, ly, count = 0.0, 0.0, 0
                    for b in list(key):
                        if b in pos:
                            ll = pos[b]
                            lx, ly, count = lx+ll[0], ly+ll[1], count+1
                    lx, ly = lx/count, ly/count
                    loc[key] = (lx + shift, ly + shift)
                nx.draw_networkx_labels(graph, loc, labels=labels, font_size=15, 
                                        font_color='darkred')


        
        if show_actions:
            plt.text(text_loc_x, text_loc_y, s='t='+str(self.t)+' Action: '+ self.str_action, 
                     fontsize=18)
        elif show_voltages:
            plt.text(text_loc_x, text_loc_y, s='t='+str(self.t), fontsize=18)

        return fig, pos

    def seed(self, seed):
        self.ActionSpace.seed(seed)

    def random_action(self):
        """Samples random action
        
        Returns:
            Array: Random control actions
        """
        return self.ActionSpace.sample()

    def dummy_action(self):
        return [1]*self.cap_num + \
               [self.reg_act_num]*self.reg_num + \
               [0.0 if isinstance(self.bat_act_num, (int, float)) and self.bat_act_num==np.inf else int(self.bat_act_num)//2]*self.bat_num
        
    def load_base_kW(self):
        '''
        get base kW of load objects.
        see class Load in circuit.py for details on Load.feature
        '''
        basekW = dict()
        for load in self.circuit.loads.keys():
            basekW[load[5:]] = self.circuit.loads[load].feature[1]
        return basekW
    

# 日志信息
logger.info("PowerZoo环境类初始化完成")
