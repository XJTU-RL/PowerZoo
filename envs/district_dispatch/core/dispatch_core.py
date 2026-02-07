# -*- coding: utf-8 -*-
"""
District Dispatch Core Environment
区域调度核心环境逻辑

编排 circuit_adapter、districts、power_exchange、loadprofile、rewards，
实现完整的 step/reset 周期。每个 step:
1. 解析动作 → 台区设备指令 + 功率交换指令
2. 应用设备控制 + 功率交换
3. OpenDSS 潮流求解
4. 读取结果 → 构建观测和奖励
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from envs.district_dispatch.constants import EPISODE, EXCHANGE, STORAGE
from envs.district_dispatch.core.circuit_adapter import DistrictCircuitAdapter
from envs.district_dispatch.core.config import (
	DistrictDeviceConfig,
	DistrictDispatchConfig,
	TieLineConfig,
)
from envs.district_dispatch.core.district import (
	District,
	EVCharger,
	PVUnit,
	StorageUnit,
)
from envs.district_dispatch.core.loadprofile import DistrictLoadProfile
from envs.district_dispatch.core.power_exchange import PowerExchangeManager
from envs.district_dispatch.rewards.dispatch_reward import DistrictDispatchReward

logger = logging.getLogger(__name__)


class DistrictDispatchCore:
	"""区域调度核心环境

	管理多台区配电网的调度决策仿真。
	每个台区作为一个 MARL agent，通过连续动作空间控制
	PV 削减、储能充放电、EV 调制和台区间功率交换。

	参数:
		config: 环境配置
		worker_idx: 并行 worker 索引
	"""

	def __init__(self, config: DistrictDispatchConfig, worker_idx: Optional[int] = None):
		self.config = config
		self.worker_idx = worker_idx
		self.config.validate()

		# 加载系统 YAML 定义
		self.system_def = self._load_system_definition()

		# 初始化 OpenDSS 电路适配器
		dss_path = self._resolve_dss_path()
		self.circuit = DistrictCircuitAdapter(dss_path, worker_idx)
		self.circuit.compile()

		# 初始化台区
		self.districts: List[District] = self._build_districts()
		self.n_agents = len(self.districts)

		# 初始化功率交换管理器
		self.exchange_manager = PowerExchangeManager(config)
		self.exchange_manager.reset(self.circuit)

		# 初始化负荷曲线
		self.load_profile = DistrictLoadProfile(config, seed=config.seed)

		# 初始化奖励函数
		self.reward_fn = DistrictDispatchReward(config)

		# 环境状态
		self.current_step = 0
		self.done = False
		self.max_action_dim = config.get_max_action_dim()
		self.max_obs_dim = config.get_max_obs_dim()
		self.share_obs_dim = config.get_share_obs_dim()

		# 首次潮流求解获取初始状态
		self.circuit.solve()
		self._update_all_district_states()

		logger.info(
			f"DistrictDispatchCore 初始化: {self.n_agents} agents, "
			f"action_dim={self.max_action_dim}, obs_dim={self.max_obs_dim}, "
			f"share_obs_dim={self.share_obs_dim}"
		)

	def _resolve_dss_path(self) -> str:
		"""解析 DSS 文件的绝对路径"""
		from utils.path_utils import get_system_folder
		dss_folder = str(get_system_folder(self.config.system_name))
		return os.path.join(dss_folder, self.config.dss_file)

	def _load_system_definition(self) -> Dict[str, Any]:
		"""加载系统 YAML 定义"""
		from utils.path_utils import get_system_folder
		sys_folder = str(get_system_folder(self.config.system_name))
		yaml_path = os.path.join(
			os.path.dirname(sys_folder.rstrip("/")),
			"..",
			"configs",
			"systems",
			f"{self.config.system_name}.yaml",
		)
		# 如果相对路径找不到，用 configs/systems 下的标准位置
		if not os.path.exists(yaml_path):
			project_root = os.path.dirname(
				os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
			)
			yaml_path = os.path.join(
				project_root, "configs", "systems", f"{self.config.system_name}.yaml"
			)

		if os.path.exists(yaml_path):
			with open(yaml_path, "r", encoding="utf-8") as f:
				return yaml.safe_load(f)
		else:
			logger.warning(f"系统定义文件未找到: {yaml_path}，使用默认配置")
			return {}

	def _build_districts(self) -> List[District]:
		"""根据系统定义和配置构建台区列表"""
		districts = []
		sys_districts = self.system_def.get("districts", {})

		for d_id in range(self.config.n_districts):
			zone_key = f"zone_{d_id}"
			zone_def = sys_districts.get(zone_key, {})
			dev_config = self.config.district_devices.get(d_id, DistrictDeviceConfig())

			# 母线列表
			buses = zone_def.get("buses", [f"bus_{d_id}_{i}" for i in range(5)])
			buses = [str(b).lower() for b in buses]

			# 边界母线
			boundary = zone_def.get("boundary_buses", zone_def.get("boundary_bus", ""))
			if isinstance(boundary, str):
				boundary_buses = [boundary.lower()] if boundary else []
			else:
				boundary_buses = [str(b).lower() for b in boundary]

			# DER 设备
			devices_def = self.system_def.get("devices", {}).get(zone_key, {})
			pv_units = self._build_pv_units(devices_def, dev_config)
			storage_units = self._build_storage_units(devices_def, dev_config)
			ev_chargers = self._build_ev_chargers(devices_def, dev_config)

			district = District(
				district_id=d_id,
				name=zone_def.get("name", f"district_{d_id}"),
				buses=buses,
				boundary_buses=boundary_buses,
				pv_units=pv_units,
				storage_units=storage_units,
				ev_chargers=ev_chargers,
			)
			districts.append(district)

		return districts

	def _build_pv_units(self, devices_def: Dict, dev_config: DistrictDeviceConfig) -> List[PVUnit]:
		"""从 YAML 定义构建 PV 单元"""
		pv_defs = devices_def.get("pv_systems", [])
		units = []
		for pv_def in pv_defs:
			units.append(PVUnit(
				name=pv_def["name"],
				bus=str(pv_def["bus"]).lower(),
				capacity_kw=pv_def.get("capacity_kw", dev_config.pv_capacity_kw),
			))
		return units

	def _build_storage_units(self, devices_def: Dict, dev_config: DistrictDeviceConfig) -> List[StorageUnit]:
		"""从 YAML 定义构建储能单元"""
		storage_defs = devices_def.get("storage", [])
		units = []
		for s_def in storage_defs:
			units.append(StorageUnit(
				name=s_def["name"],
				bus=str(s_def["bus"]).lower(),
				capacity_kwh=s_def.get("capacity_kwh", dev_config.storage_capacity_kwh),
				max_power_kw=s_def.get("max_power_kw", dev_config.storage_max_power_kw),
				soc=self.config.soc_init,
			))
		return units

	def _build_ev_chargers(self, devices_def: Dict, dev_config: DistrictDeviceConfig) -> List[EVCharger]:
		"""从 YAML 定义构建 EV 充电桩"""
		ev_defs = devices_def.get("ev_chargers", [])
		chargers = []
		for ev_def in ev_defs:
			chargers.append(EVCharger(
				name=ev_def["name"],
				bus=str(ev_def["bus"]).lower(),
				max_power_kw=ev_def.get("max_power_kw", dev_config.ev_max_power_kw),
			))
		return chargers

	# ===== step / reset =====

	def step(self, actions: np.ndarray) -> Tuple[
		List[np.ndarray],		# per-agent obs
		List[np.ndarray],		# per-agent share_obs
		np.ndarray,				# rewards (n_agents, 1)
		np.ndarray,				# dones (n_agents,)
		List[Dict[str, Any]],	# infos
	]:
		"""执行一步仿真

		参数:
			actions: shape=(n_agents, max_action_dim), 归一化连续动作

		返回:
			(obs_list, share_obs_list, rewards, dones, infos)
		"""
		self.current_step += 1

		# 1. 更新负荷曲线和时变量
		step_data = self.load_profile.get_step_data(self.current_step - 1)
		self._apply_load_profile(step_data)

		# 2. 解析并执行动作
		all_exchange_actions = {}
		for d_id, district in enumerate(self.districts):
			agent_action = actions[d_id] if d_id < len(actions) else np.zeros(self.max_action_dim)
			# 解析动作切片
			n_neighbors = len(self.config.get_neighbor_ids(d_id))
			exchange_slice, pv_slice, storage_slice, ev_slice = self._parse_action(
				agent_action, d_id, n_neighbors
			)

			# 设备控制
			district.apply_controls(
				self.circuit, pv_slice, storage_slice, ev_slice
			)

			# 功率交换
			if n_neighbors > 0:
				neighbor_ids = self.config.get_neighbor_ids(d_id)
				exchange = self.exchange_manager.decode_agent_exchange_actions(
					d_id, exchange_slice, neighbor_ids, self.config
				)
				all_exchange_actions.update(exchange)

		# 3. 应用功率交换
		if all_exchange_actions:
			self.exchange_manager.apply_exchange(self.circuit, all_exchange_actions)

		# 4. 潮流求解
		converged = self.circuit.solve()

		# 5. 读取结果更新状态
		self._update_all_district_states()

		# 6. 构建环境状态字典（供奖励函数使用）
		env_state = self._build_env_state(step_data, converged)

		# 7. 计算奖励
		rewards_list, infos_list = self.reward_fn.compute_all_agents(env_state)

		# 添加收敛信息到 infos
		for info in infos_list:
			info["converged"] = converged
			info["step"] = self.current_step

		# 8. 终止判断
		self.done = (
			self.current_step >= self.config.max_episode_steps
			or not converged
		)

		# 9. 构建观测
		obs_list = self._build_all_obs(step_data)
		share_obs_list = self._build_all_share_obs(step_data, env_state)

		# 格式化输出
		rewards = np.array([[r] for r in rewards_list], dtype=np.float32)
		dones = np.array([self.done] * self.n_agents, dtype=bool)

		# 时间截断标记
		if self.done and self.current_step >= self.config.max_episode_steps:
			for info in infos_list:
				info["TimeLimit.truncated"] = True
				info["bad_transition"] = True

		return obs_list, share_obs_list, rewards, dones, infos_list

	def reset(self, seed: Optional[int] = None) -> Tuple[
		List[np.ndarray],		# per-agent obs
		List[np.ndarray],		# per-agent share_obs
	]:
		"""重置环境

		参数:
			seed: 可选随机种子

		返回:
			(obs_list, share_obs_list)
		"""
		self.current_step = 0
		self.done = False

		# 重置负荷曲线
		self.load_profile.reset(seed=seed)

		# 重新编译电路
		self.circuit.reset()

		# 重置功率交换（含虚拟元素初始化）
		self.exchange_manager.reset(self.circuit)

		# 重置所有台区
		for district in self.districts:
			district.reset(soc_init=self.config.soc_init)

		# 初始潮流求解
		self.circuit.solve()

		# 更新初始状态
		self._update_all_district_states()

		# 构建初始观测
		step_data = self.load_profile.get_step_data(0)
		obs_list = self._build_all_obs(step_data)
		env_state = self._build_env_state(step_data, converged=True)
		share_obs_list = self._build_all_share_obs(step_data, env_state)

		return obs_list, share_obs_list

	# ===== 动作解析 =====

	def _parse_action(
		self,
		action: np.ndarray,
		district_id: int,
		n_neighbors: int,
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""将统一动作向量拆分为各类控制指令

		动作向量布局:
		[exchange_p_0..n, exchange_q_0..n, pv_0..m, storage_0..k, ev_0..j, padding...]

		参数:
			action: shape=(max_action_dim,), 归一化 [-1, 1]
			district_id: 台区ID
			n_neighbors: 邻居数

		返回:
			(exchange_slice, pv_slice, storage_slice, ev_slice)
		"""
		dev = self.config.district_devices.get(district_id, DistrictDeviceConfig())
		idx = 0

		# 功率交换 (n_neighbors * 2: P + Q)
		exchange_len = n_neighbors * 2
		exchange_slice = action[idx:idx + exchange_len]
		idx += exchange_len

		# PV 削减率 — 动作空间 [-1, 1] → 削减率 [0, 1]
		pv_len = dev.n_pv
		pv_raw = action[idx:idx + pv_len]
		pv_slice = (pv_raw + 1.0) / 2.0  # [-1,1] → [0,1]
		idx += pv_len

		# 储能控制 — 已经是 [-1, 1]
		storage_len = dev.n_storage
		storage_slice = action[idx:idx + storage_len]
		idx += storage_len

		# EV 调制 — 动作空间 [-1, 1] → 调制率 [0, 1]
		ev_len = dev.n_ev_charger
		ev_raw = action[idx:idx + ev_len]
		ev_slice = (ev_raw + 1.0) / 2.0
		idx += ev_len

		# 填充未覆盖的部分为零向量
		if len(exchange_slice) < exchange_len:
			exchange_slice = np.zeros(exchange_len, dtype=np.float32)
		if len(pv_slice) < pv_len:
			pv_slice = np.zeros(pv_len, dtype=np.float32)
		if len(storage_slice) < storage_len:
			storage_slice = np.zeros(storage_len, dtype=np.float32)
		if len(ev_slice) < ev_len:
			ev_slice = np.zeros(ev_len, dtype=np.float32)

		return exchange_slice, pv_slice, storage_slice, ev_slice

	# ===== 负荷曲线应用 =====

	def _apply_load_profile(self, step_data: Dict[str, float]) -> None:
		"""将负荷曲线应用到各台区的 DER 设备

		参数:
			step_data: 当前步的负荷数据
		"""
		pv_ratio = step_data["pv_ratio"]
		ev_ratio = step_data["ev_ratio"]

		for district in self.districts:
			# PV 可用出力 = 容量 × 日照比例
			for pv in district.pv_units:
				pv.available_power_kw = pv.capacity_kw * pv_ratio

			# EV 基础负荷 = 最大功率 × EV负荷比例
			for ev in district.ev_chargers:
				ev.current_load_kw = ev.max_power_kw * ev_ratio

	# ===== 状态更新 =====

	def _update_all_district_states(self) -> None:
		"""从电路读取更新所有台区状态"""
		for district in self.districts:
			district.update_state(self.circuit)
			# 更新交换功率
			ex_in, ex_out = self.exchange_manager.get_exchange_summary(district.district_id)
			district.exchange_in_kw = ex_in
			district.exchange_out_kw = ex_out

	# ===== 环境状态构建 =====

	def _build_env_state(self, step_data: Dict[str, float], converged: bool) -> Dict[str, Any]:
		"""构建奖励函数所需的环境状态字典"""
		loss_kw, loss_kvar = self.circuit.get_total_losses()
		total_load_kw, total_load_kvar = self.circuit.get_total_load()

		return {
			"districts": self.districts,
			"config": self.config,
			"n_districts": self.config.n_districts,
			"electricity_price": step_data["price"],
			"dt_hours": EPISODE.STEP_INTERVAL_MIN / 60.0,
			"total_loss_kw": loss_kw,
			"total_load_kw": total_load_kw,
			"exchange_records": self.exchange_manager.exchange_records,
			"converged": converged,
			"step": self.current_step,
		}

	# ===== 观测构建 =====

	def _build_all_obs(self, step_data: Dict[str, float]) -> List[np.ndarray]:
		"""构建所有台区的本地观测"""
		obs_list = []
		time_of_day = step_data["time_of_day"]
		price_norm = self.load_profile.get_price_normalized(self.current_step)

		for d_id, district in enumerate(self.districts):
			# 邻居信息
			neighbor_info = self._get_neighbor_info(d_id)

			# 本地观测
			obs = district.get_obs_vector(time_of_day, price_norm, neighbor_info)

			# 零填充到统一维度
			if len(obs) < self.max_obs_dim:
				padded = np.zeros(self.max_obs_dim, dtype=np.float32)
				padded[:len(obs)] = obs
				obs = padded
			elif len(obs) > self.max_obs_dim:
				obs = obs[:self.max_obs_dim]

			obs_list.append(obs)

		return obs_list

	def _build_all_share_obs(
		self, step_data: Dict[str, float], env_state: Dict[str, Any]
	) -> List[np.ndarray]:
		"""构建全局共享观测 (centralized critic)

		全局观测 = 所有台区本地观测拼接 + 系统级聚合 + 市场状态
		"""
		# 所有本地观测拼接
		local_obs = self._build_all_obs(step_data)
		all_local = np.concatenate(local_obs)

		# 系统级聚合 (6维)
		loss_kw = env_state["total_loss_kw"]
		total_load = env_state["total_load_kw"]
		total_gen_kw, _ = self.circuit.get_total_generation()

		all_v = []
		for d in self.districts:
			for bus, v_list in d.voltages.items():
				all_v.extend(v_list)
		if len(all_v) == 0:
			all_v = [1.0]
		sys_v_min = float(np.min(all_v))
		sys_v_max = float(np.max(all_v))
		sys_v_mean = float(np.mean(all_v))

		system_agg = np.array([
			total_load / 1000.0,		# 归一化总负荷
			total_gen_kw / 1000.0,		# 归一化总发电
			loss_kw / max(total_load, 1.0),	# 网损率
			sys_v_min,
			sys_v_max,
			sys_v_mean,
		], dtype=np.float32)

		# 市场状态 (2维)
		market = np.array([
			step_data["price"],
			self.config.carbon_intensity,
		], dtype=np.float32)

		# 拼接
		share_obs = np.concatenate([all_local, system_agg, market])

		# 确保维度对齐
		if len(share_obs) < self.share_obs_dim:
			padded = np.zeros(self.share_obs_dim, dtype=np.float32)
			padded[:len(share_obs)] = share_obs
			share_obs = padded
		elif len(share_obs) > self.share_obs_dim:
			share_obs = share_obs[:self.share_obs_dim]

		# 所有 agent 共享相同的全局观测
		return [share_obs.copy() for _ in range(self.n_agents)]

	def _get_neighbor_info(self, district_id: int) -> List[Dict[str, float]]:
		"""获取邻居台区的信息（供本地观测使用）"""
		neighbor_ids = self.config.get_neighbor_ids(district_id)
		info_list = []
		for nb_id in neighbor_ids:
			if 0 <= nb_id < len(self.districts):
				nb = self.districts[nb_id]
				_, _, v_mean = nb.get_voltage_stats()
				net_load = nb.get_net_load()
				info_list.append({"v_mean": v_mean, "net_load": net_load})
			else:
				info_list.append({"v_mean": 1.0, "net_load": 0.0})
		return info_list

	# ===== 工具方法 =====

	def seed(self, seed: int) -> None:
		"""设置随机种子"""
		np.random.seed(seed)
		self.load_profile.reset(seed=seed)

	def close(self) -> None:
		"""关闭环境并释放资源"""
		self.circuit.close()
		logger.info("DistrictDispatchCore closed")
