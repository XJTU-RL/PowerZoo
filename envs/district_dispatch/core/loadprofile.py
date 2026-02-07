# -*- coding: utf-8 -*-
"""
District Load Profile Manager
台区负荷曲线管理器

管理 24 小时负荷曲线（96 步，15 分钟分辨率）、PV 出力曲线、EV 负荷曲线。
基础曲线为确定性生成，可叠加可控随机噪声。
"""

import logging
from typing import Dict, Optional

import numpy as np

from envs.district_dispatch.constants import EPISODE, MARKET
from envs.district_dispatch.core.config import DistrictDispatchConfig

logger = logging.getLogger(__name__)


class DistrictLoadProfile:
	"""台区负荷曲线管理器

	生成并管理 24 小时运行所需的各类时序曲线：
	- 基础负荷曲线：典型日负荷（双峰型）
	- PV 出力曲线：钟形日照曲线
	- EV 负荷曲线：晚高峰充电模式
	- 分时电价曲线：谷-平-峰时段

	参数:
		config: 区域调度配置
		seed: 随机种子
	"""

	def __init__(
		self, config: DistrictDispatchConfig, seed: int = 42
	):
		self.config = config
		self.rng = np.random.RandomState(seed)
		self.max_steps = config.max_episode_steps

		# 生成基础曲线
		self._base_load_profile = self._generate_base_load()
		self._pv_profile = self._generate_pv_profile()
		self._ev_profile = self._generate_ev_profile()
		self._price_profile = self._generate_price_profile()

	def _generate_base_load(self) -> np.ndarray:
		"""生成标准日负荷曲线

		典型双峰负荷: 上午 10-11 点 + 晚间 19-20 点。

		返回:
			shape=(max_steps,), 归一化 [0, 1]
		"""
		t = np.linspace(0, 24, self.max_steps, endpoint=False)

		# 基础负荷: 双高斯峰叠加
		morning_peak = 0.7 * np.exp(
			-0.5 * ((t - 10.5) / 1.5) ** 2
		)
		evening_peak = 0.9 * np.exp(
			-0.5 * ((t - 19.5) / 2.0) ** 2
		)
		# 午间小峰
		noon_bump = 0.3 * np.exp(
			-0.5 * ((t - 13.0) / 1.0) ** 2
		)
		# 基底负荷
		base = 0.25

		profile = base + morning_peak + evening_peak + noon_bump
		# 归一化到 [0, 1]
		profile = profile / profile.max()
		return profile.astype(np.float64)

	def _generate_pv_profile(self) -> np.ndarray:
		"""生成 PV 出力曲线

		钟形日照曲线，峰值在 12:00 左右，
		日出约 06:00，日落约 18:00。

		返回:
			shape=(max_steps,), 归一化 [0, 1]
		"""
		t = np.linspace(0, 24, self.max_steps, endpoint=False)

		# 日照钟形曲线
		solar = np.exp(-0.5 * ((t - 12.0) / 2.5) ** 2)
		# 夜间清零 (06:00 前和 18:00 后)
		solar[t < 5.5] = 0.0
		solar[t > 18.5] = 0.0
		# 平滑过渡
		sunrise_mask = (t >= 5.5) & (t < 6.5)
		sunset_mask = (t > 17.5) & (t <= 18.5)
		solar[sunrise_mask] *= (t[sunrise_mask] - 5.5)
		solar[sunset_mask] *= (18.5 - t[sunset_mask])

		# 归一化
		max_val = solar.max()
		if max_val > 0:
			solar = solar / max_val
		return solar.astype(np.float64)

	def _generate_ev_profile(self) -> np.ndarray:
		"""生成 EV 负荷曲线

		晚高峰充电模式: 17:00-23:00 集中充电，
		深夜少量充电，白天低负荷。

		返回:
			shape=(max_steps,), 归一化 [0, 1]
		"""
		t = np.linspace(0, 24, self.max_steps, endpoint=False)

		# 晚高峰充电: 主峰 19:00-21:00
		evening_charge = 0.9 * np.exp(
			-0.5 * ((t - 20.0) / 1.5) ** 2
		)
		# 回家后立即充电: 17:00-19:00
		early_charge = 0.5 * np.exp(
			-0.5 * ((t - 18.0) / 1.0) ** 2
		)
		# 深夜低谷充电
		night_charge = 0.15 * np.exp(
			-0.5 * ((t - 2.0) / 2.0) ** 2
		)
		# 白天基底
		base = 0.05

		profile = base + evening_charge + early_charge + night_charge
		profile = profile / profile.max()
		return profile.astype(np.float64)

	def _generate_price_profile(self) -> np.ndarray:
		"""生成分时电价曲线

		基于 constants.MARKET.TOU_MULTIPLIERS (24小时) 插值到 max_steps。

		返回:
			shape=(max_steps,), 单位: 元/kWh
		"""
		tou = np.array(MARKET.TOU_MULTIPLIERS, dtype=np.float64)
		# 从 24 点插值到 max_steps 点
		hours = np.arange(24)
		steps = np.linspace(0, 23, self.max_steps, endpoint=False)
		price_multiplier = np.interp(steps, hours, tou)
		return price_multiplier * MARKET.BASE_PRICE_YUAN_KWH

	def get_step_data(
		self, step: int, add_noise: bool = True
	) -> Dict[str, float]:
		"""获取某一步的负荷数据

		参数:
			step: 步数索引 (0 ~ max_steps-1)
			add_noise: 是否添加随机噪声

		返回:
			{
				"load_ratio": 负荷比例,
				"pv_ratio": PV 可用比例,
				"ev_ratio": EV 负荷比例,
				"price": 电价 (元/kWh),
				"time_of_day": 归一化时间,
			}
		"""
		step = int(np.clip(step, 0, self.max_steps - 1))

		load_ratio = self._base_load_profile[step]
		pv_ratio = self._pv_profile[step]
		ev_ratio = self._ev_profile[step]
		price = self._price_profile[step]

		if add_noise and self.config.load_noise:
			noise_std = self.config.noise_std
			load_ratio *= 1.0 + self.rng.normal(0, noise_std)
			pv_ratio *= 1.0 + self.rng.normal(0, noise_std * 0.5)
			ev_ratio *= 1.0 + self.rng.normal(0, noise_std)
			# 确保非负
			load_ratio = max(0.0, load_ratio)
			pv_ratio = max(0.0, min(1.0, pv_ratio))
			ev_ratio = max(0.0, ev_ratio)

		return {
			"load_ratio": float(load_ratio),
			"pv_ratio": float(pv_ratio),
			"ev_ratio": float(ev_ratio),
			"price": float(price),
			"time_of_day": self.get_time_of_day(step),
		}

	def get_pv_available(self, step: int) -> float:
		"""获取某一步 PV 可用出力比例

		参数:
			step: 步数索引

		返回:
			PV 可用出力比例 [0, 1]
		"""
		step = int(np.clip(step, 0, self.max_steps - 1))
		return float(self._pv_profile[step])

	def get_ev_load(self, step: int) -> float:
		"""获取某一步 EV 负荷比例

		参数:
			step: 步数索引

		返回:
			EV 负荷比例 [0, 1]
		"""
		step = int(np.clip(step, 0, self.max_steps - 1))
		return float(self._ev_profile[step])

	def get_electricity_price(self, step: int) -> float:
		"""获取某一步电价

		参数:
			step: 步数索引

		返回:
			电价 (元/kWh)
		"""
		step = int(np.clip(step, 0, self.max_steps - 1))
		return float(self._price_profile[step])

	def get_time_of_day(self, step: int) -> float:
		"""获取归一化时间

		参数:
			step: 步数索引

		返回:
			归一化时间 [0, 1)，代表 00:00 到 24:00
		"""
		return float(step) / float(self.max_steps)

	def get_price_normalized(self, step: int) -> float:
		"""获取归一化电价 (0-1)

		参数:
			step: 步数索引

		返回:
			归一化电价
		"""
		price = self.get_electricity_price(step)
		max_price = MARKET.BASE_PRICE_YUAN_KWH * max(
			MARKET.TOU_MULTIPLIERS
		)
		min_price = MARKET.BASE_PRICE_YUAN_KWH * min(
			MARKET.TOU_MULTIPLIERS
		)
		if max_price == min_price:
			return 0.5
		return float(
			(price - min_price) / (max_price - min_price)
		)

	def reset(self, seed: Optional[int] = None) -> None:
		"""重置并可选地生成新的随机种子

		参数:
			seed: 新随机种子，None 则保持原种子
		"""
		if seed is not None:
			self.rng = np.random.RandomState(seed)
		# 重新生成基础曲线（确定性部分不变，噪声在 get_step_data 中实时生成）
		self._base_load_profile = self._generate_base_load()
		self._pv_profile = self._generate_pv_profile()
		self._ev_profile = self._generate_ev_profile()
		self._price_profile = self._generate_price_profile()
		logger.debug(
			f"负荷曲线已重置, seed={seed}"
		)
