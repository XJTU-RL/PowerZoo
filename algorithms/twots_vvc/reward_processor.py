# algorithms/twots_vvc/reward_processor.py
"""Reward processing for two-timescale VVC algorithm."""
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class TwoTSRewardProcessor:
	"""Process rewards for two-timescale VVC algorithm.

	This class handles algorithm-specific reward computation,
	including weighted objectives and penalty terms.
	"""

	def __init__(self, cfg: Dict[str, Any]):
		"""Initialize reward processor.

		Args:
			cfg: Configuration dictionary
		"""
		self.cfg = cfg
		reward_cfg = cfg.get("reward", {})

		# Reward weights (algorithm-specific)
		self.w_line_loss = reward_cfg.get("w_line_loss", 1.0)
		self.w_volt_violation = reward_cfg.get("w_volt_violation", 50.0)
		self.w_inverter_degradation = reward_cfg.get("w_inverter_degradation", 1.0)

		# Switching costs
		self.c_tap = reward_cfg.get("c_tap", 1.0)
		self.c_cap = reward_cfg.get("c_cap", 0.5)

		# Normalization parameters
		self.P_base = reward_cfg.get("P_base", 1000.0)  # kW base power
		self.Q_r = reward_cfg.get("Q_r", 50.0)  # PV rated reactive power kvar

		# Tracking for incremental rewards
		self.last_pv_q = {}

		logger.info(f"TwoTSRewardProcessor initialized with weights: "
					f"line_loss={self.w_line_loss}, volt_violation={self.w_volt_violation}, "
					f"inverter_degradation={self.w_inverter_degradation}")

	def compute_minute_reward(self, info: Dict[str, Any], delta_q: float = 0.0) -> float:
		"""Compute minute-level (fast timescale) reward.

		Args:
			info: Information dictionary from environment step
			delta_q: Change in reactive power output

		Returns:
			Weighted reward value
		"""
		# Check convergence
		if not info.get("convergence", True):
			# Large penalty for non-convergence
			return -(1000.0 * self.w_line_loss + 100.0 * self.w_volt_violation)

		# Extract raw metrics from environment
		line_loss = info.get("line_loss", 0.0)
		volt_violation_cost = info.get("volt_violation_cost", 0.0)

		# Cost components with algorithm-specific weights
		J_L = line_loss * self.w_line_loss
		J_V = volt_violation_cost * self.w_volt_violation
		J_I = (delta_q ** 2) * self.w_inverter_degradation

		# Return negative cost as reward
		reward = -(J_L + J_V + J_I)

		return reward

	def compute_interval_reward(self, interval_data: List[Dict[str, Any]],
								switch_cost: float = 0.0) -> Dict[str, Any]:
		"""Compute interval-level (slow timescale) reward.

		Args:
			interval_data: List of step data from the interval
			switch_cost: Total switching cost from slow actions

		Returns:
			Dictionary containing reward components and total reward
		"""
		total_minute_rewards = 0.0
		line_loss_sum = 0.0
		volt_violation_sum = 0.0
		convergence_count = 0
		total_delta_q = 0.0

		for step_data in interval_data:
			info = step_data.get("infos", [{}])[0] if isinstance(step_data.get("infos"), list) else step_data.get("infos", {})

			# Compute delta Q for this step
			delta_q = self._compute_step_delta_q(step_data)
			total_delta_q += delta_q ** 2

			# Compute minute reward
			minute_reward = self.compute_minute_reward(info, delta_q)
			total_minute_rewards += minute_reward

			# Track components for logging
			line_loss_sum += info.get("line_loss", 0.0)
			volt_violation_sum += info.get("volt_violation_cost", 0.0)
			if info.get("convergence", True):
				convergence_count += 1

		# Compute interval reward
		interval_reward = total_minute_rewards - switch_cost

		# Average metrics
		num_steps = max(len(interval_data), 1)

		return {
			"interval_reward": interval_reward,
			"minute_rewards_sum": total_minute_rewards,
			"switch_cost": switch_cost,
			"avg_line_loss": line_loss_sum / num_steps,
			"avg_volt_violation": volt_violation_sum / num_steps,
			"convergence_rate": convergence_count / num_steps,
			"rms_delta_q": np.sqrt(total_delta_q / num_steps)
		}

	def compute_switching_cost(self, device_id: str, old_state: int, new_state: int) -> float:
		"""Compute switching cost for a slow device.

		Args:
			device_id: Device identifier
			old_state: Previous state
			new_state: New state

		Returns:
			Switching cost
		"""
		if "Regulator" in device_id:
			# Tap change cost proportional to change magnitude
			tap_change = abs(new_state - old_state)
			return tap_change * self.c_tap
		elif "Capacitor" in device_id:
			# Fixed cost for capacitor switching
			if new_state != old_state:
				return self.c_cap
		elif "Battery" in device_id:
			# Battery mode switching cost (if discrete)
			if new_state != old_state:
				return self.c_cap * 0.5  # Lower cost for battery mode changes

		return 0.0

	def _compute_step_delta_q(self, step_data: Dict[str, Any]) -> float:
		"""Compute reactive power change for a step.

		Args:
			step_data: Step data dictionary

		Returns:
			RMS of reactive power changes
		"""
		obs_dict = step_data.get("obs_dict", {})
		if not obs_dict:
			return 0.0

		total_delta = 0.0
		count = 0

		# Check PV devices for reactive power changes
		for device_id, obs in obs_dict.items():
			if "PV" in device_id and isinstance(obs, dict):
				current_q = obs.get("q_kvar", 0.0)

				if device_id in self.last_pv_q:
					delta = (current_q - self.last_pv_q[device_id]) / self.Q_r  # Normalize
					total_delta += delta ** 2
					count += 1

				self.last_pv_q[device_id] = current_q

		# Return RMS of deltas
		if count > 0:
			return np.sqrt(total_delta / count)
		return 0.0

	def reset_tracking(self):
		"""Reset internal tracking states."""
		self.last_pv_q = {}

	def get_reward_info(self) -> Dict[str, float]:
		"""Get current reward configuration.

		Returns:
			Dictionary of reward weights and parameters
		"""
		return {
			"w_line_loss": self.w_line_loss,
			"w_volt_violation": self.w_volt_violation,
			"w_inverter_degradation": self.w_inverter_degradation,
			"c_tap": self.c_tap,
			"c_cap": self.c_cap,
			"P_base": self.P_base,
			"Q_r": self.Q_r
		}