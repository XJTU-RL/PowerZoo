# algorithms/twots_vvc/env_manager.py
"""Environment manager for two-timescale VVC algorithm."""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TwoTSEnvManager:
	"""Manages environment interaction logic for two-timescale VVC.

	This class bridges the gap between the standard PowerZoo environment
	and the two-timescale VVC algorithm, handling:
	- Device grouping into fast/slow layers
	- Time interval management
	- Action format conversion
	- State tracking across intervals
	"""

	def __init__(self, env, cfg: Dict[str, Any]):
		"""Initialize the environment manager.

		Args:
			env: Standard PowerZoo environment instance
			cfg: Configuration dictionary
		"""
		self.env = env
		self.cfg = cfg

		# Time scale configuration
		ts_cfg = cfg.get("timescale", {})
		self.T = ts_cfg.get("T", 15)  # Steps per interval
		self.slow_period_minutes = ts_cfg.get("slow_period_minutes", 15)

		# Initialize device groups
		self._init_device_groups()

		# State tracking
		self.current_step = 0
		self.interval_data = []
		self.last_obs = None
		self.last_share_obs = None
		self.last_avail_actions = None

		# Device state tracking for slow actions
		self.slow_device_states = {}

		logger.info(f"TwoTSEnvManager initialized: T={self.T}, "
					f"slow_devices={len(self.slow_ids)}, fast_devices={len(self.fast_ids)}")

	def _init_device_groups(self):
		"""Initialize device groups based on environment configuration."""
		self.slow_ids = []  # Capacitors and regulators
		self.fast_ids = []   # PV and batteries (if continuous control)
		self.all_device_ids = []

		# Map device names to indices for action conversion
		self.device_to_idx = {}
		idx = 0

		# Capacitors (slow layer - discrete switching)
		if hasattr(self.env, 'cap_names'):
			for cap_name in self.env.cap_names:
				self.slow_ids.append(cap_name)
				self.all_device_ids.append(cap_name)
				self.device_to_idx[cap_name] = idx
				idx += 1

		# Regulators (slow layer - discrete tap positions)
		if hasattr(self.env, 'reg_names'):
			for reg_name in self.env.reg_names:
				self.slow_ids.append(reg_name)
				self.all_device_ids.append(reg_name)
				self.device_to_idx[reg_name] = idx
				idx += 1

		# Batteries (can be fast or slow depending on control mode)
		if hasattr(self.env, 'bat_names'):
			bat_continuous = getattr(self.env, 'bat_act_num', 33) == float('inf')
			for bat_name in self.env.bat_names:
				if bat_continuous:
					self.fast_ids.append(bat_name)
				else:
					self.slow_ids.append(bat_name)
				self.all_device_ids.append(bat_name)
				self.device_to_idx[bat_name] = idx
				idx += 1

		# PV systems (fast layer - continuous reactive power control)
		if hasattr(self.env, 'pv_names') and getattr(self.env, 'pv_control_enabled', False):
			for pv_name in self.env.pv_names:
				self.fast_ids.append(pv_name)
				self.all_device_ids.append(pv_name)
				self.device_to_idx[pv_name] = idx
				idx += 1

		# Get action dimensions for each device
		self.slow_action_dims = []
		for device_id in self.slow_ids:
			if "Capacitor" in device_id:
				self.slow_action_dims.append(2)  # ON/OFF
			elif "Regulator" in device_id:
				dim = getattr(self.env, 'reg_act_num', 33)
				self.slow_action_dims.append(dim)
			elif "Battery" in device_id:
				dim = getattr(self.env, 'bat_act_num', 33)
				self.slow_action_dims.append(dim)
			else:
				self.slow_action_dims.append(2)  # Default

		self.fast_action_dim = 0
		for device_id in self.fast_ids:
			if "PV" in device_id:
				self.fast_action_dim += 2  # P and Q control
			elif "Battery" in device_id:
				self.fast_action_dim += 1  # Power control
			else:
				self.fast_action_dim += 1  # Default

		logger.info(f"Device groups initialized: {len(self.slow_ids)} slow, {len(self.fast_ids)} fast")

	def reset_interval(self, seed: Optional[int] = None) -> Dict[str, Any]:
		"""Reset for a new slow interval.

		Args:
			seed: Random seed for environment reset

		Returns:
			Dictionary containing initial observations for the interval
		"""
		# Reset environment using standard interface
		if seed is not None:
			self.env.seed(seed)

		obs, share_obs, avail_actions = self.env.reset()

		# Reset tracking
		self.current_step = 0
		self.interval_data = []
		self.last_obs = obs
		self.last_share_obs = share_obs
		self.last_avail_actions = avail_actions

		# Initialize slow device states
		for device_id in self.slow_ids:
			self.slow_device_states[device_id] = 0  # Default state

		return {
			"obs": obs,
			"share_obs": share_obs,
			"avail_actions": avail_actions,
			"obs_dict": self._build_obs_dict(obs)
		}

	def apply_slow_actions(self, slow_actions: List[int]) -> float:
		"""Apply slow layer actions (discrete switches).

		Args:
			slow_actions: List of discrete actions for slow devices

		Returns:
			Total switching cost
		"""
		total_cost = 0.0

		# Track state changes and calculate costs
		for i, (device_id, action) in enumerate(zip(self.slow_ids, slow_actions)):
			old_state = self.slow_device_states.get(device_id, 0)

			if "Regulator" in device_id:
				# Tap change cost
				tap_change = abs(action - old_state)
				cost = tap_change * self.cfg.get("reward", {}).get("c_tap", 1.0)
				total_cost += cost
			elif "Capacitor" in device_id:
				# Switching cost
				if action != old_state:
					cost = self.cfg.get("reward", {}).get("c_cap", 0.5)
					total_cost += cost

			self.slow_device_states[device_id] = action

		return total_cost

	def step_fast(self, fast_actions: np.ndarray) -> Tuple[Dict[str, Any], bool]:
		"""Execute one fast step.

		Args:
			fast_actions: Continuous actions for fast devices

		Returns:
			Tuple of (step_info, interval_done)
		"""
		# Convert fast actions to environment format
		env_actions = self._build_env_actions(fast_actions)

		# Execute standard environment step
		obs, share_obs, rewards, dones, infos, avail_actions = self.env.step(env_actions)

		# Store step data
		self.interval_data.append({
			"obs": obs,
			"share_obs": share_obs,
			"actions": fast_actions,
			"rewards": rewards,
			"dones": dones,
			"infos": infos
		})

		# Update state
		self.last_obs = obs
		self.last_share_obs = share_obs
		self.last_avail_actions = avail_actions
		self.current_step += 1

		# Check if interval is complete
		interval_done = (self.current_step >= self.T)

		return {
			"obs": obs,
			"share_obs": share_obs,
			"reward": rewards,
			"info": infos[0] if isinstance(infos, list) else infos,
			"done": dones,
			"obs_dict": self._build_obs_dict(obs)
		}, interval_done

	def get_interval_summary(self) -> Dict[str, Any]:
		"""Summarize the completed interval.

		Returns:
			Dictionary containing interval summary data
		"""
		if not self.interval_data:
			return {
				"sequence": np.zeros((self.T, 10)),  # Placeholder
				"total_reward": 0.0,
				"avg_convergence": 1.0
			}

		# Extract sequence data for embedding
		sequence_data = []
		total_reward = 0.0
		convergence_count = 0

		for step_data in self.interval_data:
			# Flatten observations and actions for sequence
			obs_flat = self._flatten_obs(step_data["obs"])
			action_flat = step_data["actions"]
			reward = step_data["rewards"][0][0] if isinstance(step_data["rewards"], list) else step_data["rewards"]

			# Combine into sequence step
			step_features = np.concatenate([obs_flat, action_flat, [reward]])
			sequence_data.append(step_features)

			# Accumulate metrics
			total_reward += reward
			if step_data["infos"][0].get("convergence", True):
				convergence_count += 1

		# Pad sequence if needed
		while len(sequence_data) < self.T:
			sequence_data.append(np.zeros_like(sequence_data[0]))

		return {
			"sequence": np.array(sequence_data[:self.T], dtype=np.float32),
			"total_reward": total_reward,
			"avg_convergence": convergence_count / max(len(self.interval_data), 1)
		}

	def get_current_state(self) -> Dict[str, Any]:
		"""Get current environment state.

		Returns:
			Dictionary containing current observations and device states
		"""
		return {
			"obs": self.last_obs,
			"share_obs": self.last_share_obs,
			"avail_actions": self.last_avail_actions,
			"slow_device_states": self.slow_device_states.copy(),
			"obs_dict": self._build_obs_dict(self.last_obs) if self.last_obs is not None else {}
		}

	def _build_env_actions(self, fast_actions: np.ndarray) -> np.ndarray:
		"""Build complete action array for environment.

		Args:
			fast_actions: Actions from fast layer algorithm

		Returns:
			Complete action array for environment step
		"""
		# Create full action array
		total_devices = len(self.all_device_ids)
		env_actions = np.zeros(total_devices)

		# Set slow device actions from stored states
		for device_id in self.slow_ids:
			idx = self.device_to_idx[device_id]
			env_actions[idx] = self.slow_device_states.get(device_id, 0)

		# Set fast device actions
		fast_idx = 0
		for device_id in self.fast_ids:
			idx = self.device_to_idx[device_id]
			if "PV" in device_id:
				# PV uses two action dimensions
				if fast_idx + 1 < len(fast_actions):
					env_actions[idx] = fast_actions[fast_idx]  # Active power
					# Note: Reactive power would need special handling
					fast_idx += 2
			else:
				if fast_idx < len(fast_actions):
					env_actions[idx] = fast_actions[fast_idx]
					fast_idx += 1

		return env_actions

	def _build_obs_dict(self, obs) -> Dict[str, Any]:
		"""Build observation dictionary from raw observations.

		Args:
			obs: Raw observation from environment

		Returns:
			Dictionary mapping device IDs to observations
		"""
		obs_dict = {}

		# Handle different observation formats
		if isinstance(obs, dict):
			return obs
		elif isinstance(obs, (list, np.ndarray)):
			# Map observations to devices
			if len(obs) == len(self.all_device_ids):
				for i, device_id in enumerate(self.all_device_ids):
					obs_dict[device_id] = obs[i] if isinstance(obs[i], dict) else {"value": obs[i]}
			else:
				# Fallback: create system observation
				obs_dict["system"] = {"obs": obs}

		return obs_dict

	def _flatten_obs(self, obs) -> np.ndarray:
		"""Flatten observation for sequence processing.

		Args:
			obs: Raw observation

		Returns:
			Flattened numpy array
		"""
		if isinstance(obs, dict):
			flat_list = []
			for key in sorted(obs.keys()):
				val = obs[key]
				if isinstance(val, (list, np.ndarray)):
					flat_list.extend(np.array(val).flatten())
				elif isinstance(val, dict):
					flat_list.extend(self._flatten_obs(val))
				else:
					flat_list.append(float(val))
			return np.array(flat_list, dtype=np.float32)
		elif isinstance(obs, (list, np.ndarray)):
			return np.array(obs).flatten().astype(np.float32)
		else:
			return np.array([float(obs)], dtype=np.float32)