# algorithms/twots_vvc/obs_processor.py
"""Observation processing for two-timescale VVC algorithm."""
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class TwoTSObsProcessor:
	"""Process observations for two-timescale VVC algorithm.

	This class handles observation transformation from raw environment
	observations to algorithm-specific representations for fast and slow layers.
	"""

	def __init__(self, cfg: Dict[str, Any]):
		"""Initialize observation processor.

		Args:
			cfg: Configuration dictionary
		"""
		self.cfg = cfg

		# Normalization parameters
		norm_cfg = cfg.get("normalization", {})
		self.P_base = norm_cfg.get("P_base", 1000.0)  # kW base power
		self.P_r = norm_cfg.get("P_r", 100.0)  # PV rated power kW
		self.Q_r = norm_cfg.get("Q_r", 50.0)  # PV rated reactive power kvar
		self.V_base = norm_cfg.get("V_base", 1.0)  # Per-unit voltage base
		self.V_range = norm_cfg.get("V_range", 0.05)  # Voltage normalization range

		# Time encoding parameters
		self.use_time_encoding = cfg.get("use_time_encoding", True)

		# Forecast parameters
		self.forecast_horizon = cfg.get("timescale", {}).get("T", 15)

		logger.info("TwoTSObsProcessor initialized")

	def build_fast_obs(self, raw_obs: Any, share_obs: Any,
					   device_groups: Dict[str, List[str]]) -> np.ndarray:
		"""Build fast layer observation vector.

		Args:
			raw_obs: Raw observation from environment
			share_obs: Shared observation from environment
			device_groups: Dictionary with 'fast' and 'slow' device lists

		Returns:
			Fast layer observation vector
		"""
		features = []

		# Extract system-level features
		system_features = self._extract_system_features(raw_obs, share_obs)
		features.extend(system_features)

		# Extract fast device features
		fast_devices = device_groups.get('fast', [])
		obs_dict = self._obs_to_dict(raw_obs)

		for device_id in fast_devices:
			if device_id in obs_dict:
				device_obs = obs_dict[device_id]
				device_features = self._extract_device_features(device_id, device_obs)
				features.extend(device_features)
			else:
				# Default features if device not in observation
				features.extend(self._get_default_device_features(device_id))

		return np.array(features, dtype=np.float32)

	def build_slow_obs(self, raw_obs: Any, share_obs: Any,
					   device_groups: Dict[str, List[str]],
					   forecasts: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
		"""Build slow layer observation vector.

		Args:
			raw_obs: Raw observation from environment
			share_obs: Shared observation from environment
			device_groups: Dictionary with 'fast' and 'slow' device lists
			forecasts: Optional forecast data

		Returns:
			Slow layer observation vector
		"""
		features = []

		# Add forecasts if available
		if forecasts:
			# Load forecast
			if "load" in forecasts:
				features.extend(forecasts["load"])
			else:
				# Default load forecast pattern
				features.extend(self._generate_default_forecast("load"))

			# PV generation forecast
			if "pv" in forecasts:
				features.extend(forecasts["pv"])
			else:
				# Default PV forecast pattern
				features.extend(self._generate_default_forecast("pv"))
		else:
			# No forecasts available, use default patterns
			features.extend(self._generate_default_forecast("load"))
			features.extend(self._generate_default_forecast("pv"))

		# Extract slow device states
		slow_devices = device_groups.get('slow', [])
		obs_dict = self._obs_to_dict(raw_obs)

		for device_id in slow_devices:
			if device_id in obs_dict:
				device_obs = obs_dict[device_id]
				state = self._extract_device_state(device_id, device_obs)
				features.append(state)
			else:
				features.append(0.0)  # Default state

		# Add time encoding if enabled
		if self.use_time_encoding:
			time_features = self._get_time_encoding()
			features.extend(time_features)

		return np.array(features, dtype=np.float32)

	def process_sequence(self, obs_sequence: List[np.ndarray],
						action_sequence: List[np.ndarray],
						reward_sequence: List[float]) -> np.ndarray:
		"""Process a sequence of observations for embedding.

		Args:
			obs_sequence: List of observation vectors
			action_sequence: List of action vectors
			reward_sequence: List of rewards

		Returns:
			Processed sequence array for embedding
		"""
		sequence_data = []

		for i in range(len(obs_sequence)):
			# Combine observation, action, and reward
			step_features = []

			# Add observation
			obs = obs_sequence[i] if i < len(obs_sequence) else np.zeros_like(obs_sequence[0])
			step_features.extend(obs.flatten())

			# Add action
			if i < len(action_sequence):
				action = action_sequence[i]
				step_features.extend(action.flatten())
			else:
				# Pad with zeros
				step_features.extend(np.zeros(action_sequence[0].shape[0]))

			# Add reward
			if i < len(reward_sequence):
				step_features.append(reward_sequence[i])
			else:
				step_features.append(0.0)

			sequence_data.append(step_features)

		return np.array(sequence_data, dtype=np.float32)

	def _extract_system_features(self, raw_obs: Any, share_obs: Any) -> List[float]:
		"""Extract system-level features.

		Args:
			raw_obs: Raw observation
			share_obs: Shared observation

		Returns:
			List of system features
		"""
		features = []

		# Try to extract from observation dictionary
		obs_dict = self._obs_to_dict(raw_obs)
		system_obs = obs_dict.get("system", {})

		# Extract voltage statistics
		if "v_bus" in system_obs:
			v_bus = np.array(system_obs["v_bus"])
			avg_v = np.mean(v_bus)
			min_v = np.min(v_bus)
			max_v = np.max(v_bus)
		else:
			# Default values
			avg_v = 1.0
			min_v = 0.95
			max_v = 1.05

		# Normalize voltages
		avg_v_norm = (avg_v - self.V_base) / self.V_range
		min_v_norm = (min_v - self.V_base) / self.V_range
		max_v_norm = (max_v - self.V_base) / self.V_range

		features.extend([avg_v_norm, min_v_norm, max_v_norm])

		# Extract load information
		total_load = system_obs.get("total_load_kw", 0.0)
		load_norm = total_load / self.P_base
		features.append(load_norm)

		# Extract generation information
		total_gen = system_obs.get("total_gen_kw", 0.0)
		gen_norm = total_gen / self.P_base
		features.append(gen_norm)

		return features

	def _extract_device_features(self, device_id: str, device_obs: Dict[str, Any]) -> List[float]:
		"""Extract features for a specific device.

		Args:
			device_id: Device identifier
			device_obs: Device observation dictionary

		Returns:
			List of device features
		"""
		features = []

		if "PV" in device_id:
			# PV system features
			p_kw = device_obs.get("p_kw", 0.0)
			q_kvar = device_obs.get("q_kvar", 0.0)
			v_local = device_obs.get("local_voltage", 1.0)
			q_max = device_obs.get("q_max", self.Q_r)

			# Normalize
			p_norm = p_kw / self.P_r
			q_norm = q_kvar / self.Q_r
			v_norm = (v_local - self.V_base) / self.V_range
			qmax_norm = q_max / self.Q_r

			features.extend([p_norm, q_norm, v_norm, qmax_norm])

		elif "Battery" in device_id:
			# Battery features
			soc = device_obs.get("soc", 0.5)
			power = device_obs.get("power", 0.0)
			capacity = device_obs.get("capacity", 100.0)

			# Normalize
			power_norm = power / self.P_base
			capacity_norm = capacity / self.P_base

			features.extend([soc, power_norm, capacity_norm])

		elif "Capacitor" in device_id:
			# Capacitor features
			state = device_obs.get("switch_state", 0)
			v_local = device_obs.get("local_voltage", 1.0)

			v_norm = (v_local - self.V_base) / self.V_range
			features.extend([float(state), v_norm])

		elif "Regulator" in device_id:
			# Regulator features
			tap_pos = device_obs.get("tap_position", 0)
			v_local = device_obs.get("local_voltage", 1.0)

			tap_norm = tap_pos / 16.0  # Normalize to [-1, 1]
			v_norm = (v_local - self.V_base) / self.V_range
			features.extend([tap_norm, v_norm])

		return features

	def _extract_device_state(self, device_id: str, device_obs: Dict[str, Any]) -> float:
		"""Extract normalized state for a slow device.

		Args:
			device_id: Device identifier
			device_obs: Device observation

		Returns:
			Normalized device state
		"""
		if "Regulator" in device_id:
			tap = device_obs.get("tap_position", 0)
			return tap / 16.0  # Normalize to [-1, 1]
		elif "Capacitor" in device_id:
			state = device_obs.get("switch_state", 0)
			return float(state)  # Already 0/1
		elif "Battery" in device_id:
			mode = device_obs.get("mode", 0)
			return float(mode) / 2.0  # Normalize if discrete modes
		else:
			return 0.0

	def _get_default_device_features(self, device_id: str) -> List[float]:
		"""Get default features for a device.

		Args:
			device_id: Device identifier

		Returns:
			List of default features
		"""
		if "PV" in device_id:
			return [0.0, 0.0, 1.0, 1.0]  # p, q, v, q_max normalized
		elif "Battery" in device_id:
			return [0.5, 0.0, 0.5]  # soc, power, capacity
		elif "Capacitor" in device_id:
			return [0.0, 1.0]  # state, voltage
		elif "Regulator" in device_id:
			return [0.0, 1.0]  # tap, voltage
		else:
			return [0.0]

	def _generate_default_forecast(self, forecast_type: str) -> List[float]:
		"""Generate default forecast pattern.

		Args:
			forecast_type: 'load' or 'pv'

		Returns:
			List of forecast values
		"""
		hours = np.arange(self.forecast_horizon) / 4  # Convert to hours

		if forecast_type == "load":
			# Load forecast (peaks during day)
			forecast = 0.5 + 0.3 * np.sin(2 * np.pi * hours / 24 - np.pi/2)
		elif forecast_type == "pv":
			# PV forecast (peaks at noon)
			forecast = np.maximum(0, 0.5 * np.sin(2 * np.pi * hours / 24 - np.pi/2))
		else:
			forecast = np.ones(self.forecast_horizon) * 0.5

		return forecast.tolist()

	def _get_time_encoding(self) -> List[float]:
		"""Get cyclical time encoding features.

		Returns:
			List of time encoding features
		"""
		import time
		t = time.localtime()

		hour = t.tm_hour
		day = t.tm_wday

		# Cyclical encoding
		hour_sin = np.sin(2 * np.pi * hour / 24)
		hour_cos = np.cos(2 * np.pi * hour / 24)
		day_sin = np.sin(2 * np.pi * day / 7)
		day_cos = np.cos(2 * np.pi * day / 7)

		return [hour_sin, hour_cos, day_sin, day_cos]

	def _obs_to_dict(self, obs: Any) -> Dict[str, Any]:
		"""Convert observation to dictionary format.

		Args:
			obs: Raw observation

		Returns:
			Observation dictionary
		"""
		if isinstance(obs, dict):
			return obs
		elif isinstance(obs, (list, np.ndarray)):
			# Create system observation from array
			return {"system": {"obs": obs}}
		else:
			return {"system": {"value": obs}}