# -*- coding: utf-8 -*-
"""
Circuit Adapter for Stackelberg Game

This module provides an adapter interface between the Stackelberg game
environment and the PowerZoo circuit simulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging


class StackelbergCircuitAdapter:
	"""
	Adapter for translating Stackelberg game actions to circuit controls.
	
	Handles:
	- UC pricing signals to circuit control
	- Consumer DR actions to load adjustments
	- DER and ESS control mapping
	"""
	
	def __init__(self, circuit, config: Optional[Dict[str, Any]] = None):
		"""
		Initialize circuit adapter.
		
		Args:
			circuit: PowerZoo circuit object
			config: Configuration dictionary
		"""
		self.circuit = circuit
		self.config = config or {}
		self.logger = logging.getLogger('CircuitAdapter')
		
		# Control mappings
		self.load_control_mapping = {}
		self.der_control_mapping = {}
		self.ess_control_mapping = {}
		
		# Initialize mappings
		self._init_control_mappings()
		
	def _init_control_mappings(self):
		"""Initialize control element mappings."""
		# Map loads to control elements
		for load_name, load_obj in self.circuit.loads.items():
			self.load_control_mapping[load_name] = {
				'obj': load_obj,
				'base_kw': load_obj.feature[1] if hasattr(load_obj, 'feature') else load_obj.kW,
				'base_kvar': load_obj.feature[2] if hasattr(load_obj, 'feature') else load_obj.kvar
			}
		
		# Map DER elements (PV systems)
		if hasattr(self.circuit.dss, 'PVSystems'):
			for pv_name in self.circuit.dss.ActiveCircuit.PVSystems.AllNames:
				self.der_control_mapping[pv_name] = {
					'name': pv_name,
					'rated_kw': self._get_pv_rated_power(pv_name)
				}
		
		# Map ESS elements (Storage)
		if hasattr(self.circuit.dss, 'Storages'):
			for storage_name in self.circuit.dss.ActiveCircuit.Storages.AllNames:
				self.ess_control_mapping[storage_name] = {
					'name': storage_name,
					'rated_kw': self._get_storage_rated_power(storage_name)
				}
	
	def apply_uc_actions(self, uc_action: np.ndarray) -> Dict[str, Any]:
		"""
		Apply UC actions to circuit control elements.
		
		Args:
			uc_action: UC action vector [price_mult, dr_incentive, dr_target, ess_charge, ess_discharge]
			
		Returns:
			Control results dictionary
		"""
		results = {}
		
		# Extract UC actions
		price_multiplier = uc_action[0] if len(uc_action) > 0 else 1.0
		dr_incentive = uc_action[1] if len(uc_action) > 1 else 0.0
		dr_target = uc_action[2] if len(uc_action) > 2 else 0.0
		ess_charge = uc_action[3] if len(uc_action) > 3 else 0.0
		ess_discharge = uc_action[4] if len(uc_action) > 4 else 0.0
		
		# Apply ESS control
		if self.ess_control_mapping:
			ess_results = self._control_ess(ess_charge, ess_discharge)
			results['ess_control'] = ess_results
		
		# Store UC signals for consumer observation
		results['uc_signals'] = {
			'price_multiplier': price_multiplier,
			'dr_incentive': dr_incentive,
			'dr_target': dr_target
		}
		
		return results
	
	def apply_consumer_actions(self, 
							  consumer_actions: Dict[int, np.ndarray],
							  load_to_agent: Dict[str, int]) -> Dict[str, Any]:
		"""
		Apply consumer actions to loads.
		
		Args:
			consumer_actions: Dictionary of agent_id -> action array
			load_to_agent: Mapping from load names to agent IDs
			
		Returns:
			Control results dictionary
		"""
		results = {
			'load_adjustments': {},
			'der_control': {}
		}
		
		# Apply load adjustments
		for load_name, agent_id in load_to_agent.items():
			if agent_id in consumer_actions and load_name in self.load_control_mapping:
				action = consumer_actions[agent_id]
				load_adjustment = action[0] if len(action) > 0 else 0.0
				
				# Apply load adjustment
				adjusted_kw = self._adjust_load(load_name, load_adjustment)
				results['load_adjustments'][load_name] = {
					'agent_id': agent_id,
					'adjustment': load_adjustment,
					'new_kw': adjusted_kw
				}
		
		# Apply DER control if consumers have DER
		for agent_id, action in consumer_actions.items():
			if len(action) > 1:
				der_output = action[1]
				# Apply DER control for this agent
				# (Implementation depends on DER-agent mapping)
		
		return results
	
	def _adjust_load(self, load_name: str, adjustment: float) -> float:
		"""
		Adjust load power based on consumer action.
		
		Args:
			load_name: Name of the load
			adjustment: Adjustment factor (-0.3 to 0.1)
			
		Returns:
			New load power in kW
		"""
		if load_name not in self.load_control_mapping:
			return 0.0
		
		load_info = self.load_control_mapping[load_name]
		base_kw = load_info['base_kw']
		
		# Apply adjustment with constraints
		adjustment = np.clip(adjustment, -0.3, 0.1)
		new_kw = base_kw * (1 + adjustment)
		
		# Update circuit load
		try:
			self.circuit.dss.ActiveCircuit.SetActiveElement(f'Load.{load_name}')
			self.circuit.dss.ActiveCircuit.ActiveElement.Properties('kW').Val = new_kw
			
			# Maintain power factor
			base_kvar = load_info['base_kvar']
			if base_kw > 0:
				pf = base_kw / np.sqrt(base_kw**2 + base_kvar**2)
				new_kvar = new_kw * np.sqrt(1/pf**2 - 1) if pf < 1 else 0
				self.circuit.dss.ActiveCircuit.ActiveElement.Properties('kvar').Val = new_kvar
				
		except Exception as e:
			self.logger.warning(f"Failed to adjust load {load_name}: {e}")
		
		return new_kw
	
	def _control_ess(self, charge_action: float, discharge_action: float) -> Dict[str, Any]:
		"""
		Control ESS charging/discharging.
		
		Args:
			charge_action: Charging action (0-1)
			discharge_action: Discharging action (0-1)
			
		Returns:
			ESS control results
		"""
		results = {}
		
		# Ensure charge and discharge don't happen simultaneously
		if charge_action > 0 and discharge_action > 0:
			discharge_action = 0.0
		
		for storage_name, storage_info in self.ess_control_mapping.items():
			try:
				self.circuit.dss.ActiveCircuit.SetActiveElement(f'Storage.{storage_name}')
				
				if charge_action > 0:
					# Set to charging mode
					charge_kw = charge_action * storage_info['rated_kw']
					self.circuit.dss.ActiveCircuit.ActiveElement.Properties('State').Val = 'Charging'
					self.circuit.dss.ActiveCircuit.ActiveElement.Properties('kW').Val = charge_kw
					results[storage_name] = {'mode': 'charging', 'kW': charge_kw}
					
				elif discharge_action > 0:
					# Set to discharging mode
					discharge_kw = discharge_action * storage_info['rated_kw']
					self.circuit.dss.ActiveCircuit.ActiveElement.Properties('State').Val = 'Discharging'
					self.circuit.dss.ActiveCircuit.ActiveElement.Properties('kW').Val = discharge_kw
					results[storage_name] = {'mode': 'discharging', 'kW': discharge_kw}
					
				else:
					# Set to idle mode
					self.circuit.dss.ActiveCircuit.ActiveElement.Properties('State').Val = 'Idling'
					results[storage_name] = {'mode': 'idling', 'kW': 0}
					
			except Exception as e:
				self.logger.warning(f"Failed to control ESS {storage_name}: {e}")
		
		return results
	
	def _get_pv_rated_power(self, pv_name: str) -> float:
		"""Get rated power of PV system."""
		try:
			self.circuit.dss.ActiveCircuit.SetActiveElement(f'PVSystem.{pv_name}')
			return float(self.circuit.dss.ActiveCircuit.ActiveElement.Properties('Pmpp').Val)
		except:
			return 0.0
	
	def _get_storage_rated_power(self, storage_name: str) -> float:
		"""Get rated power of storage system."""
		try:
			self.circuit.dss.ActiveCircuit.SetActiveElement(f'Storage.{storage_name}')
			return float(self.circuit.dss.ActiveCircuit.ActiveElement.Properties('kWrated').Val)
		except:
			return 0.0
	
	def get_circuit_state(self) -> Dict[str, Any]:
		"""Get current circuit state for observations."""
		state = {
			'bus_voltages': {},
			'line_flows': {},
			'load_powers': {},
			'der_outputs': {},
			'ess_states': {}
		}
		
		# Get bus voltages
		for bus_name in self.circuit.dss.ActiveCircuit.AllBusNames:
			state['bus_voltages'][bus_name] = self.circuit.bus_voltage(bus_name)
		
		# Get load powers
		for load_name, load_info in self.load_control_mapping.items():
			try:
				self.circuit.dss.ActiveCircuit.SetActiveElement(f'Load.{load_name}')
				kw = float(self.circuit.dss.ActiveCircuit.ActiveElement.Properties('kW').Val)
				kvar = float(self.circuit.dss.ActiveCircuit.ActiveElement.Properties('kvar').Val)
				state['load_powers'][load_name] = {'kW': kw, 'kvar': kvar}
			except:
				pass
		
		# Get DER outputs
		for pv_name in self.der_control_mapping:
			try:
				self.circuit.dss.ActiveCircuit.SetActiveElement(f'PVSystem.{pv_name}')
				power = self.circuit.dss.ActiveCircuit.ActiveElement.Powers
				if len(power) >= 2:
					state['der_outputs'][pv_name] = {'kW': -power[0], 'kvar': -power[1]}
			except:
				pass
		
		# Get ESS states
		for storage_name in self.ess_control_mapping:
			try:
				self.circuit.dss.ActiveCircuit.SetActiveElement(f'Storage.{storage_name}')
				soc = float(self.circuit.dss.ActiveCircuit.ActiveElement.Properties('%stored').Val) / 100.0
				state['ess_states'][storage_name] = {
					'soc': soc,
					'state': self.circuit.dss.ActiveCircuit.ActiveElement.Properties('State').Val
				}
			except:
				pass
		
		return state
	
	def reset_controls(self):
		"""Reset all control elements to default states."""
		# Reset loads to base values
		for load_name, load_info in self.load_control_mapping.items():
			try:
				self.circuit.dss.ActiveCircuit.SetActiveElement(f'Load.{load_name}')
				self.circuit.dss.ActiveCircuit.ActiveElement.Properties('kW').Val = load_info['base_kw']
				self.circuit.dss.ActiveCircuit.ActiveElement.Properties('kvar').Val = load_info['base_kvar']
			except:
				pass
		
		# Reset ESS to idle
		for storage_name in self.ess_control_mapping:
			try:
				self.circuit.dss.ActiveCircuit.SetActiveElement(f'Storage.{storage_name}')
				self.circuit.dss.ActiveCircuit.ActiveElement.Properties('State').Val = 'Idling'
			except:
				pass