# -*- coding: utf-8 -*-
"""
Circuit Adapter for Stackelberg Game Environment

This module provides an adapter to interface with the PowerZoo circuit module
while maintaining independence of the Stackelberg game environment.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from envs.powerzoo.powerzoo.circuit import Circuits


class StackelbergCircuitAdapter:
    """
    Adapter class to interface Stackelberg game with PowerZoo circuits.
    
    This adapter provides necessary circuit operations while maintaining
    the independence of the Stackelberg game implementation.
    """
    
    def __init__(self, circuit: Circuits):
        """
        Initialize the circuit adapter.
        
        Args:
            circuit: PowerZoo Circuits instance
        """
        self.circuit = circuit
        self._cache_system_info()
        
    def _cache_system_info(self):
        """Cache frequently used system information."""
        self.bus_names = self.circuit.get_bus_names()
        self.load_names = self.circuit.get_load_names()
        self.line_names = self.circuit.get_line_names()
        self.n_buses = len(self.bus_names)
        self.n_loads = len(self.load_names)
        self.n_lines = len(self.line_names)
        
    def get_system_state(self) -> Dict[str, Any]:
        """
        Get current system state for Stackelberg game.
        
        Returns:
            Dictionary containing system state information
        """
        state = {
            'voltages': self.circuit.get_bus_voltages(),
            'powers': self.circuit.get_bus_powers(),
            'line_flows': self.circuit.get_line_flows(),
            'total_loss': self.circuit.get_total_loss(),
            'load_powers': self.circuit.get_load_powers(),
        }
        
        # Add voltage statistics
        voltages = state['voltages']
        state['min_voltage'] = np.min(voltages)
        state['max_voltage'] = np.max(voltages)
        state['avg_voltage'] = np.mean(voltages)
        
        # Add power statistics
        powers = state['powers']
        state['total_load'] = np.sum([p for p in powers if p > 0])
        state['total_generation'] = np.sum([abs(p) for p in powers if p < 0])
        
        # Calculate voltage violations
        voltage_lower = 0.95
        voltage_upper = 1.05
        state['voltage_violations'] = np.sum(
            (voltages < voltage_lower) | (voltages > voltage_upper)
        )
        
        return state
    
    def apply_uc_actions(self, uc_actions: np.ndarray) -> Dict[str, Any]:
        """
        Apply UC actions to the circuit.
        
        Args:
            uc_actions: UC action array [price, dr_incentive, capacity, ess, der_curtail]
            
        Returns:
            Dictionary with action results
        """
        # UC actions don't directly modify circuit, but we track them
        results = {
            'price_signal': uc_actions[0],
            'dr_incentive': uc_actions[1], 
            'capacity_allocation': uc_actions[2],
            'ess_action': uc_actions[3],
            'der_curtailment': uc_actions[4]
        }
        
        # ESS action affects circuit (if ESS is modeled)
        if hasattr(self.circuit, 'set_storage_power'):
            ess_power_mw = uc_actions[3] * 0.5  # Max 0.5 MW charge/discharge
            self.circuit.set_storage_power(ess_power_mw)
            
        return results
    
    def apply_consumer_actions(self, 
                             consumer_actions: Dict[int, np.ndarray],
                             consumer_bus_mapping: Dict[int, List[str]]) -> Dict[str, Any]:
        """
        Apply consumer actions to the circuit.
        
        Args:
            consumer_actions: Dictionary of consumer actions
            consumer_bus_mapping: Mapping of consumer IDs to bus names
            
        Returns:
            Dictionary with action results
        """
        results = {
            'load_adjustments': {},
            'der_outputs': {},
            'total_dr_response': 0.0
        }
        
        for consumer_id, action in consumer_actions.items():
            load_adjustment = action[0]  # Load adjustment factor
            der_output = action[1] if len(action) > 1 else 0.0
            
            # Get buses controlled by this consumer
            buses = consumer_bus_mapping.get(consumer_id, [])
            
            for bus in buses:
                # Apply load adjustment
                if hasattr(self.circuit, 'adjust_load_at_bus'):
                    original_load = self.circuit.get_load_at_bus(bus)
                    new_load = original_load * (1 + load_adjustment)
                    self.circuit.set_load_at_bus(bus, new_load)
                    
                    dr_response = original_load - new_load
                    results['load_adjustments'][bus] = dr_response
                    results['total_dr_response'] += dr_response
                
                # Apply DER output (if DER is at this bus)
                if hasattr(self.circuit, 'set_der_output_at_bus'):
                    self.circuit.set_der_output_at_bus(bus, der_output)
                    results['der_outputs'][bus] = der_output
        
        return results
    
    def run_power_flow(self) -> bool:
        """
        Run power flow analysis.
        
        Returns:
            True if converged, False otherwise
        """
        return self.circuit.run_power_flow()
    
    def get_bus_mapping_for_consumers(self, n_consumers: int) -> Dict[int, List[str]]:
        """
        Create default bus mapping for consumers.
        
        Args:
            n_consumers: Number of consumer agents
            
        Returns:
            Dictionary mapping consumer IDs to bus names
        """
        # Exclude slack bus and distribute loads among consumers
        load_buses = [bus for bus in self.load_names if 'slack' not in bus.lower()]
        
        mapping = {}
        buses_per_consumer = len(load_buses) // n_consumers
        
        for i in range(n_consumers):
            start_idx = i * buses_per_consumer
            end_idx = start_idx + buses_per_consumer if i < n_consumers - 1 else len(load_buses)
            mapping[i] = load_buses[start_idx:end_idx]
            
        return mapping
    
    def check_n1_security(self, critical_lines: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Check N-1 security constraints.
        
        Args:
            critical_lines: List of critical lines to check
            
        Returns:
            Dictionary with N-1 security analysis results
        """
        if critical_lines is None:
            critical_lines = self.line_names[:5]  # Check first 5 lines by default
            
        results = {
            'secure': True,
            'violations': [],
            'max_flow_increase': 0.0
        }
        
        # Save current state
        original_state = self.circuit.save_state()
        
        for line in critical_lines:
            if line in self.line_names:
                # Disconnect line
                self.circuit.disconnect_line(line)
                
                # Run power flow
                converged = self.run_power_flow()
                
                if not converged:
                    results['secure'] = False
                    results['violations'].append({
                        'line': line,
                        'issue': 'non_convergence'
                    })
                else:
                    # Check for overloads
                    flows = self.circuit.get_line_flows()
                    for other_line, flow in flows.items():
                        if flow > 0.9:  # 90% loading threshold
                            results['secure'] = False
                            results['violations'].append({
                                'line': line,
                                'issue': 'overload',
                                'affected_line': other_line,
                                'loading': flow
                            })
                    
                    # Check voltage violations
                    voltages = self.circuit.get_bus_voltages()
                    voltage_violations = np.sum((voltages < 0.95) | (voltages > 1.05))
                    if voltage_violations > 0:
                        results['secure'] = False
                        results['violations'].append({
                            'line': line,
                            'issue': 'voltage_violation',
                            'count': voltage_violations
                        })
                
                # Restore line
                self.circuit.restore_state(original_state)
        
        return results
    
    def calculate_carbon_emissions(self, 
                                 generation_mix: Dict[str, float],
                                 carbon_intensities: Dict[str, float]) -> float:
        """
        Calculate total carbon emissions.
        
        Args:
            generation_mix: Dictionary of generation by type (MW)
            carbon_intensities: Carbon intensity by generation type (kg CO2/MWh)
            
        Returns:
            Total carbon emissions (kg CO2)
        """
        total_emissions = 0.0
        
        for gen_type, power_mw in generation_mix.items():
            intensity = carbon_intensities.get(gen_type, 0.5)  # Default 0.5 kg/kWh
            total_emissions += power_mw * intensity
            
        return total_emissions
    
    def reset(self):
        """Reset circuit to initial state."""
        self.circuit.reset()
        
    def get_circuit_info(self) -> Dict[str, Any]:
        """Get circuit information."""
        return {
            'n_buses': self.n_buses,
            'n_loads': self.n_loads,
            'n_lines': self.n_lines,
            'bus_names': self.bus_names,
            'load_names': self.load_names,
            'line_names': self.line_names
        }