# -*- coding: utf-8 -*-
"""
Sensitivity calculation module for multi-agent reinforcement learning.

This module provides abstract interfaces and concrete implementations for
computing agent sensitivity values, enabling decoupling between environment-specific
sensitivity calculations and the core SHOM algorithm.

Classes:
    SensitivityCalculator: Abstract base class for sensitivity calculation
    PowerGridSensitivity: Implementation for power grid environments
    UniformSensitivity: Default implementation returning uniform sensitivity
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class SensitivityCalculator(ABC):
    """
    Abstract base class for sensitivity calculation.

    This interface decouples the sensitivity computation logic from specific
    environment implementations, allowing different environments to provide
    their own sensitivity metrics.
    """

    @abstractmethod
    def compute(self, env_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute sensitivity values for each agent.

        Args:
            env_info: Dictionary containing environment-specific information
                     needed for sensitivity calculation.

        Returns:
            Dictionary mapping agent identifiers to their sensitivity values.
        """
        pass

    @abstractmethod
    def aggregate(self, sensitivity_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate sensitivity values over multiple timesteps.

        Args:
            sensitivity_history: List of sensitivity info dicts from multiple steps.

        Returns:
            Dictionary mapping agent identifiers to aggregated sensitivity values.
        """
        pass


class PowerGridSensitivity(SensitivityCalculator):
    """
    Sensitivity calculator for power grid environments.

    Computes sensitivity based on reactive power-voltage sensitivity matrix (S matrix).
    This implementation aggregates sensitivity values for buses associated with each agent.

    Args:
        agents_bus_mapping: Dictionary mapping agent names to their associated bus IDs.
    """

    def __init__(self, agents_bus_mapping: Dict[str, List[str]]):
        """
        Initialize PowerGridSensitivity calculator.

        Args:
            agents_bus_mapping: Mapping from agent names to list of bus identifiers.
                              e.g., {"Regulator.reg1": ["650.1", "650.2"], ...}
        """
        self.agents_bus = agents_bus_mapping

    def compute(self, env_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute sensitivity from environment info containing S matrix.

        Args:
            env_info: Must contain 'S' key with sensitivity matrix data.

        Returns:
            Dictionary mapping agent names to their sensitivity values.
        """
        S = env_info.get('S', {})
        if not S:
            return {}

        agent_sensitivity = {}
        for agent, buses in self.agents_bus.items():
            total_sensitivity = sum(S.get(bus, 0) for bus in buses)
            agent_sensitivity[agent] = total_sensitivity

        return agent_sensitivity

    def aggregate(self, sensitivity_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate sensitivity values from multiple timesteps.

        First aggregates raw sensitivity values by bus, then computes
        per-agent sensitivity by summing over associated buses.

        Args:
            sensitivity_history: List of info dicts, each potentially containing 'S' data.

        Returns:
            Dictionary mapping agent names to aggregated sensitivity values.
        """
        # Aggregate raw sensitivity by bus ID
        bus_sensitivity = {}
        for step_info in sensitivity_history:
            if not isinstance(step_info, dict):
                continue
            for key, value in step_info.items():
                if '.' in key:
                    # Parse bus ID format: "label.phase"
                    label, phase = key.split('.', 1)
                    full_label = f"{label}.{phase}"
                    bus_sensitivity[full_label] = bus_sensitivity.get(full_label, 0) + value

        # Compute per-agent sensitivity
        agent_sensitivity = {}
        for agent, buses in self.agents_bus.items():
            total_value = sum(bus_sensitivity.get(bus, 0) for bus in buses)
            agent_sensitivity[agent] = total_value

        return agent_sensitivity


class UniformSensitivity(SensitivityCalculator):
    """
    Default sensitivity calculator returning uniform values.

    Used when no environment-specific sensitivity information is available.
    All agents receive equal sensitivity values.
    """

    def __init__(self, num_agents: int = 0):
        """
        Initialize UniformSensitivity calculator.

        Args:
            num_agents: Number of agents (optional, for generating default mappings).
        """
        self.num_agents = num_agents

    def compute(self, env_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Return empty dict indicating uniform/default sensitivity.

        Args:
            env_info: Ignored for uniform sensitivity.

        Returns:
            Empty dictionary (signals to use default ordering).
        """
        return {}

    def aggregate(self, sensitivity_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Return empty dict for uniform sensitivity aggregation.

        Args:
            sensitivity_history: Ignored for uniform sensitivity.

        Returns:
            Empty dictionary.
        """
        return {}


def create_sensitivity_calculator(
    env_name: str,
    env_args: Dict[str, Any],
    agents_bus_mapping: Optional[Dict[str, List[str]]] = None,
    num_agents: int = 0
) -> SensitivityCalculator:
    """
    Factory function to create appropriate sensitivity calculator.

    Args:
        env_name: Name of the environment (e.g., "powerzoo").
        env_args: Environment configuration arguments.
        agents_bus_mapping: Agent-to-bus mapping (required for power grid envs).
        num_agents: Number of agents (used for uniform sensitivity).

    Returns:
        Appropriate SensitivityCalculator instance.
    """
    use_sensitivity = env_args.get("useS", False)

    if not use_sensitivity:
        return UniformSensitivity(num_agents)

    if env_name == "powerzoo" and agents_bus_mapping is not None:
        return PowerGridSensitivity(agents_bus_mapping)

    return UniformSensitivity(num_agents)
