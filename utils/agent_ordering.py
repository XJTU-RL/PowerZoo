# -*- coding: utf-8 -*-
"""
Agent ordering strategies for sequential multi-agent updates.

This module provides strategy pattern implementations for determining the order
in which agents are updated during training. This enables decoupling of ordering
logic from the core training runner.

Classes:
    AgentOrderStrategy: Abstract base class for ordering strategies
    SensitivityOrder: Order agents based on sensitivity values
    FixedOrder: Fixed sequential order (0, 1, 2, ...)
    RandomOrder: Random permutation each episode
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import random

# Optional imports for full functionality
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from utils.sensitivity import SensitivityCalculator


class AgentOrderStrategy(ABC):
    """
    Abstract base class for agent ordering strategies.

    Defines the interface for determining the order in which agents
    should be updated during sequential multi-agent training.
    """

    @abstractmethod
    def get_order(self, num_agents: int, **kwargs) -> List[int]:
        """
        Determine the order of agent updates.

        Args:
            num_agents: Total number of agents.
            **kwargs: Strategy-specific parameters.

        Returns:
            List of agent indices in the order they should be updated.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the ordering strategy."""
        pass


class SensitivityOrder(AgentOrderStrategy):
    """
    Order agents based on sensitivity values.

    Agents can be ordered from highest to lowest sensitivity (descending)
    or from lowest to highest (ascending).

    Args:
        sensitivity_calculator: Calculator for computing agent sensitivities.
        agent_id_mapping: Mapping from agent names to integer IDs.
        descending: If True, order from high to low sensitivity.
    """

    def __init__(
        self,
        sensitivity_calculator: SensitivityCalculator,
        agent_id_mapping: Dict[str, int],
        descending: bool = True
    ):
        """
        Initialize SensitivityOrder strategy.

        Args:
            sensitivity_calculator: Instance for computing sensitivity values.
            agent_id_mapping: Dict mapping agent names (e.g., "Regulator.reg1") to indices.
            descending: If True, agents with higher sensitivity are updated first.
        """
        self.sensitivity_calc = sensitivity_calculator
        self.agent_id_mapping = agent_id_mapping
        self.descending = descending

    def get_order(
        self,
        num_agents: int,
        sensitivity_values: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> List[int]:
        """
        Get agent order based on sensitivity values.

        Args:
            num_agents: Total number of agents.
            sensitivity_values: Pre-computed sensitivity values per agent.
                               If None, returns default sequential order.

        Returns:
            List of agent indices ordered by sensitivity.
        """
        if sensitivity_values is None or not sensitivity_values:
            # Fall back to default order if no sensitivity data
            return list(range(num_agents))

        # Sort agent names by their sensitivity values
        sorted_agents = sorted(
            self.agent_id_mapping.keys(),
            key=lambda x: sensitivity_values.get(x, 0),
            reverse=self.descending
        )

        # Convert to integer indices
        order = [self.agent_id_mapping[agent] for agent in sorted_agents]
        return order

    @property
    def name(self) -> str:
        direction = "big2small" if self.descending else "small2big"
        return f"sensitivity_{direction}"


class FixedOrder(AgentOrderStrategy):
    """
    Fixed sequential order strategy.

    Always returns agents in the same order: [0, 1, 2, ..., n-1].
    """

    def __init__(self, custom_order: Optional[List[int]] = None):
        """
        Initialize FixedOrder strategy.

        Args:
            custom_order: Optional custom fixed order. If None, uses [0, 1, ..., n-1].
        """
        self.custom_order = custom_order

    def get_order(self, num_agents: int, **kwargs) -> List[int]:
        """
        Get fixed sequential order.

        Args:
            num_agents: Total number of agents.

        Returns:
            Sequential list [0, 1, 2, ..., num_agents-1] or custom order.
        """
        if self.custom_order is not None:
            return self.custom_order[:num_agents]
        return list(range(num_agents))

    @property
    def name(self) -> str:
        return "fixed"


class RandomOrder(AgentOrderStrategy):
    """
    Random permutation order strategy.

    Returns a random permutation of agents for each call.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize RandomOrder strategy.

        Args:
            seed: Optional random seed for reproducibility.
        """
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    def get_order(self, num_agents: int, **kwargs) -> List[int]:
        """
        Get random permutation order.

        Args:
            num_agents: Total number of agents.

        Returns:
            Random permutation of [0, 1, ..., num_agents-1].
        """
        if TORCH_AVAILABLE:
            return list(torch.randperm(num_agents).numpy())
        else:
            # Fallback to pure Python random
            order = list(range(num_agents))
            random.shuffle(order)
            return order

    @property
    def name(self) -> str:
        return "random"


class AgentOrderManager:
    """
    Manager class for handling agent ordering in training.

    Combines sensitivity calculation and ordering strategy to provide
    a clean interface for the training runner.

    Args:
        strategy: The ordering strategy to use.
        sensitivity_calculator: Calculator for agent sensitivities (optional).
    """

    def __init__(
        self,
        strategy: AgentOrderStrategy,
        sensitivity_calculator: Optional[SensitivityCalculator] = None
    ):
        """
        Initialize AgentOrderManager.

        Args:
            strategy: Ordering strategy instance.
            sensitivity_calculator: Optional sensitivity calculator for strategies that need it.
        """
        self.strategy = strategy
        self.sensitivity_calc = sensitivity_calculator

    def compute_order(
        self,
        num_agents: int,
        buffer_infos: Optional[Dict] = None,
        verbose: bool = False
    ) -> List[int]:
        """
        Compute agent order for current training iteration.

        Args:
            num_agents: Total number of agents.
            buffer_infos: Buffer containing environment info with sensitivity data.
            verbose: If True, print the computed order.

        Returns:
            List of agent indices in update order.
        """
        sensitivity_values = None

        # Compute sensitivity if calculator is available and buffer has data
        if self.sensitivity_calc is not None and buffer_infos is not None:
            # Extract and flatten info data from buffer
            sensitivity_history = []
            for step_data in buffer_infos.values():
                for item in step_data:
                    if isinstance(item, dict):
                        sensitivity_history.append(item)

            sensitivity_values = self.sensitivity_calc.aggregate(sensitivity_history)

        # Get order from strategy
        order = self.strategy.get_order(
            num_agents,
            sensitivity_values=sensitivity_values
        )

        if verbose:
            print(f"{self.strategy.name}_sorted_order: {order}")

        return order

    @property
    def strategy_name(self) -> str:
        """Return the name of the current ordering strategy."""
        return self.strategy.name


def create_order_manager(
    env_name: str,
    env_args: Dict[str, Any],
    algo_args: Dict[str, Any],
    num_agents: int,
    agents_bus_mapping: Optional[Dict[str, List[str]]] = None,
    agent_id_mapping: Optional[Dict[str, int]] = None
) -> AgentOrderManager:
    """
    Factory function to create appropriate AgentOrderManager.

    Args:
        env_name: Name of the environment.
        env_args: Environment configuration.
        algo_args: Algorithm configuration.
        num_agents: Number of agents.
        agents_bus_mapping: Agent-to-bus mapping (for power grid envs).
        agent_id_mapping: Agent name to index mapping.

    Returns:
        Configured AgentOrderManager instance.
    """
    use_sensitivity = env_args.get("useS", False)
    big_to_small = env_args.get("big2small", True)
    use_ordered = algo_args.get("algo", {}).get("ordered", False)

    # Import here to avoid circular dependency
    from utils.sensitivity import (
        PowerGridSensitivity,
        UniformSensitivity,
        create_sensitivity_calculator
    )

    if not use_ordered:
        # Random order - no sensitivity needed
        return AgentOrderManager(
            strategy=RandomOrder(),
            sensitivity_calculator=None
        )

    if use_sensitivity and env_name == "powerzoo" and agents_bus_mapping and agent_id_mapping:
        # Sensitivity-based ordering for power grid
        sensitivity_calc = PowerGridSensitivity(agents_bus_mapping)
        strategy = SensitivityOrder(
            sensitivity_calculator=sensitivity_calc,
            agent_id_mapping=agent_id_mapping,
            descending=big_to_small
        )
        return AgentOrderManager(
            strategy=strategy,
            sensitivity_calculator=sensitivity_calc
        )

    # Default to fixed order
    return AgentOrderManager(
        strategy=FixedOrder(),
        sensitivity_calculator=None
    )
