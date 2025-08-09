# -*- coding: utf-8 -*-
"""
Intelligent Load Aggregator for Stackelberg Game Environment

This module implements various load aggregation strategies to map
physical loads to consumer agents in the power system.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
from sklearn.cluster import KMeans, SpectralClustering
from scipy.spatial.distance import cdist
import logging


class IntelligentLoadAggregator:
    """
    Implements intelligent load aggregation for multi-agent power systems.
    
    Supports multiple aggregation methods:
    - Zone-based: Geographic/electrical proximity
    - Priority-based: Load importance and criticality
    - Graph-based: Network topology aware clustering
    - Adaptive: Dynamic re-aggregation based on system state
    """
    
    def __init__(self, 
                 circuit,
                 method: str = 'zone',
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the load aggregator.
        
        Args:
            circuit: Circuit object from OpenDSS
            method: Aggregation method ('zone', 'priority', 'graph', 'adaptive')
            config: Configuration dictionary
        """
        self.circuit = circuit
        self.method = method
        self.config = config or {}
        
        # Aggregation results
        self.load_to_agent = {}
        self.agent_to_loads = defaultdict(list)
        self.agent_characteristics = {}
        
        # Load information
        self.loads = dict(circuit.loads)
        self.load_buses = {name: load.bus1 for name, load in self.loads.items()}
        self.load_features = self._extract_load_features()
        
        # Network topology
        self.topology = circuit.topology.copy()
        
        # Logging
        self.logger = logging.getLogger('LoadAggregator')
        
        # Method-specific initialization
        self._init_method_specific()
    
    def _init_method_specific(self):
        """Initialize method-specific parameters."""
        if self.method == 'zone':
            self.zone_definitions = self.config.get('zone_definitions', None)
            self.use_electrical_distance = self.config.get('use_electrical_distance', True)
        elif self.method == 'priority':
            self.priority_levels = self.config.get('priority_levels', 3)
            self.priority_weights = self.config.get('priority_weights', {
                'residential': 1.0,
                'commercial': 2.0,
                'industrial': 3.0,
                'critical': 10.0
            })
        elif self.method == 'graph':
            self.spectral_clusters = self.config.get('spectral_clusters', True)
            self.consider_power_flow = self.config.get('consider_power_flow', True)
        elif self.method == 'adaptive':
            self.adaptation_interval = self.config.get('adaptation_interval', 100)
            self.adaptation_threshold = self.config.get('adaptation_threshold', 0.1)
    
    def aggregate_loads(self, n_agents: int) -> Dict[str, int]:
        """
        Aggregate loads into specified number of agents.
        
        Args:
            n_agents: Number of consumer agents
            
        Returns:
            Mapping from load name to agent ID
        """
        if len(self.loads) == 0:
            self.logger.warning("No loads found in the circuit")
            return {}
        
        if n_agents <= 0:
            raise ValueError(f"Invalid number of agents: {n_agents}")
        
        if n_agents >= len(self.loads):
            # One agent per load
            self.logger.info(f"Assigning one agent per load ({len(self.loads)} loads)")
            return self._one_to_one_mapping()
        
        # Apply selected aggregation method
        if self.method == 'zone':
            mapping = self._zone_based_aggregation(n_agents)
        elif self.method == 'priority':
            mapping = self._priority_based_aggregation(n_agents)
        elif self.method == 'graph':
            mapping = self._graph_based_aggregation(n_agents)
        elif self.method == 'adaptive':
            mapping = self._adaptive_aggregation(n_agents)
        else:
            self.logger.warning(f"Unknown method {self.method}, using zone-based")
            mapping = self._zone_based_aggregation(n_agents)
        
        # Store results
        self.load_to_agent = mapping
        self._build_agent_to_loads()
        self._compute_agent_characteristics()
        
        # Log aggregation summary
        self._log_aggregation_summary(n_agents)
        
        return mapping
    
    def _extract_load_features(self) -> Dict[str, Dict[str, Any]]:
        """Extract features for each load."""
        features = {}
        
        for name, load in self.loads.items():
            features[name] = {
                'bus': load.bus1,
                'kw': load.feature[1] if hasattr(load, 'feature') else load.kW,
                'kvar': load.feature[2] if hasattr(load, 'feature') else load.kvar,
                'phases': load.phases if hasattr(load, 'phases') else 3,
                'conn': getattr(load, 'conn', 'wye'),
                'class': self._classify_load(name, load)
            }
        
        return features
    
    def _classify_load(self, name: str, load) -> str:
        """Classify load type based on name and characteristics."""
        name_lower = name.lower()
        
        # Simple classification based on naming convention
        if 'res' in name_lower or 'home' in name_lower:
            return 'residential'
        elif 'com' in name_lower or 'shop' in name_lower or 'office' in name_lower:
            return 'commercial'
        elif 'ind' in name_lower or 'factory' in name_lower:
            return 'industrial'
        elif 'hosp' in name_lower or 'critical' in name_lower or 'emerg' in name_lower:
            return 'critical'
        
        # Classification based on load size
        kw = load.feature[1] if hasattr(load, 'feature') else load.kW
        if kw < 50:
            return 'residential'
        elif kw < 500:
            return 'commercial'
        else:
            return 'industrial'
    
    def _one_to_one_mapping(self) -> Dict[str, int]:
        """Create one-to-one mapping of loads to agents."""
        mapping = {}
        for i, load_name in enumerate(sorted(self.loads.keys())):
            mapping[load_name] = i + 1  # Agent IDs start from 1
        return mapping
    
    def _zone_based_aggregation(self, n_agents: int) -> Dict[str, int]:
        """
        Aggregate loads based on electrical zones.
        
        Uses electrical distance or geographic proximity to cluster loads.
        """
        self.logger.info(f"Performing zone-based aggregation with {n_agents} agents")
        
        # Get distance matrix
        if self.use_electrical_distance:
            distances = self._compute_electrical_distances()
        else:
            distances = self._compute_geographic_distances()
        
        # Perform clustering
        if distances is not None and distances.shape[0] > 0:
            # Use K-means on distance matrix
            clustering = KMeans(n_clusters=n_agents, random_state=42)
            
            # Convert distance matrix to feature matrix for clustering
            # Using MDS-like approach
            features = self._distance_to_features(distances)
            labels = clustering.fit_predict(features)
            
            # Create mapping
            load_names = sorted(self.loads.keys())
            mapping = {}
            for i, load_name in enumerate(load_names):
                mapping[load_name] = int(labels[i]) + 1  # Agent IDs start from 1
            
            return mapping
        else:
            # Fallback to round-robin
            return self._round_robin_mapping(n_agents)
    
    def _priority_based_aggregation(self, n_agents: int) -> Dict[str, int]:
        """
        Aggregate loads based on priority levels.
        
        Critical loads get dedicated agents, others are grouped.
        """
        self.logger.info(f"Performing priority-based aggregation with {n_agents} agents")
        
        # Categorize loads by priority
        priority_groups = defaultdict(list)
        for load_name, features in self.load_features.items():
            load_class = features['class']
            priority = self.priority_weights.get(load_class, 1.0)
            priority_groups[priority].append(load_name)
        
        # Sort priority levels (highest first)
        sorted_priorities = sorted(priority_groups.keys(), reverse=True)
        
        # Allocate agents to priority groups
        mapping = {}
        agent_id = 1
        
        for priority in sorted_priorities:
            loads_in_group = priority_groups[priority]
            
            if priority >= 10.0:  # Critical loads
                # Dedicate one agent per critical load if possible
                for load_name in loads_in_group:
                    if agent_id <= n_agents:
                        mapping[load_name] = agent_id
                        agent_id += 1
                    else:
                        # Group remaining critical loads
                        mapping[load_name] = n_agents
            else:
                # Distribute non-critical loads among remaining agents
                remaining_agents = n_agents - agent_id + 1
                if remaining_agents > 0:
                    loads_per_agent = max(1, len(loads_in_group) // remaining_agents)
                    
                    for i, load_name in enumerate(loads_in_group):
                        assigned_agent = min(
                            agent_id + i // loads_per_agent,
                            n_agents
                        )
                        mapping[load_name] = assigned_agent
                else:
                    # All agents used, assign to last agent
                    for load_name in loads_in_group:
                        mapping[load_name] = n_agents
        
        return mapping
    
    def _graph_based_aggregation(self, n_agents: int) -> Dict[str, int]:
        """
        Aggregate loads based on network topology.
        
        Uses graph clustering algorithms to group electrically connected loads.
        """
        self.logger.info(f"Performing graph-based aggregation with {n_agents} agents")
        
        # Build load connectivity graph
        load_graph = self._build_load_graph()
        
        if len(load_graph.nodes()) == 0:
            return self._round_robin_mapping(n_agents)
        
        # Apply spectral clustering if requested
        if self.spectral_clusters and len(load_graph.nodes()) > n_agents:
            # Get adjacency matrix
            adjacency = nx.adjacency_matrix(load_graph)
            
            # Perform spectral clustering
            clustering = SpectralClustering(
                n_clusters=n_agents,
                affinity='precomputed',
                random_state=42
            )
            
            # Create similarity matrix from adjacency
            similarity = adjacency.toarray()
            if self.consider_power_flow:
                # Weight by load sizes
                load_weights = self._get_load_weights_vector(load_graph.nodes())
                similarity = similarity * np.outer(load_weights, load_weights)
            
            labels = clustering.fit_predict(similarity)
            
            # Create mapping
            mapping = {}
            for i, load_name in enumerate(load_graph.nodes()):
                mapping[load_name] = int(labels[i]) + 1
            
            return mapping
        else:
            # Use community detection for smaller graphs
            return self._community_based_aggregation(load_graph, n_agents)
    
    def _adaptive_aggregation(self, n_agents: int) -> Dict[str, int]:
        """
        Adaptive aggregation that can change based on system state.
        
        Initially uses zone-based, then adapts based on load patterns.
        """
        self.logger.info(f"Performing adaptive aggregation with {n_agents} agents")
        
        # Start with zone-based aggregation
        initial_mapping = self._zone_based_aggregation(n_agents)
        
        # Store initial mapping with adaptation capability
        self.adaptive_state = {
            'current_mapping': initial_mapping,
            'adaptation_count': 0,
            'load_history': defaultdict(list),
            'imbalance_history': []
        }
        
        return initial_mapping
    
    def adapt_aggregation(self, 
                         current_loads: Dict[str, float],
                         n_agents: int) -> Optional[Dict[str, int]]:
        """
        Adapt aggregation based on current system state.
        
        Args:
            current_loads: Current load values
            n_agents: Number of agents
            
        Returns:
            New mapping if adaptation occurred, None otherwise
        """
        if not hasattr(self, 'adaptive_state'):
            return None
        
        # Update load history
        for load_name, value in current_loads.items():
            self.adaptive_state['load_history'][load_name].append(value)
        
        # Check if adaptation is needed
        if self._should_adapt():
            self.logger.info("Adapting load aggregation based on system state")
            
            # Compute new aggregation based on load patterns
            new_mapping = self._compute_adaptive_mapping(n_agents)
            
            # Update state
            self.adaptive_state['current_mapping'] = new_mapping
            self.adaptive_state['adaptation_count'] += 1
            
            # Update internal mappings
            self.load_to_agent = new_mapping
            self._build_agent_to_loads()
            self._compute_agent_characteristics()
            
            return new_mapping
        
        return None
    
    def _should_adapt(self) -> bool:
        """Determine if aggregation should be adapted."""
        if not self.adaptive_state['load_history']:
            return False
        
        # Check load imbalance among agents
        agent_loads = defaultdict(float)
        for load_name, agent_id in self.adaptive_state['current_mapping'].items():
            if load_name in self.adaptive_state['load_history']:
                recent_loads = self.adaptive_state['load_history'][load_name][-10:]
                avg_load = np.mean(recent_loads) if recent_loads else 0
                agent_loads[agent_id] += avg_load
        
        if not agent_loads:
            return False
        
        # Calculate imbalance
        loads = list(agent_loads.values())
        if len(loads) > 1:
            imbalance = np.std(loads) / (np.mean(loads) + 1e-6)
            self.adaptive_state['imbalance_history'].append(imbalance)
            
            # Adapt if imbalance exceeds threshold
            return imbalance > self.adaptation_threshold
        
        return False
    
    def _compute_adaptive_mapping(self, n_agents: int) -> Dict[str, int]:
        """Compute new mapping based on load patterns."""
        # Get average loads
        avg_loads = {}
        for load_name, history in self.adaptive_state['load_history'].items():
            if history:
                avg_loads[load_name] = np.mean(history[-20:])  # Recent average
            else:
                avg_loads[load_name] = 0
        
        # Sort loads by average consumption
        sorted_loads = sorted(avg_loads.items(), key=lambda x: x[1], reverse=True)
        
        # Distribute loads to balance agent totals
        agent_totals = [0.0] * n_agents
        mapping = {}
        
        for load_name, avg_load in sorted_loads:
            # Assign to agent with minimum current total
            min_agent = np.argmin(agent_totals)
            mapping[load_name] = min_agent + 1  # Agent IDs start from 1
            agent_totals[min_agent] += avg_load
        
        return mapping
    
    def _compute_electrical_distances(self) -> Optional[np.ndarray]:
        """Compute electrical distances between loads."""
        load_names = sorted(self.loads.keys())
        n_loads = len(load_names)
        
        if n_loads == 0:
            return None
        
        # Build bus-to-bus shortest path distances
        bus_distances = dict(nx.shortest_path_length(self.topology))
        
        # Create distance matrix
        distances = np.zeros((n_loads, n_loads))
        
        for i, load1 in enumerate(load_names):
            bus1 = self.load_buses[load1].split('.')[0].lower()
            for j, load2 in enumerate(load_names):
                if i == j:
                    distances[i, j] = 0
                else:
                    bus2 = self.load_buses[load2].split('.')[0].lower()
                    
                    # Try to get distance
                    if bus1 in bus_distances and bus2 in bus_distances[bus1]:
                        distances[i, j] = bus_distances[bus1][bus2]
                    else:
                        # Set large distance if not connected
                        distances[i, j] = 1000
        
        return distances
    
    def _compute_geographic_distances(self) -> Optional[np.ndarray]:
        """Compute geographic distances between loads (if coordinates available)."""
        # This would use actual geographic coordinates if available
        # For now, fallback to electrical distances
        return self._compute_electrical_distances()
    
    def _distance_to_features(self, distances: np.ndarray) -> np.ndarray:
        """Convert distance matrix to feature matrix for clustering."""
        # Use MDS-like approach to embed distances in feature space
        n = distances.shape[0]
        
        # Double centering
        row_mean = distances.mean(axis=1, keepdims=True)
        col_mean = distances.mean(axis=0, keepdims=True)
        total_mean = distances.mean()
        
        B = -0.5 * (distances - row_mean - col_mean + total_mean)
        
        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(B)
        
        # Take top k components
        k = min(10, n - 1)
        idx = np.argsort(eigvals)[::-1][:k]
        
        features = eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 0))
        
        return features
    
    def _build_load_graph(self) -> nx.Graph:
        """Build graph of load connectivity."""
        load_graph = nx.Graph()
        
        # Add all loads as nodes
        for load_name in self.loads:
            load_graph.add_node(load_name)
        
        # Add edges based on electrical connectivity
        for load1 in self.loads:
            bus1 = self.load_buses[load1].split('.')[0].lower()
            for load2 in self.loads:
                if load1 < load2:  # Avoid duplicates
                    bus2 = self.load_buses[load2].split('.')[0].lower()
                    
                    # Check if buses are connected
                    if nx.has_path(self.topology, bus1, bus2):
                        distance = nx.shortest_path_length(self.topology, bus1, bus2)
                        if distance <= 3:  # Only nearby loads
                            load_graph.add_edge(load1, load2, weight=1.0/distance)
        
        return load_graph
    
    def _get_load_weights_vector(self, load_names) -> np.ndarray:
        """Get load size weights as vector."""
        weights = []
        for load_name in load_names:
            if load_name in self.load_features:
                weights.append(self.load_features[load_name]['kw'])
            else:
                weights.append(1.0)
        
        weights = np.array(weights)
        return weights / (weights.max() + 1e-6)  # Normalize
    
    def _community_based_aggregation(self, 
                                   load_graph: nx.Graph,
                                   n_agents: int) -> Dict[str, int]:
        """Use community detection for graph-based aggregation."""
        # Try to import community detection
        try:
            import community as community_louvain
            
            # Detect communities
            partition = community_louvain.best_partition(load_graph)
            
            # Map communities to agents
            communities = defaultdict(list)
            for node, comm in partition.items():
                communities[comm].append(node)
            
            # Assign agents to communities
            mapping = {}
            agent_id = 1
            
            for comm_id, nodes in sorted(communities.items()):
                for node in nodes:
                    mapping[node] = min(agent_id, n_agents)
                agent_id += 1
            
            return mapping
            
        except ImportError:
            # Fallback to simple connected components
            return self._connected_components_aggregation(load_graph, n_agents)
    
    def _connected_components_aggregation(self,
                                        load_graph: nx.Graph,
                                        n_agents: int) -> Dict[str, int]:
        """Aggregate based on connected components."""
        components = list(nx.connected_components(load_graph))
        
        mapping = {}
        agent_id = 1
        
        for component in components:
            for node in component:
                mapping[node] = min(agent_id, n_agents)
            agent_id += 1
        
        return mapping
    
    def _round_robin_mapping(self, n_agents: int) -> Dict[str, int]:
        """Simple round-robin assignment."""
        mapping = {}
        load_names = sorted(self.loads.keys())
        
        for i, load_name in enumerate(load_names):
            mapping[load_name] = (i % n_agents) + 1
        
        return mapping
    
    def _build_agent_to_loads(self):
        """Build reverse mapping from agents to loads."""
        self.agent_to_loads.clear()
        
        for load_name, agent_id in self.load_to_agent.items():
            self.agent_to_loads[agent_id].append(load_name)
    
    def _compute_agent_characteristics(self):
        """Compute aggregate characteristics for each agent."""
        self.agent_characteristics.clear()
        
        for agent_id, load_list in self.agent_to_loads.items():
            total_kw = sum(
                self.load_features[load]['kw'] 
                for load in load_list
            )
            total_kvar = sum(
                self.load_features[load]['kvar']
                for load in load_list
            )
            
            load_classes = [
                self.load_features[load]['class']
                for load in load_list
            ]
            
            # Determine dominant class
            class_counts = defaultdict(int)
            for lc in load_classes:
                class_counts[lc] += 1
            dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
            
            self.agent_characteristics[agent_id] = {
                'total_kw': total_kw,
                'total_kvar': total_kvar,
                'n_loads': len(load_list),
                'dominant_class': dominant_class,
                'load_diversity': len(set(load_classes)) / len(load_classes)
            }
    
    def _log_aggregation_summary(self, n_agents: int):
        """Log summary of aggregation results."""
        self.logger.info(f"Aggregation complete: {len(self.loads)} loads -> {n_agents} agents")
        self.logger.info(f"Method: {self.method}")
        
        # Log agent statistics
        for agent_id in sorted(self.agent_to_loads.keys()):
            if agent_id in self.agent_characteristics:
                char = self.agent_characteristics[agent_id]
                self.logger.info(
                    f"Agent {agent_id}: {char['n_loads']} loads, "
                    f"{char['total_kw']:.1f} kW, "
                    f"class: {char['dominant_class']}"
                )
    
    def get_agent_load_profile(self, agent_id: int) -> Dict[str, Any]:
        """Get aggregated load profile for an agent."""
        if agent_id not in self.agent_to_loads:
            return {}
        
        profile = {
            'loads': self.agent_to_loads[agent_id],
            'characteristics': self.agent_characteristics.get(agent_id, {}),
            'buses': [
                self.load_buses[load]
                for load in self.agent_to_loads[agent_id]
            ]
        }
        
        return profile
    
    def visualize_aggregation(self, save_path: Optional[str] = None):
        """Visualize the load aggregation (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            # Create a figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get unique agent IDs and assign colors
            agent_ids = sorted(set(self.load_to_agent.values()))
            colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(agent_ids)))
            color_map = {aid: colors[i] for i, aid in enumerate(agent_ids)}
            
            # Plot topology with colored nodes
            pos = nx.spring_layout(self.topology, k=2, iterations=50)
            
            # Draw edges
            nx.draw_networkx_edges(self.topology, pos, alpha=0.2)
            
            # Draw nodes colored by agent
            for agent_id in agent_ids:
                agent_loads = self.agent_to_loads[agent_id]
                agent_buses = set()
                
                for load in agent_loads:
                    bus = self.load_buses[load].split('.')[0].lower()
                    agent_buses.add(bus)
                
                nx.draw_networkx_nodes(
                    self.topology,
                    pos,
                    nodelist=list(agent_buses),
                    node_color=[color_map[agent_id]],
                    node_size=100,
                    label=f'Agent {agent_id}'
                )
            
            # Add legend
            ax.legend(loc='upper right')
            ax.set_title(f'Load Aggregation Visualization ({self.method} method)')
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            self.logger.warning("Matplotlib not available for visualization")