"""Algorithm and environment metadata registry for PowerZoo Training UI.

Provides structured metadata for all 15 MARL algorithms and 8 power system
environments, organized by algorithm family. Used by UI components for
populating dropdowns, validation, and config generation.
"""

from dataclasses import dataclass, field


@dataclass
class AlgoMeta:
	"""Metadata for a single MARL algorithm."""

	name: str              # e.g. "happo"
	display_name: str      # e.g. "HAPPO"
	family: str            # "on_policy_ha" | "on_policy_ma" | "off_policy_ha" | "off_policy_ma" | "qmix"
	config_file: str       # e.g. "happo.yaml"
	description: str
	compatible_envs: list[str] = field(default_factory=list)  # empty = all envs


@dataclass
class EnvMeta:
	"""Metadata for a single training environment."""

	name: str              # e.g. "vvc"
	display_name: str
	config_file: str       # e.g. "vvc.yaml"
	description: str
	default_agents: int
	default_system_ref: str = ""
	available_systems: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Algorithm Registry (15 algorithms)
# ---------------------------------------------------------------------------

ALGO_REGISTRY: dict[str, AlgoMeta] = {
	# --- On-Policy HA Series ---
	"happo": AlgoMeta(
		name="happo",
		display_name="HAPPO",
		family="on_policy_ha",
		config_file="happo.yaml",
		description="Heterogeneous-Agent Proximal Policy Optimization",
	),
	"hatrpo": AlgoMeta(
		name="hatrpo",
		display_name="HATRPO",
		family="on_policy_ha",
		config_file="hatrpo.yaml",
		description="Heterogeneous-Agent Trust Region Policy Optimization",
	),
	"haa2c": AlgoMeta(
		name="haa2c",
		display_name="HAA2C",
		family="on_policy_ha",
		config_file="haa2c.yaml",
		description="Heterogeneous-Agent Advantage Actor-Critic",
	),
	"shom": AlgoMeta(
		name="shom",
		display_name="SHOM",
		family="on_policy_ha",
		config_file="shom.yaml",
		description="Shared Hierarchical On-policy MARL",
	),
	"sn_mappo": AlgoMeta(
		name="sn_mappo",
		display_name="SN-MAPPO",
		family="on_policy_ha",
		config_file="sn_mappo.yaml",
		description="Stackelberg Network MAPPO",
	),
	"dan_happo": AlgoMeta(
		name="dan_happo",
		display_name="DAN-HAPPO",
		family="on_policy_ha",
		config_file="dan_happo.yaml",
		description="Dynamic Attention Network HAPPO",
	),
	# --- On-Policy MA Series ---
	"mappo": AlgoMeta(
		name="mappo",
		display_name="MAPPO",
		family="on_policy_ma",
		config_file="mappo.yaml",
		description="Multi-Agent Proximal Policy Optimization",
	),
	# --- Off-Policy HA Series ---
	"haddpg": AlgoMeta(
		name="haddpg",
		display_name="HADDPG",
		family="off_policy_ha",
		config_file="haddpg.yaml",
		description="Heterogeneous-Agent Deep Deterministic Policy Gradient",
	),
	"hatd3": AlgoMeta(
		name="hatd3",
		display_name="HATD3",
		family="off_policy_ha",
		config_file="hatd3.yaml",
		description="Heterogeneous-Agent Twin Delayed DDPG",
	),
	"hasac": AlgoMeta(
		name="hasac",
		display_name="HASAC",
		family="off_policy_ha",
		config_file="hasac.yaml",
		description="Heterogeneous-Agent Soft Actor-Critic",
	),
	"had3qn": AlgoMeta(
		name="had3qn",
		display_name="HAD3QN",
		family="off_policy_ha",
		config_file="had3qn.yaml",
		description="Heterogeneous-Agent Dueling Double DQN",
	),
	# --- Off-Policy MA Series ---
	"maddpg": AlgoMeta(
		name="maddpg",
		display_name="MADDPG",
		family="off_policy_ma",
		config_file="maddpg.yaml",
		description="Multi-Agent Deep Deterministic Policy Gradient",
	),
	"matd3": AlgoMeta(
		name="matd3",
		display_name="MATD3",
		family="off_policy_ma",
		config_file="matd3.yaml",
		description="Multi-Agent Twin Delayed DDPG",
	),
	# --- QMix ---
	"qmix": AlgoMeta(
		name="qmix",
		display_name="QMIX",
		family="qmix",
		config_file="qmix.yaml",
		description="QMIX Value Decomposition",
	),
}


# ---------------------------------------------------------------------------
# Algorithm Family Grouping
# ---------------------------------------------------------------------------

ALGO_FAMILIES: dict[str, list[str]] = {
	"On-Policy HA": ["happo", "hatrpo", "haa2c", "shom", "sn_mappo", "dan_happo"],
	"On-Policy MA": ["mappo"],
	"Off-Policy HA": ["haddpg", "hatd3", "hasac", "had3qn"],
	"Off-Policy MA": ["maddpg", "matd3"],
	"QMix": ["qmix"],
}


# ---------------------------------------------------------------------------
# Environment Registry (8 environments)
# ---------------------------------------------------------------------------

ENV_REGISTRY: dict[str, EnvMeta] = {
	"vvc": EnvMeta(
		name="vvc",
		display_name="VVC (Volt-VAR Control)",
		config_file="vvc.yaml",
		description="Volt-VAR control environment based on OpenDSS simulation",
		default_agents=6,
		default_system_ref="13Bus",
		available_systems=["13Bus", "34Bus", "34Bus_PV", "123Bus"],
	),
	"smartgrid": EnvMeta(
		name="smartgrid",
		display_name="SmartGrid",
		config_file="smartgrid.yaml",
		description="Modular smart grid simulation environment",
		default_agents=12,
		default_system_ref="34Bus_PV_Aggressive",
		available_systems=[
			"13Bus", "34Bus", "34Bus_PV", "34Bus_PV_Aggressive",
			"34Bus_PV_Conservative", "34Bus_PV_Optimized", "123Bus", "8500Node",
		],
	),
	"stackelberg_13bus": EnvMeta(
		name="stackelberg_13bus",
		display_name="Stackelberg 13-Bus",
		config_file="stackelberg_13bus.yaml",
		description="13-bus Stackelberg leader-follower game environment",
		default_agents=9,
		default_system_ref="13Bus",
		available_systems=["13Bus"],
	),
	"stackelberg_34bus": EnvMeta(
		name="stackelberg_34bus",
		display_name="Stackelberg 34-Bus",
		config_file="stackelberg_34bus.yaml",
		description="34-bus Stackelberg leader-follower game environment",
		default_agents=11,
		default_system_ref="34Bus",
		available_systems=["34Bus"],
	),
	"stackelberg_123bus": EnvMeta(
		name="stackelberg_123bus",
		display_name="Stackelberg 123-Bus",
		config_file="stackelberg_123bus.yaml",
		description="123-bus large-scale Stackelberg game environment",
		default_agents=123,
		default_system_ref="123Bus",
		available_systems=["123Bus"],
	),
	"dsr": EnvMeta(
		name="dsr",
		display_name="DSR (Distribution Restoration)",
		config_file="dsr.yaml",
		description="Distribution service restoration environment",
		default_agents=10,
		default_system_ref="123Bus",
		available_systems=["123Bus", "8500Node"],
	),
	"dsr_13bus": EnvMeta(
		name="dsr_13bus",
		display_name="DSR 13-Bus",
		config_file="dsr_13bus.yaml",
		description="13-bus distribution service restoration environment",
		default_agents=8,
		default_system_ref="13Bus",
		available_systems=["13Bus"],
	),
	"dsr_8500node": EnvMeta(
		name="dsr_8500node",
		display_name="DSR 8500-Node",
		config_file="dsr_8500node.yaml",
		description="8500-node large-scale distribution restoration environment",
		default_agents=50,
		default_system_ref="8500Node",
		available_systems=["8500Node"],
	),
}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_algos_for_family(family: str) -> list[AlgoMeta]:
	"""Get all algorithm metadata objects belonging to a display family.

	Args:
		family: Display family name, e.g. "On-Policy HA".

	Returns:
		List of AlgoMeta for that family, or empty list if not found.
	"""
	names = ALGO_FAMILIES.get(family, [])
	return [ALGO_REGISTRY[n] for n in names if n in ALGO_REGISTRY]


def get_algo(name: str) -> AlgoMeta | None:
	"""Lookup a single algorithm by internal name."""
	return ALGO_REGISTRY.get(name)


def get_env(name: str) -> EnvMeta | None:
	"""Lookup a single environment by internal name."""
	return ENV_REGISTRY.get(name)


def get_family_choices() -> list[str]:
	"""Return the list of algorithm family display names."""
	return list(ALGO_FAMILIES.keys())


def get_algo_choices(family: str) -> list[str]:
	"""Return algorithm display names for a given family.

	Args:
		family: Display family name, e.g. "On-Policy HA".

	Returns:
		List of display_name strings, e.g. ["HAPPO", "HATRPO", ...].
	"""
	return [
		ALGO_REGISTRY[n].display_name
		for n in ALGO_FAMILIES.get(family, [])
		if n in ALGO_REGISTRY
	]
