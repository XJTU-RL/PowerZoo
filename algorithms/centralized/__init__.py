"""集中控制（单智能体）算法库：与 MARL 算法注册表（algorithms/actors）并列。

集中控制算法通过 `utils/single_agent/backends.py` 的后端层接入
`utils/centralized_cli.py` 入口，不进入 ALGO_REGISTRY / MARL runner
（单智能体走独立训练循环是仓库明文认可的形态，
见 docs/training-entrypoints.md）。
"""

from .ldcppo import COST_KEYS, LDCPPO, LdcpoConfig

# 集中控制算法注册表：后端层按名取用。
# LDCPPO：拉格朗日专用 critic PPO（Huang et al., IEEE TSG 2026,
# arXiv:2610.33654 式(27)-(36)；复现与结果见 research/pistg_ldcpo/
# 与 docs/pistg-repro-results.md）。
CENTRALIZED_ALGOS = {
    "ldcppo": LDCPPO,
}
