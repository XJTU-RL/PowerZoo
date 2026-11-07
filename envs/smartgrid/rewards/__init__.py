"""SmartGrid CMDP 奖励计算、拉格朗日更新与可视化。

旧权重标定工具使用占位控制差异和随机策略，已退役；本包不再导出标定接口。
"""

from envs.smartgrid.rewards.vvc_reward import VVCReward
from envs.smartgrid.rewards.lagrangian import (
    LagrangianUpdater,
    MultiConstraintLagrangian,
)
from envs.smartgrid.rewards.visualize import RewardVisualizer

__all__ = [
    "VVCReward",
    "LagrangianUpdater",
    "MultiConstraintLagrangian",
    "RewardVisualizer",
]

__version__ = "2.0.0"  # CMDP 版本
