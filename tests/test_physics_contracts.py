"""物理重构公共契约：不能用训练更新成功掩盖任务目标丢失。"""

from unittest.mock import Mock

import numpy as np
import pytest


def test_ep_rejects_non_shared_rewards_before_mutating_buffers():
    from runners.on_policy_base_runner import OnPolicyBaseRunner

    runner = OnPolicyBaseRunner.__new__(OnPolicyBaseRunner)
    runner.state_type = "EP"
    runner.actor_buffer = [Mock(), Mock()]
    runner.critic_buffer = Mock()
    rewards = np.array([[[1.0], [9.0]]])
    with pytest.raises(ValueError, match="共享团队奖励"):
        runner.insert(
            (None, None, rewards, None, None, None, None, None, None, None, None)
        )
    runner.critic_buffer.insert.assert_not_called()
    for buffer in runner.actor_buffer:
        buffer.insert.assert_not_called()


@pytest.mark.parametrize("family",["dsr","district_dispatch","stackelberg"])
def test_unrebuilt_physics_is_rejected_before_training_preparation(family):
    from utils.config_pipeline import ConfigLoader
    from utils.training_setup import prepare_run

    cfg = ConfigLoader().load("happo",family)
    with pytest.raises(ValueError,match="待重建研究实现"):
        prepare_run(cfg)
