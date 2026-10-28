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


def test_retired_smartgrid_sources_are_recoverable_and_not_active():
    import json
    from hashlib import sha256
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "archive/physics_refactor_smartgrid/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for record in manifest["files"]:
        assert not (root / record["original"]).exists()
        assert (
            sha256((root / record["archive"]).read_bytes()).hexdigest()
            == record["sha256"]
        )


@pytest.mark.parametrize("family", ["dsr", "stackelberg"])
def test_unrebuilt_physics_is_rejected_before_training_preparation(family):
    from utils.config_pipeline import ConfigLoader
    from utils.training_setup import prepare_run

    cfg = ConfigLoader().load("happo", family)
    with pytest.raises(ValueError, match="待重建研究实现"):
        prepare_run(cfg)
