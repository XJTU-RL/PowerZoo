"""SmartGrid检查点入口复用公共物理/动作/观测/任务校验。"""

from envs.powergrid.checkpoint import load_checkpoint as _load


def load_checkpoint(checkpoint_dir, device="cpu"):
    return _load(checkpoint_dir, device, expected_family="smartgrid")
