"""一个事件目录保存所有指标，避免 Windows 指标子目录超过路径长度限制。"""

import time
from tensorboardX import SummaryWriter


class ScalarWriter(SummaryWriter):
    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, **kwargs):
        walltime = time.time() if walltime is None else walltime
        super().add_scalar(tag, scalar_value, global_step, walltime, **kwargs)
        value = (
            scalar_value.item()
            if hasattr(scalar_value, "item")
            else float(scalar_value)
        )
        self.scalar_dict.setdefault(tag, []).append([walltime, global_step, value])

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        for tag, value in tag_scalar_dict.items():
            self.add_scalar(f"{main_tag}/{tag}", value, global_step, walltime)
