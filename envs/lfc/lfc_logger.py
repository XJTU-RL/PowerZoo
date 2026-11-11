# -*- coding: utf-8 -*-
"""LFC 任务 logger。

说明：off-policy 路径（ohasac/hasac）的指标直接经 runner 的 tensorboard
writer 记录，logger 主要服务 on-policy 入径与目录命名；任务名取 Simulink
基础模型名（论文 ``LFC_Logger.get_task_name`` 同语义）。
"""

from common.base_logger import BaseLogger


class LFCLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args["env_name"]
