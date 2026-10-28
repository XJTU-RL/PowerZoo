"""VVC快照直接使用任务的公共物理帧；旧DSS-only构造不足以确定dt。"""

from envs.powergrid.snapshot import assemble_snapshot


class SnapshotAssembler:
    def __init__(self, env):
        if not hasattr(env, "physical_state"):
            raise TypeError("请传入正式VVC任务内核；DSS-only读取不能确定数据/时钟契约")
        self.env = env

    def assemble(self, step, actions=None, rewards=None, infos=None):
        return assemble_snapshot(self.env, step, actions, rewards, infos)
