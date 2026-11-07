"""经典VVC任务：原生机械/Storage控制与独立电压、损耗、操作成本。"""

from envs.powergrid.control_env import DeviceControlEnv
from envs.vvc.vvc.vvc_config import VVCConfig


class Env(DeviceControlEnv):
    def __init__(self, folder_path, config):
        if not isinstance(config, VVCConfig):
            raise TypeError("VVC需要唯一加载器构造的VVCConfig")
        super().__init__(folder_path, config, config.overlays)

    def _reward(self, capdiff, regdiff, soc_err, dis_err, pvdiff):
        c = self.config
        # 保留旧VVC的逐母线最差相上下界越限定义；真实失电相不被过滤。
        violations = {
            name: min(0, c.voltage_max - max(values))
            + min(0, min(values) - c.voltage_min)
            for name, values in self.obs["bus_voltages"].items()
        }
        voltage = sum(violations.values())
        power = -c.power_w * min(1, max(0, self.obs["power_loss_ratio"]))
        # regdiff是整个Tap范围的比例，换成原生实际档位数后计机械操作成本。
        taps = sum(
            d * r.numtaps for d, r in zip(regdiff, self.circuit.regulators.values())
        )
        control = -(
            c.cap_w * sum(capdiff)
            + c.reg_w * taps
            + c.dis_w * sum(dis_err)
            + (c.soc_w * sum(soc_err) if self.t == self.horizon else 0)
        )
        return power + voltage + control, {
            "vol_reward": voltage,
            "powerloss_reward": power,
            "ctrl_reward": control,
            "violated_nodes": [name for name, value in violations.items() if value < 0],
        }
