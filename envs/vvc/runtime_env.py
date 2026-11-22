"""经典VVC任务：原生机械/Storage控制与独立电压、损耗、操作成本。"""

from math import isclose

from dss import DSSException

from envs.powergrid.control_env import DeviceControlEnv
from envs.vvc.voltage_sensitivity import NativeVoltageProbe, native_reactive_matrix
from envs.vvc.vvc.vvc_config import VVCConfig


class Env(DeviceControlEnv):
    def __init__(self, folder_path, config):
        if not isinstance(config, VVCConfig):
            raise TypeError("VVC需要唯一加载器构造的VVCConfig")
        self._sensitivity_probe = None
        super().__init__(
            folder_path,
            config,
            config.overlays,
            light_state_read=getattr(config, "light_state_read", False),
            observation_mode=getattr(config, "observation_mode", "global"),
        )
        self._sensitivity_binding = (
            self.get_physics_manifest()
            if config.sensitivity_measurement != "disabled"
            else None
        )

    def _reset_task(self, seed):
        self._sensitivity_info = {}

    def _after_solve(self):
        if self._sensitivity_binding is None:
            return
        binding = self._sensitivity_binding
        schema = binding["action_schema"]
        mode = self.config.sensitivity_measurement
        try:
            if mode == "paper":
                nodes, mapping, matrix = native_reactive_matrix(self.circuit, schema)
                details = {"device_nodes": mapping}
            else:
                if self._sensitivity_probe is None:
                    self._sensitivity_probe = NativeVoltageProbe(self.circuit)
                devices = sorted(entry["element"] for entry in schema)
                nodes, matrix, probes = self._sensitivity_probe.measure(self, devices)
                details = {
                    "control_basis": "one_normalized_device_command",
                    "device_columns": devices,
                    "perturbations": probes,
                }
        except (DSSException, ValueError, RuntimeError) as exc:
            self._sensitivity_info = {"voltage_sensitivity_unavailable": str(exc)}
        else:
            self._sensitivity_info = {
                "voltage_sensitivity": {
                    "mode": mode,
                    "model_sha256": binding["model_sha256"],
                    "action_schema_sha256": binding["action_schema_sha256"],
                    "scenario_id": self.entry["id"],
                    "operating_time_s": self.circuit.time_s,
                    "time_s": self.circuit.time_s + self.circuit.step_seconds,
                    "nodes": nodes,
                    "matrix": matrix,
                    **details,
                }
            }

    def _info(self):
        info = super()._info()
        sample = self._sensitivity_info.get("voltage_sensitivity")
        if sample is not None and isclose(
            sample["time_s"], info["time_s"], rel_tol=0, abs_tol=1e-6
        ):
            info["voltage_sensitivity"] = {**sample, "time_s": info["time_s"]}
        elif sample is not None:
            info["voltage_sensitivity_unavailable"] = "SHOM 采样时钟与转移不一致"
        else:
            info.update(self._sensitivity_info)
        return info

    def close(self):
        if self._sensitivity_probe is not None:
            self._sensitivity_probe.close()
        super().close()

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
