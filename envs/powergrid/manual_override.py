"""按正式设备ID和真实动作空间进行手动覆盖。"""

import numpy as np
from envs.render_common.engine.base_manual_override import BaseManualOverride


class DeviceManualOverride(BaseManualOverride):
    def __init__(self, env):
        self.spaces = env.agent_spaces
        self.schema = env.get_physics_manifest()["action_schema"]
        super().__init__(
            len(self.spaces),
            [1 if hasattr(s, "n") else s.shape[0] for s in self.spaces],
        )

    @classmethod
    def from_env(cls, env):
        return cls(env)

    def get_action_labels(self, agent_id):
        if "controls" in self.schema[agent_id]:
            return [
                c["element"] + " " + component
                for c in self.schema[agent_id]["controls"]
                for component in c["components"]
            ]
        element = self.schema[agent_id]["element"]
        if element.startswith("PVSystem.") and self.action_dims[agent_id] == 2:
            return [
                element + " available P command [-1,1]",
                element + " Q command [-1,1]",
            ]
        return [element + " " + self.schema[agent_id]["space"]]

    def set_agent_override(self, agent_id, values):
        self._validate_indices(agent_id, 0)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        space = self.spaces[agent_id]
        if len(values) != self.action_dims[agent_id] or not np.isfinite(values).all():
            raise ValueError("动作长度或数值无效")
        if hasattr(space, "n"):
            valid = values[0] == int(values[0]) and space.contains(int(values[0]))
        else:
            valid = space.contains(values)
        if not valid:
            raise ValueError(f"动作超出{self.schema[agent_id]}的{space}")
        for index, value in enumerate(values):
            self.set_override(agent_id, index, float(value))
        self.enabled = True

    def has_overrides(self):
        return self.enabled and bool(self._overrides)

    def set_override(self, agent_id, action_idx, value):
        self._validate_indices(agent_id, action_idx)
        space = self.spaces[agent_id]
        if not np.isfinite(value):
            raise ValueError("动作必须有限")
        if hasattr(space, "n"):
            valid = value == int(value) and space.contains(int(value))
        else:
            valid = space.low[action_idx] <= value <= space.high[action_idx]
        if not valid:
            raise ValueError(f"动作超出{self.schema[agent_id]}的{space}")
        super().set_override(agent_id, action_idx, value)
        self.enabled = True
