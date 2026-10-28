"""VVC团队收益及有单位物理指标；按真实回合长度累计，训练与验证不混计。"""

import numpy as np
from common.base_logger import BaseLogger


class VVCLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args["system_name"]

    @staticmethod
    def _empty():
        return dict(
            steps=0,
            reward=0.0,
            loss_kwh=0.0,
            grid_import_kwh=0.0,
            pv_kwh=0.0,
            phase_violation_minutes=0.0,
            capacitor_operations=0.0,
            regulator_steps=0.0,
            voltage_min_pu=float("inf"),
            voltage_max_pu=float("-inf"),
        )

    def init(self, episodes):
        super().init(episodes)
        self._train = [
            self._empty() for _ in range(self.algo_args["train"]["n_rollout_threads"])
        ]
        self._train_done = []
        self._last_train = {}
        self._last_eval = {}

    @staticmethod
    def _accumulate(record, infos):
        while isinstance(infos, (list, tuple, np.ndarray)) and len(infos):
            infos = infos[0]
        if not isinstance(infos, dict):
            raise ValueError("VVC缺少真实info，不能用零指标替代")
        record["steps"] += 1
        record["reward"] += infos["reward_total"]
        for target, key in (
            ("loss_kwh", "power_loss_kw"),
            ("grid_import_kwh", "total_power_kw"),
            ("pv_kwh", "pv_power_kw"),
        ):
            record[target] += infos[key] * infos["step_seconds"] / 3600
        record["phase_violation_minutes"] += (
            infos["voltage_violation_count"] * infos["step_seconds"] / 60
        )
        record["capacitor_operations"] += infos["capacitor_ctrl"]
        record["regulator_steps"] += infos["regulator_ctrl"]
        record["voltage_min_pu"] = min(
            record["voltage_min_pu"], infos["voltage_min_pu"]
        )
        record["voltage_max_pu"] = max(
            record["voltage_max_pu"], infos["voltage_max_pu"]
        )

    def per_step(self, data):
        super().per_step(data)
        for i, info in enumerate(data[4]):
            self._accumulate(self._train[i], info)
            if np.all(data[3][i]):
                self._train_done.append(self._train[i])
                self._train[i] = self._empty()

    def _publish(self, records, prefix):
        if not records:
            return {}
        result = {key: float(np.mean([r[key] for r in records])) for key in records[0]}
        for key, value in result.items():
            self.writer.add_scalar(prefix + "/" + key, value, self.total_num_steps)
        return result

    def episode_log(self, *args):
        super().episode_log(*args)
        result = self._publish(self._train_done, "vvc/train")
        if result:
            self._last_train = result
        self._train_done = []

    def eval_init(self):
        super().eval_init()
        self._eval = [
            self._empty()
            for _ in range(self.algo_args["eval"]["n_eval_rollout_threads"])
        ]
        self._eval_done = []

    def eval_per_step(self, data):
        super().eval_per_step(data)
        for i, info in enumerate(data[4]):
            self._accumulate(self._eval[i], info)

    def eval_thread_done(self, tid):
        super().eval_thread_done(tid)
        self._eval_done.append(self._eval[tid])
        self._eval[tid] = self._empty()

    def eval_log(self, episode):
        super().eval_log(episode)
        self._last_eval = self._publish(self._eval_done, "vvc/eval")

    def get_result(self):
        return {
            "train_avg_reward": self._last_train.get("reward"),
            "eval_avg_reward": self._last_eval.get("reward"),
            "train_physics": self._last_train,
            "eval_physics": self._last_eval,
        }
