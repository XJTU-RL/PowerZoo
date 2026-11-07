"""
SmartGrid Stress Test Runner
SmartGrid 压力测试运行器

继承 BaseStressTestRunner，定义 SmartGrid 特定的
参数扫描范围和指标提取逻辑。
"""

import copy
import logging
from typing import Any, Dict, Optional

import numpy as np

from envs.render_common.engine.base_episode_runner import BaseEpisodeRunner
from envs.render_common.engine.base_stress_test_runner import BaseStressTestRunner
from envs.render_common.engine.episode_reader import EpisodeData
from envs.render_common.engine.inference_engine import InferenceEngine
from envs.smartgrid.render.engine.episode_runner import SmartGridEpisodeRunner

logger = logging.getLogger(__name__)


class SmartGridStressTestRunner(BaseStressTestRunner):
    """SmartGrid 压力测试运行器

    Args:
            config: SmartGridConfig 或等效字典
            inference_engine: 推理引擎
            project_root: 项目根目录
    """

    SWEEP_DEFAULTS: Dict[str, Dict[str, Any]] = {
        "load_mult": {
            "min": 0.5,
            "max": 2.0,
            "default": 1.0,
            "description": "Load multiplier (uniform scaling)",
        },
        "irradiance_scale": {
            "min": 0.1,
            "max": 1.5,
            "default": 1.0,
            "description": "Irradiance multiplier",
        },
        "lambda_init": {
            "min": 0.01,
            "max": 10.0,
            "default": 1.0,
            "description": "Initial Lagrangian lambda value",
        },
    }

    def __init__(
        self,
        config: Any,
        inference_engine: Optional[InferenceEngine] = None,
        project_root: str = "",
    ):
        super().__init__(config, inference_engine)
        self.project_root = project_root

    def _extract_metrics(self, episode_data: EpisodeData) -> Dict[str, float]:
        """从 episode 数据中提取关键指标

        Args:
                episode_data: 完整 episode 数据

        Returns:
                {指标名: 值}
        """
        metrics: Dict[str, float] = {
            "total_reward": episode_data.total_reward,
            "episode_length": float(episode_data.episode_length),
        }

        snapshots = episode_data.snapshots
        if not snapshots:
            return metrics

        # 电压统计
        v_min_vals = []
        v_max_vals = []
        v_mean_vals = []
        loss_vals = []
        violation_vals = []

        for snap in snapshots:
            circuit = snap.get("circuit", {})
            v_min = circuit.get("v_min_pu")
            v_max = circuit.get("v_max_pu")
            v_mean = circuit.get("v_mean_pu")
            loss = circuit.get("total_loss_kw")

            if isinstance(v_min, (int, float)):
                v_min_vals.append(v_min)
            if isinstance(v_max, (int, float)):
                v_max_vals.append(v_max)
            if isinstance(v_mean, (int, float)):
                v_mean_vals.append(v_mean)
            if isinstance(loss, (int, float)):
                loss_vals.append(loss)

            info = snap.get("info", {})
            viol = info.get("voltage_violations_per_bus")
            if isinstance(viol, (int, float)):
                violation_vals.append(viol)

        if v_min_vals:
            metrics["min_voltage_pu"] = float(np.min(v_min_vals))
        if v_max_vals:
            metrics["max_voltage_pu"] = float(np.max(v_max_vals))
        if v_mean_vals:
            metrics["avg_voltage_pu"] = float(np.mean(v_mean_vals))
        if loss_vals:
            metrics["avg_loss_kw"] = float(np.mean(loss_vals))
            metrics["total_loss_kwh"] = sum(
                s["circuit"]["total_loss_kw"] * s["circuit"]["step_seconds"] / 3600
                for s in snapshots
                if s.get("step", 1) > 0
            )
        if violation_vals:
            metrics["avg_violation_rate"] = float(np.mean(violation_vals))

        return metrics

    def _create_episode_runner(self, config: Any) -> BaseEpisodeRunner:
        """创建 episode runner 实例

        Args:
                config: 修改后的环境配置

        Returns:
                SmartGridEpisodeRunner 实例
        """
        return SmartGridEpisodeRunner(
            config=config,
            inference_engine=self.inference_engine,
            collect_snapshots=True,
            project_root=self.project_root,
        )

    def _apply_params_to_config(
        self,
        config: Any,
        params: Dict[str, float],
    ) -> Any:
        """将压力测试参数应用到环境配置

        Args:
                config: 环境配置副本
                params: {参数名: 值}

        Returns:
                修改后的配置
        """
        from dataclasses import asdict
        from envs.smartgrid.base_env.env_config import SmartGridConfig
        from utils.config_validation import build_dataclass

        values = (
            asdict(config)
            if isinstance(config, SmartGridConfig)
            else copy.deepcopy(config)
        )
        data = values.setdefault("data", {})
        if data.get("source", "synthetic") != "synthetic":
            raise ValueError(
                "数据倍率扫描要求合成场景；files模式请显式提供独立场景文件"
            )
        data.pop("path", None)
        data.pop("manifest_sha256", None)
        for name, value in params.items():
            if name == "load_mult":
                data["load_scale"] = float(value)
            elif name == "irradiance_scale":
                data["irradiance_scale"] = float(value)
            elif name == "lambda_init":
                values["lambda_init"] = float(value)
            else:
                raise ValueError(f"不支持的扫描参数：{name}")
        return build_dataclass(SmartGridConfig, values)
