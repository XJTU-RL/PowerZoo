"""实际日志生命周期、单位、JSON及多worker信息聚合的断言。"""

import json
import logging
from uuid import uuid4

import numpy as np
import pytest
import yaml

from envs.smartgrid.logging.base_logger import UnifiedLogger
from envs.smartgrid.logging.smartgrid_logger import SmartGridLogger
from envs.smartgrid.logging.unified_logger import UnifiedLogManager
from envs.smartgrid.logging.visualization_manager import VisualizationManager


@pytest.mark.parametrize(
    "method", ["info", "train_info", "reward_debug", "action_debug"]
)
def test_file_logger_flush_and_windows_release(tmp_path, method):
    name = "test-" + uuid4().hex
    path = tmp_path / "logs/test.log"
    logger = UnifiedLogger.get_logger(name, str(path), level=logging.DEBUG)
    try:
        assert logger is UnifiedLogger.get_logger(name)
        getattr(logger, method)("原生SVG量测 -37 kvar")
        for handler in logger.handlers:
            handler.flush()
        assert "原生SVG量测 -37 kvar" in path.read_text(encoding="utf-8")
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        UnifiedLogger._instances.pop(name)
    path.unlink()  # Windows下仍持文件句柄会在这里失败


def test_log_manager_uses_existing_run_and_records_config(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    manager = UnifiedLogManager(
        "smartgrid",
        "34Bus_PV",
        "happo",
        "test",
        7,
        base_dir=str(tmp_path),
        existing_run_dir=str(run),
        config={"seed": 7},
    )
    assert manager.run_dir == run
    assert manager.get_log_file("training.log") == run / "logs/training.log"
    cfg = yaml.safe_load((run / "training_config.yaml").read_text())
    assert cfg["meta"]["seed"] == 7 and cfg["config"] == {"seed": 7}


@pytest.mark.parametrize(
    "infos,expected",
    [
        ([[{"x": 1}, {"x": 1}], [{"x": 3}, {"x": 3}]], [1, 3]),
        ([{"x": 2}, {}], [2, -1]),
        ([], [-1]),
    ],
)
def test_worker_info_is_counted_once_without_fake_base_initialization(infos, expected):
    logger = SmartGridLogger.__new__(SmartGridLogger)
    np.testing.assert_array_equal(logger._extract_info_value(infos, "x", -1), expected)


def test_visualization_buffer_has_actual_units_and_numpy_json(tmp_path):
    manager = VisualizationManager(str(tmp_path), plot_interval=10000, buffer_size=2)
    for value in (1, 2, 3):
        manager.update(
            {
                "reward_main": value,
                "cost_voltage": 0.01,
                "lambda": np.float32(2),
                "voltage_violation_rate": 0.02,
                "reward_total": value - 0.02,
            }
        )
    assert manager.step_count == 3
    assert list(manager.data_buffers["reward_main"]) == [2, 3]
    assert list(manager.data_buffers["voltage_violation_rate"]) == [2, 2]  # 百分数
    assert manager._calculate_statistics()["reward_main"]["mean"] == 2.5
    manager.update_episode_end(np.float32(2), np.float32(0.01), np.float32(2))
    manager._save_statistics()
    saved = json.loads((tmp_path / "statistics_ep1.json").read_text())
    assert saved["episode_stats"][0]["reward"] == 2
    assert saved["buffer_stats"]["reward_main"]["mean"] == 2.5


def test_disabled_visualization_does_not_collect(tmp_path):
    manager = VisualizationManager(str(tmp_path), enable_plotting=False)
    manager.update({"reward_main": 1})
    assert manager.step_count == 0
    assert not any(manager.data_buffers.values())
