"""唯一时序入口。"""

from .dataset import ScenarioDataset, generate_dataset, prepare_dataset

__all__ = ["ScenarioDataset", "generate_dataset", "prepare_dataset"]
