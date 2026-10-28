"""兼容SmartGrid旧入口；正式实现位于公共集中训练CLI。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from utils.centralized_cli import main

if __name__ == "__main__":
    main(
        "configs/experiments/smartgrid_synthetic_smoke.yaml",
        expected_family="smartgrid",
    )
