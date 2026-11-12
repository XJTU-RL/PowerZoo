"""六类电力环境的公共 MARL 训练入口；--config 选择实验，--set 覆盖参数。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.train_cli import main


if __name__ == "__main__":
    main()
