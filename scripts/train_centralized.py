"""VVC/SmartGrid公共集中式PPO入口，--config只选择任务与实验差异。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.centralized_cli import main

if __name__ == "__main__":
    main()
