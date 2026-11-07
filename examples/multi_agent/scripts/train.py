"""PowerZoo 统一训练入口。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from utils.train_cli import main

if __name__ == '__main__':
    main()
