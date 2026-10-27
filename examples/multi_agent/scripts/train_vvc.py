"""统一配置入口的快捷方式；参数说明见 train.py --help。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from utils.train_cli import main

if __name__ == '__main__':
    main(default_env='smartgrid')
