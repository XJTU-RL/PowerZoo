"""生成可独立检查/重放的 SmartGrid 数据集；无 DSS、Torch 或网络依赖。"""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.smartgrid.data_process.dataset import DataConfig, generate_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--steps", type=int, default=360)
    p.add_argument("--step-seconds", type=int, default=60)
    p.add_argument("--episodes-per-split", type=int, default=12)
    p.add_argument("--load-scale", type=float, default=1)
    p.add_argument("--irradiance-scale", type=float, default=1)
    p.add_argument("--noise", type=float, default=0.025)
    a = p.parse_args()
    cfg = DataConfig(
        seed=a.seed,
        step_seconds=a.step_seconds,
        episodes_per_split=a.episodes_per_split,
        load_scale=a.load_scale,
        irradiance_scale=a.irradiance_scale,
        noise=a.noise,
    )
    print(generate_dataset(a.output, cfg, a.steps))


if __name__ == "__main__":
    main()
