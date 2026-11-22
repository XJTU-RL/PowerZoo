"""列出训练能力或静态预检一套已解析配置，无需启动仿真。"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.acceptance import run_acceptance
from utils.capabilities import discover_capabilities, preflight
from utils.config_pipeline import ConfigLoader, parse_assignments
from utils.config_validation import ConfigError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="列出全部公开能力")
    parser.add_argument("--config", dest="experiment")
    parser.add_argument("--algo")
    parser.add_argument("--backend")
    parser.add_argument("--env")
    parser.add_argument("--system", dest="system_ref")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--mask", choices=("structural", "feasible"), default="none")
    parser.add_argument("--checkpoint", choices=("policy_reload", "training_resume"))
    parser.add_argument("--independent-replay", action="store_true")
    parser.add_argument(
        "--checkpoint-contract",
        choices=("binding_v1", "legacy_physics_only"),
        help="集中 checkpoint 格式；默认按当前后端写出的 binding_v1 预检",
    )
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps(discover_capabilities(), indent=2, ensure_ascii=False))
        return 0
    try:
        config = ConfigLoader().load(
            experiment=args.experiment,
            algo=args.algo,
            env=args.env,
            system_ref=args.system_ref,
            overrides=parse_assignments(args.set),
        )
        result = preflight(
            config,
            backend=args.backend,
            require_mask=args.mask,
            checkpoint=args.checkpoint,
            independent_replay=args.independent_replay,
            checkpoint_contract=args.checkpoint_contract,
        )
    except (ConfigError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(run_acceptance(result), indent=2, ensure_ascii=False))
    return 0 if result.compatible else 1


if __name__ == "__main__":
    sys.exit(main())
