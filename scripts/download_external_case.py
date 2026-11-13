"""下载并校验外部 OpenDSS 算例（当前支持英国 MV-LV 联合网络 HV_UG 版）。

数据不进入版本库；来源、上游提交、SHA-256 与许可说明写入数据目录的
manifest.json，登记值固定在 utils/external_cases.py。示例：

    python scripts/download_external_case.py --case uk_mvlv_hv_ug
    python scripts/download_external_case.py --case uk_mvlv_hv_ug --data-dir /data/external --verify
"""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.external_cases import CASES, download, external_root, verify  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--data-dir", default=None, help="外部数据根目录（默认 data/external）")
    parser.add_argument("--force", action="store_true", help="已存在时重新下载")
    parser.add_argument("--no-keep-zip", action="store_true", help="解压后删除压缩包")
    parser.add_argument("--verify", action="store_true", help="只做校验，不下载")
    args = parser.parse_args()

    root = external_root(args.data_dir)
    if args.verify:
        report = verify(args.case, root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report is None or report["status"] != "ok":
            sys.exit(1)
        return

    result = download(
        args.case,
        root,
        keep_zip=not args.no_keep_zip,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    case = CASES[args.case]
    print(f"来源：{case.repo_url} @ {case.upstream_commit}")
    print(f"引用：{case.citation}")
    print(f"许可：{case.license_note}")


if __name__ == "__main__":
    main()
