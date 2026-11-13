"""下载并校验外部 OpenDSS 算例。

zip 算例（uk_mvlv_hv_ug）整体下载校验；文件清单算例（smart_ds_*）按仓库内
清单逐文件下载，只取被引用的最小子集，不下载完整区域。示例：

    python scripts/download_external_case.py --case uk_mvlv_hv_ug
    python scripts/download_external_case.py --case smart_ds_sfo_p4udt4
    python scripts/download_external_case.py --case smart_ds_sfo_p4udt4 --verify
    python scripts/download_external_case.py --case smart_ds_sfo_p4udt4 --discover
        # --discover 需联网：从上游列举生成文件清单 JSON（含逐文件 SHA-256），
        # 供审查后提交仓库；日常下载只依赖已提交清单。
"""

from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.external_cases import (  # noqa: E402
    CASES,
    FILE_SET_CASES,
    download,
    external_root,
    s3_list_objects,
    verify,
)

# 清单发现：模型目录前缀（相对 source_base）；LoadShape 引用的 profiles 与
# solar_data 会被自动解析补入。新增算例在此登记发现参数。
DISCOVER_PRESETS = {
    "smart_ds_sfo_p4udt4": {
        "source_base": "https://oedi-data-lake.s3.amazonaws.com/SMART-DS/v1.0/2017/SFO/P4U",
        "scenario_prefixes": [
            "scenarios/base_timeseries/opendss/p4uhs0_4/p4uhs0_4--p4udt4/",
            "scenarios/solar_high_batteries_none_timeseries/opendss/p4uhs0_4/p4uhs0_4--p4udt4/",
        ],
        "solar_data_for_profiles": True,
    },
    "smart_ds_gso_rdt1534": {
        "source_base": "https://oedi-data-lake.s3.amazonaws.com/SMART-DS/v1.0/2017/GSO/rural",
        "scenario_prefixes": [
            "scenarios/solar_high_batteries_none_timeseries/opendss/rhs0_1247/rhs0_1247--rdt1534/",
        ],
        "solar_data_for_profiles": True,
    },
}


def _discover(case_id):
    """联网枚举文件并生成清单 JSON（stdout 输出；--write-manifest 写入仓库）。"""
    from urllib.request import urlopen

    case = FILE_SET_CASES[case_id]
    preset = DISCOVER_PRESETS[case_id]
    base = preset["source_base"].rstrip("/")
    files = {}

    def fetch(rel):
        with urlopen(f"{base}/{rel}", timeout=120) as response:
            payload = response.read()
        files[rel] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        return payload

    model_files = []
    for prefix in preset["scenario_prefixes"]:
        for key in s3_list_objects(base, prefix):
            model_files.append(key)
    layout_prefix_len = len(case.layout_root.rstrip("/")) + 1
    # 相对引用按布局根绝对锚点解析（resolve() 不能落在进程 CWD 上）。
    anchor = Path("/layout")
    for key in model_files:
        rel = key[layout_prefix_len:]
        if rel.lower().endswith("loadshapes.dss"):
            text = fetch(rel).decode("utf-8")
            shapes_dir = str(Path(rel).parent)
            for match in re.finditer(r"file=([^)\s]+)", text):
                target = (anchor / shapes_dir / match.group(1).replace("\\", "/")).resolve()
                try:
                    rel_target = str(target.relative_to(anchor))
                except ValueError:
                    raise SystemExit(
                        f"LoadShape 引用越界：{match.group(1)!r}（来自 {rel}）"
                    )
                if rel_target not in files:
                    fetch(rel_target)
                    stem = Path(rel_target).stem
                    # PV 辐照曲线以城市前缀命名（SFO_/GSO_/AUS_…）；负荷
                    # 曲线为 res_/com_ 前缀，无对应 solar_data。
                    is_pv_curve = not stem.startswith(("res_", "com_"))
                    if preset.get("solar_data_for_profiles") and is_pv_curve:
                        solar = f"solar_data/{stem}_full.csv"
                        if solar not in files:
                            fetch(solar)
        else:
            if rel not in files:
                fetch(rel)

    manifest = {
        "schema_version": 1,
        "case_id": case_id,
        "source_base": base,
        "upstream": {"dataset_url": case.dataset_url, "publisher": "NREL via OEDI"},
        "generator": (
            f"python scripts/download_external_case.py --case {case_id} --discover"
        ),
        "files": [
            {"path": rel, **info} for rel, info in sorted(files.items())
        ],
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=sorted(CASES) + sorted(FILE_SET_CASES),
        required=True,
    )
    parser.add_argument("--data-dir", default=None, help="外部数据根目录（默认 data/external）")
    parser.add_argument("--force", action="store_true", help="已存在时重新下载")
    parser.add_argument("--no-keep-zip", action="store_true", help="解压后删除压缩包（仅 zip 算例）")
    parser.add_argument("--verify", action="store_true", help="只做校验，不下载")
    parser.add_argument(
        "--discover",
        action="store_true",
        help="联网枚举生成文件清单 JSON（文件清单算例专用）",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="--discover 时把清单写入仓库登记路径（否则打印到 stdout）",
    )
    args = parser.parse_args()

    if args.discover:
        manifest = _discover(args.case)
        text = json.dumps(manifest, ensure_ascii=False, indent=1) + "\n"
        if args.write_manifest:
            target = ROOT / FILE_SET_CASES[args.case].file_manifest
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            print(f"已写入 {target}（{len(manifest['files'])} 个文件）")
        else:
            print(text)
        return

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
    case = CASES.get(args.case) or FILE_SET_CASES.get(args.case)
    print(f"来源：{getattr(case, 'dataset_url', None) or getattr(case, 'repo_url', None)}")
    print(f"引用：{case.citation}")
    print(f"许可：{case.license_note}")


if __name__ == "__main__":
    main()
