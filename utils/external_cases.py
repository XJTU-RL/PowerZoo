"""外部 OpenDSS 算例的来源登记、下载校验与定位。

外部算例指不由本仓库分发的大规模模型。两类登记：

- ``ExternalCase``（单 zip）：如英国 MV-LV 联合网络。上游许可未确认
  （uk-mvlv-models 无 LICENSE，默认保留所有权利），压缩包与解压产物不进
  版本库。
- ``FileSetCase``（文件清单）：如 NREL SMART-DS（OEDI 分发）。逐文件登记
  SHA-256（清单 JSON 提交本仓库），只下载被引用的最小子集而非整个区域；
  必要的引擎兼容补丁（如 2017 版参数在当前 DSS C-API 的改名）以精确文本
  替换应用并逐条记录理由，原始 SHA-256 一并写入 manifest，可从上游复原。

数据目录默认为 ``data/external/``（已在 .gitignore 忽略），可用环境变量
``POWERZOO_EXTERNAL_CASES`` 指到仓库外。目录布局：

    data/external/<case_id>/
    ├── manifest.json          # 下载时生成的来源记录
    ├── <zip 名>.zip           # 单 zip 算例：原始压缩包（供 SHA-256 复核）
    └── <布局根>/…             # 模型文件（FileSetCase 保留上游相对布局）
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1
DOWNLOAD_CHUNK_BYTES = 1 << 20


class ExternalCaseError(RuntimeError):
    """外部算例不可用；消息中给出定位到的文件与补救命令。"""


@dataclass(frozen=True)
class ExternalCase:
    case_id: str
    repo_url: str
    upstream_commit: str
    zip_url: str
    zip_name: str
    zip_sha256: str
    zip_bytes: int
    zip_top_dir: str
    master: str
    description: str
    citation: str
    license_note: str


UK_MVLV_HV_UG = ExternalCase(
    case_id="uk_mvlv_hv_ug",
    repo_url="https://github.com/deakinmt/uk-mvlv-models",
    upstream_commit="e97e260ed46b2ad818f1a701ba2b95f269c9c12c",
    zip_url="https://raw.githubusercontent.com/deakinmt/uk-mvlv-models/master/HV_UG_full.zip",
    zip_name="HV_UG_full.zip",
    zip_sha256="23ab52a9bc3d678ae2e9e3d0c0376a005aacd4471bb3498d8318a4cf64074808",
    zip_bytes=9257503,
    zip_top_dir="HV_UG_full",
    master="master_mvlv.dss",
    description=(
        "英国 33/11 kV 电缆中压网 + 75 个低压子网的联合模型（HV_UG 版）："
        "主入口 master_mvlv.dss 经 redirect 引用 2 台 26.4 MVA 主变（2 个原生"
        " RegControl）、75 条中压线路、75 个被 lds_edit.dss 禁用的中压集中负荷"
        "（需求由低压明细网承载，不可全部重新启用）与 75 个 LV slave 子网。"
    ),
    citation=(
        "M. Deakin, D. Greenwood, S. Walker and P. C. Taylor, "
        '"Hybrid European MV-LV Network Models for Smart Distribution Network '
        'Modelling", 2021 IEEE Madrid PowerTech. Preprint: arXiv:2009.14240'
    ),
    license_note=(
        "上游仓库无 LICENSE 文件（GitHub API license=null，默认保留所有权利），"
        "且其来源 UKGDS/LVNS 各有条款：压缩包、完整展开模型与完整派生模型不得"
        "提交本仓库或发布包；只能通过下载脚本从上游获取。"
    ),
)

CASES = {case.case_id: case for case in (UK_MVLV_HV_UG,)}


@dataclass(frozen=True)
class FilePatch:
    """对布局内指定文件的精确文本替换；每条必须给出理由并写入 manifest。"""

    path: str
    find: str
    replace: str
    reason: str


@dataclass(frozen=True)
class CaseEntry:
    folder: str  # 相对布局根的模型目录
    master: str  # 相对模型目录的主入口


@dataclass(frozen=True)
class FileSetCase:
    """文件清单型外部算例：逐文件登记 SHA-256，按需下载最小子集。

    ``file_manifest`` 指向提交在仓库内的清单 JSON（含 source_base 与逐文件
    sha256/bytes）；布局保留上游相对结构（LoadShape 的 ``file=`` 相对引用
    依赖该结构）。
    """

    case_id: str
    dataset_url: str
    file_manifest: str  # 仓库内相对路径
    layout_root: str
    entries: dict  # {入口名: CaseEntry}
    patches: tuple  # (FilePatch, ...)
    description: str
    citation: str
    license_note: str


SMART_DS_SFO_P4UDT4 = FileSetCase(
    case_id="smart_ds_sfo_p4udt4",
    dataset_url="https://data.openei.org/submissions/2981",
    file_manifest="configs/external/smart_ds_sfo_p4udt4_files.json",
    layout_root="SMART-DS/v1.0/2017/SFO/P4U",
    entries={
        "base": CaseEntry(
            folder="scenarios/base_timeseries/opendss/p4uhs0_4/p4uhs0_4--p4udt4",
            master="Master.dss",
        ),
        "solar_high_none": CaseEntry(
            folder=(
                "scenarios/solar_high_batteries_none_timeseries/opendss/"
                "p4uhs0_4/p4uhs0_4--p4udt4"
            ),
            master="Master.dss",
        ),
    },
    patches=(
        FilePatch(
            path=(
                "scenarios/solar_high_batteries_none_timeseries/opendss/"
                "p4uhs0_4/p4uhs0_4--p4udt4/PVSystems.dss"
            ),
            find="kvarlimit=",
            replace="kvarMax=",
            reason="2017 版 PVSystem 参数 kvarlimit 在当前 DSS C-API 0.14.5 已"
            "拆分为 kvarMax/kvarMaxAbs；映射到发出侧上限 kvarMax，吸收侧"
            "保持引擎默认（=kvarMax）",
        ),
        FilePatch(
            path=(
                "scenarios/solar_high_batteries_none_timeseries/opendss/"
                "p4uhs0_4/p4uhs0_4--p4udt4/PVSystems.dss"
            ),
            find="VV_RefReactivePower",
            replace="RefReactivePower",
            reason="InvControl 参数 VV_RefReactivePower 于 2020 年更名为"
            " RefReactivePower（引擎报错提示的官方改名）",
        ),
        FilePatch(
            path=(
                "scenarios/base_timeseries/opendss/p4uhs0_4/p4uhs0_4--p4udt4/"
                "Master.dss"
            ),
            find="Solve mode=yearly stepsize=15m number=35040",
            replace="! Solve mode=yearly stepsize=15m number=35040（派生注释："
            "训练会话按 15 分钟步长显式驱动，compile 期全年求解只为导出"
            "官方监控数据，且会留下控制器遗留状态）",
            reason="compile 期 35040 步全年求解在每次会话 reset 重放（rdt1534 "
            "实测 278s），且其遗留的 InvControl 中间状态偏离模型文件初始值",
        ),
        FilePatch(
            path=(
                "scenarios/solar_high_batteries_none_timeseries/opendss/"
                "p4uhs0_4/p4uhs0_4--p4udt4/Master.dss"
            ),
            find="Solve mode=yearly stepsize=15m number=35040",
            replace="! Solve mode=yearly stepsize=15m number=35040（派生注释，"
            "理由同 base 入口）",
            reason="同 base 入口：去掉 compile 期全年求解",
        ),
    ),
    description=(
        "NREL SMART-DS v1.0（2017 年合成数据）SFO/P4U 场景的 p4udt4 馈线："
        "4kV 城市馈线，71 母线/36 负荷/13 变压器；base_timeseries 无 PV，"
        "solar_high_batteries_none 变体含 6 台 PVSystem（其中 1 台绑定原生"
        " InvControl Volt-Var）。负荷 pu 曲线（铭牌=年峰值）与 PV 辐照曲线"
        "（kW/m²）均为 15 分钟×35040 点，当地标准时（SFO=UTC-8，全年无夏令时）；"
        "中心抽头用户=两个负荷对象各挂一半额定（官方指南明确，勿重复计负荷）。"
    ),
    citation=(
        "B. Palmintier et al., “SMART-DS Synthetic Electrical Network Data "
        "OpenDSS Models for SFO, GSO, and AUS,” OEDI submission 2981, 2020. "
        "https://data.openei.org/submissions/2981；方法论文：B. Palmintier "
        "et al., “Experiences developing large-scale synthetic U.S.-style "
        "distribution test systems,” Electric Power Systems Research, vol. "
        "190, 106665, 2021, doi:10.1016/j.epsr.2020.106665"
    ),
    license_note=(
        "OEDI 提交页标注 CC BY 4.0（https://creativecommons.org/licenses/"
        "by/4.0/）：下载、内部派生与再分发均允许，须署名（引用见上）并注明"
        "修改。本仓库仍按文件清单从上游获取、不打包原始数据；派生修改逐条"
        "登记于 manifest。"
    ),
)

SMART_DS_GSO_RDT1534 = FileSetCase(
    case_id="smart_ds_gso_rdt1534",
    dataset_url="https://data.openei.org/submissions/2981",
    file_manifest="configs/external/smart_ds_gso_rdt1534_files.json",
    layout_root="SMART-DS/v1.0/2017/GSO/rural",
    entries={
        "solar_high_none": CaseEntry(
            folder=(
                "scenarios/solar_high_batteries_none_timeseries/opendss/"
                "rhs0_1247/rhs0_1247--rdt1534"
            ),
            master="Master.dss",
        ),
    },
    patches=(
        FilePatch(
            path=(
                "scenarios/solar_high_batteries_none_timeseries/opendss/"
                "rhs0_1247/rhs0_1247--rdt1534/PVSystems.dss"
            ),
            find="kvarlimit=",
            replace="kvarMax=",
            reason="同 p4udt4：2017 版参数 kvarlimit 在当前引擎已拆分为 kvarMax/"
            "kvarMaxAbs，映射到发出侧上限",
        ),
        FilePatch(
            path=(
                "scenarios/solar_high_batteries_none_timeseries/opendss/"
                "rhs0_1247/rhs0_1247--rdt1534/PVSystems.dss"
            ),
            find="VV_RefReactivePower",
            replace="RefReactivePower",
            reason="同 p4udt4：InvControl 参数 2020 年官方改名",
        ),
        FilePatch(
            path=(
                "scenarios/solar_high_batteries_none_timeseries/opendss/"
                "rhs0_1247/rhs0_1247--rdt1534/Master.dss"
            ),
            find="Solve mode=yearly stepsize=15m number=35040",
            replace="! Solve mode=yearly stepsize=15m number=35040（派生注释，"
            "理由同 p4udt4）",
            reason="compile 期全年求解实测 278s 且遗留控制器中间状态",
        ),
    ),
    description=(
        "NREL SMART-DS v1.0（2017）GSO rural 场景 rhs0_1247 变电站的 rdt1534 "
        "馈线：12.47kV 农村馈线，472 台 PVSystem（218 台带 kvarlimit、217 台"
        "绑定原生 InvControl）。15 分钟×35040 点曲线，当地标准时"
        "（GSO=UTC-5，全年无夏令时）。"
    ),
    citation=SMART_DS_SFO_P4UDT4.citation,
    license_note=SMART_DS_SFO_P4UDT4.license_note,
)

FILE_SET_CASES = {
    case.case_id: case for case in (SMART_DS_SFO_P4UDT4, SMART_DS_GSO_RDT1534)
}


def _registry(case_id):
    if case_id in CASES:
        return CASES[case_id], "zip"
    if case_id in FILE_SET_CASES:
        return FILE_SET_CASES[case_id], "file_set"
    raise ExternalCaseError(
        f"未登记的外部算例：{case_id}"
        f"（zip：{sorted(CASES)}；文件清单：{sorted(FILE_SET_CASES)}）"
    )


def external_root(root=None):
    """外部数据根目录；POWERZOO_EXTERNAL_CASES 显式优先。"""
    import os

    configured = root or os.environ.get("POWERZOO_EXTERNAL_CASES")
    return Path(configured).expanduser().resolve() if configured else ROOT / "data" / "external"


def case_dir(case_id, root=None):
    _registry(case_id)
    return external_root(root) / case_id


def model_folder(case_id, root=None):
    """算例解压后的模型目录（上游原样；主入口位于其下）。"""
    return case_dir(case_id, root) / CASES[case_id].zip_top_dir


def manifest_path(case_id, root=None):
    return case_dir(case_id, root) / MANIFEST_NAME


def download_hint(case_id):
    return (
        f"请先运行：python scripts/download_external_case.py --case {case_id}"
        "（或用 --data-dir 指定已有数据目录，并设置 POWERZOO_EXTERNAL_CASES）"
    )




def _dependency_missing(model_dir, master):
    """主入口 redirect/include/LoadShape 数据依赖闭包的缺失清单（不修改文件）。"""
    from envs.powergrid.case_inventory import dependency_closure

    return dependency_closure(Path(model_dir), master)["missing"]


def _sha256_file(path, chunk=DOWNLOAD_CHUNK_BYTES):
    digest = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_members(archive, case):
    top = case.zip_top_dir
    for member in archive.namelist():
        parts = Path(member).parts
        if member.startswith("/") or ".." in parts or (parts and parts[0] != top):
            raise ExternalCaseError(
                f"{case.zip_name} 含越界成员 {member!r}；拒绝解压"
            )


def download(case_id, root=None, *, keep_zip=True, force=False, timeout=120):
    """下载并校验外部算例（按登记类型分派 zip / 文件清单）。"""
    _, kind = _registry(case_id)
    if kind == "file_set":
        return _download_file_set(case_id, root, force=force, timeout=timeout)
    return _download_zip(
        case_id, root, keep_zip=keep_zip, force=force, timeout=timeout
    )


def _download_zip(case_id, root=None, *, keep_zip=True, force=False, timeout=120):
    """按登记的 URL 与 SHA-256 下载、校验并解压 zip 算例；返回校验报告。

    已存在且校验通过的算例默认跳过（force=True 时重新下载）。
    解压前逐成员检查路径，拒绝绝对路径与 ``..`` 越界。
    """
    case = CASES[case_id]
    target = case_dir(case_id, root)
    present = verify(case_id, root)
    if not force and present is not None and present["status"] == "ok":
        return {"case_id": case_id, "status": "already_present", "dir": str(target)}

    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / case.zip_name
    try:
        with urlopen(case.zip_url, timeout=timeout) as response, open(
            zip_path, "wb"
        ) as handle:
            while True:
                block = response.read(DOWNLOAD_CHUNK_BYTES)
                if not block:
                    break
                handle.write(block)
    except Exception as error:
        zip_path.unlink(missing_ok=True)
        raise ExternalCaseError(
            f"下载失败：{case.zip_url}（{error}）；请检查网络或手工放置 "
            f"{zip_path} 后重试（SHA-256 应为 {case.zip_sha256}）"
        ) from error

    actual_sha, actual_bytes = _sha256_file(zip_path), zip_path.stat().st_size
    if actual_sha != case.zip_sha256 or actual_bytes != case.zip_bytes:
        zip_path.unlink(missing_ok=True)
        raise ExternalCaseError(
            f"{zip_path} 校验失败：SHA-256={actual_sha}（登记值 {case.zip_sha256}），"
            f"大小 {actual_bytes}（登记值 {case.zip_bytes}）；上游可能已更新，"
            "未经审查不得换用新版本"
        )

    extracted = model_folder(case_id, root)
    if extracted.exists():
        raise ExternalCaseError(
            f"解压目标已存在：{extracted}；请手工确认后删除或使用 --force"
        )
    with zipfile.ZipFile(zip_path) as archive:
        _safe_members(archive, case)
        archive.extractall(target)
    if not (extracted / case.master).is_file():
        raise ExternalCaseError(
            f"解压完成但未找到主入口：{extracted / case.master}"
        )

    file_hashes = {
        path.relative_to(extracted).as_posix(): _sha256_file(path)
        for path in sorted(extracted.rglob("*"))
        if path.is_file()
    }
    # 完整性门禁：入口依赖闭包（redirect/include/LoadShape file=）必须无缺失，
    # 否则视为下载/展开损坏，立即失败并留下诊断。
    missing = _dependency_missing(extracted, case.master)
    if missing:
        raise ExternalCaseError(
            f"{case_id} 依赖闭包缺失（ refusing 安装）：{missing[:5]}"
            f"（共 {len(missing)} 项）"
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "case_id": case_id,
        "source": {
            "repo_url": case.repo_url,
            "upstream_commit": case.upstream_commit,
            "zip_url": case.zip_url,
            "citation": case.citation,
            "license_note": case.license_note,
        },
        "zip": {"name": case.zip_name, "sha256": case.zip_sha256, "bytes": case.zip_bytes},
        "extraction": {
            "top_dir": case.zip_top_dir,
            "file_count": len(file_hashes),
            # 逐文件 sha：verify(deep=True) 与损坏检测的依据（审核复现缺项）。
            "files": file_hashes,
        },
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "scripts/download_external_case.py",
    }
    manifest_path(case_id, root).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not keep_zip:
        zip_path.unlink(missing_ok=True)
    return {
        "case_id": case_id,
        "status": "downloaded",
        "dir": str(target),
        "model_folder": str(extracted),
        "master": case.master,
        "sha256": case.zip_sha256,
        "file_count": len(file_hashes),
    }


def _load_file_manifest(case):
    """读仓库内提交的文件清单 JSON；清单即下载契约（逐文件 sha256/bytes）。"""
    path = ROOT / case.file_manifest
    if not path.is_file():
        raise ExternalCaseError(f"文件清单缺失：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("case_id") != case.case_id or data.get("schema_version") != 1:
        raise ExternalCaseError(f"{path} 的 case_id/schema_version 与登记不符")
    return data


def _download_file_set(case_id, root=None, *, force=False, timeout=120):
    """按仓库内清单逐文件下载校验，应用登记补丁并写 manifest。"""
    case = FILE_SET_CASES[case_id]
    target = case_dir(case_id, root)
    present = verify(case_id, root)
    if not force and present is not None and present["status"] == "ok":
        return {"case_id": case_id, "status": "already_present", "dir": str(target)}

    listing = _load_file_manifest(case)
    source_base = listing["source_base"].rstrip("/")
    layout = target / case.layout_root
    layout.mkdir(parents=True, exist_ok=True)
    patches_by_path = {}
    for patch in case.patches:
        patches_by_path.setdefault(patch.path, []).append(patch)

    records = []
    for item in listing["files"]:
        rel = item["path"]
        parts = Path(rel).parts
        if rel.startswith("/") or ".." in parts:
            raise ExternalCaseError(f"清单含越界路径：{rel}")
        destination = layout / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urlopen(f"{source_base}/{rel}", timeout=timeout) as response, open(
                destination, "wb"
            ) as handle:
                while True:
                    block = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not block:
                        break
                    handle.write(block)
        except Exception as error:
            raise ExternalCaseError(
                f"下载失败：{source_base}/{rel}（{error}）"
            ) from error
        raw_sha = _sha256_file(destination)
        if raw_sha != item["sha256"] or destination.stat().st_size != item["bytes"]:
            destination.unlink(missing_ok=True)
            raise ExternalCaseError(
                f"{destination} 校验失败：SHA-256={raw_sha}（清单值 "
                f"{item['sha256']}）；上游可能已更新，未经审查不得换用新版本"
            )
        record = {"path": rel, "sha256": raw_sha, "bytes": item["bytes"], "patched": False}
        for patch in patches_by_path.get(rel, ()):
            text = destination.read_text(encoding="utf-8")
            count = text.count(patch.find)
            if count == 0:
                raise ExternalCaseError(
                    f"登记补丁未命中：{rel} 中找不到 {patch.find!r}（{patch.reason}）；"
                    "上游文件可能已变化"
                )
            destination.write_text(
                text.replace(patch.find, patch.replace), encoding="utf-8"
            )
            record["patched"] = True
            record.setdefault("patches", []).append(
                {
                    "find": patch.find,
                    "replace": patch.replace,
                    "count": count,
                    "reason": patch.reason,
                    "sha256_before": raw_sha,
                }
            )
        if record["patched"]:
            record["patched_sha256"] = _sha256_file(destination)
        records.append(record)

    for name, entry in case.entries.items():
        if not (layout / entry.folder / entry.master).is_file():
            raise ExternalCaseError(
                f"下载完成但入口 {name} 缺少主入口：{layout / entry.folder / entry.master}"
            )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "case_id": case_id,
        "kind": "file_set",
        "source": {
            "dataset_url": case.dataset_url,
            "source_base": source_base,
            "file_manifest": case.file_manifest,
            "citation": case.citation,
            "license_note": case.license_note,
        },
        "layout_root": case.layout_root,
        "entries": {n: {"folder": e.folder, "master": e.master} for n, e in case.entries.items()},
        "files": records,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "scripts/download_external_case.py",
    }
    manifest_path(case_id, root).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    patched = [r for r in records if r["patched"]]
    return {
        "case_id": case_id,
        "status": "downloaded",
        "dir": str(target),
        "file_count": len(records),
        "patched_files": len(patched),
        "entries": sorted(case.entries),
    }


def verify(case_id, root=None, deep=False):
    """校验算例可用性；返回 ``status=ok`` 报告或含 problems 清单的 unavailable 报告。

    快检（默认）：manifest/压缩包哈希/主入口存在 + 依赖闭包无缺失（缺依赖
    文件立即失败——审核复现的漏洞）。deep=True 额外全量比对展开文件与
    manifest 登记的逐文件 SHA-256（检测展开后被改动的损坏，代价为全量读）。
    """
    case, kind = _registry(case_id)
    if kind == "file_set":
        return _verify_file_set(case_id, root)
    return _verify_zip(case_id, root, deep=deep)


def _verify_file_set(case_id, root=None):
    case = FILE_SET_CASES[case_id]
    directory = case_dir(case_id, root)
    manifest_file = directory / MANIFEST_NAME
    layout = directory / case.layout_root
    problems = []
    if not directory.is_dir():
        problems.append(f"数据目录不存在：{directory}")
        return {"case_id": case_id, "status": "unavailable", "problems": problems}
    if not manifest_file.is_file():
        problems.append(f"缺少来源记录：{manifest_file}")
        for name, entry in case.entries.items():
            master = layout / entry.folder / entry.master
            if not master.is_file():
                problems.append(f"入口 {name} 缺少主入口：{master}")
        return {"case_id": case_id, "status": "unavailable", "problems": problems}

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("case_id") != case_id or manifest.get("kind") != "file_set":
        raise ExternalCaseError(
            f"{manifest_file} 内容与算例登记不一致；请删除该目录后重新下载。"
            f"{download_hint(case_id)}"
        )
    for record in manifest["files"]:
        path = layout / record["path"]
        expected = record.get("patched_sha256", record["sha256"])
        if not path.is_file():
            problems.append(f"缺少清单文件：{path}")
        elif _sha256_file(path) != expected:
            problems.append(f"{path} SHA-256 与 manifest 记录不符（数据被改动？）")
    for name, entry in case.entries.items():
        master = layout / entry.folder / entry.master
        if not master.is_file():
            problems.append(f"入口 {name} 缺少主入口：{master}")
    if problems:
        return {"case_id": case_id, "status": "unavailable", "problems": problems}
    return {
        "case_id": case_id,
        "status": "ok",
        "layout_root": str(layout),
        "entries": {
            name: {"folder": str(layout / e.folder), "master": e.master}
            for name, e in case.entries.items()
        },
        "downloaded_at": manifest["downloaded_at"],
        # 文件清单算例的校验本就逐文件比对哈希，deep 开关恒等价。
        "deep_checked": True,
    }


def _verify_zip(case_id, root=None, deep=False):
    case = CASES[case_id]
    directory = case_dir(case_id, root)
    manifest_file = directory / MANIFEST_NAME
    model = model_folder(case_id, root)
    master_file = model / case.master
    problems = []
    if not directory.is_dir():
        problems.append(f"数据目录不存在：{directory}")
    else:
        if not manifest_file.is_file():
            problems.append(f"缺少来源记录：{manifest_file}")
        zip_path = directory / case.zip_name
        if zip_path.is_file():
            actual = _sha256_file(zip_path)
            if actual != case.zip_sha256:
                problems.append(
                    f"{zip_path} SHA-256={actual} 与登记值 {case.zip_sha256} 不符"
                )
        if not master_file.is_file():
            problems.append(f"缺少主入口：{master_file}")
    manifest = None
    if manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("case_id") != case_id or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ExternalCaseError(
                f"{manifest_file} 内容与算例登记不一致；请删除该目录后重新下载。"
                f"{download_hint(case_id)}"
            )
        if deep:
            recorded = (manifest.get("extraction") or {}).get("files") or {}
            if not recorded:
                problems.append(f"{manifest_file} 缺少逐文件哈希（旧版 manifest）；请重新下载后启用深度校验")
            else:
                for rel, expected in recorded.items():
                    path = model / rel
                    if not path.is_file():
                        problems.append(f"深度校验缺少文件：{path}")
                    elif _sha256_file(path) != expected:
                        problems.append(f"{path} 与 manifest 哈希不符（文件被改动或损坏）")
        if master_file.is_file():
            # 快检也做依赖闭包：缺失依赖（缺文件）不允许通过。
            for item in _dependency_missing(model, case.master):
                problems.append(
                    f"依赖缺失（{item.get('from', '?')} 引用 {item.get('path', '?')}）"
                )
    if problems:
        return {"case_id": case_id, "status": "unavailable", "problems": problems}
    if manifest is None:
        return {"case_id": case_id, "status": "unavailable", "problems": problems}
    return {
        "case_id": case_id,
        "status": "ok",
        "model_folder": str(model),
        "master": case.master,
        "upstream_commit": manifest["source"]["upstream_commit"],
        "zip_sha256": manifest["zip"]["sha256"],
        "downloaded_at": manifest["downloaded_at"],
        "deep_checked": bool(deep),
    }


def locate(case_id, root=None):
    """zip 算例：返回模型目录 Path；文件清单算例请用 locate_entry。"""
    _, kind = _registry(case_id)
    if kind == "file_set":
        raise ExternalCaseError(
            f"{case_id} 是文件清单算例（多入口）；请用 locate_entry(case_id, entry)"
        )
    report = verify(case_id, root)
    if report["status"] != "ok":
        raise ExternalCaseError(
            f"外部算例 {case_id} 未就绪：{'；'.join(report['problems'])}。"
            f"{download_hint(case_id)}"
        )
    return Path(report["model_folder"])


def locate_entry(case_id, entry, root=None):
    """文件清单算例：返回 (模型目录 Path, 主入口文件名)。"""
    case, kind = _registry(case_id)
    if kind != "file_set":
        raise ExternalCaseError(f"{case_id} 不是文件清单算例；请用 locate()")
    if entry not in case.entries:
        raise ExternalCaseError(
            f"入口 {entry!r} 未登记（可用：{sorted(case.entries)}）"
        )
    report = verify(case_id, root)
    if report["status"] != "ok":
        raise ExternalCaseError(
            f"外部算例 {case_id} 未就绪：{'；'.join(report['problems'])}。"
            f"{download_hint(case_id)}"
        )
    return Path(report["entries"][entry]["folder"]), case.entries[entry].master


def read_manifest(case_id, root=None):
    return json.loads(manifest_path(case_id, root).read_text(encoding="utf-8"))


def s3_list_objects(source_base, prefix, *, exclude=("/analysis/",), timeout=60):
    """匿名列出 S3/HTTP 对象存储前缀下全部对象 key（分页；排除子目录片段）。

    仅用于清单发现（--discover）；日常下载校验不依赖网络列举。
    """
    import re
    from urllib.parse import quote, urlsplit

    # listing 查询必须打在 bucket 根上；source_base 的路径并入前缀。
    parts = urlsplit(source_base.rstrip("/"))
    bucket_root = f"{parts.scheme}://{parts.netloc}"
    full_prefix = "/".join(p for p in (parts.path.strip("/"), prefix.strip("/")) if p)
    keys, token = [], None
    while True:
        url = f"{bucket_root}/?list-type=2&prefix={quote(full_prefix, safe='/')}"
        if token:
            url += f"&continuation-token={quote(token)}"
        with urlopen(url, timeout=timeout) as response:
            data = response.read().decode("utf-8")
        keys.extend(re.findall(r"<Key>([^<]+)</Key>", data))
        token = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", data)
        if not token:
            break
        token = token.group(1)
    return [k for k in keys if not any(part in k for part in exclude)]
