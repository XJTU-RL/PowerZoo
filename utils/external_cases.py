"""外部 OpenDSS 算例的来源登记、下载校验与定位。

外部算例指不由本仓库分发的大规模模型（如英国 MV-LV 联合网络）。上游数据
许可未确认（uk-mvlv-models 仓库无 LICENSE 文件，默认保留所有权利），因此
压缩包与解压产物一律不进入版本库，只提交：来源登记、下载脚本、适配配置、
清单检查与测试。

数据目录默认为 ``data/external/``（已在 .gitignore 忽略），可用环境变量
``POWERZOO_EXTERNAL_CASES`` 指到仓库外。目录布局：

    data/external/<case_id>/
    ├── manifest.json          # 下载时生成的来源记录
    ├── <zip 名>.zip           # 原始压缩包（保留供 SHA-256 复核）
    └── <zip 顶层目录>/        # 解压产物，模型目录（上游原样，不做修改）
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


def external_root(root=None):
    """外部数据根目录；POWERZOO_EXTERNAL_CASES 显式优先。"""
    import os

    configured = root or os.environ.get("POWERZOO_EXTERNAL_CASES")
    return Path(configured).expanduser().resolve() if configured else ROOT / "data" / "external"


def case_dir(case_id, root=None):
    if case_id not in CASES:
        raise ExternalCaseError(
            f"未登记的外部算例：{case_id}（已登记：{sorted(CASES)}）"
        )
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
    """按登记的 URL 与 SHA-256 下载、校验并解压外部算例；返回校验报告。

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

    file_count = sum(1 for path in extracted.rglob("*") if path.is_file())
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
        "extraction": {"top_dir": case.zip_top_dir, "file_count": file_count},
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
        "file_count": file_count,
    }


def verify(case_id, root=None):
    """校验算例可用性；返回 ``status=ok`` 报告或含 problems 清单的 unavailable 报告。"""
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
    if problems:
        return {"case_id": case_id, "status": "unavailable", "problems": problems}
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("case_id") != case_id or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ExternalCaseError(
            f"{manifest_file} 内容与算例登记不一致；请删除该目录后重新下载。"
            f"{download_hint(case_id)}"
        )
    return {
        "case_id": case_id,
        "status": "ok",
        "model_folder": str(model),
        "master": case.master,
        "upstream_commit": manifest["source"]["upstream_commit"],
        "zip_sha256": manifest["zip"]["sha256"],
        "downloaded_at": manifest["downloaded_at"],
    }


def locate(case_id, root=None):
    """返回模型目录 Path；不可用时抛出带补救命令的异常。"""
    report = verify(case_id, root)
    if report["status"] != "ok":
        raise ExternalCaseError(
            f"外部算例 {case_id} 未就绪：{'；'.join(report['problems'])}。"
            f"{download_hint(case_id)}"
        )
    return Path(report["model_folder"])


def read_manifest(case_id, root=None):
    return json.loads(manifest_path(case_id, root).read_text(encoding="utf-8"))
