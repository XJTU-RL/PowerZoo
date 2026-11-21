"""列出 VVC 算例，或生成供标准训练入口读取的 34/123 场景目录。"""

import argparse
from hashlib import sha1, sha256
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "scenarios/vvc/cases.yaml"


def load_catalog():
    return yaml.safe_load(CATALOG.read_text(encoding="utf-8"))


def _storage_commands(contents):
    return b"".join(
        line
        for line in contents.splitlines(keepends=True)
        if line.lstrip().lower().startswith(b"new storage.")
    )


def reference_inputs(case):
    """仅从锁定的 PowerZoo 静态闭包剥离研究储能，拓扑与依赖保持原样。"""
    from envs.powergrid.case_inventory import dependency_closure

    feeder = case["feeder"]
    folder = ROOT / feeder["folder"]
    master = feeder["master"]
    pins = feeder["tracked_sha256"]
    closure = dependency_closure(folder, master)
    hashes = {name: entry["sha256"] for name, entry in closure["files"].items()}
    if closure["missing"] or closure["notes"] or hashes != pins:
        raise ValueError(f"运行电路与登记的闭包不符：{folder}")

    files = {name: (folder / name).read_bytes() for name in pins}
    if any(
        sha256(contents).hexdigest() != pins[name] for name, contents in files.items()
    ):
        raise ValueError(f"读取过程中电路文件已更改：{folder}")
    original_master = files[master]
    storage = _storage_commands(original_master)
    scenario = case["scenario"]
    overlay = (ROOT / scenario["overlay"]).read_bytes()
    if (
        not storage
        or storage != _storage_commands(overlay)
        or sha256(overlay).hexdigest() != scenario["overlay_sha256"]
    ):
        raise ValueError("储能叠加层与登记的运行电路不符")

    public_master = original_master.replace(storage, b"", 1)
    if sha256(public_master).hexdigest() != feeder["base_master_sha256"]:
        raise ValueError("未能按登记摘要分离储能定义")
    files[master] = public_master
    return files, overlay


def verify_source(case_id, folder):
    """核对调用者从上游取得的原始文件，保留原字节。"""
    case = load_catalog()["cases"].get(case_id)
    if case is None:
        raise ValueError(f"未知算例：{case_id}")
    source = case["source"]
    folder = Path(folder)
    if "upstream_sha256" in source:
        expected = source["upstream_sha256"]
        actual = {
            name: sha256((folder / name).read_bytes()).hexdigest() for name in expected
        }
    elif "git_blob_sha1" in source:
        name = source["upstream_filename"]
        expected = {name: source["git_blob_sha1"]}
        contents = (folder / name).read_bytes()
        header = f"blob {len(contents)}\0".encode("ascii")
        actual = {name: sha1(header + contents, usedforsecurity=False).hexdigest()}
    else:
        raise ValueError(f"{case_id} 未登记逐文件上游摘要")
    mismatches = [name for name, digest in actual.items() if digest != expected[name]]
    if mismatches:
        raise ValueError(f"{case_id} 上游文件摘要不符：{mismatches}")
    return actual


def stage_reference(case_id, destination):
    """生成独立 DSS 输入、场景叠加层、训练 YAML 和输入摘要。"""
    catalog = load_catalog()
    case = catalog["cases"].get(case_id)
    if case is None or case["role"] != "reference":
        raise ValueError(f"{case_id} 没有可生成的参考场景")
    files, overlay = reference_inputs(case)
    destination = Path(destination).resolve()
    model = destination / "model"
    overlay_rel = "overlays/storage.dss"

    experiment = yaml.safe_load(
        (ROOT / catalog["reference_experiment"]).read_text(encoding="utf-8")
    )
    experiment["system_ref"] = case["system_ref"]
    experiment["env"].update(
        dss_folder=str(model), dss_file=case["feeder"]["master"], overlays=[overlay_rel]
    )
    experiment_text = yaml.safe_dump(experiment, sort_keys=False, allow_unicode=True)

    destination.mkdir(parents=True, exist_ok=False)
    for name, contents in {**files, overlay_rel: overlay}.items():
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    experiment_path = destination / "experiment.yaml"
    experiment_path.write_text(experiment_text, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "base_revision": catalog["base_revision"],
        "case": case_id,
        "source": case["source"],
        "source_runtime_sha256": case["feeder"]["tracked_sha256"],
        "base_model_sha256": {
            f"model/{name}": sha256(contents).hexdigest()
            for name, contents in files.items()
        },
        "overlay_sha256": {f"model/{overlay_rel}": sha256(overlay).hexdigest()},
        "experiment_sha256": sha256(experiment_path.read_bytes()).hexdigest(),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return experiment_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="显示算例登记及来源 JSON")
    stage = commands.add_parser("stage", help="生成 34 或 123 参考场景")
    stage.add_argument("case", choices=("ieee34", "ieee123"))
    stage.add_argument("destination", type=Path)
    verify = commands.add_parser("verify-source", help="核对本地公开原件")
    verify.add_argument("case")
    verify.add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.command == "list":
        print(json.dumps(load_catalog(), indent=2, ensure_ascii=False))
    elif args.command == "verify-source":
        print(json.dumps(verify_source(args.case, args.folder), indent=2))
    else:
        print(stage_reference(args.case, args.destination))


if __name__ == "__main__":
    main()
