"""Validate the metadata and bytes of the first standard-case registry tranche.

No solver, loader changes, network fetches, or external payload staging.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = "node_systems/case_registry.json"


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _path(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*", value):
        raise ValueError(f"invalid registry asset path: {value}")
    path = PurePosixPath(value)
    if any(part in (".", "..") for part in value.split("/")):
        raise ValueError(f"invalid registry asset path: {value}")
    return path


def _local_file(root, value):
    p = root
    for part in _path(value).parts:
        if part not in {x.name for x in p.iterdir()}:
            raise ValueError(f"missing or differently cased input: {value}")
        p /= part
        if p.is_symlink():
            raise ValueError(f"symlink not allowed in pinned asset path: {value}")
    if not p.is_file():
        raise ValueError(f"registry input is not a file: {value}")
    return p


def _https(value):
    if not isinstance(value, str):
        raise ValueError("expected a pinned HTTPS source")
    url = urlsplit(value)
    if (url.scheme != "https" or not url.hostname or url.username or url.password
            or url.query or url.fragment or not url.path.startswith("/")):
        raise ValueError(f"expected a pinned HTTPS source: {value}")


def _verify_assets(assets, root):
    for asset in assets:
        payload = _local_file(root, asset["path"]).read_bytes()
        if asset["hash_kind"] == "sha256":
            actual = hashlib.sha256(payload).hexdigest()
        else:
            actual = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        if actual != asset["digest"]:
            raise ValueError(f"asset digest mismatch: {asset['path']}")


def _check_dss_redirects(case, root):
    # Scope is the existing simple Redirect syntax of these exact static
    # bundles; not an allow-all parser for unknown OpenDSS scripts.
    paths = {asset["path"] for asset in case["assets"]}
    visiting, visited = set(), set()

    def walk(value):
        if value in visiting:
            raise ValueError("cycle in pinned model redirects")
        if value in visited:
            return
        visiting.add(value)
        for line in _local_file(root, value).read_text(encoding="utf-8-sig").splitlines():
            line = line.partition("!")[0].strip()
            if re.match(r"^redirect\b", line, re.I):
                target = line.split(maxsplit=1)[1].strip('\"')
                link = (_path(value).parent / _path(target)).as_posix()
                if link not in paths:
                    raise ValueError(f"unregistered DSS redirect: {link}")
                walk(link)
            elif re.search(r"^compile\b|^buscoords\b|\bfile\s*=", line, re.I):
                raise ValueError("unsupported external input syntax in bundled deck")
        visiting.remove(value)
        visited.add(value)

    walk(case["entrypoint"])
    if visited != paths:
        raise ValueError("bundled assets exceed the active DSS redirect closure")


def validate(registry=None, *, root=ROOT, external=None):
    root = Path(root)
    if registry is None:
        registry = json.loads(_local_file(root, REGISTRY).read_bytes(), object_pairs_hook=_unique_pairs)
    if registry["schema_version"] != 1 or not registry["cases"]:
        raise ValueError("unsupported/empty standard case registry")
    indexed = {}
    external = {} if external is None else external
    for case in registry["cases"]:
        id_ = case["id"]
        if not re.fullmatch("[a-z0-9]+(?:-[a-z0-9]+)*", id_) or id_ in indexed:
            raise ValueError(f"invalid or duplicate case id: {id_}")
        indexed[id_] = case
        if type(case["nominal_bus_count"]) is not int or case["nominal_bus_count"] < 1:
            raise ValueError(f"invalid nominal case label count: {id_}")
        source = case["source"]
        _https(source["url"])
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9]+\.[0-9]+(?:\.[0-9]+)?)", source["revision"]):
            raise ValueError(f"invalid pinned revision: {id_}")
        if source["revision"] not in urlsplit(source["url"]).path.split("/"):
            raise ValueError(f"source URL omits the declared revision: {id_}")
        paths = set()
        for asset in case["assets"]:
            path = _path(asset["path"]).as_posix()
            n = {"sha256": 64, "git_blob_sha1": 40}.get(asset["hash_kind"])
            if n is None or not re.fullmatch(f"[0-9a-f]{{{n}}}", asset["digest"]) or path.casefold() in paths:
                raise ValueError(f"invalid/duplicate pinned asset: {asset['path']}")
            paths.add(path.casefold())
        entry = _path(case["entrypoint"])
        if entry.as_posix() not in {a["path"] for a in case["assets"]}:
            raise ValueError(f"entrypoint is not pinned: {id_}")
        level, vvc = case["availability"], case["tasks"]["vvc"]
        if level == "bundled":
            if (id_ in external or case["representation"] != "opendss"
                    or len(entry.parts) < 3 or entry.parts[0] != "node_systems"
                    or any(_path(a["path"]).parent != entry.parent for a in case["assets"])
                    or source["distribution"] != "existing_repository_assets"
                    or vvc["status"] != "smoke_validated"):
                raise ValueError(f"invalid bundled model boundary: {id_}")
            _verify_assets(case["assets"], root)
            _check_dss_redirects(case, root)
            config = _path(vvc["config"])
            if (config.parent.as_posix() != "configs/systems"
                    or config.stem != vvc["system_ref"] or entry.parent.name != vvc["system_ref"]):
                raise ValueError(f"VVC reference must target the same existing case: {id_}")
            # Existing loader continues to own default system configuration.
            import yaml
            system = yaml.safe_load(_local_file(root, config.as_posix()).read_bytes())["system"]
            if (system["name"] != vvc["system_ref"]
                    or system["dss_folder"] != entry.parent.as_posix()
                    or system["dss_file"] != entry.name):
                raise ValueError(f"VVC path disagrees with existing system config: {id_}")
            _local_file(root, vvc["evidence"])
        elif level in ("external_source", "external_source_conversion"):
            expected = "opendss" if level == "external_source" else "matpower"
            if (case["representation"] != expected or source["distribution"] != "redistribution_unverified"
                    or vvc != {"status": "unverified"}
                    or any(path.split("/")[0] == "node_systems" for path in paths)):
                raise ValueError(f"external case cannot advertise local package or VVC readiness: {id_}")
            if id_ in external:
                _verify_assets(case["assets"], Path(external[id_]))
        else:
            raise ValueError(f"invalid availability category: {id_}")
        evidence = case["static_evidence"]
        if not re.fullmatch("[0-9a-f]{40}", evidence["revision"]):
            raise ValueError(f"static evidence revision is not pinned: {id_}")
        _path(evidence["reference"])
        _https(evidence["url"])
        if not urlsplit(evidence["url"]).path.endswith(
            "/" + evidence["revision"] + "/" + evidence["reference"]
        ):
            raise ValueError(f"static evidence URL is not pinned to the declared file: {id_}")
    if external.keys() - indexed.keys():
        raise ValueError(f"unknown external cases: {external.keys() - indexed.keys()}")
    return indexed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", nargs=2, action="append", default=[], metavar=("CASE", "SOURCE_ROOT"))
    args = parser.parse_args()
    if len(dict(args.external)) != len(args.external):
        parser.error("duplicate --external case")
    for name, metadata in validate(external=dict(args.external)).items():
        print(f"{name}: {metadata['availability']}; VVC {metadata['tasks']['vvc']['status']}")
