"""在不加载 DSS 或任何数据分片的条件下核对 VVC 算例绑定。"""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from scripts.vvc_case_library import (
    ROOT,
    load_catalog,
    reference_inputs,
    stage_reference,
)
from utils.config_pipeline import ConfigLoader


@pytest.mark.parametrize("case_id", ["ieee34", "ieee123"])
@pytest.mark.parametrize("algo", ["happo", "mappo"])
def test_staged_reference_is_consumed_by_the_experiment_loader(case_id, algo, tmp_path):
    catalog = load_catalog()
    case = catalog["cases"][case_id]
    destination = tmp_path / case_id
    config = stage_reference(case_id, destination)

    loader = ConfigLoader()
    staged = loader.load(algo=algo, experiment=config)
    expected = loader.load(experiment=case["legacy_experiment"])
    assert staged.main_args["algo"] == algo
    assert staged.algo_args["train"]["num_env_steps"] == 240
    assert staged.algo_args["algo"]["share_param"] is False
    assert staged.env_args == {
        **expected.env_args,
        "dss_folder": str(destination / "model"),
        "overlays": ["overlays/storage.dss"],
    }
    assert staged.env_args["data"]["split"] == "train"

    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["case"] == case_id
    assert manifest["experiment_sha256"] == sha256(config.read_bytes()).hexdigest()
    model_files = manifest["base_model_sha256"]
    overlay_files = manifest["overlay_sha256"]
    for name, digest in {**model_files, **overlay_files}.items():
        assert sha256((destination / name).read_bytes()).hexdigest() == digest

    master = case["feeder"]["master"]
    bare = (destination / "model" / master).read_bytes()
    original = (ROOT / case["feeder"]["folder"] / master).read_bytes()
    assert b"new storage." not in bare.lower()
    assert b"new storage." in original.lower()
    assert set(
        p.relative_to(destination).as_posix()
        for p in destination.rglob("*")
        if p.is_file()
    ) == {*model_files, *overlay_files, "experiment.yaml", "manifest.json"}

    with pytest.raises(FileExistsError):
        stage_reference(case_id, destination)
    assert json.loads((destination / "manifest.json").read_text()) == manifest


def test_reference_rejects_changed_closure_and_overlay_pin():
    reference = load_catalog()["cases"]["ieee123"]
    changed = deepcopy(reference)
    changed["feeder"]["tracked_sha256"]["network.dss"] = "0" * 64
    with pytest.raises(ValueError, match="闭包"):
        reference_inputs(changed)
    changed = deepcopy(reference)
    changed["scenario"]["overlay_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="储能叠加层"):
        reference_inputs(changed)


def test_compatibility_entry_does_not_materialize_external_assets(tmp_path):
    destination = tmp_path / "external"
    for case in ("ieee4", "ieee37", "baran-wu69"):
        assert (
            load_catalog()["cases"][case]["role"]
            == "compatibility_with_voltage_violations"
        )
        with pytest.raises(ValueError, match="没有可生成"):
            stage_reference(case, destination)
    assert not destination.exists()


def test_verify_source_compares_original_bytes(monkeypatch, tmp_path):
    from scripts import vvc_case_library

    upstream = b"! original\r\n"
    file = tmp_path / "stock.dss"
    file.write_bytes(upstream)
    expected = {"stock.dss": sha256(upstream).hexdigest()}
    monkeypatch.setattr(
        vvc_case_library,
        "load_catalog",
        lambda: {"cases": {"stock": {"source": {"upstream_sha256": expected}}}},
    )
    assert vvc_case_library.verify_source("stock", tmp_path) == expected
    assert file.read_bytes() == upstream
    file.write_bytes(b"! original\n")
    with pytest.raises(ValueError, match="摘要不符"):
        vvc_case_library.verify_source("stock", tmp_path)
