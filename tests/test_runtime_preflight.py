"""The UI preflight reports missing packages without starting rollout workers."""

import sys
from types import SimpleNamespace

import pytest

from scripts import check_runtime


@pytest.mark.parametrize("missing_module", [None, "gymnasium", "dss"])
def test_preflight_reports_independent_checks(missing_module, monkeypatch, capsys):
    def import_module(name):
        if name == missing_module:
            raise ModuleNotFoundError(f"No module named '{name}'")
        return SimpleNamespace(__version__="1.0")

    def unexpected_run(*args, **kwargs):
        pytest.fail("Preflight started a rollout or training")

    monkeypatch.setattr(
        check_runtime, "importlib", SimpleNamespace(import_module=import_module)
    )
    monkeypatch.setattr(
        check_runtime,
        "check_torch",
        lambda require_cuda: {"torch": "2.6", "devices": ["cpu"], "cuda": None},
    )
    monkeypatch.setattr(
        check_runtime,
        "check_native_dss",
        lambda: {"dss_version": "DSS C-API", "direct_version": "0.9.4"},
    )
    monkeypatch.setattr(check_runtime, "check_vvc", unexpected_run)
    monkeypatch.setattr(check_runtime, "check_training", unexpected_run)
    monkeypatch.setattr(sys, "argv", ["check_runtime.py", "--preflight"])

    with pytest.raises(SystemExit) as result:
        check_runtime.main()

    assert result.value.code == (1 if missing_module else 0)
    output = capsys.readouterr().out
    assert "[通过] gym  1.0" in output
    assert "[通过] PyTorch 张量运算  2.6  实测设备 CPU" in output
    if missing_module:
        assert f"[失败] {missing_module}  缺少模块 ModuleNotFoundError" in output
    if missing_module == "dss":
        assert "[略过] OpenDSS 原生求解" in output
    else:
        assert "[通过] OpenDSS 原生求解  OpenDSSDirect.py 0.9.4" in output
        assert "DSS C-API" in output
