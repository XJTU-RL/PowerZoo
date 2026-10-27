"""Regression checks for Windows and arbitrarily named development worktrees."""

from pathlib import Path
import subprocess
import sys


def test_legacy_vvc_path_lookup_terminates_in_worktree():
    """Legacy lookup must terminate without requiring a folder named PowerZoo."""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from envs.vvc.vvc.env_register import get_info_and_folder; "
            "from pathlib import Path; "
            "_, folder = get_info_and_folder('13Bus'); "
            "assert Path(folder).resolve() == (Path.cwd() / 'node_systems').resolve()",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_both_native_dss_apis_solve():
    """Both Python bindings must load their native library and solve a circuit."""
    from scripts.check_runtime import check_native_dss

    result = check_native_dss()
    assert "DSS" in result["dss_version"]


def test_pv_config_respects_another_process_lock(tmp_path):
    """A busy lock must prevent writing; generation can resume after release."""
    from filelock import FileLock

    from envs.smartgrid.data_process.loadprofile_config import ConfigGenerator

    root = Path(__file__).resolve().parents[1]
    with FileLock(str(tmp_path / ".pv_config.lock")):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; "
                "from envs.smartgrid.data_process.loadprofile_config import ConfigGenerator; "
                "assert not ConfigGenerator(sys.argv[1], 24).ensure_default_pv_config()",
                str(tmp_path),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert not list(tmp_path.glob("*.dss"))

    assert ConfigGenerator(str(tmp_path), 24).ensure_default_pv_config()
    assert (tmp_path / "pv_data.dss").is_file()
    assert len(list(tmp_path.glob("pv_data_*.dss"))) == 8
