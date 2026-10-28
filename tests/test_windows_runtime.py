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


def test_dataset_respects_another_process_lock(tmp_path, monkeypatch):
    """真正的数据入口在另一进程持锁时不能写文件，释放后只生成一套数据。"""
    import hashlib
    import json
    import time
    from filelock import FileLock
    from envs.smartgrid.base_env.env_config import DataConfig
    from envs.smartgrid.data_process import dataset

    cfg = DataConfig(episodes_per_split=6)
    key = hashlib.sha256(
        json.dumps(dataset._generation(cfg, 24), sort_keys=True).encode()
    ).hexdigest()[:20]
    folder = tmp_path / "results/smartgrid_data" / key
    folder.parent.mkdir(parents=True)
    marker = tmp_path / "waiting"
    code = """import sys
from pathlib import Path
from envs.smartgrid.data_process import dataset
from envs.smartgrid.base_env.env_config import DataConfig
dataset.ROOT=Path(sys.argv[1])
original=dataset.FileLock
class SignalingLock(original):
    def acquire(self,*a,**k):
        Path(sys.argv[2]).write_text('waiting')
        return super().acquire(*a,**k)
dataset.FileLock=SignalingLock
dataset.prepare_dataset(DataConfig(episodes_per_split=6),24)
"""
    lock = FileLock(str(folder) + ".lock")
    lock.acquire()
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path), str(marker)],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 20
        while (
            not marker.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert marker.exists() and process.poll() is None
        assert not folder.exists()
        lock.release()
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, (stdout, stderr)
        before = sorted(
            (p.relative_to(folder), p.read_bytes())
            for p in folder.rglob("*")
            if p.is_file()
        )
        monkeypatch.setattr(dataset, "ROOT", tmp_path)
        dataset.prepare_dataset(cfg, 24)
        assert before == sorted(
            (p.relative_to(folder), p.read_bytes())
            for p in folder.rglob("*")
            if p.is_file()
        )
    finally:
        lock.release()
        if process.poll() is None:
            process.kill()
            process.communicate()
