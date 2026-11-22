"""公共 MARL CLI 的静态拒绝与运行期资源回收。"""

from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

from gymnasium.spaces import Box, Discrete
import pytest

from utils.train_cli import main


ROOT = Path(__file__).resolve().parents[1]


def test_config_preview_does_not_load_runtime():
    preview = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from scripts.train import main\n"
            "main(['--env', 'vvc', '--algo', 'happo', '--validate-config'])\n"
            "assert not {'utils.training_setup', 'gymnasium', 'torch', 'dss', "
            "'opendssdirect', 'runners', 'stable_baselines3'} & sys.modules.keys()",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert "配置校验通过" in preview.stdout


def test_unsupported_configuration_rejected_before_prepare_or_save(
    monkeypatch, tmp_path, capsys
):
    def unexpected_prepare(config):
        pytest.fail("不兼容配置不能调用 prepare_run")

    monkeypatch.setitem(
        sys.modules,
        "utils.training_setup",
        SimpleNamespace(prepare_run=unexpected_prepare),
    )
    config_path = tmp_path / "unreachable.yaml"
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "--env",
                "vvc",
                "--algo",
                "matd3",
                "--save-config",
                str(config_path),
            ]
        )
    assert raised.value.code == 2
    assert "matd3 动作支持" in capsys.readouterr().err
    assert not config_path.exists()


def _mock_runtime(monkeypatch, algorithm, events, wrapper=None):
    setup = ModuleType("utils.training_setup")
    setup.prepare_run = lambda config: events.append("prepare")
    runners = ModuleType("runners")

    def close():
        if wrapper is not None:
            wrapper.close()
        events.append("close")

    def make_runner(main_args, algo_args, env_args):
        assert main_args["algo"] == algorithm
        events.append("construct")
        return SimpleNamespace(
            envs=wrapper,
            run=lambda: events.append("run"),
            close=close,
        )

    runners.RUNNER_REGISTRY = {algorithm: make_runner}
    monkeypatch.setitem(sys.modules, "utils.training_setup", setup)
    monkeypatch.setitem(sys.modules, "runners", runners)


@pytest.mark.parametrize("matches", [True, False])
def test_runtime_preflight_checks_existing_worker_before_run(
    monkeypatch, capsys, matches
):
    events = []
    action = Discrete(3) if matches else Box(-1.0, 1.0, (1,))

    def manifest():
        events.append("inspect")
        return {"action_schema": [{"element": "Storage.example", "space": str(action)}]}

    wrapper = SimpleNamespace(
        action_space=[action],
        observation_space=[Box(-1.0, 1.0, (4,))],
        get_physics_manifest=manifest,
        close=lambda: events.append("worker close"),
    )
    _mock_runtime(monkeypatch, "happo", events, wrapper)

    args = ["--env", "vvc", "--algo", "happo"]
    if matches:
        main(args)
        assert events == [
            "prepare",
            "construct",
            "inspect",
            "run",
            "worker close",
            "close",
        ]
    else:
        with pytest.raises(SystemExit) as raised:
            main(args)
        assert raised.value.code == 2
        assert "与静态声明" in capsys.readouterr().err
        assert events == ["prepare", "construct", "inspect", "worker close", "close"]


@pytest.mark.parametrize(
    "environment,algorithm,extra",
    [
        ("dsr", "happo", []),
        ("vvc", "hasac", []),
        ("vvc", "happo", ["--set", "render.use_render=true"]),
    ],
)
def test_existing_unprofiled_or_render_paths_do_not_require_layout(
    monkeypatch, environment, algorithm, extra
):
    events = []
    _mock_runtime(monkeypatch, algorithm, events)
    main(["--env", environment, "--algo", algorithm, *extra])
    assert events == ["prepare", "construct", "run", "close"]
