"""检查当前训练解释器的依赖与原生计算。"""

import subprocess
import sys

import gradio as gr

from training_ui.core.process_manager import PROJECT_ROOT, subprocess_environment

CHECK_TIMEOUT_SECONDS = 60


def check_environment() -> tuple[str, str]:
    """用训练子进程的 Python 和工作目录运行轻量检查。"""
    script = PROJECT_ROOT / "scripts" / "check_runtime.py"
    location = f"Python  {sys.executable}\n工作目录  {PROJECT_ROOT}"
    try:
        result = subprocess.run(
            [sys.executable, "-u", str(script), "--preflight"],
            cwd=PROJECT_ROOT,
            env=subprocess_environment(),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=CHECK_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"检查超过 {CHECK_TIMEOUT_SECONDS} 秒，已结束。", location
    except OSError as exc:
        return "无法启动环境检查。", f"{location}\n\n{exc}"

    report = result.stdout.strip() or location
    if result.returncode == 0:
        return "环境检查通过。", report
    if result.stderr.strip():
        report += f"\n\n附加输出\n{result.stderr.strip()}"
    return "环境检查发现问题，请查看明细。", report


def build_environment_tab() -> gr.Tab:
    with gr.Tab(label="环境检查", elem_classes="experiment-tab") as tab:
        gr.Markdown(
            "## 训练运行环境\n"
            "检查当前训练解释器的依赖导入、PyTorch 张量运算和 OpenDSS 原生求解。"
            "检查过程只读，无需启动训练。",
            elem_classes="experiment-intro",
        )
        run_check = gr.Button("运行环境检查", variant="primary", scale=0)
        status = gr.Textbox(
            label="检查状态",
            value="尚未运行",
            interactive=False,
            elem_id="environment-check-status",
        )
        report = gr.Code(
            label="检查明细",
            lines=17,
            max_lines=26,
            interactive=False,
            elem_id="environment-check-report",
        )
        gr.Markdown(
            '缺少依赖时，先核对报告中的 Python 路径。可切换到完整环境，'
            '或在目标环境中从仓库根目录运行 `python -m pip install -e ".[viz]"`，'
            '再重启训练工作台。',
            elem_classes="experiment-note",
        )
        run_check.click(check_environment, outputs=[status, report])
    return tab
