# -*- coding: utf-8 -*-
"""
Tab 2: Live Inference - 实时单步推理

提供单步推理控制、手动动作覆写、
拓扑图 + 电压剖面 + 设备状态实时显示。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np

from envs.render_common.engine.inference_engine import InferenceEngine
from envs.vvc.render.engine.episode_runner import VVCEpisodeRunner
from envs.vvc.render.engine.manual_override import VVCManualOverride
from envs.vvc.render.viz.plotly.topology_graph import create_topology_figure
from envs.vvc.render.viz.plotly.voltage_profile import create_voltage_profile
from envs.vvc.render.viz.plotly.device_schedule_chart import create_device_schedule

logger = logging.getLogger(__name__)


def _build_default_config(system: str = "13Bus") -> Dict[str, Any]:
    """构建默认 VVC 推理配置。"""
    from utils.config_pipeline import ConfigLoader
    from utils.config_validation import build_dataclass, environment_class

    cfg = ConfigLoader().load("happo", "vvc", system_ref=system)
    return build_dataclass(environment_class("vvc"), cfg.env_args)


def _init_runner(
    inference_engine: Optional[InferenceEngine],
    system: str,
) -> Tuple[Optional[VVCEpisodeRunner], str]:
    """初始化 episode runner。"""
    config = (
        inference_engine.config
        if inference_engine is not None
        else _build_default_config(system)
    )
    try:
        runner = VVCEpisodeRunner(
            config=config,
            inference_engine=inference_engine,
            collect_snapshots=True,
        )
        runner._init_env(seed=42)
        return runner, "Runner initialized"
    except Exception as exc:
        logger.error(f"Runner init failed: {exc}", exc_info=True)
        return None, f"Init failed: {exc}"


def _run_single_step(
    runner: Optional[VVCEpisodeRunner],
    override: Optional[VVCManualOverride],
    use_model: bool,
    snapshots: List[Dict[str, Any]],
    step_count: int,
    bus_coords: Dict,
) -> Tuple[
    Optional[VVCEpisodeRunner],
    List[Dict[str, Any]],
    int,
    Any,
    Any,
    Any,
    str,
]:
    """执行单步推理。"""
    if runner is None:
        empty = gr.Plot()
        return runner, snapshots, step_count, empty, empty, empty, "No runner"

    try:
        if step_count == 0:
            runner._init_env(seed=42)

        actions = None
        if use_model and runner.inference_engine is not None:
            actions = runner.inference_engine.infer(
                runner._current_obs,
                available_actions=runner._current_avail_actions,
            )
        else:
            actions = runner._random_actions()

        if override is not None:
            actions = override.apply(actions)

        snap = runner.run_step(step=step_count + 1, actions=actions)
        if snap is not None:
            snapshots = snapshots + [snap]

        step_count += 1

        # 更新图表
        topo_fig = None
        vp_fig = None
        ds_fig = None

        if snap:
            topo_fig = create_topology_figure(
                bus_data=snap.get("buses", {}),
                line_data=snap.get("lines", {}),
                bus_coords=bus_coords if bus_coords else None,
                device_data=snap.get("devices", {}),
                step=snap.get("step", step_count),
            )
            vp_fig = create_voltage_profile(snap, step=snap.get("step", step_count))

        if snapshots:
            ds_fig = create_device_schedule(snapshots)

        reward = snap.get("step_reward", 0.0) if snap else 0.0
        status = f"Step {step_count} | Reward: {reward:.4f}"

        return runner, snapshots, step_count, topo_fig, vp_fig, ds_fig, status

    except Exception as exc:
        logger.error(f"Step failed: {exc}", exc_info=True)
        empty = gr.Plot()
        return runner, snapshots, step_count, empty, empty, empty, f"Error: {exc}"


def _run_full_episode(
    runner: Optional[VVCEpisodeRunner],
    use_model: bool,
    bus_coords: Dict,
) -> Tuple[
    Optional[VVCEpisodeRunner],
    List[Dict[str, Any]],
    int,
    Any,
    Any,
    Any,
    str,
]:
    """运行完整 episode (24步)。"""
    if runner is None:
        empty = gr.Plot()
        return runner, [], 0, empty, empty, empty, "No runner"

    try:
        episode_data = runner.run_episode()
        snaps = episode_data.snapshots
        n_steps = episode_data.episode_length

        topo_fig = None
        vp_fig = None
        ds_fig = None

        if snaps:
            last = snaps[-1]
            topo_fig = create_topology_figure(
                bus_data=last.get("buses", {}),
                line_data=last.get("lines", {}),
                bus_coords=bus_coords if bus_coords else None,
                device_data=last.get("devices", {}),
                step=last.get("step", n_steps),
            )
            vp_fig = create_voltage_profile(last, step=last.get("step", n_steps))
            ds_fig = create_device_schedule(snaps)

        status = (
            f"Episode done: {n_steps} steps | "
            f"Total Reward: {episode_data.total_reward:.4f}"
        )
        return runner, snaps, n_steps, topo_fig, vp_fig, ds_fig, status

    except Exception as exc:
        logger.error(f"Episode failed: {exc}", exc_info=True)
        empty = gr.Plot()
        return runner, [], 0, empty, empty, empty, f"Error: {exc}"


def _update_override(
    override: Optional[VVCManualOverride],
    agent_id: int,
    action_idx: int,
    value: float,
) -> Tuple[Optional[VVCManualOverride], str]:
    """设置手动覆写值。"""
    if override is None:
        return override, "No override manager"
    override.set_override(agent_id, action_idx, value)
    return (
        override,
        f"Override set: agent {agent_id}, action {action_idx} = {value:.2f}",
    )


def _clear_overrides(
    override: Optional[VVCManualOverride],
) -> Tuple[Optional[VVCManualOverride], str]:
    """清除所有覆写。"""
    if override is None:
        return override, "No override manager"
    override.clear_all()
    return override, "All overrides cleared"


def create_tab(shared_states: Dict[str, gr.State]) -> Dict[str, Any]:
    """创建 Live Inference Tab 的 UI 布局和事件绑定。

    Args:
            shared_states: 跨 Tab 共享状态字典

    Returns:
            该 Tab 内关键组件的引用字典
    """
    state_model_loaded = shared_states["model_loaded"]
    state_inference_engine = shared_states["inference_engine"]
    state_bus_coords = shared_states["bus_coords"]

    # 本地状态
    state_runner = gr.State(None)
    state_override = gr.State(None)
    state_snapshots = gr.State([])
    state_step_count = gr.State(0)

    # === 控制面板 ===
    with gr.Group():
        gr.Markdown("### Inference Control")
        with gr.Row():
            system_dropdown = gr.Dropdown(
                choices=["13Bus", "34Bus", "123Bus"],
                value="13Bus",
                label="System",
                scale=1,
            )
            init_btn = gr.Button("Init Runner", variant="primary", scale=1)
            step_btn = gr.Button("Step", variant="secondary", scale=1)
            episode_btn = gr.Button("Run Episode", variant="secondary", scale=1)
            reset_btn = gr.Button("Reset", scale=1)

        with gr.Row():
            use_model_cb = gr.Checkbox(
                label="Use Model (vs Random)",
                value=True,
                scale=2,
            )

    # === 手动覆写 ===
    with gr.Accordion("Manual Override", open=False):
        with gr.Row():
            ov_agent = gr.Number(label="Agent ID", value=0, precision=0, scale=1)
            ov_action = gr.Number(label="Action Index", value=0, precision=0, scale=1)
            ov_value = gr.Number(label="Value", value=0.0, scale=1)
            ov_set_btn = gr.Button("Set", scale=1)
            ov_clear_btn = gr.Button("Clear All", scale=1)

    # === 可视化 ===
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Network Topology")
            topology_plot = gr.Plot(label="Topology")
        with gr.Column(scale=2):
            gr.Markdown("### Voltage Profile")
            voltage_plot = gr.Plot(label="Voltage Profile")

    with gr.Row():
        gr.Markdown("### Device Schedule")
    device_plot = gr.Plot(label="Device Schedule")

    status_box = gr.Textbox(label="Status", interactive=False, lines=1)

    # --- 事件绑定 ---
    def on_init(engine, system, bus_coords):
        runner, msg = _init_runner(engine, system)
        override = None
        if runner is not None:
            override = VVCManualOverride.from_env(runner._env)

        return runner, override, [], 0, msg

    def on_step(runner, override, use_model, snapshots, step_count, bus_coords):
        return _run_single_step(
            runner,
            override,
            use_model,
            snapshots,
            step_count,
            bus_coords,
        )

    def on_episode(runner, use_model, bus_coords):
        return _run_full_episode(runner, use_model, bus_coords)

    def on_reset(runner):
        if runner is not None:
            try:
                runner._init_env(seed=42)
            except Exception:
                pass
        return runner, [], 0, gr.Plot(), gr.Plot(), gr.Plot(), "Reset done"

    def on_set_override(override, agent_id, action_idx, value):
        return _update_override(override, int(agent_id), int(action_idx), value)

    def on_clear_override(override):
        return _clear_overrides(override)

    init_btn.click(
        fn=on_init,
        inputs=[state_inference_engine, system_dropdown, state_bus_coords],
        outputs=[
            state_runner,
            state_override,
            state_snapshots,
            state_step_count,
            status_box,
        ],
    )
    step_btn.click(
        fn=on_step,
        inputs=[
            state_runner,
            state_override,
            use_model_cb,
            state_snapshots,
            state_step_count,
            state_bus_coords,
        ],
        outputs=[
            state_runner,
            state_snapshots,
            state_step_count,
            topology_plot,
            voltage_plot,
            device_plot,
            status_box,
        ],
    )
    episode_btn.click(
        fn=on_episode,
        inputs=[state_runner, use_model_cb, state_bus_coords],
        outputs=[
            state_runner,
            state_snapshots,
            state_step_count,
            topology_plot,
            voltage_plot,
            device_plot,
            status_box,
        ],
    )
    reset_btn.click(
        fn=on_reset,
        inputs=[state_runner],
        outputs=[
            state_runner,
            state_snapshots,
            state_step_count,
            topology_plot,
            voltage_plot,
            device_plot,
            status_box,
        ],
    )
    ov_set_btn.click(
        fn=on_set_override,
        inputs=[state_override, ov_agent, ov_action, ov_value],
        outputs=[state_override, status_box],
    )
    ov_clear_btn.click(
        fn=on_clear_override,
        inputs=[state_override],
        outputs=[state_override, status_box],
    )

    return {
        "topology_plot": topology_plot,
        "voltage_plot": voltage_plot,
        "device_plot": device_plot,
        "status_box": status_box,
        "step_btn": step_btn,
        "episode_btn": episode_btn,
    }
