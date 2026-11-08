"""直接编辑 YAML，显示最终参数和来源；新增环境字段无需维护第二套表单映射。"""

from pathlib import Path

import gradio as gr
import yaml

from training_ui.core.config_builder import config_to_yaml_str, resolve_config, save_config
from training_ui.core.vvc_config import (
    STANDARD_FEEDERS,
    YAML_FEEDER,
    describe_feeder,
    load_short_preset,
    select_feeder,
)
from utils.config_pipeline import ALGORITHM_SECTIONS, UniqueKeyLoader, read_yaml
from utils.config_validation import ConfigError
from utils.environment_registry import environment_spec


class BaseEnvironmentTab:
    env_name = "vvc"
    tab_label = "VVC"
    show_feeder_selection = False

    def build(self):
        root = Path(__file__).resolve().parents[2]
        sequential = environment_spec(self.env_name).interaction == "sequential"
        profiles = sorted(
            path.stem
            for path in (root / "configs/envs_cfgs").glob("*.yaml")
            if path.stem.startswith(self.env_name.split("_")[0])
        )
        algorithms = sorted(
            path.stem
            for path in (root / "configs/algos_cfgs").glob("*.yaml")
            if (path.stem == "sequential_ppo") == sequential
            if all(
                isinstance(read_yaml(path).get(key), dict) for key in ALGORITHM_SECTIONS
            )
        )

        def template(name):
            return config_to_yaml_str(
                read_yaml(root / "configs/envs_cfgs" / f"{name}.yaml")
            )

        def parse(text):
            values = yaml.load(text, Loader=UniqueKeyLoader)
            if not isinstance(values, dict):
                raise ConfigError("配置必须是 YAML 映射")
            return values

        def preview(algo, env, name, text, path, feeder=None):
            try:
                overrides = parse(text)
                config = resolve_config(algo, env, name, overrides)
                resolved_yaml = config_to_yaml_str(config.to_dict())
                summary = ""
                if feeder is not None:
                    summary = "\n" + describe_feeder(overrides, config.env_args, feeder)
            except Exception as exc:
                return "", f"配置校验失败\n{exc}", ""

            source = ""
            if path := (path or "").strip():
                try:
                    source = config_to_yaml_str(config.explain(path))
                except (ConfigError, KeyError) as exc:
                    return resolved_yaml, f"配置校验通过，未找到参数来源\n{exc}{summary}", ""
            return resolved_yaml, f"配置校验通过{summary}", source

        def launch(algo, env, name, text, feeder=None):
            try:
                overrides = parse(text)
                config = resolve_config(algo, env, name, overrides)
                summary = ""
                if feeder is not None:
                    summary = "\n" + describe_feeder(overrides, config.env_args, feeder)
                path, document = save_config(config)
                run = document["run"]
                from training_ui.core.process_manager import process_manager

                task = process_manager.start(
                    run["algo"],
                    run["env"],
                    run["exp_name"],
                    path,
                    system_name=document["env"].get("system_name", ""),
                )
                return f"已启动 {task}。请在「训练监控」查看日志。{summary}\n最终配置：{path}"
            except Exception as exc:
                return f"启动失败\n{exc}"

        with gr.Tab(label=self.tab_label, elem_classes="experiment-tab") as tab:
            gr.Markdown(
                "## 实验配置\n"
                "选择算法和环境配置，编辑左侧 YAML。点击「校验并预览」查看合并默认值后的完整配置。",
                elem_classes="experiment-intro",
            )
            if self.show_feeder_selection:
                with gr.Row(elem_classes="experiment-actions"):
                    feeder = gr.Dropdown(
                        choices=[*STANDARD_FEEDERS, YAML_FEEDER],
                        value=STANDARD_FEEDERS[0],
                        label="配电网络",
                        elem_id="vvc-feeder",
                    )
                    preset = gr.Button("载入短训练预设", scale=0, elem_id="vvc-short-preset")
                gr.Markdown(
                    "切换配电网络只修改系统引用。手写自定义模型时，请选「由 YAML 指定」。"
                    "短训练预设会替换左侧 YAML，选择 HAPPO，并载入 CPU 训练参数，保留实验名称。"
                    "配置修改后，启动前须重新校验。",
                    elem_classes="experiment-note",
                )
            with gr.Row():
                algo = gr.Dropdown(
                    choices=algorithms,
                    value="sequential_ppo" if sequential else "happo",
                    label="算法",
                )
                env = gr.Dropdown(
                    choices=profiles, value=self.env_name, label="环境配置"
                )
                name = gr.Textbox(value="experiment", label="实验名称")
            with gr.Row(elem_classes="experiment-config"):
                editor = gr.Code(
                    value=template(self.env_name),
                    language="yaml",
                    label="实验覆盖 YAML",
                    lines=16,
                    max_lines=16,
                    min_width=285,
                    elem_id=f"{self.env_name}-overrides",
                )
                resolved = gr.Code(
                    language="yaml",
                    label="最终生效配置",
                    interactive=False,
                    lines=16,
                    max_lines=16,
                    min_width=285,
                    elem_id=f"{self.env_name}-resolved",
                )
            with gr.Row(elem_classes="experiment-actions"):
                path = gr.Textbox(
                    value="env.max_episode_steps",
                    label="查询参数来源（完整路径）",
                    min_width=245,
                )
                check = gr.Button("校验并预览", scale=0)
                start = gr.Button("保存配置并训练", variant="primary", scale=0)
            status = gr.Textbox(
                label="校验 / 运行状态",
                placeholder="校验和启动结果显示在这里。",
                lines=2,
                max_lines=4,
                interactive=False,
                elem_id=f"{self.env_name}-status",
                elem_classes="experiment-status",
            )
            with gr.Accordion("参数来源详情", open=False):
                source_details = gr.Textbox(
                    label="来源明细",
                    placeholder="填写参数路径并校验后，这里会列出覆盖顺序。",
                    lines=6,
                    max_lines=10,
                    interactive=False,
                    elem_classes="experiment-status",
                )
            gr.Markdown(
                "`train.episode_length` 是采样长度，"
                "`env.max_episode_steps` 是单局步数，`seed.seed` 是随机种子。",
                elem_classes="experiment-note",
            )
            if self.show_feeder_selection:
                verified = gr.State()
                events = {"concurrency_id": "vvc-config", "concurrency_limit": 1}

                def change_feeder(text, selected):
                    if selected == YAML_FEEDER:
                        return (
                            gr.skip(), "", "由 YAML 指定。编辑后请校验并预览。",
                            "", None, gr.update(interactive=False),
                        )
                    try:
                        overrides = select_feeder(parse(text), selected)
                    except Exception as exc:
                        return (
                            gr.skip(), "", f"切换未应用，保留原 YAML。\n{exc}",
                            "", None, gr.update(interactive=True),
                        )
                    return (
                        config_to_yaml_str(overrides), "",
                        f"已选择 {selected}，请校验并预览。", "", None,
                        gr.update(interactive=True),
                    )

                def change_profile(profile):
                    selected = STANDARD_FEEDERS[0] if profile == self.env_name else YAML_FEEDER
                    return (
                        template(profile), "", "", "", None, selected,
                        gr.update(interactive=selected in STANDARD_FEEDERS),
                    )

                def apply_preset(selected):
                    try:
                        algorithm, overrides = load_short_preset(selected)
                    except Exception as exc:
                        return gr.skip(), gr.skip(), "", f"未载入短训练预设。\n{exc}", "", None
                    return (
                        algorithm, config_to_yaml_str(overrides), "",
                        f"已载入 {selected} 的 CPU 短训练预设，请校验并预览。", "", None,
                    )

                def preview_feeder(algo, env, name, text, path, selected):
                    result = preview(algo, env, name, text, path, selected)
                    current = (algo, env, name, text, selected) if result[0] else None
                    return (*result, current)

                def launch_feeder(algo, env, name, text, selected, current):
                    if current != (algo, env, name, text, selected):
                        return "当前设置尚未校验，请先点击「校验并预览」。"
                    return launch(algo, env, name, text, selected)

                inputs = [algo, env, name, editor]
                outputs = [editor, resolved, status, source_details, verified]
                env.change(
                    change_profile,
                    inputs=[env],
                    outputs=[*outputs, feeder, preset],
                    **events,
                )
                feeder.input(
                    change_feeder,
                    inputs=[editor, feeder],
                    outputs=[*outputs, preset],
                    trigger_mode="always_last",
                    **events,
                )
                preset.click(
                    apply_preset, inputs=[feeder], outputs=[algo, *outputs], **events,
                )
                check.click(
                    preview_feeder,
                    inputs=[*inputs, path, feeder],
                    outputs=[resolved, status, source_details, verified],
                    show_progress="minimal",
                    **events,
                )
                start.click(
                    launch_feeder,
                    inputs=[*inputs, feeder, verified],
                    outputs=[status],
                    **events,
                )
            else:
                env.change(
                    lambda name: (template(name), "", "", ""),
                    inputs=[env],
                    outputs=[editor, resolved, status, source_details],
                )
                check.click(
                    preview,
                    inputs=[algo, env, name, editor, path],
                    outputs=[resolved, status, source_details],
                    show_progress="minimal",
                )
                start.click(launch, inputs=[algo, env, name, editor], outputs=[status])
        return tab
