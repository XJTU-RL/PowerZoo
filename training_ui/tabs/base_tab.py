"""直接编辑 YAML，显示最终参数和来源；新增环境字段无需维护第二套表单映射。"""

from pathlib import Path

import gradio as gr
import yaml

from training_ui.core.config_builder import build_config, config_to_yaml_str, resolve_config
from utils.config_pipeline import ALGORITHM_SECTIONS, UniqueKeyLoader, read_yaml
from utils.config_validation import ConfigError
from utils.environment_registry import environment_spec


class BaseEnvironmentTab:
    env_name = "vvc"
    tab_label = "VVC"

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

        def preview(algo, env, name, text, path):
            try:
                config = resolve_config(algo, env, name, parse(text))
                resolved_yaml = config_to_yaml_str(config.to_dict())
            except Exception as exc:
                return "", f"配置校验失败\n{exc}", ""

            source = ""
            if path := (path or "").strip():
                try:
                    source = config_to_yaml_str(config.explain(path))
                except (ConfigError, KeyError) as exc:
                    return resolved_yaml, f"配置校验通过，未找到参数来源\n{exc}", ""
            return resolved_yaml, "配置校验通过", source

        def launch(algo, env, name, text):
            try:
                path, document = build_config(algo, env, name, parse(text))
                run = document["run"]
                from training_ui.core.process_manager import process_manager

                task = process_manager.start(
                    run["algo"], run["env"], run["exp_name"], path
                )
                return f"已启动 {task}。请在「训练监控」查看日志。\n最终配置：{path}"
            except Exception as exc:
                return f"启动失败\n{exc}"

        with gr.Tab(label=self.tab_label, elem_classes="experiment-tab") as tab:
            gr.Markdown(
                "## 实验配置\n"
                "选择算法和环境配置，编辑左侧 YAML。点击「校验并预览」查看合并默认值后的完整配置。",
                elem_classes="experiment-intro",
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
