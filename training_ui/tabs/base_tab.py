"""直接编辑 YAML，显示最终参数和来源；新增环境字段无需维护第二套表单映射。"""

from pathlib import Path
import gradio as gr
import yaml
from utils.config_pipeline import read_yaml, UniqueKeyLoader, ALGORITHM_SECTIONS
from utils.config_validation import ConfigError
from utils.environment_registry import environment_spec
from training_ui.core.config_builder import (
    resolve_config,
    build_config,
    config_to_yaml_str,
)


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
                source = config_to_yaml_str(config.explain(path)) if path else ""
                return config_to_yaml_str(config.to_dict()), "配置校验通过\n" + source
            except Exception as exc:
                return "", str(exc)

        def launch(algo, env, name, text):
            try:
                path, document = build_config(algo, env, name, parse(text))
                run = document["run"]
                from training_ui.core.process_manager import process_manager

                task = process_manager.start(
                    run["algo"], run["env"], run["exp_name"], path
                )
                return f"已启动 {task}，最终配置：{path}"
            except Exception as exc:
                return str(exc)

        with gr.Tab(label=self.tab_label) as tab:
            gr.Markdown(
                "配置采用与命令行相同的 YAML。左侧只写需要覆盖的参数；右侧预览包含默认值的完整配置。"
                "采样长度用 train.episode_length，单局长度用 env.max_episode_steps；随机种子只设置 seed.seed。"
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
            with gr.Row():
                editor = gr.Code(
                    value=template(self.env_name),
                    language="yaml",
                    label="实验覆盖 YAML",
                    lines=26,
                )
                resolved = gr.Code(
                    language="yaml", label="最终生效配置", interactive=False, lines=26
                )
            env.change(template, inputs=[env], outputs=[editor])
            path = gr.Textbox(
                value="env.max_episode_steps", label="查询参数来源（完整路径）"
            )
            status = gr.Textbox(
                label="校验 / 来源 / 运行状态", lines=8, interactive=False
            )
            with gr.Row():
                check = gr.Button("校验并预览")
                start = gr.Button("保存配置并训练", variant="primary")
            check.click(
                preview,
                inputs=[algo, env, name, editor, path],
                outputs=[resolved, status],
            )
            start.click(launch, inputs=[algo, env, name, editor], outputs=[status])
        return tab
