"""Browse training outputs and copy paths for replay."""

from pathlib import Path

import gradio as gr

from training_ui.core.config_builder import config_to_yaml_str
from training_ui.core.model_discovery import filter_models, scan_models
from training_ui.i18n import t


def build_models_tab() -> gr.Tab:
    with gr.Tab(t("tab_models")) as tab:
        gr.Markdown(t("models_heading"))
        gr.Markdown("扫描项目的 `results/`。选择实验行查看生效配置与模型文件。")

        with gr.Row():
            env_filter = gr.Dropdown(
                label=t("label_env_filter"), choices=[("全部", "")], value="", scale=2
            )
            algo_filter = gr.Dropdown(
                label=t("label_algo_filter"), choices=[("全部", "")], value="", scale=2
            )
            scan_btn = gr.Button(t("btn_scan_results"), variant="primary", scale=1)

        records_state = gr.State([])
        models_table = gr.Dataframe(
            headers=[
                t("col_env"),
                t("col_system"),
                t("col_algorithm"),
                t("col_experiment"),
                t("col_run_id"),
                t("col_checkpoints"),
            ],
            label=t("label_discovered_models"),
            value=[],
            interactive=False,
        )

        with gr.Row():
            checkpoint_dropdown = gr.Dropdown(
                label=t("label_checkpoint"), choices=[], interactive=True, scale=2
            )
            model_path_box = gr.Textbox(
                label=t("label_model_path"), interactive=False, show_copy_button=True, scale=3
            )
        model_config = gr.Code(
            label=t("label_training_config"), language="yaml", interactive=False, lines=16
        )

        def row(record):
            return [
                record.env, record.system, record.algo, record.exp_name,
                record.run_id, len(record.checkpoints),
            ]

        def refresh(env, algo):
            records = scan_models()
            env_choices = sorted({record.env for record in records})
            algo_choices = sorted({record.algo for record in records})
            env = env if env in env_choices else ""
            algo = algo if algo in algo_choices else ""
            filtered = filter_models(records, env=env, algo=algo)
            return (
                gr.update(choices=[("全部", ""), *env_choices], value=env),
                gr.update(choices=[("全部", ""), *algo_choices], value=algo),
                [row(record) for record in filtered],
                filtered,
                gr.update(choices=[], value=None),
                "",
                "",
            )

        def select_record(records, event: gr.SelectData):
            selected_row = getattr(event, "row_value", None)
            if selected_row is not None:
                record = next(
                    (item for item in records if row(item)[:5] == list(selected_row[:5])),
                    None,
                )
            else:
                index = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
                record = records[index] if isinstance(index, int) and 0 <= index < len(records) else None
            if record is None:
                return gr.update(choices=[], value=None), "", ""

            choices = [
                (name, str(Path(record.run_dir) / name)) for name in record.checkpoints
            ]
            return (
                gr.update(choices=choices, value=None),
                record.run_dir,
                config_to_yaml_str(record.config) if record.config else "",
            )

        inputs = [env_filter, algo_filter]
        outputs = [
            *inputs, models_table, records_state, checkpoint_dropdown, model_path_box,
            model_config,
        ]
        scan_btn.click(refresh, inputs=inputs, outputs=outputs)
        env_filter.input(refresh, inputs=inputs, outputs=outputs)
        algo_filter.input(refresh, inputs=inputs, outputs=outputs)
        models_table.select(
            select_record,
            inputs=records_state,
            outputs=[checkpoint_dropdown, model_path_box, model_config],
        )
        checkpoint_dropdown.input(
            lambda path: path or "", inputs=checkpoint_dropdown, outputs=model_path_box
        )

    return tab
