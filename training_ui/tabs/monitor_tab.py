"""Monitor training tasks and their subprocess logs."""

from datetime import datetime

import gradio as gr

from training_ui.core.process_manager import process_manager
from training_ui.i18n import t


def build_monitor_tab() -> gr.Tab:
    with gr.Tab(t("tab_monitor")) as tab:
        gr.Markdown(t("monitor_heading"))

        with gr.Row():
            task_selector = gr.Dropdown(
                label=t("label_active_task"), choices=[], interactive=True, scale=3
            )
            refresh_btn = gr.Button(t("btn_refresh"), scale=1)
            stop_btn = gr.Button(
                t("btn_stop_training"), variant="stop", interactive=False, scale=1
            )
        stop_receipt = gr.Textbox(label="停止操作结果", interactive=False, visible=False)

        with gr.Row():
            task_info = gr.JSON(label=t("label_task_info"), value={})
            log_output = gr.Textbox(
                label=t("label_log_output"),
                lines=16,
                max_lines=16,
                interactive=False,
                scale=2,
                elem_classes=["training-log"],
            )

        gr.Markdown(t("monitor_history_heading"))
        history_table = gr.Dataframe(
            headers=[
                t("col_task_id"),
                t("col_algorithm"),
                t("col_environment"),
                "网络",
                t("col_experiment"),
                t("col_status"),
                t("col_start_time"),
                t("col_duration"),
            ],
            label=t("label_all_tasks"),
            interactive=False,
        )
        timer = gr.Timer(value=3)

        def selected_details(task_id):
            task = process_manager.status(task_id) if task_id else None
            if task is None:
                return {}, "", gr.update(interactive=False)
            return (
                {
                    "task_id": task.task_id,
                    "algorithm": task.algo,
                    "environment": task.env,
                    "system_name": task.system_name,
                    "experiment": task.exp_name,
                    "status": task.status,
                    "pid": task.pid,
                    "start_time": task.start_time,
                    "end_time": task.end_time,
                    "config_path": task.config_path,
                    "log_file": task.log_file,
                },
                process_manager.log_tail(task_id),
                gr.update(interactive=task.status == "running"),
            )

        def refresh_tasks(task_id):
            tasks = process_manager.list_all()
            choices = [
                (f"{task.task_id} · {task.exp_name} · {task.algo}/{task.env}", task.task_id)
                for task in tasks
            ]
            if not any(task.task_id == task_id for task in tasks):
                task_id = tasks[0].task_id if tasks else None

            rows = []
            for task in tasks:
                start = datetime.fromisoformat(task.start_time)
                end = datetime.fromisoformat(task.end_time) if task.end_time else datetime.now()
                rows.append([
                    task.task_id, task.algo, task.env, task.system_name or "—",
                    task.exp_name, task.status,
                    task.start_time, str(end - start).split(".")[0],
                ])

            return gr.update(choices=choices, value=task_id), rows, *selected_details(task_id)

        def stop_task(task_id):
            task = process_manager.status(task_id) if task_id else None
            if task is None:
                message = t("msg_no_task")
            elif task.status != "running":
                message = f"{task_id} 当前状态为 {task.status}。"
            elif process_manager.stop(task_id):
                message = f"{task_id} {t('msg_task_stopped')}"
            else:
                message = f"{t('msg_task_stop_failed')} {task_id}"
            return gr.update(value=message, visible=True), *refresh_tasks(task_id)

        detail_outputs = [task_info, log_output, stop_btn]
        refresh_outputs = [task_selector, history_table, *detail_outputs]
        refresh_btn.click(refresh_tasks, inputs=task_selector, outputs=refresh_outputs)
        task_selector.input(selected_details, inputs=task_selector, outputs=detail_outputs)
        stop_btn.click(stop_task, inputs=task_selector, outputs=[stop_receipt, *refresh_outputs])
        timer.tick(
            refresh_tasks, inputs=task_selector, outputs=refresh_outputs, show_progress="hidden"
        )

    return tab
