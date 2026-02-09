"""Training monitoring tab - view running tasks, logs, and history."""

import gradio as gr

from training_ui.i18n import t


def build_monitor_tab() -> gr.Tab:
	"""Build the training monitor tab.

	Layout:
		Row 1: Active task selector (Dropdown) + Refresh button
		Row 2: Task info card (algo, env, status, PID, elapsed time)
		Row 3: Stop button
		Row 4: Real-time log viewer (auto-refresh every 3s)
		Row 5: Task history table
	"""
	with gr.Tab(t("tab_monitor")) as tab:
		gr.Markdown(t("monitor_heading"))

		with gr.Row():
			task_selector = gr.Dropdown(
				label=t("label_active_task"),
				choices=[],
				interactive=True,
				scale=3,
			)
			refresh_btn = gr.Button(t("btn_refresh"), variant="secondary", scale=1)

		# Task info card
		with gr.Row():
			with gr.Column():
				task_info = gr.JSON(label=t("label_task_info"), value={})
			with gr.Column():
				with gr.Row():
					stop_btn = gr.Button(t("btn_stop_training"), variant="stop")
					# empty column for spacing

		# Log viewer
		gr.Markdown(t("monitor_log_heading"))
		log_output = gr.Textbox(
			label=t("label_log_output"),
			lines=25,
			interactive=False,
		)

		# Auto-refresh timer
		timer = gr.Timer(value=3)

		# Task history
		gr.Markdown(t("monitor_history_heading"))
		history_table = gr.Dataframe(
			headers=[t("col_task_id"), t("col_algorithm"), t("col_environment"), t("col_status"), t("col_start_time"), t("col_duration")],
			label=t("label_all_tasks"),
			interactive=False,
		)

		# === Event Bindings ===

		def refresh_tasks():
			"""Refresh task list and history."""
			from training_ui.core.process_manager import process_manager
			tasks = process_manager.list_all()

			# Active tasks for dropdown
			running = [t for t in tasks if t.status == "running"]
			choices = [f"{t.task_id} - {t.algo}/{t.env}" for t in running]

			# History table
			rows = []
			for t in tasks:
				duration = ""
				if t.start_time and t.end_time:
					# Calculate duration from start to end
					try:
						from datetime import datetime
						start = datetime.fromisoformat(t.start_time)
						end = datetime.fromisoformat(t.end_time)
						delta = end - start
						duration = str(delta).split(".")[0]
					except Exception:
						duration = "N/A"
				elif t.start_time and t.status == "running":
					# Calculate elapsed time for running tasks
					try:
						from datetime import datetime
						start = datetime.fromisoformat(t.start_time)
						delta = datetime.now() - start
						duration = str(delta).split(".")[0]
					except Exception:
						duration = "running..."
				rows.append([t.task_id, t.algo, t.env, t.status, t.start_time, duration])

			return (
				gr.Dropdown(choices=choices, value=choices[0] if choices else None),
				rows,
			)

		def get_task_info(task_selection):
			"""Get selected task info as a JSON dict."""
			if not task_selection:
				return {}
			task_id = task_selection.split(" - ")[0]
			from training_ui.core.process_manager import process_manager
			info = process_manager.status(task_id)
			if info:
				return {
					"task_id": info.task_id,
					"algorithm": info.algo,
					"environment": info.env,
					"experiment": info.exp_name,
					"status": info.status,
					"pid": info.pid,
					"start_time": info.start_time,
				}
			return {}

		def get_log(task_selection):
			"""Get task log tail (last 100 lines)."""
			if not task_selection:
				return ""
			task_id = task_selection.split(" - ")[0]
			from training_ui.core.process_manager import process_manager
			return process_manager.log_tail(task_id, lines=100)

		def stop_task(task_selection):
			"""Stop the selected running task."""
			if not task_selection:
				return t("msg_no_task")
			task_id = task_selection.split(" - ")[0]
			from training_ui.core.process_manager import process_manager
			success = process_manager.stop(task_id)
			return f"Task {task_id} {t('msg_task_stopped')}" if success else f"{t('msg_task_stop_failed')} {task_id}"

		# Bind events
		refresh_btn.click(refresh_tasks, outputs=[task_selector, history_table])
		task_selector.change(get_task_info, inputs=[task_selector], outputs=[task_info])
		task_selector.change(get_log, inputs=[task_selector], outputs=[log_output])
		stop_btn.click(stop_task, inputs=[task_selector], outputs=[gr.Textbox(visible=False)])

		# Timer auto-refresh log
		timer.tick(get_log, inputs=[task_selector], outputs=[log_output])

	return tab
