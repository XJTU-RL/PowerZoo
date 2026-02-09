"""Model management tab - browse and load trained models."""

import gradio as gr

from training_ui.i18n import t


def build_models_tab() -> gr.Tab:
	"""Build the models management tab.

	Layout:
		Row 1: Filter dropdowns (env, algo) + Scan button
		Row 2: Models table (env, system, algo, exp, run_id, checkpoints count)
		Row 3: Selected model details (JSON) + config preview
		Row 4: Action buttons (Copy path, Load config)
	"""
	with gr.Tab(t("tab_models")) as tab:
		gr.Markdown(t("models_heading"))

		# Filters
		with gr.Row():
			env_filter = gr.Dropdown(
				label=t("label_env_filter"),
				choices=["All", "vvc", "smartgrid", "stackelberg_13bus", "stackelberg_34bus",
						"stackelberg_123bus", "dsr", "dsr_13bus", "dsr_8500node"],
				value="All",
				scale=2,
			)
			algo_filter = gr.Dropdown(
				label=t("label_algo_filter"),
				choices=["All", "happo", "hatrpo", "haa2c", "mappo", "shom", "sn_mappo",
						"dan_happo", "haddpg", "hatd3", "hasac", "had3qn",
						"maddpg", "matd3", "qmix"],
				value="All",
				scale=2,
			)
			scan_btn = gr.Button(t("btn_scan_results"), variant="primary", scale=1)

		# Models table
		models_table = gr.Dataframe(
			headers=[t("col_env"), t("col_system"), t("col_algorithm"), t("col_experiment"), t("col_run_id"), t("col_checkpoints")],
			label=t("label_discovered_models"),
			interactive=False,
		)

		# Selected model details
		with gr.Row():
			with gr.Column():
				model_details = gr.JSON(label=t("label_model_details"), value={})
			with gr.Column():
				model_config = gr.Code(
					label=t("label_training_config"),
					language="json",
					interactive=False,
					lines=15,
				)

		# Actions
		with gr.Row():
			model_path_box = gr.Textbox(label=t("label_model_path"), interactive=False)
			copy_path_btn = gr.Button(t("btn_copy_path"), variant="secondary")
			checkpoint_dropdown = gr.Dropdown(label=t("label_checkpoint"), choices=[], interactive=True)

		# === Event Bindings ===

		def scan_models_handler(env_f, algo_f):
			"""Scan results directory and return filtered model records."""
			from training_ui.core.model_discovery import scan_models, filter_models

			records = scan_models()
			env_val = None if env_f == "All" else env_f
			algo_val = None if algo_f == "All" else algo_f
			filtered = filter_models(records, env=env_val, algo=algo_val)

			rows = []
			for r in filtered:
				rows.append([
					r.env, r.system, r.algo, r.exp_name,
					r.run_id, len(r.checkpoints),
				])
			return rows

		def on_model_select(evt: gr.SelectData, table_data):
			"""When a row is selected in the models table, show details."""
			if evt.index is None or table_data is None:
				return {}, "", "", []

			row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
			if row_idx >= len(table_data):
				return {}, "", "", []

			row = table_data[row_idx]
			env, system, algo, exp, run_id = row[0], row[1], row[2], row[3], row[4]

			from training_ui.core.model_discovery import (
				scan_models, get_model_config, get_checkpoints,
			)
			import json

			records = scan_models()
			# Find matching record
			match = None
			for r in records:
				if r.env == env and r.algo == algo and r.exp_name == exp and r.run_id == run_id:
					match = r
					break

			if not match:
				return {}, "", "", []

			config = get_model_config(match.run_dir)
			checkpoints = get_checkpoints(match.run_dir)

			details = {
				"env": match.env,
				"system": match.system,
				"algo": match.algo,
				"experiment": match.exp_name,
				"run_id": match.run_id,
				"path": match.run_dir,
				"checkpoints": len(checkpoints),
			}

			config_str = json.dumps(config, indent=2, ensure_ascii=False) if config else "{}"

			return (
				details,
				config_str,
				match.run_dir,
				checkpoints,
			)

		# Bind events
		scan_btn.click(scan_models_handler, inputs=[env_filter, algo_filter], outputs=[models_table])
		models_table.select(
			on_model_select,
			inputs=[models_table],
			outputs=[model_details, model_config, model_path_box, checkpoint_dropdown],
		)

	return tab
