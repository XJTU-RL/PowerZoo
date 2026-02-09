"""Real-time log viewer component.

Provides a non-interactive text area that auto-refreshes via ``gr.Timer``
to display live training log output.
"""

import gradio as gr

from training_ui.i18n import t


def build_log_viewer() -> tuple[gr.Textbox, gr.Timer]:
	"""Build log viewer with auto-refresh timer.

	The textbox is read-only and styled for monospace log output.
	The timer fires every 3 seconds; the caller is responsible for
	wiring it to a function that reads the latest log content.

	Returns:
		Tuple of (log_textbox, refresh_timer).
	"""
	log_textbox = gr.Textbox(
		label=t("label_training_log"),
		lines=25,
		interactive=False,
		placeholder=t("placeholder_log"),
	)
	# gr.Timer triggers its .tick event at the specified interval (seconds).
	refresh_timer = gr.Timer(value=3)
	return log_textbox, refresh_timer


def bind_refresh(
	refresh_timer: gr.Timer,
	log_textbox: gr.Textbox,
	read_fn: callable,
) -> None:
	"""Bind the timer tick to a log-reading function.

	Args:
		refresh_timer: The gr.Timer that fires periodically.
		log_textbox: The textbox to update with fresh log content.
		read_fn: A zero-argument callable that returns the latest log
			text as a string.
	"""
	refresh_timer.tick(
		fn=read_fn,
		outputs=[log_textbox],
	)
