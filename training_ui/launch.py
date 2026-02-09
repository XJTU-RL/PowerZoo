"""CLI launcher for PowerZoo Training Management System.

Usage:
	python -m training_ui.launch [--port 7860] [--share] [--host 0.0.0.0]
"""

import argparse
import sys
from pathlib import Path

import gradio as gr

from training_ui.i18n import t

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


def main():
	"""Launch the Gradio app."""
	parser = argparse.ArgumentParser(description=t("cli_description"))
	parser.add_argument("--port", type=int, default=7860, help=t("cli_port_help"))
	parser.add_argument("--host", type=str, default="0.0.0.0", help=t("cli_host_help"))
	parser.add_argument("--share", action="store_true", help=t("cli_share_help"))
	args = parser.parse_args()

	from training_ui.app import CUSTOM_CSS_PATH, create_app

	app = create_app()

	# Gradio 6.x: theme/css moved from Blocks() to launch()
	css_paths = []
	if CUSTOM_CSS_PATH.exists():
		css_paths.append(str(CUSTOM_CSS_PATH))

	app.launch(
		server_name=args.host,
		server_port=args.port,
		share=args.share,
		theme=gr.themes.Soft(),
		css_paths=css_paths,
		pwa=False,
	)


if __name__ == "__main__":
	main()
