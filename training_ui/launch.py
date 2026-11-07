"""CLI launcher for PowerZoo Training Management System.

Usage:
    python -m training_ui [--port 7860] [--host 127.0.0.1] [--share]
"""

import argparse

from training_ui.i18n import t


def main():
    """Launch the Gradio app."""
    parser = argparse.ArgumentParser(description=t("cli_description"))
    parser.add_argument("--port", type=int, default=7860, help=t("cli_port_help"))
    parser.add_argument("--host", default="127.0.0.1", help=t("cli_host_help"))
    parser.add_argument("--share", action="store_true", help=t("cli_share_help"))
    args = parser.parse_args()

    from training_ui.app import create_app

    create_app().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
