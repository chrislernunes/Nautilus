"""
src/nautilus/cli.py
===================
``nautilus`` command-line entry point.

Usage
-----
    nautilus          # launch dashboard (default)
    nautilus --help
"""
from __future__ import annotations

import sys


def main() -> None:
    """Launch the Nautilus regime dashboard via Streamlit."""
    try:
        import streamlit.web.cli as stcli
    except ImportError as exc:
        print(f"streamlit is required to run the dashboard: {exc}", file=sys.stderr)
        sys.exit(1)

    import pathlib

    dashboard = pathlib.Path(__file__).with_name("dashboard") / "regime_dashboard.py"
    sys.argv = ["streamlit", "run", str(dashboard), "--server.headless", "false"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
