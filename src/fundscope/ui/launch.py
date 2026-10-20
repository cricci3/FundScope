"""
launch.py — The `fundscope` command: start the web UI and open it in the browser

    uv run fundscope                 # http://localhost:8501 (next free port if taken)
    uv run fundscope --port 8600
    uv run fundscope --no-browser    # start the server only

The terminal agent is `fundscope-cli`.
"""

import argparse
import os
import sys
from pathlib import Path

from fundscope.config import EMBEDDING_MODEL

APP = Path(__file__).with_name("app.py")

THEME = {
    "theme.base": "light",            # the diagram and chart colours are picked for a light surface
    "theme.primaryColor": "#2a78d6",
    "theme.font": "sans serif",
}


def _embedding_model_cached() -> bool:
    """True when the Hugging Face cache already holds the embedding model (checked without importing hf)."""
    hub = os.getenv("HF_HUB_CACHE") or (
        Path(os.getenv("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")
    snapshots = Path(hub) / f"models--{EMBEDDING_MODEL.replace('/', '--')}" / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def main():
    p = argparse.ArgumentParser(description="FundScope web UI (Streamlit)")
    p.add_argument("--port", type=int, default=None, help="Port (default 8501, or the next free one)")
    p.add_argument("--no-browser", action="store_true", help="Do not open the browser")
    args = p.parse_args()

    # Skip Hugging Face's network checks once the model is downloaded (startup is
    # ~10 s faster); must be set before huggingface_hub is imported.
    if _embedding_model_cached():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(APP),
                "--server.headless", "true" if args.no_browser else "false",
                "--browser.gatherUsageStats", "false",
                "--client.toolbarMode", "minimal",
                # the file watcher walks every imported module: with transformers loaded
                # it imports hundreds of optional submodules and floods the log
                "--server.fileWatcherType", "none"]
    if args.port:
        sys.argv += ["--server.port", str(args.port)]
    for key, value in THEME.items():
        sys.argv += [f"--{key}", value]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
