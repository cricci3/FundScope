"""
log.py — Logging setup shared by the command-line entrypoints

Modules log through `logging.getLogger(__name__)`; only the CLIs configure
handlers. Diagnostics go to stderr, so stdout carries only the program output
(answers, reports, --json). Default level WARNING, INFO with --verbose.
"""

import argparse
import logging
import sys


def add_verbose_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Log progress and diagnostics (INFO level) to stderr")


def setup_logging(verbose: bool = False) -> None:
    # Output piped on Windows defaults to the ANSI code page and fails on ←, ✓, —
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="[%(module)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    # Third-party libraries are chatty at INFO (HTTP requests, model loading)
    for name in ("httpx", "chromadb", "sentence_transformers", "urllib3", "yfinance"):
        logging.getLogger(name).setLevel(logging.WARNING)
