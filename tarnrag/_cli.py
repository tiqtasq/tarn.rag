"""Console-script entry point — a dependency-light wrapper around the rich console UI.

Kept separate from ``tarnrag.console`` (which imports ``rich`` at module load) so the installed
``tarnrag`` command exits with a clear "install the console extra" message rather than a raw
``ModuleNotFoundError`` when the optional ``console`` extra isn't present. The ``console_scripts``
entry point in ``pyproject.toml`` targets this.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    """Run the interactive console, or exit with install guidance if its ``console`` extra is missing."""
    try:
        from tarnrag.console import main as console_main
    except ModuleNotFoundError as exc:
        if exc.name != "rich":  # a genuinely-different missing module — don't mask it
            raise
        raise SystemExit(
            "the tarnrag console needs the 'console' extra — install it with:\n"
            "    pip install 'tarn-rag[console]'"
        ) from exc
    console_main(argv)
