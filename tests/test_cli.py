"""The `tarnrag` console-script wrapper (``tarnrag._cli``): delegates to the console, or exits with a
clear hint when the optional ``console`` extra (rich) is missing."""

import builtins
import importlib
import sys
import types

import pytest

from tarnrag import _cli


def test_cli_delegates_to_the_console(monkeypatch):
    """With the console importable, the wrapper forwards argv to ``tarnrag.console.main``."""
    calls: list = []
    fake = types.ModuleType("tarnrag.console")
    fake.main = lambda argv=None: calls.append(argv)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tarnrag.console", fake)

    _cli.main(["config.json"])
    assert calls == [["config.json"]]


def test_cli_exits_with_hint_when_rich_missing(monkeypatch):
    """When importing the console fails because rich isn't installed, the wrapper raises a friendly
    SystemExit pointing at the `console` extra (not a raw ModuleNotFoundError)."""
    monkeypatch.delitem(sys.modules, "tarnrag.console", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tarnrag.console":
            raise ModuleNotFoundError("No module named 'rich'", name="rich")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit, match="console.*extra"):
        _cli.main([])


def test_cli_reraises_unrelated_import_error(monkeypatch):
    """A ModuleNotFoundError for some *other* module is not masked as the console-extra hint."""
    monkeypatch.delitem(sys.modules, "tarnrag.console", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tarnrag.console":
            raise ModuleNotFoundError("No module named 'wat'", name="wat")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ModuleNotFoundError, match="wat"):
        _cli.main([])


def test_cli_module_is_importable_without_rich():
    """The wrapper module itself must not import rich at load time (that's the whole point)."""
    mod = importlib.import_module("tarnrag._cli")
    assert hasattr(mod, "main")
