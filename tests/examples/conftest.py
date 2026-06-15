"""Gate example tests on the embedding model + runtime being available.

Any test marked ``requires_model`` is auto-skipped when the ONNX model hasn't been fetched or
``onnxruntime``/``tokenizers`` aren't installed — so a bare CI stays green, and the tests run
wherever the model is present (a dev box, or a CI job that runs ``scripts/fetch_model.py``).
This is the example-suite counterpart of the gate in ``tests/test_embedder.py``.
"""

from __future__ import annotations

import importlib.util

import pytest

from examples.common import MODEL_DIR


def _model_available() -> bool:
    return (
        (MODEL_DIR / "model.onnx").exists()
        and importlib.util.find_spec("onnxruntime") is not None
        and importlib.util.find_spec("tokenizers") is not None
    )


def pytest_collection_modifyitems(config, items):
    if _model_available():
        return
    skip = pytest.mark.skip(
        reason="embedding model/runtime unavailable — run `python scripts/fetch_model.py` "
        'and install the "[onnx]" extra'
    )
    for item in items:
        if "requires_model" in item.keywords:
            item.add_marker(skip)
