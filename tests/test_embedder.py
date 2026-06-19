"""Real OnnxEmbedder over the bundled model — gated on the model + onnx deps being present
(so a minimal env just skips). Run `python scripts/fetch_model.py` to populate MODEL_DIR.
"""

from pathlib import Path

import pytest

MODEL_DIR = Path("models/gte-small")
pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.onnx").exists(), reason="model not fetched (scripts/fetch_model.py)"
)


def _embedder():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    from tarnrag.core.resources.embedder import OnnxEmbedder

    return OnnxEmbedder(
        str(MODEL_DIR), model_id="thenlper/gte-small", embedding_dim=384
    )


def test_dim_and_determinism():
    e = _embedder()
    v1 = e.embed_query("How do I inspect a storage tank?")
    v2 = e.embed_query("How do I inspect a storage tank?")
    assert len(v1) == 384
    assert v1 == v2  # deterministic on CPU


def test_semantics_related_gt_unrelated():
    import numpy as np

    e = _embedder()
    q = np.array(e.embed_query("How do I inspect a storage tank?"))
    related = np.array(e.embed_passages(["storage tank inspection procedure"])[0])
    unrelated = np.array(e.embed_passages(["the quokka is a small marsupial"])[0])
    assert float(q @ related) > float(q @ unrelated)
    assert float(q @ related) > 0.5


def test_fingerprint_stable_and_meta_complete():
    e = _embedder()
    assert e.config_fingerprint() == _embedder().config_fingerprint()  # config-derived
    meta = e.embed_meta()
    for k in (
        "embedding_model_id", "embedding_dim", "tokenizer_sha256",
        "pooling", "normalize", "embedding_config_fingerprint",
    ):
        assert k in meta
    assert meta["embedding_config_fingerprint"] == e.config_fingerprint()
    assert meta["embedding_dim"] == "384"


def test_inject_header_path_changes_fingerprint():
    from tarnrag.core.resources.embedder import OnnxEmbedder

    base = _embedder()  # inject_header_path defaults False
    injected = OnnxEmbedder(
        str(MODEL_DIR), model_id="thenlper/gte-small",
        embedding_dim=384, inject_header_path=True,
    )
    # The injection flag is part of the embedding identity -> a distinct index fingerprint, so an
    # injected index won't open() with a non-injecting embedder (the two are compared as separate indexes).
    assert base.config_fingerprint() != injected.config_fingerprint()
    assert base.embed_meta()["inject_header_path"] == "False"
    assert injected.embed_meta()["inject_header_path"] == "True"
