"""Real OnnxEmbedder over the bundled model — gated on the model + onnx deps being present
(so a minimal env just skips). Run `python scripts/fetch_model.py` to populate MODEL_DIR.
"""

from pathlib import Path

import pytest

MODEL_DIR = Path("models/all-MiniLM-L6-v2")
pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.onnx").exists(), reason="model not fetched (scripts/fetch_model.py)"
)


def _embedder():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    from tarnrag.embedder import OnnxEmbedder

    return OnnxEmbedder(
        str(MODEL_DIR), model_id="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=384
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
