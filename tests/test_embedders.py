"""Pluggable embedder backends: ONNX pooling, the provider selector, and the API request/parse logic.

No network and no ONNX model needed — the pooling is tested on the numpy helper directly, and each API
backend is exercised by stubbing its single ``_post`` seam (the real API paths are gated, like the model).
"""

import numpy as np
import pytest

from tarnrag.core.components import Component
from tarnrag.core.config import EmbeddingSettings
from tarnrag.core.embedder import Embedder, OnnxEmbedder, build_embedder
from tarnrag.core.embedder_api import GeminiEmbedder, OpenAIEmbedder, VoyageEmbedder
from tarnrag.core.resource import Resource


# ---------------- ONNX pooling / normalize (now config-driven) ----------------

def _onnx(pooling="mean", normalize="l2"):
    return OnnxEmbedder("d", model_id="m", pooling=pooling, normalize=normalize)


def test_pooling_modes():
    # out [B=1, L=3, dim=2]; the 3rd token is padding (mask 0) — Qwen-style 'last' must ignore it.
    out = np.array([[[1.0, 0.0], [3.0, 0.0], [9.0, 9.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]], dtype=np.int64)
    assert _onnx("mean")._pool(out, mask).tolist() == [[2.0, 0.0]]  # ([1,0]+[3,0])/2
    assert _onnx("cls")._pool(out, mask).tolist() == [[1.0, 0.0]]   # first token
    assert _onnx("last")._pool(out, mask).tolist() == [[3.0, 0.0]]  # last non-pad token, not the [9,9] pad


def test_pooling_rejects_unknown():
    with pytest.raises(ValueError, match="unknown pooling"):
        _onnx("bogus")._pool(np.zeros((1, 1, 2), np.float32), np.ones((1, 1), np.int64))


def test_normalize_modes():
    pooled = np.array([[3.0, 4.0]], dtype=np.float32)  # norm 5
    assert _onnx(normalize="l2")._normalize(pooled).tolist()[0] == pytest.approx([0.6, 0.8])  # unit length
    assert _onnx(normalize="none")._normalize(pooled).tolist() == [[3.0, 4.0]]


# ---------------- the provider selector ----------------

def test_build_embedder_dispatch():
    assert isinstance(build_embedder(EmbeddingSettings(), 384), OnnxEmbedder)  # local default
    cases = [("openai", OpenAIEmbedder, 1536), ("voyage", VoyageEmbedder, 1024), ("gemini", GeminiEmbedder, 3072)]
    for provider, cls, dim in cases:
        e = build_embedder(EmbeddingSettings(provider=provider, model="m"), dim)
        assert isinstance(e, cls) and e.dim() == dim


def test_api_embedders_are_resources_not_components():
    e = OpenAIEmbedder(model="m", embedding_dim=2, api_key="k")
    assert isinstance(e, (Resource, Embedder)) and not isinstance(e, Component)


def test_provider_is_part_of_the_fingerprint():
    a = build_embedder(EmbeddingSettings(provider="openai", model="x"), 1536)
    b = build_embedder(EmbeddingSettings(provider="voyage", model="x"), 1536)
    assert a.config_fingerprint() != b.config_fingerprint()  # same model, different backend -> different index
    assert a.embed_meta()["embedding_provider"] == "openai"
    assert a.embed_meta()["embedding_config_fingerprint"] == a.config_fingerprint()


# ---------------- API request shaping + response parse (stub the _post seam) ----------------

def test_openai_request_and_parse(monkeypatch):
    e = OpenAIEmbedder(model="text-embedding-3-small", embedding_dim=2, api_key="k")
    seen = {}

    def fake_post(url, body, headers):
        seen.update(url=url, body=body, headers=headers)
        return {"data": [{"index": 1, "embedding": [0.0, 2.0]}, {"index": 0, "embedding": [3.0, 0.0]}]}

    monkeypatch.setattr(e, "_post", fake_post)
    vecs = e.embed_passages(["a", "b"])
    assert seen["url"].endswith("/embeddings")
    assert seen["body"] == {"model": "text-embedding-3-small", "input": ["a", "b"], "dimensions": 2}
    assert seen["headers"]["Authorization"] == "Bearer k"
    assert vecs == [[1.0, 0.0], [0.0, 1.0]]  # re-ordered by index, then L2-normalized


def test_voyage_uses_input_type_and_output_dimension(monkeypatch):
    e = VoyageEmbedder(model="voyage-3.5", embedding_dim=2, api_key="k")
    bodies = []
    monkeypatch.setattr(e, "_post", lambda url, body, headers: bodies.append(body) or {"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
    e.embed_query("q")
    e.embed_passages(["p"])
    assert bodies[0]["input_type"] == "query" and bodies[1]["input_type"] == "document"
    assert bodies[0]["output_dimension"] == 2


def test_gemini_request_and_parse(monkeypatch):
    e = GeminiEmbedder(model="gemini-embedding-001", embedding_dim=2, api_key="k")
    seen = {}

    def fake_post(url, body, headers):
        seen.update(url=url, body=body, headers=headers)
        return {"embeddings": [{"values": [0.0, 5.0]}]}

    monkeypatch.setattr(e, "_post", fake_post)
    vec = e.embed_query("q")
    assert seen["url"].endswith("models/gemini-embedding-001:batchEmbedContents")
    req = seen["body"]["requests"][0]
    assert req["model"] == "models/gemini-embedding-001"
    assert req["taskType"] == "RETRIEVAL_QUERY" and req["outputDimensionality"] == 2
    assert seen["headers"]["x-goog-api-key"] == "k"
    assert vec == [0.0, 1.0]  # L2-normalized


def test_api_key_is_required(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        OpenAIEmbedder(model="m", embedding_dim=2).embed_passages(["x"])
