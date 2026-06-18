"""Pluggable embedder backends: ONNX pooling, the provider selector, and the API request/response path.

No network and no ONNX model needed — the pooling is tested on the numpy helper directly, and each API
backend is driven through a real ``httpx`` client backed by ``httpx.MockTransport`` (so the full
``_http()`` / ``_post()`` path and the API key reaching the request headers are exercised, with a mock
handler standing in for the network).
"""

import json

import httpx
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


# ---------------- API request/response path (real httpx client over a mock transport) ----------------

def _mock_client(handler):
    """A real httpx client whose transport runs ``handler(request) -> httpx.Response`` — no network, but
    the embedder's ``_http()`` / ``_post()`` (and the request it builds) run for real."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_full_http_path():
    cap = {}

    def handler(request):
        cap.update(url=str(request.url), method=request.method, headers=request.headers,
                   body=json.loads(request.content))
        # out-of-order indices (exercise the sort) + known vectors (check parsing/normalization)
        return httpx.Response(200, json={"data": [
            {"index": 1, "embedding": [0.0, 2.0]}, {"index": 0, "embedding": [3.0, 0.0]},
        ]})

    e = OpenAIEmbedder(model="text-embedding-3-small", embedding_dim=2, api_key="sk-test")
    e._client = _mock_client(handler)
    vecs = e.embed_passages(["a", "b"])
    assert cap["method"] == "POST" and cap["url"].endswith("/embeddings")
    assert cap["headers"]["authorization"] == "Bearer sk-test"  # the key reached the real request
    assert cap["body"] == {"model": "text-embedding-3-small", "input": ["a", "b"], "dimensions": 2}
    assert vecs == [[1.0, 0.0], [0.0, 1.0]]  # index-sorted, then L2-normalized


def test_voyage_full_http_path():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3.0, 4.0]}]})

    e = VoyageEmbedder(model="voyage-3.5", embedding_dim=2, api_key="vk")
    e._client = _mock_client(handler)
    qv = e.embed_query("q")
    e.embed_passages(["p"])
    assert bodies[0]["input_type"] == "query" and bodies[1]["input_type"] == "document"
    assert bodies[0]["output_dimension"] == 2
    assert qv == pytest.approx([0.6, 0.8])  # [3,4] L2-normalized


def test_gemini_full_http_path():
    cap = {}

    def handler(request):
        cap.update(url=str(request.url), headers=request.headers, body=json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [{"values": [0.0, 5.0]}]})

    e = GeminiEmbedder(model="gemini-embedding-001", embedding_dim=2, api_key="gk")
    e._client = _mock_client(handler)
    vec = e.embed_query("q")
    assert cap["url"].endswith("models/gemini-embedding-001:batchEmbedContents")
    req = cap["body"]["requests"][0]
    assert req["model"] == "models/gemini-embedding-001"
    assert req["taskType"] == "RETRIEVAL_QUERY" and req["outputDimensionality"] == 2
    assert cap["headers"]["x-goog-api-key"] == "gk"
    assert vec == [0.0, 1.0]  # L2-normalized


def test_api_key_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    cap = {}

    def handler(request):
        cap["headers"] = request.headers
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    e = OpenAIEmbedder(model="m", embedding_dim=2)  # no explicit key -> env-var fallback
    e._client = _mock_client(handler)
    e.embed_query("q")
    assert cap["headers"]["authorization"] == "Bearer sk-from-env"


def test_api_key_is_required(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        OpenAIEmbedder(model="m", embedding_dim=2).embed_passages(["x"])
