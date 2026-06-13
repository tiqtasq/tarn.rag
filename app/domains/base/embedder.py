"""The shared embedding pipeline — used by ingestion (passages) and retrieval (queries).

Writing it once is what guarantees the §5.3 requirement that retrieval replays *exactly* the
pipeline ingestion used. The pipeline is: prefix → tokenize (HF ``tokenizers``) → ONNX Runtime
(CPU) → mean-pool (attention mask) → L2-normalize. Its identity (model id+revision, dim,
tokenizer sha256, pooling, normalize, prefixes, max_length) is summarized by
``config_fingerprint()`` and recorded in ``index_meta``; retrieval refuses to ``open()`` on
mismatch.

Heavy deps (``onnxruntime``, ``tokenizers``) are imported lazily so this module — and
``config_fingerprint()``/``embed_meta()`` — work without the model loaded; only ``embed`` needs
the runtime. Tests use a fake ``Embedder``.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

from app.core.config import EmbeddingSettings


class Embedder(ABC):
    @abstractmethod
    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed passages (ingestion side) — uses the passage prefix."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query (retrieval side) — uses the query prefix."""

    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def config_fingerprint(self) -> str: ...

    @abstractmethod
    def embed_meta(self) -> dict[str, str]:
        """The ``index_meta`` embedding keys (incl. ``embedding_config_fingerprint``)."""


class OnnxEmbedder(Embedder):
    def __init__(
        self,
        model_dir: str,
        *,
        model_id: str,
        revision: str = "",
        embedding_dim: int = 384,
        max_length: int = 512,
        pooling: str = "mean",
        normalize: str = "l2",
        query_prefix: str = "",
        passage_prefix: str = "",
        model_file: str = "model.onnx",
        tokenizer_file: str = "tokenizer.json",
    ):
        self.model_dir = Path(model_dir)
        self.model_id = model_id
        self.revision = revision
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.pooling = pooling
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.model_path = self.model_dir / model_file
        self.tokenizer_path = self.model_dir / tokenizer_file
        self._session = None
        self._tokenizer = None
        self._input_names: set[str] | None = None

    @classmethod
    def create(cls, embedding: EmbeddingSettings, embedding_dimension: int) -> OnnxEmbedder:
        """Build the shared ONNX embedder from its config slice (model loads lazily on first
        ``embed``). Ingestion and retrieval both go through here, so they share one pipeline.
        ``embedding_dimension`` is passed separately — it's cross-cutting (index + repo match it)."""
        return cls(
            embedding.model_dir,
            model_id=embedding.model,
            revision=embedding.revision,
            embedding_dim=embedding_dimension,
            max_length=embedding.max_seq_length,
            query_prefix=embedding.query_prefix,
            passage_prefix=embedding.passage_prefix,
        )

    # ---------------- identity (no model load needed) ----------------

    def dim(self) -> int:
        return self.embedding_dim

    def _tokenizer_sha256(self) -> str:
        return hashlib.sha256(self.tokenizer_path.read_bytes()).hexdigest()

    def config_fingerprint(self) -> str:
        identity = {
            "model_id": self.model_id,
            "revision": self.revision,
            "embedding_dim": self.embedding_dim,
            "tokenizer_sha256": self._tokenizer_sha256(),
            "pooling": self.pooling,
            "normalize": self.normalize,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "max_length": self.max_length,
        }
        blob = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def embed_meta(self) -> dict[str, str]:
        return {
            "embedding_model_id": self.model_id,
            "embedding_model_revision": self.revision,
            "embedding_dim": str(self.embedding_dim),
            "tokenizer_sha256": self._tokenizer_sha256(),
            "pooling": self.pooling,
            "normalize": self.normalize,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "max_length": str(self.max_length),
            "embedding_config_fingerprint": self.config_fingerprint(),
        }

    # ---------------- inference ----------------

    def _load(self):
        if self._session is None:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(self.tokenizer_path))
            tok.enable_truncation(self.max_length)
            tok.enable_padding()
            self._tokenizer = tok
            self._session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            self._input_names = {i.name for i in self._session.get_inputs()}

    def _embed(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        import numpy as np

        encs = self._tokenizer.encode_batch([prefix + t for t in texts])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)

        out = self._session.run(None, feed)[0]  # [B, L, dim]
        m = mask[..., None].astype(np.float32)
        pooled = (out * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)  # mean pool
        norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        return (pooled / norms).astype(np.float32).tolist()  # L2

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, self.passage_prefix)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], self.query_prefix)[0]
