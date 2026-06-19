"""``Resource`` — the base for heavy, engine-built runtime models (the ``Embedder``, the
``CrossEncoder``), the sibling of ``Component``.

The distinction the two types draw:

* a **Component** is a config-driven *strategy* composed into a pipeline by spec — built by the
  ``ComponentFactory``, registered by ``class_name``, and placed in a container tree (retrievers,
  fusers, mergers, rerankers).
* a **Resource** is the *model a strategy consumes* — built by the engine from its typed ``Settings``
  slice, long-lived and shared (the embedder spans ingestion + retrieval and feeds the index
  fingerprint), and injected into components at call time via the ``RetrievalContext`` (so tests fake
  it by passing an instance, not by registering a class).

They are **siblings, not parent/child** (``Resource`` is *not* a ``Component``). Keeping resources out
of the ``Component`` hierarchy avoids inheriting container semantics (``_build_children`` / the factory
build flow) that a leaf model has no use for, and keeps model/ops config (model dir, revision) in the
``Settings`` slices rather than the pipeline-composition spec.

The shared contract is just ``identity()`` — a stable id for provenance / eval / logging. Construction
stays per-resource (each has a ``create(...)`` classmethod over its ``Settings`` slice): the inputs
genuinely differ — the embedder also needs the cross-cutting ``EMBEDDING_DIMENSION`` and is built from a
slice + dimension (which the embed stage carries, not a whole ``Settings``) — so a single uniform
``create`` signature would be a forced abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Resource(ABC):
    """An engine-built, injected runtime model (the embedder, the cross-encoder) — see the module
    docstring for why it's a sibling of ``Component`` rather than a subclass."""

    @abstractmethod
    def identity(self) -> str:
        """A stable identity string (e.g. ``model_id@revision``) for provenance, eval, and logging.
        Distinct from any compatibility fingerprint a concrete resource may also expose (e.g. the
        embedder's ``config_fingerprint``, which gates index ``open()``)."""
