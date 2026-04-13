"""
MCP server exposing semantic search over FAISS indexes.

Tools:
  - vault_search(query, top_k=5): search Obsidian vault index
  - documents_search(query, top_k=5, content_type, source_dir): search all Documents
  - vault_reindex(): rebuild vault-only FAISS index
  - documents_reindex(): rebuild full Documents FAISS index
"""

import json
import pickle
import subprocess
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).parent
INDEX_DIR = PROJECT_DIR / "index_data"
CONFIG_PATH = PROJECT_DIR / "config.json"
MODEL_NAME = "all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mcp = FastMCP("vault-index", log_level="WARNING")

# Lazy-loaded globals
_model = None
_indexes = {}  # scope -> (faiss_index, metadata)
_config = None


def _get_config():
    global _config
    if _config is None:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, encoding="utf-8") as f:
                _config = json.load(f)
        else:
            # Fallback defaults matching indexer.py
            _config = {"scopes": {
                "vault": {"index": "vault.index", "metadata": "metadata.json"},
                "docs": {"index": "documents.index", "metadata": "documents_metadata.json"},
            }}
    return _config


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    return _model


def _load_index(scope: str = "vault"):
    """Lazy-load an index by scope (FAISS + optional BM25)."""
    if scope in _indexes:
        return _indexes[scope]

    config = _get_config()
    scopes = config.get("scopes", {})
    if scope not in scopes:
        raise ValueError(f"Unknown scope '{scope}'. Available: {list(scopes.keys())}")

    cfg = scopes[scope]
    index_path = INDEX_DIR / cfg["index"]
    meta_path = INDEX_DIR / cfg["metadata"]

    if not index_path.exists():
        raise FileNotFoundError(
            f"No {scope} index found at {index_path}. Run the appropriate reindex tool."
        )

    idx = faiss.read_index(str(index_path))
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    bm25 = None
    bm25_path = Path(str(index_path) + ".bm25.pkl")
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)

    _indexes[scope] = (idx, metadata, bm25)
    return idx, metadata, bm25


def _reload_index(scope: str):
    """Force reload an index from disk."""
    _indexes.pop(scope, None)
    _load_index(scope)


def _do_search(
    query: str,
    scope: str,
    top_k: int = 5,
    content_type: str | None = None,
    source_dir: str | None = None,
    alpha: float = 0.6,
) -> str:
    model = _get_model()
    index, metadata, bm25 = _load_index(scope)

    fetch_k = top_k * 10 if (content_type or source_dir or bm25) else top_k
    embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    faiss_scores, faiss_indices = index.search(
        embedding.astype(np.float32), min(fetch_k, index.ntotal)
    )

    if bm25 is None:
        # Pure FAISS fallback
        results = []
        for score, idx in zip(faiss_scores[0], faiss_indices[0]):
            if idx < 0:
                continue
            meta = metadata[idx]
            if content_type and meta.get("content_type") != content_type:
                continue
            if source_dir and meta.get("source_dir") != source_dir:
                continue
            results.append({
                "score": round(float(score), 3),
                "source": meta["source"],
                "content_type": meta.get("content_type", "markdown"),
                "source_dir": meta.get("source_dir", ""),
                "excerpt": meta["text"][:500],
            })
            if len(results) >= top_k:
                break
    else:
        # Hybrid: FAISS + BM25 score fusion
        query_tokens = query.lower().split()
        bm25_all_scores = bm25.get_scores(query_tokens)
        bm25_max = bm25_all_scores.max()
        if bm25_max > 0:
            bm25_norm = bm25_all_scores / bm25_max
        else:
            bm25_norm = bm25_all_scores

        combined = {}
        for score, idx in zip(faiss_scores[0], faiss_indices[0]):
            if idx < 0:
                continue
            i = int(idx)
            combined[i] = alpha * float(score) + (1 - alpha) * float(bm25_norm[i])

        bm25_top = np.argsort(bm25_all_scores)[::-1][:fetch_k]
        for idx in bm25_top:
            i = int(idx)
            if i not in combined:
                combined[i] = (1 - alpha) * float(bm25_norm[i])

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        results = []
        for idx, fused_score in ranked:
            meta = metadata[idx]
            if content_type and meta.get("content_type") != content_type:
                continue
            if source_dir and meta.get("source_dir") != source_dir:
                continue
            results.append({
                "score": round(fused_score, 3),
                "source": meta["source"],
                "content_type": meta.get("content_type", "markdown"),
                "source_dir": meta.get("source_dir", ""),
                "excerpt": meta["text"][:500],
            })
            if len(results) >= top_k:
                break

    return json.dumps(results, indent=2, ensure_ascii=False)


@mcp.tool()
def vault_search(query: str, top_k: int = 5) -> str:
    """Search the Obsidian vault semantically.

    Returns the most relevant note chunks ranked by similarity.

    Args:
        query: Natural language search query
        top_k: Number of results to return (default 5)
    """
    return _do_search(query, "vault", top_k)


@mcp.tool()
def documents_search(
    query: str,
    top_k: int = 5,
    content_type: str = "",
    source_dir: str = "",
) -> str:
    """Search all of Documents/ semantically.

    Searches across markdown, code, PDFs, and text files in ~/Documents/.

    Args:
        query: Natural language search query
        top_k: Number of results to return (default 5)
        content_type: Filter by type: markdown, text, code, pdf (empty = all)
        source_dir: Filter by top-level directory, e.g. "Projects" (empty = all)
    """
    return _do_search(
        query, "docs", top_k,
        content_type=content_type or None,
        source_dir=source_dir or None,
    )


@mcp.tool()
def vault_reindex() -> str:
    """Rebuild the FAISS index from the Obsidian vault.

    Run this after adding or significantly editing notes.
    """
    indexer_path = Path(__file__).parent / "indexer.py"
    result = subprocess.run(
        [sys.executable, str(indexer_path), "--scope", "vault"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        return f"Indexing failed:\n{result.stderr}"
    _reload_index("vault")
    return result.stdout


@mcp.tool()
def documents_reindex() -> str:
    """Rebuild the FAISS index from all of ~/Documents/.

    Indexes markdown, text, code, and PDF files. Skips binaries and caches.
    This may take a few minutes depending on the size of Documents/.
    """
    indexer_path = Path(__file__).parent / "indexer.py"
    result = subprocess.run(
        [sys.executable, str(indexer_path), "--scope", "docs"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        return f"Indexing failed:\n{result.stderr}"
    _reload_index("docs")
    return result.stdout


if __name__ == "__main__":
    mcp.run(transport="stdio")
