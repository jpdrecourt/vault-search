"""
CLI hybrid search over FAISS + BM25 indexes.

Usage:
  python search.py "query" [--top_k N] [--scope vault|docs]
  python search.py "query" --type code --dir Projects
  python search.py "query" --alpha 0.4   (lower = more keyword weight)
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    os.environ["PYTHONIOENCODING"] = "utf-8"

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).parent
INDEX_DIR = PROJECT_DIR / "index_data"
CONFIG_PATH = PROJECT_DIR / "config.json"
MODEL_NAME = "all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config() -> dict:
    """Load config.json."""
    if not CONFIG_PATH.exists():
        print(f"No config found at {CONFIG_PATH}. Run indexer.py first.", file=sys.stderr)
        sys.exit(1)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_scope_files(config: dict, scope: str) -> tuple[Path, Path]:
    """Get index and metadata paths for a scope."""
    scopes = config.get("scopes", {})
    if scope not in scopes:
        print(f"Unknown scope '{scope}'. Available: {list(scopes.keys())}", file=sys.stderr)
        sys.exit(1)
    cfg = scopes[scope]
    return INDEX_DIR / cfg["index"], INDEX_DIR / cfg["metadata"]


def search(
    query: str,
    top_k: int = 5,
    scope: str = "vault",
    content_type: str | None = None,
    source_dir: str | None = None,
    alpha: float = 0.6,
):
    config = load_config()
    index_path, meta_path = get_scope_files(config, scope)

    if not index_path.exists():
        print(f"No index found at {index_path}. Run: python indexer.py --scope {scope}",
              file=sys.stderr)
        sys.exit(1)

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    index = faiss.read_index(str(index_path))

    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    # Load BM25 index if available
    bm25_path = Path(str(index_path) + ".bm25.pkl")
    bm25 = None
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)

    # Over-fetch to allow fusion and filtering
    fetch_k = top_k * 10 if (content_type or source_dir or bm25) else top_k

    # FAISS semantic search
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

        # Collect FAISS hits with fused scores
        combined = {}
        for score, idx in zip(faiss_scores[0], faiss_indices[0]):
            if idx < 0:
                continue
            i = int(idx)
            combined[i] = alpha * float(score) + (1 - alpha) * float(bm25_norm[i])

        # Add BM25-only hits not already covered by FAISS
        bm25_top = np.argsort(bm25_all_scores)[::-1][:fetch_k]
        for idx in bm25_top:
            i = int(idx)
            if i not in combined:
                combined[i] = (1 - alpha) * float(bm25_norm[i])

        # Rank by fused score, apply filters, take top_k
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

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid search over Documents")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    parser.add_argument("--scope", default="vault",
                        help="Which index to search (defined in config.json)")
    parser.add_argument("--type", dest="content_type",
                        choices=["markdown", "text", "code", "pdf"],
                        help="Filter by content type")
    parser.add_argument("--dir", dest="source_dir",
                        help="Filter by top-level source directory")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Semantic weight (0.0=pure BM25, 1.0=pure FAISS, default 0.6)")
    args = parser.parse_args()
    search(args.query, args.top_k, args.scope, args.content_type, args.source_dir, args.alpha)
