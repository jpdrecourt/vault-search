"""
Documents semantic indexer — builds a FAISS index from configurable roots.

Supports markdown, text, code, and PDF files.
Uses sentence-transformers with GPU acceleration.
Incremental: only re-embeds new or modified files.

Scopes are defined in config.json (auto-created on first run).
"""

import argparse
import hashlib
import json
import os
import pickle
import re
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import torch

PROJECT_DIR = Path(__file__).parent
INDEX_DIR = PROJECT_DIR / "index_data"
CONFIG_PATH = PROJECT_DIR / "config.json"
MODEL_NAME = "all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Chunking config
MIN_CHUNK_LENGTH = 120
MAX_CHUNK_LENGTH = 1500

# File extensions by content type
TEXT_EXTENSIONS = {".md", ".txt", ".rst"}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".lua", ".maxpat", ".sh", ".css", ".html",
                   ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".r", ".R"}
PDF_EXTENSIONS = {".pdf"}

# Directories to skip
IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".cache", ".venv", "venv",
    ".env", "env", ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".obsidian", ".trash", ".DS_Store", "index_data",
    # App-managed content (not user knowledge)
    "Factory Packs", "Library", "Lessons",
}

# Top-level directories to skip entirely (app data, not knowledge)
IGNORE_TOP_DIRS = {
    "My Games", "WindowsPowerShell", "PowerShell",
    "Visual Studio 2022", "IISExpress",
    "Custom Office Templates",
}

# Files to skip
IGNORE_FILES = {".env", ".gitignore", "package-lock.json", "yarn.lock",
                "pnpm-lock.yaml", "Thumbs.db", ".DS_Store"}


DEFAULT_CONFIG = {
    "scopes": {
        "vault": {
            "root": str(Path.home() / "Documents" / "Obsidian" / "Notes"),
            "index": "vault.index",
            "metadata": "metadata.json",
        },
        "docs": {
            "root": str(Path.home() / "Documents"),
            "index": "documents.index",
            "metadata": "documents_metadata.json",
        },
    }
}


def load_config() -> dict:
    """Load config, creating with defaults if missing."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    print(f"Created default config at {CONFIG_PATH}")
    print(f"Edit the 'root' paths in config.json to point at your notes directory, then run again.")
    raise SystemExit(1)


def get_scope_config(config: dict, scope: str) -> dict:
    """Get root, index name, and metadata name for a scope."""
    scopes = config.get("scopes", {})
    if scope not in scopes:
        raise ValueError(f"Unknown scope '{scope}'. Available: {list(scopes.keys())}")
    return scopes[scope]


def file_hash(filepath: Path) -> str:
    """Fast hash: mtime + size. Catches content changes without reading every file."""
    stat = filepath.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def should_skip(path: Path) -> bool:
    """Check if a path should be skipped based on ignore patterns."""
    for part in path.parts:
        if part in IGNORE_DIRS:
            return True
    if path.name in IGNORE_FILES:
        return True
    return False


def detect_content_type(filepath: Path) -> str | None:
    """Classify a file by content type. Returns None for unsupported files."""
    ext = filepath.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return "markdown" if ext == ".md" else "text"
    if ext in CODE_EXTENSIONS:
        return "code"
    if ext in PDF_EXTENSIONS:
        return "pdf"
    return None


def extract_markdown(filepath: Path) -> str:
    """Read markdown/text file, strip frontmatter."""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3:]
    return text.strip()


def extract_code(filepath: Path) -> str:
    """Read a code file."""
    return filepath.read_text(encoding="utf-8", errors="replace").strip()


def extract_pdf(filepath: Path) -> str:
    """Extract text from PDF via pymupdf."""
    import fitz
    text_parts = []
    try:
        with fitz.open(filepath) as doc:
            for page in doc:
                text_parts.append(page.get_text())
    except Exception as e:
        print(f"  Warning: failed to read PDF {filepath}: {e}")
        return ""
    return "\n".join(text_parts).strip()


def chunk_markdown(text: str, source: str, content_type: str, source_dir: str) -> list[dict]:
    """Split text into chunks by headings and double newlines."""
    sections = re.split(r'(?=^#{1,3}\s)|(?:\n\n+)', text, flags=re.MULTILINE)
    return _assemble_chunks(sections, source, content_type, source_dir)


def chunk_code(text: str, source: str, content_type: str, source_dir: str) -> list[dict]:
    """Split code into chunks by function/class/block boundaries."""
    sections = re.split(
        r'(?=^(?:class |def |function |const |let |var |export |async function |module\.exports))',
        text,
        flags=re.MULTILINE,
    )
    if len(sections) <= 1:
        sections = re.split(r'\n\n+', text)
    return _assemble_chunks(sections, source, content_type, source_dir)


def _assemble_chunks(
    sections: list[str], source: str, content_type: str, source_dir: str
) -> list[dict]:
    """Common chunking logic: size-check sections and build chunk dicts."""
    chunks = []
    for section in sections:
        section = section.strip()
        if len(section) < MIN_CHUNK_LENGTH:
            continue

        if len(section) > MAX_CHUNK_LENGTH:
            sub_parts = section.split("\n")
            current = ""
            for part in sub_parts:
                if len(current) + len(part) > MAX_CHUNK_LENGTH and current:
                    chunks.append({
                        "text": current.strip(),
                        "source": source,
                        "content_type": content_type,
                        "source_dir": source_dir,
                    })
                    current = part
                else:
                    current += "\n" + part if current else part
            if current.strip() and len(current.strip()) >= MIN_CHUNK_LENGTH:
                chunks.append({
                    "text": current.strip(),
                    "source": source,
                    "content_type": content_type,
                    "source_dir": source_dir,
                })
        else:
            chunks.append({
                "text": section,
                "source": source,
                "content_type": content_type,
                "source_dir": source_dir,
            })
    return chunks


def collect_files(root: Path) -> list[tuple[Path, str]]:
    """Walk root and return (filepath, content_type) for all indexable files."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        rel = Path(dirpath).relative_to(root)
        if rel == Path("."):
            dirnames[:] = [d for d in dirnames
                           if d not in IGNORE_DIRS and d not in IGNORE_TOP_DIRS]
        else:
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]

        for name in filenames:
            if name in IGNORE_FILES:
                continue
            fp = Path(dirpath) / name
            ct = detect_content_type(fp)
            if ct is not None:
                files.append((fp, ct))

    files.sort(key=lambda x: x[0])
    return files


def process_file(filepath: Path, content_type: str, root: Path) -> list[dict]:
    """Extract and chunk a single file."""
    rel_path = str(filepath.relative_to(root)).replace("\\", "/")
    parts = filepath.relative_to(root).parts
    source_dir = parts[0] if len(parts) > 1 else ""

    if content_type in ("markdown", "text"):
        text = extract_markdown(filepath)
        if not text:
            return []
        return chunk_markdown(text, rel_path, content_type, source_dir)

    if content_type == "code":
        text = extract_code(filepath)
        if not text:
            return []
        return chunk_code(text, rel_path, content_type, source_dir)

    if content_type == "pdf":
        text = extract_pdf(filepath)
        if not text:
            return []
        return chunk_markdown(text, rel_path, content_type, source_dir)

    return []


def load_cache(scope_cfg: dict) -> tuple[dict, list[dict], np.ndarray | None]:
    """Load the hash manifest, metadata, and embeddings cache for a scope.

    Returns (hashes, metadata, embeddings) where hashes maps relative
    file paths to their last-indexed hash string.
    """
    hash_path = INDEX_DIR / f"{scope_cfg['index']}.hashes.json"
    meta_path = INDEX_DIR / scope_cfg["metadata"]
    embed_path = INDEX_DIR / f"{scope_cfg['index']}.embeddings.npy"

    hashes = {}
    metadata = []
    embeddings = None

    if hash_path.exists():
        with open(hash_path, encoding="utf-8") as f:
            hashes = json.load(f)

    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

    if embed_path.exists():
        embeddings = np.load(embed_path)

    # Validate consistency
    if embeddings is not None and len(metadata) != embeddings.shape[0]:
        print("  Cache inconsistency (metadata/embeddings mismatch) — full rebuild")
        return {}, [], None

    return hashes, metadata, embeddings


def save_cache(scope_cfg: dict, hashes: dict, metadata: list[dict], embeddings: np.ndarray):
    """Save hash manifest, metadata, embeddings cache, and FAISS index."""
    INDEX_DIR.mkdir(exist_ok=True)

    # Save hashes
    hash_path = INDEX_DIR / f"{scope_cfg['index']}.hashes.json"
    with open(hash_path, "w", encoding="utf-8") as f:
        json.dump(hashes, f, ensure_ascii=False)

    # Save metadata
    meta_path = INDEX_DIR / scope_cfg["metadata"]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    # Save embeddings cache
    embed_path = INDEX_DIR / f"{scope_cfg['index']}.embeddings.npy"
    np.save(embed_path, embeddings)

    # Build and save FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(INDEX_DIR / scope_cfg["index"]))

    print(f"Index saved: {INDEX_DIR / scope_cfg['index']} ({index.ntotal} vectors)")

    # Build and save BM25 index
    tokenized = [doc["text"].lower().split() for doc in metadata]
    bm25 = BM25Okapi(tokenized)
    bm25_path = INDEX_DIR / f"{scope_cfg['index']}.bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved: {bm25_path}")


def build_index(scope: str = "docs", force: bool = False):
    """Build FAISS index incrementally from files in the given scope."""
    config = load_config()
    scope_cfg = get_scope_config(config, scope)
    root = Path(scope_cfg["root"]).expanduser().resolve()

    if not root.exists():
        print(f"Error: root path does not exist: {root}")
        return

    INDEX_DIR.mkdir(exist_ok=True)

    # Load existing cache
    old_hashes, old_metadata, old_embeddings = ({}, [], None) if force else load_cache(scope_cfg)
    has_cache = old_embeddings is not None and len(old_metadata) > 0

    print(f"Loading model '{MODEL_NAME}' (GPU)...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    # Collect current files and compute hashes
    files = collect_files(root)
    type_counts = {}
    for _, ct in files:
        type_counts[ct] = type_counts.get(ct, 0) + 1
    print(f"Found {len(files)} files in {root}")
    for ct, count in sorted(type_counts.items()):
        print(f"  {ct}: {count}")

    current_hashes = {}
    for fp, _ in files:
        rel = str(fp.relative_to(root)).replace("\\", "/")
        try:
            current_hashes[rel] = file_hash(fp)
        except OSError:
            pass

    if has_cache:
        # Determine what changed
        old_files = set(old_hashes.keys())
        new_files_set = set(current_hashes.keys())

        unchanged = {f for f in old_files & new_files_set
                     if old_hashes[f] == current_hashes[f]}
        changed = {f for f in old_files & new_files_set
                   if old_hashes[f] != current_hashes[f]}
        added = new_files_set - old_files
        deleted = old_files - new_files_set

        to_process = changed | added

        print(f"Incremental: {len(unchanged)} unchanged, {len(changed)} changed, "
              f"{len(added)} added, {len(deleted)} deleted")

        if not to_process and not deleted:
            # Check if BM25 index needs building from existing metadata
            bm25_path = INDEX_DIR / f"{scope_cfg['index']}.bm25.pkl"
            if not bm25_path.exists() and old_metadata:
                print("FAISS up to date, building missing BM25 index...")
                tokenized = [doc["text"].lower().split() for doc in old_metadata]
                bm25 = BM25Okapi(tokenized)
                with open(bm25_path, "wb") as f:
                    pickle.dump(bm25, f)
                print(f"BM25 index saved: {bm25_path}")
            else:
                print("Nothing to update.")
            return

        # Keep chunks/embeddings for unchanged files
        keep_indices = []
        for i, meta in enumerate(old_metadata):
            if meta["source"] in unchanged:
                keep_indices.append(i)

        if keep_indices:
            kept_metadata = [old_metadata[i] for i in keep_indices]
            kept_embeddings = old_embeddings[keep_indices]
        else:
            kept_metadata = []
            kept_embeddings = np.empty((0, old_embeddings.shape[1]), dtype=np.float32)
    else:
        print("No cache found — full build")
        to_process = set(current_hashes.keys())
        kept_metadata = []
        kept_embeddings = None

    # Process new/changed files
    file_map = {str(fp.relative_to(root)).replace("\\", "/"): (fp, ct) for fp, ct in files}
    new_chunks = []
    for rel in sorted(to_process):
        if rel not in file_map:
            continue
        fp, ct = file_map[rel]
        try:
            chunks = process_file(fp, ct, root)
            new_chunks.extend(chunks)
        except Exception as e:
            print(f"  Error processing {fp}: {e}")

    print(f"Processing {len(to_process)} files -> {len(new_chunks)} new chunks")

    # Embed new chunks
    if new_chunks:
        texts = [c["text"] for c in new_chunks]
        print(f"Embedding {len(texts)} chunks...")
        t0 = time.time()
        new_embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        elapsed = time.time() - t0
        print(f"Embedded in {elapsed:.1f}s")
    else:
        dim = kept_embeddings.shape[1] if kept_embeddings is not None and kept_embeddings.shape[0] > 0 else 768
        new_embeddings = np.empty((0, dim), dtype=np.float32)

    # Combine kept + new
    new_meta = [{"text": c["text"], "source": c["source"],
                 "content_type": c["content_type"], "source_dir": c["source_dir"]}
                for c in new_chunks]

    all_metadata = kept_metadata + new_meta

    if kept_embeddings is not None and kept_embeddings.shape[0] > 0:
        all_embeddings = np.vstack([kept_embeddings, new_embeddings]).astype(np.float32)
    else:
        all_embeddings = new_embeddings.astype(np.float32)

    if len(all_metadata) == 0:
        print("No chunks to index!")
        return

    # Save everything
    save_cache(scope_cfg, current_hashes, all_metadata, all_embeddings)
    print(f"Total: {len(all_metadata)} chunks from {len(current_hashes)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS semantic index")
    parser.add_argument(
        "--scope", default="vault",
        help="Index scope (defined in config.json). Default: vault",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force full rebuild, ignoring cache",
    )
    args = parser.parse_args()
    build_index(args.scope, args.force)
