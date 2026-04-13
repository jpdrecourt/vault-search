# vault-search

Hybrid semantic + keyword search over markdown vaults and document collections. Combines FAISS vector similarity with BM25 full-text matching, all running locally.

## Requirements

- Python 3.11+
- CUDA GPU (optional — falls back to CPU)

## Setup

```bash
pip install -r requirements.txt
```

On first run, `config.json` is auto-created with default scopes pointing to `~/Documents/Obsidian/Notes/` (vault) and `~/Documents/` (docs). Edit it to match your setup.

## Usage

### Build the index

```bash
python indexer.py --scope vault
```

Incremental by default — only re-embeds new or modified files. Use `--force` for a full rebuild. Supports markdown, text, code, and PDF files.

### Search

```bash
python search.py "query" --top_k 10
```

| Flag | Default | Description |
|------|---------|-------------|
| `--top_k` | 5 | Number of results |
| `--scope` | vault | Index scope (`vault` or `docs`) |
| `--type` | all | Filter: `markdown`, `text`, `code`, `pdf` |
| `--dir` | all | Filter by top-level directory |
| `--alpha` | 0.6 | Semantic vs keyword weight (0.0 = pure BM25, 1.0 = pure FAISS) |

Output is JSON: `[{score, source, content_type, source_dir, excerpt}, ...]`

### MCP server

```bash
python server.py
```

Exposes `vault_search`, `documents_search`, `vault_reindex`, and `documents_reindex` as MCP tools over stdio.

## How it works

The indexer chunks documents by headings/semantic boundaries, embeds them with `all-mpnet-base-v2` (768-dim), and builds two parallel indexes:

- **FAISS** (`IndexFlatIP`) for vector similarity search
- **BM25** (`rank_bm25`) for keyword matching

At query time, both indexes are queried. Scores are normalised to [0,1] and fused:

```
score = alpha * faiss_score + (1 - alpha) * bm25_score
```

This catches both conceptual similarity (FAISS) and exact keyword matches (BM25) that pure vector search misses.

## Claude Code integration

To use this as a skill in a Claude Code project, create `.claude/skills/vault-search/SKILL.md` with the search and indexer paths, available flags, and scope definitions. Commands and workflows can then reference the skill instead of hardcoding paths.

Example skill file:

```markdown
# Vault Search

Hybrid semantic + keyword search. See project README for full details.

## Search

\```bash
python /path/to/vault-search/search.py "query" --top_k 20
\```

## Reindex

\```bash
python /path/to/vault-search/indexer.py --scope vault
\```
```

Replace `/path/to/vault-search/` with wherever you cloned this project.
