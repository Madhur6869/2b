"""Embeddings + LanceDB vector store for conversation retrieval.

Fastembed produces the vectors, LanceDB stores and searches them. Re-indexing a
file is idempotent: existing rows for that source are deleted before re-adding.

Fail-hard: real errors (bad model, corrupt table, dimension mismatch) raise.
"""
from pathlib import Path

import fastembed
import lancedb

from secondbrain import config, store

_EMBEDDER = None
_DB = None


def _get_embedder():
    """Lazy singleton fastembed embedder."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = fastembed.TextEmbedding(model_name=config.EMBED_MODEL)
    return _EMBEDDER


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts into 384-float vectors.

    fastembed's .embed() yields numpy arrays lazily; materialize and convert.
    """
    embedder = _get_embedder()
    return [vec.tolist() for vec in embedder.embed(texts)]


def _connect():
    """Lazy singleton LanceDB connection at config.INDEX_DIR."""
    global _DB
    if _DB is None:
        _DB = lancedb.connect(str(config.INDEX_DIR))
    return _DB


def chunk_text(
    text: str,
    size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Character-window chunks with overlap; skip empty/whitespace-only chunks."""
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must be in [0, size), got {overlap} (size {size})")

    step = size - overlap
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start : start + size]
        if piece.strip():
            chunks.append(piece)
    return chunks


def index_file(md_path: Path) -> int:
    """Parse, chunk, embed, and (re)index a single conversation file.

    Returns the number of chunks indexed. Empty body -> 0 chunks, no crash.
    Deletes existing rows for this source first, so re-indexing is idempotent.
    """
    parsed = store.parse_conversation(md_path)
    stem = parsed["id"]
    fm = parsed["frontmatter"] or {}

    chunks = chunk_text(parsed["body"])
    if not chunks:
        # Nothing to index, but still purge any stale rows for this source.
        db = _connect()
        if config.LANCE_TABLE in db.table_names():
            db.open_table(config.LANCE_TABLE).delete(f"source = '{stem}'")
        return 0

    vectors = embed_texts(chunks)
    participants = ", ".join(fm.get("participants") or [])
    title = fm.get("title", "") or ""
    date = str(fm.get("date", "") or "")

    rows = [
        {
            "id": f"{stem}#{i}",
            "source": stem,
            "title": title,
            "date": date,
            "participants": participants,
            "text": chunk,
            "vector": vector,
        }
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    db = _connect()
    if config.LANCE_TABLE in db.table_names():
        table = db.open_table(config.LANCE_TABLE)
        table.delete(f"source = '{stem}'")
        table.add(rows)
    else:
        # First batch defines the schema (vector dim inferred from the data).
        db.create_table(config.LANCE_TABLE, data=rows)

    return len(rows)


def reindex_all() -> int:
    """Drop the table and rebuild it from every conversation. Returns total chunks."""
    db = _connect()
    if config.LANCE_TABLE in db.table_names():
        db.drop_table(config.LANCE_TABLE)

    total = 0
    for md_path in store.list_conversations():
        total += index_file(md_path)
    return total


def search(query: str, k: int = config.MCP_DEFAULT_K) -> list[dict]:
    """Vector-search the index. Returns up to k result dicts (empty if no table)."""
    db = _connect()
    if config.LANCE_TABLE not in db.table_names():
        return []

    table = db.open_table(config.LANCE_TABLE)
    if table.count_rows() == 0:
        return []

    query_vector = embed_texts([query])[0]
    hits = table.search(query_vector).limit(k).to_list()

    results = []
    for hit in hits:
        results.append(
            {
                "text": hit["text"],
                "source": hit["source"],
                "title": hit["title"],
                "date": hit["date"],
                "participants": hit["participants"],
                # LanceDB returns L2 distance in _distance (lower = closer).
                "score": hit["_distance"],
            }
        )
    return results
