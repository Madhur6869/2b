"""Human-readable formatting for search results."""

_SNIPPET_LEN = 300


def format_results(results: list[dict]) -> str:
    """Render search results as a pretty multi-line string."""
    if not results:
        return "No results."

    blocks = []
    for rank, r in enumerate(results, start=1):
        title = r.get("title") or "(untitled)"
        source = r.get("source", "")
        date = r.get("date", "")
        score = r.get("score", 0.0)

        text = (r.get("text") or "").strip()
        snippet = text.replace("\n", " ")
        if len(snippet) > _SNIPPET_LEN:
            snippet = snippet[:_SNIPPET_LEN].rstrip() + "…"

        header = f"{rank}. {title}  [{source}]"
        meta = f"   date: {date or '—'}   score: {score:.4f}"
        body = f"   {snippet}"
        blocks.append("\n".join((header, meta, body)))

    return "\n\n".join(blocks)
