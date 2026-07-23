"""Higher-level knowledge queries over the brain: thoughts, topics, action items.

These back the MCP tools that let Claude both write to and reason over the brain.
Fail-hard: real errors raise.
"""
import re
from pathlib import Path

from secondbrain import config, store, util


def add_thought(text: str, topics: list[str] | None = None) -> Path:
    """Capture a free-form thought: store it, index it, and file it under any topics."""
    path = store.write_thought(text, tags=topics or [], source="session")

    from secondbrain.index import index_file
    index_file(path)

    if topics:
        from secondbrain.distill import _append_topic
        parsed = store.parse_conversation(path)
        fm = parsed["frontmatter"]
        for raw_slug in topics:
            slug = util.slugify(raw_slug)
            if not slug:
                continue
            title = slug.replace("-", " ").title()
            _append_topic(slug, title, f"- {text.strip()}", parsed["id"], fm["title"], fm["date"])
    return path


def list_topics() -> list[dict]:
    """All living topic notes with their title and how many conversations fed them."""
    out = []
    for p in sorted(config.TOPICS_DIR.glob("*.md")):
        lines = p.read_text().splitlines()
        title = lines[0][2:].strip() if lines and lines[0].startswith("# ") else p.stem
        entries = sum(1 for line in lines if line.startswith("## From:"))
        out.append({"slug": p.stem, "title": title, "entries": entries})
    return out


def get_topic(slug: str) -> str | None:
    """Full living-note content for a topic slug, or None if absent."""
    p = config.TOPICS_DIR / f"{util.slugify(slug)}.md"
    return p.read_text() if p.exists() else None


def list_action_items(open_only: bool = True) -> list[dict]:
    """Scan distilled conversations for action items in their '## Action items' section."""
    items = []
    for p in store.list_conversations():
        parsed = store.parse_conversation(p)
        body = parsed["body"]
        m = re.search(r"(?ms)^## Action items\s*\n(.*?)(?=^## |\Z)", body)
        if not m:
            continue
        for line in m.group(1).splitlines():
            mm = re.match(r"\s*- \[([ xX])\]\s*(.+)", line)
            if not mm:
                continue
            done = mm.group(1).lower() == "x"
            if open_only and done:
                continue
            items.append({
                "task": mm.group(2).strip(),
                "done": done,
                "source": parsed["id"],
                "title": (parsed["frontmatter"] or {}).get("title", ""),
                "date": (parsed["frontmatter"] or {}).get("date", ""),
            })
    return items
