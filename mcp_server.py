"""MCP server exposing the local Second Brain to Claude Code.

Registers tools for semantic search across recorded desk conversations,
reading a full conversation, listing recent conversations, and looking up
colleagues.

`--http` runs ONE shared server on http://127.0.0.1:MCP_HTTP_PORT/mcp (the
com.terran.secondbrain.mcp LaunchAgent). Claude Code connects to that URL.
Without `--http` it runs over stdio, which starts one copy per session. That
loads fastembed + lancedb (~160 MB) into every session: 51 copies held 8 GB
on 2026-10-27.

Fail-hard philosophy: no broad exception-swallowing. Real errors raise.
"""
import sys
from pathlib import Path

# Bootstrap sys.path so `secondbrain` is importable no matter what cwd
# Claude Code launches this server from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP

from secondbrain import config, index, query, store
from secondbrain.config import CONVERSATIONS_DIR, MCP_DEFAULT_K
from secondbrain.store import list_conversations, parse_conversation

mcp = FastMCP("second-brain")


def _fmt_participants(participants) -> str:
    """Render a participants field (list or scalar) as a readable string."""
    if participants is None:
        return "unknown"
    if isinstance(participants, (list, tuple)):
        return ", ".join(str(p) for p in participants) if participants else "unknown"
    return str(participants)


@mcp.tool()
def search_conversations(query: str, k: int = MCP_DEFAULT_K) -> str:
    """Semantic search across all recorded desk conversations. Returns the most
    relevant transcript excerpts with their source conversation id, title, date,
    and participants."""
    results = index.search(query, k)
    if not results:
        return f"No conversations matched query: {query!r}"

    blocks = []
    for rank, r in enumerate(results, start=1):
        blocks.append(
            f"#{rank}  {r['title']}\n"
            f"    date:         {r['date']}\n"
            f"    participants: {_fmt_participants(r['participants'])}\n"
            f"    source id:    {r['source']}\n"
            f"    score:        {r['score']:.4f}\n"
            f"    excerpt:\n"
            f"{r['text']}"
        )
    header = f"Top {len(results)} result(s) for {query!r}:\n"
    return header + "\n\n".join(blocks)


# Claude Code rejects a tool result above ~25k tokens outright (it lands in a
# tool-results file instead of the context). A 71k-char transcript did that
# on 2026-10-17, so transcripts are paged: `max_chars` per call, `offset` to
# continue. 0 disables the cap.
MCP_TRANSCRIPT_PAGE_CHARS = 40_000


@mcp.tool()
def get_conversation(
    conversation_id: str = "",
    source_id: str = "",
    max_chars: int = MCP_TRANSCRIPT_PAGE_CHARS,
    offset: int = 0,
) -> str:
    """Return the transcript and metadata for one conversation by its id
    (the markdown filename stem, e.g. 2026-07-16_1430_planner-latency — the
    'source id' shown by search_conversations). Use after search to read the whole
    conversation. `source_id` is accepted as an alias for `conversation_id`.

    Long transcripts come in pages: at most `max_chars` characters of transcript
    per call (default 40000; 0 = no limit), starting at character `offset`. A
    truncated page ends with the offset to pass for the next page. The distilled
    summary sits at the top of the transcript, so the first page usually answers
    the question."""
    conversation_id = conversation_id or source_id
    if not conversation_id:
        return "get_conversation needs conversation_id=<the 'source id' from search_conversations>."
    md_path = CONVERSATIONS_DIR / f"{conversation_id}.md"
    if not md_path.exists():
        available = [p.stem for p in list_conversations()[:10]]
        listing = "\n".join(f"  - {a}" for a in available) or "  (none recorded yet)"
        return (
            f"Conversation not found: {conversation_id!r}\n"
            f"A few available conversation ids:\n{listing}"
        )

    conv = parse_conversation(md_path)
    fm = conv["frontmatter"] or {}
    meta = "\n".join(f"{key}: {value}" for key, value in fm.items())
    body = conv["body"].strip()
    total = len(body)
    offset = max(0, int(offset or 0))
    page = body[offset:]
    trailer = ""
    if max_chars and max_chars > 0 and len(page) > max_chars:
        page = page[:max_chars]
        nxt = offset + max_chars
        trailer = (
            f"\n--- truncated: transcript chars {offset}-{nxt} of {total}; "
            f"call get_conversation again with offset={nxt} for the next page ---"
        )
    elif offset:
        trailer = f"\n--- end of transcript (chars {offset}-{total} of {total}) ---"
    return (
        f"id: {conv['id']}\n"
        f"path: {conv['path']}\n"
        f"--- metadata ---\n"
        f"{meta}\n"
        f"--- transcript ({total} chars) ---\n"
        f"{page}{trailer}"
    )


@mcp.tool()
def list_recent_conversations(n: int = 10) -> str:
    """List the n most recent conversations (id, date, title, participants) so you
    can pick one to read."""
    paths = list_conversations()[:n]
    if not paths:
        return "No conversations recorded yet."

    lines = [f"{len(paths)} most recent conversation(s):"]
    for p in paths:
        conv = parse_conversation(p)
        fm = conv["frontmatter"] or {}
        lines.append(
            f"  - {conv['id']}\n"
            f"      date:         {fm.get('date', 'unknown')}\n"
            f"      title:        {fm.get('title', '(untitled)')}\n"
            f"      participants: {_fmt_participants(fm.get('participants'))}"
        )
    return "\n".join(lines)


@mcp.tool()
def who_is(name: str) -> str:
    """Look up a colleague in the roster (people.yaml) — returns their role and
    aliases if known."""
    needle = name.strip().casefold()
    for person in config.load_people():
        candidates = [str(person.get("name", ""))]
        candidates.extend(str(a) for a in person.get("aliases", []) or [])
        if any(c.strip().casefold() == needle for c in candidates if c):
            aliases = person.get("aliases") or []
            aliases_str = ", ".join(str(a) for a in aliases) if aliases else "(none)"
            return (
                f"{person.get('name', name)}\n"
                f"  role:    {person.get('role', 'unknown')}\n"
                f"  aliases: {aliases_str}"
            )
    return f"{name!r} is not in the roster (people.yaml)."


@mcp.tool()
def add_thought(text: str, topics: str = "") -> str:
    """Capture a free-form THOUGHT from the user into their second brain (not from a
    recording). Use this whenever the user says things like "add to my brain / note this
    / remember that ...". `topics` is an optional comma-separated list of topic slugs to
    file the thought under (e.g. "odrive-safety, thermal"). The thought becomes
    immediately searchable and is filed into the given topic notes."""
    topic_list = [t.strip() for t in topics.split(",") if t.strip()]
    path = query.add_thought(text, topic_list)
    filed = f" filed under: {', '.join(topic_list)}" if topic_list else ""
    return f"Saved thought → {path.name}{filed}"


@mcp.tool()
def list_topics() -> str:
    """List all living topic notes (accumulated knowledge across conversations) with how
    many conversations have contributed to each. Use to see what the brain knows about."""
    topics = query.list_topics()
    if not topics:
        return "No topic notes yet."
    lines = [f"{len(topics)} topic(s):"]
    for t in topics:
        lines.append(f"  - {t['slug']}  ({t['entries']} contribution(s)) — {t['title']}")
    return "\n".join(lines)


@mcp.tool()
def get_topic(topic: str = "", slug: str = "") -> str:
    """Return the full living topic note for a slug (e.g. 'robot-safety-control-architecture').
    This is synthesized, accumulated knowledge — prefer it over raw transcript search when
    the user asks 'what do we know / what's our thinking about <topic>'. `slug` is accepted
    as an alias for `topic`."""
    topic = topic or slug
    if not topic:
        avail = ", ".join(t["slug"] for t in query.list_topics()) or "(none)"
        return f"get_topic needs topic=<slug>. Available: {avail}"
    note = query.get_topic(topic)
    if note is None:
        avail = ", ".join(t["slug"] for t in query.list_topics()) or "(none)"
        return f"No topic note {topic!r}. Available: {avail}"
    return note


@mcp.tool()
def list_action_items(open_only: bool = True) -> str:
    """List action items extracted from distilled conversations (with owner/task and
    source conversation). Use for 'what do I need to do', 'open action items', or to help
    answer 'who should I talk to today' by seeing what's outstanding and who owns it."""
    items = query.list_action_items(open_only=open_only)
    if not items:
        return "No open action items found." if open_only else "No action items found."
    lines = [f"{len(items)} action item(s):"]
    for it in items:
        mark = " " if not it["done"] else "x"
        lines.append(f"  - [{mark}] {it['task']}\n      (from {it['source']}, {it['date']})")
    return "\n".join(lines)


MCP_HTTP_PORT = 8765


if __name__ == "__main__":
    if sys.argv[1:] == ["--http"]:
        mcp.settings.port = MCP_HTTP_PORT  # host stays 127.0.0.1: local clients only
        mcp.run(transport="streamable-http")
    elif sys.argv[1:]:
        sys.exit(f"usage: {sys.argv[0]} [--http]")
    else:
        mcp.run()  # stdio
