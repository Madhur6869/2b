"""Distillation layer — turn raw transcripts into clean structured knowledge.

Runs the `claude` CLI in headless mode (`claude -p`), which uses your Claude
subscription rather than the metered API. For each conversation that hasn't been
distilled yet it produces:
  - a clean structured summary (TL;DR / decisions / action items / open questions /
    entities), inserted at the top of the conversation file, and
  - appends durable knowledge to living per-topic notes in topics/.

Fail-hard: if the CLI errors or returns an unparseable response, we raise rather
than silently writing a bad note.
"""
import re
import subprocess
from pathlib import Path

import yaml

from secondbrain import config, util
from secondbrain.store import parse_conversation, list_conversations

SUMMARY_MARK = "<!-- DISTILLED SUMMARY (auto-generated) -->"
RAW_MARK = "<!-- RAW TRANSCRIPT -->"

PROMPT_TEMPLATE = """You are the distillation engine for a personal "second brain". \
You are given the RAW, error-prone speech-to-text transcript of a spoken conversation. \
The audio was far-field so there are transcription errors (e.g. "ODrive" may appear as \
"O drive", "overdrive", "hoedrive", "OREF"; "lock" may appear as "log"). Use context to \
silently correct obvious errors. Do NOT invent facts that are not in the transcript.

Participants in this conversation: {participants}

Existing topic notes you may append to (slug | title). Reuse an existing slug when the \
conversation touches that topic; otherwise create a new concise kebab-case slug:
{topic_index}

Output EXACTLY in the following format and NOTHING else (no preamble, no code fences):

===TITLE===
<a concise 4-8 word title naming what this conversation was actually about; no date, no quotes>
===SUMMARY===
## TL;DR
<2-4 sentence plain-language summary of what this conversation was about>

## Decisions
<bullet list of decisions/conclusions reached; "- none" if truly none>

## Open questions
<bullet list of unresolved questions raised; "- none" if none>

## Entities
- People: <names mentioned>
- Projects/Systems: <projects, systems, components, tools mentioned>

===TOPICS===
<For each durable topic this conversation contributes knowledge to, output a block:>
### <slug> | <Human Title>
<1-5 standalone bullets of durable knowledge to remember about this topic — written so \
they make sense months later without the transcript. "- none" blocks are not allowed; \
only include topics that genuinely gained knowledge.>
===END===

RAW TRANSCRIPT:
{transcript}
"""


def find_undistilled() -> list[Path]:
    out = []
    for p in list_conversations():
        fm = parse_conversation(p)["frontmatter"]
        if not fm.get("distilled"):
            out.append(p)
    return out


def _topic_index() -> str:
    lines = []
    for p in sorted(config.TOPICS_DIR.glob("*.md")):
        title = p.stem
        first = p.read_text().splitlines()
        if first and first[0].startswith("# "):
            title = first[0][2:].strip()
        lines.append(f"- {p.stem} | {title}")
    return "\n".join(lines) if lines else "(none yet)"


def _run_claude(prompt: str) -> str:
    claude = config.resolve_claude_bin()
    proc = subprocess.run(
        [claude, "-p", prompt],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"`claude -p` failed (exit {proc.returncode}): {proc.stderr.strip()[:500]}"
        )
    return proc.stdout


def _parse_output(out: str) -> tuple[str, str, list[tuple[str, str, str]]]:
    """Return (title, summary_markdown, [(slug, title, body), ...])."""
    if "===SUMMARY===" not in out or "===TOPICS===" not in out:
        raise RuntimeError(
            "Distillation output missing required markers; refusing to write a bad note.\n"
            f"Got: {out[:400]!r}"
        )
    conv_title = ""
    if "===TITLE===" in out:
        _, rest = out.split("===TITLE===", 1)
        title_block, rest = rest.split("===SUMMARY===", 1)
        lines = [ln.strip() for ln in title_block.strip().splitlines() if ln.strip()]
        conv_title = lines[0].strip().strip('"').strip("#").strip() if lines else ""
    else:
        _, rest = out.split("===SUMMARY===", 1)
    summary, rest = rest.split("===TOPICS===", 1)
    topics_block = rest.split("===END===", 1)[0]
    summary = summary.strip()

    topics: list[tuple[str, str, str]] = []
    for block in re.split(r"(?m)^###\s+", topics_block):
        block = block.strip()
        if not block:
            continue
        head, _, body = block.partition("\n")
        if "|" not in head:
            continue
        slug, title = (s.strip() for s in head.split("|", 1))
        slug = util.slugify(slug)
        body = body.strip()
        if slug and body and body.lower() != "- none":
            topics.append((slug, title, body))
    return conv_title, summary, topics


def _write_summary_into_file(md_path: Path, summary: str, title: str = "") -> None:
    parsed = parse_conversation(md_path)
    fm = dict(parsed["frontmatter"])
    fm["distilled"] = True
    if title:
        fm["title"] = title  # auto-name the conversation from its content

    # Body may already contain a prior summary block (if force re-distilling) — strip it
    # back to just the raw transcript before re-inserting.
    body = parsed["body"]
    if RAW_MARK in body:
        body = body.split(RAW_MARK, 1)[1].lstrip("\n")

    fm_yaml = yaml.safe_dump(fm, default_flow_style=None, sort_keys=False, allow_unicode=True).strip()
    new = (
        f"---\n{fm_yaml}\n---\n"
        f"{SUMMARY_MARK}\n{summary}\n\n"
        f"{RAW_MARK}\n{body.rstrip()}\n"
    )
    md_path.write_text(new)


def _remove_conv_from_topics(conv_id: str) -> None:
    """Remove any existing entries for this conversation from all topic notes, so
    re-distilling replaces rather than duplicates them."""
    pattern = re.compile(
        r"(?ms)^## From:.*?\[\[" + re.escape(conv_id) + r"\]\].*?(?=^## From:|\Z)"
    )
    for p in config.TOPICS_DIR.glob("*.md"):
        text = p.read_text()
        new = pattern.sub("", text)
        if new != text:
            p.write_text(new.rstrip() + "\n")


def _append_topic(slug: str, title: str, body: str, conv_id: str, conv_title: str, date: str) -> None:
    path = config.TOPICS_DIR / f"{slug}.md"
    if not path.exists():
        path.write_text(
            f"# {title}\n\n_Living note — knowledge accumulated across conversations. "
            f"Auto-maintained by the second brain._\n"
        )
    entry = (
        f"\n## From: {conv_title} ({date}) — [[{conv_id}]]\n{body.rstrip()}\n"
    )
    with path.open("a") as f:
        f.write(entry)


def distill_file(md_path: Path) -> dict:
    parsed = parse_conversation(md_path)
    fm = parsed["frontmatter"]

    # Distill from the RAW transcript only (strip any existing summary block).
    raw = parsed["body"]
    if RAW_MARK in raw:
        raw = raw.split(RAW_MARK, 1)[1].lstrip("\n")

    participants = ", ".join(fm.get("participants") or []) or "unknown"
    prompt = PROMPT_TEMPLATE.format(
        participants=participants,
        topic_index=_topic_index(),
        transcript=raw,
    )

    print(f"[distill] {md_path.name}: calling claude...")
    out = _run_claude(prompt)
    title, summary, topics = _parse_output(out)

    _write_summary_into_file(md_path, summary, title)

    conv_id = parsed["id"]
    conv_title = title or fm.get("title", conv_id)
    date = str(fm.get("date", ""))
    _remove_conv_from_topics(conv_id)  # idempotent: drop prior entries before re-adding
    for slug, title, body in topics:
        _append_topic(slug, title, body, conv_id, conv_title, date)

    # Re-index so the clean summary is searchable.
    from secondbrain.index import index_file
    index_file(md_path)

    print(f"[distill] {md_path.name}: done ({len(topics)} topic note(s) updated).")
    return {"summary": summary, "topics": [t[0] for t in topics]}


def distill_all(force: bool = False) -> int:
    targets = list_conversations() if force else find_undistilled()
    if not targets:
        print("[distill] nothing to distill.")
        return 0
    for p in targets:
        distill_file(p)
    return len(targets)
