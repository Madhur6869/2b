"""Conversation markdown store: write, parse, and list conversation files.

File format (frontmatter dumped as YAML between --- lines, then the body):

    ---
    date: 2026-07-16 14:30
    participants: [Alex Chen, Priya R]
    duration: 12m34s
    audio: audio/2026-07-16_1430.wav
    topics: []
    language: en
    title: Planner latency chat
    ---
    <transcript body>

Fail-hard: real errors raise.
"""
from datetime import datetime
from pathlib import Path

import yaml

from secondbrain import config, util


def write_conversation(
    *,
    title: str,
    participants: list[str],
    audio_path: Path | None,
    transcript: dict,
    started_at: datetime,
    diarized: bool = False,
) -> Path:
    """Write a conversation .md file and return its Path.

    `diarized` records whether speaker labelling actually succeeded. It is written
    to the frontmatter because diarization is a soft-fail step: without this flag an
    unlabelled transcript is indistinguishable from a labelled one that happened to
    have few turns, and audio retention (secondbrain/prune.py) depends on knowing
    the difference.
    """
    if audio_path is not None:
        audio_rel = str(Path(audio_path).resolve().relative_to(config.ROOT))
    else:
        audio_rel = ""

    frontmatter = {
        "date": util.human_time(started_at),
        "participants": list(participants),
        "duration": util.human_duration(transcript["duration"]),
        "audio": audio_rel,
        "topics": [],
        "language": transcript["language"],
        "title": title,
        "diarized": bool(diarized),
    }

    md_path = (
        config.CONVERSATIONS_DIR
        / f"{util.stamp(started_at)}_{util.slugify(title)}.md"
    )

    fm_yaml = yaml.safe_dump(
        frontmatter,
        sort_keys=False,
        default_flow_style=None,
        allow_unicode=True,
    )
    content = f"---\n{fm_yaml}---\n{transcript['text']}\n"
    md_path.write_text(content, encoding="utf-8")

    print(f"[store] wrote {md_path}")
    return md_path


def write_thought(text: str, tags: list[str] | None = None, source: str = "session") -> Path:
    """Write a free-form 'thought' note (typed, no audio) and return its Path.

    Thoughts live in the same store as conversations so they share one search index.
    They are marked distilled=true so the conversation distiller skips them (they are
    already clean); topic filing is handled by the caller via `tags`.
    """
    now = util.now()
    text = (text or "").strip()
    if not text:
        raise ValueError("Refusing to write an empty thought")

    # Title = first few words of the thought.
    words = text.split()
    title = " ".join(words[:8]) + ("…" if len(words) > 8 else "")

    frontmatter = {
        "date": util.human_time(now),
        "type": "thought",
        "participants": ["Madhur"],
        "tags": list(tags or []),
        "source": source,
        "title": title,
        "distilled": True,
    }
    md_path = config.CONVERSATIONS_DIR / f"{util.stamp(now)}_thought_{util.slugify(title)}.md"
    fm_yaml = yaml.safe_dump(
        frontmatter, sort_keys=False, default_flow_style=None, allow_unicode=True
    )
    md_path.write_text(f"---\n{fm_yaml}---\n{text}\n", encoding="utf-8")
    print(f"[store] wrote thought {md_path}")
    return md_path


def parse_conversation(md_path: Path) -> dict:
    """Parse a conversation .md into id, path, frontmatter, and body."""
    md_path = Path(md_path)
    raw = md_path.read_text(encoding="utf-8")

    if not raw.startswith("---\n"):
        raise ValueError(f"{md_path} is missing a leading '---' frontmatter block")

    # Split off the frontmatter: everything between the first two '---' lines.
    _, fm_text, body = raw.split("---\n", 2)
    frontmatter = yaml.safe_load(fm_text)

    return {
        "id": md_path.stem,
        "path": str(md_path),
        "frontmatter": frontmatter,
        "body": body,
    }


def list_conversations() -> list[Path]:
    """All conversation .md files, newest first (filenames start with a stamp)."""
    return sorted(
        config.CONVERSATIONS_DIR.glob("*.md"),
        key=lambda p: p.name,
        reverse=True,
    )
