"""Small shared helpers used across modules."""
import re
from datetime import datetime


def slugify(text: str, maxlen: int = 50) -> str:
    """Filesystem-safe slug: lowercase, hyphen-separated, trimmed."""
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return (text or "untitled")[:maxlen].strip("-")


def now() -> datetime:
    return datetime.now()


def stamp(dt: datetime) -> str:
    """Filename stamp: 2026-07-16_1430"""
    return dt.strftime("%Y-%m-%d_%H%M")


def human_time(dt: datetime) -> str:
    """Frontmatter/display time: 2026-07-16 14:30"""
    return dt.strftime("%Y-%m-%d %H:%M")


def human_duration(seconds: float) -> str:
    """e.g. 754 -> '12m34s', 42 -> '42s'."""
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"
