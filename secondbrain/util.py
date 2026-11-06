"""Small shared helpers used across modules."""
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


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


def notify_failure(job: str, message: str, stamp_dir: Path) -> None:
    """macOS notification that a background job is failing, at most once a day.

    A launchd job's traceback only reaches a log nobody reads: distill failed every 15
    minutes for 8 days, and diarization for 2 weeks, before anyone noticed.
    """
    stamp = stamp_dir / f".{job}_alerted"
    if stamp.exists() and time.time() - stamp.stat().st_mtime < 86400:
        return
    stamp.touch()
    subprocess.run(
        ["osascript", "-e", "on run argv",
         "-e", "display notification (item 1 of argv) with title (item 2 of argv)",
         "-e", "end run", "--", message[:200], f"Second Brain: {job} is failing"],
        check=True,
    )
