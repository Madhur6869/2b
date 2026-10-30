"""Audio retention — delete .wav files whose content is safely captured elsewhere.

Audio is by far the largest thing this app stores (~100 MB/hour). Once a recording
has been transcribed AND diarized, the .wav has done its job and can be pruned.

The bar for deleting is deliberately high, because audio is the ONLY copy — there is
no backup, and re-transcription/re-diarization is impossible once it's gone. A file
is pruned only when every one of these holds:

  1. a conversation .md exists and points at it
  2. that conversation is distilled
  3. its transcript is DIARIZED (speaker-labelled turns)
  4. the transcript is substantial (not a failed/empty transcription)
  5. the recording is older than the TTL

Rule 3 matters most. Diarization is a soft-fail enrichment in ingest.py: when it
fails, the transcript is still saved, just without labels and with only a warning to
stdout. Those transcripts look complete but have lost who-said-what forever, and the
audio is the only route back — so an unlabelled transcript blocks the delete.

Pruning stamps `audio_pruned` into the frontmatter so the loss is recorded rather
than looking like the file went missing on its own.
"""
import re
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from secondbrain import config
from secondbrain.store import parse_conversation, list_conversations

SUMMARY_MARK = "<!-- DISTILLED SUMMARY (auto-generated) -->"
RAW_MARK = "<!-- RAW TRANSCRIPT -->"

# A diarized transcript is a sequence of "<label>: text" turns. Labels are either
# pyannote's raw clusters (SPEAKER_00, SPEAKER_?), a human name substituted in later
# (Alice:), or a bracketed note for an unresolved cluster ([Alice/Bob - unresolved]:).
LABEL_RE = re.compile(
    r"^\s*(?:SPEAKER_[0-9?]+|\[[^\]]{1,60}\]|[A-Za-z][A-Za-z0-9 .'\-]{0,30})\s*:\s",
)

# Below this, treat the transcript as a failed transcription and keep the audio.
MIN_TRANSCRIPT_CHARS = 500
# Fraction of non-empty transcript lines that must carry a speaker label.
MIN_LABELLED_RATIO = 0.5


def raw_transcript(body: str) -> str:
    """The transcript itself, with any distilled summary stripped off the front."""
    if RAW_MARK in body:
        return body.split(RAW_MARK, 1)[1]
    if SUMMARY_MARK in body:
        # Summary present but no raw marker — nothing reliable to measure.
        return ""
    return body


def is_diarized(frontmatter: dict, body: str) -> bool:
    """True if the transcript carries speaker attribution.

    A hand-written `speakers:` map in the frontmatter is definitive — someone
    resolved the clusters to names. Otherwise require that most transcript lines
    are labelled turns, so a plain transcript that happens to contain a stray
    "Yeah: ok" line can't pass.
    """
    # An explicit flag written at ingest time is definitive, in both directions.
    if isinstance(frontmatter.get("diarized"), bool):
        return frontmatter["diarized"]
    if frontmatter.get("speakers"):
        return True

    lines = [ln for ln in raw_transcript(body).splitlines() if ln.strip()]
    if not lines:
        return False
    labelled = sum(1 for ln in lines if LABEL_RE.match(ln))
    return labelled >= 3 and (labelled / len(lines)) >= MIN_LABELLED_RATIO


def recording_date(frontmatter: dict, audio: Path) -> datetime:
    """When the conversation happened. Prefer the frontmatter date over file mtime,
    which changes if the file is ever copied or touched."""
    raw = frontmatter.get("date")
    if isinstance(raw, datetime):
        return raw
    if raw:
        try:
            return datetime.fromisoformat(str(raw))
        except ValueError:
            pass
    return datetime.fromtimestamp(audio.stat().st_mtime)


def evaluate(md_path: Path, ttl_days: int, require_diarized: bool) -> dict:
    """Decide whether one conversation's audio can be pruned.

    Returns {'path', 'audio', 'bytes', 'prunable', 'reason'}. `reason` explains the
    decision either way, so a dry run is self-documenting.
    """
    parsed = parse_conversation(md_path)
    fm = parsed["frontmatter"] or {}
    body = parsed["body"]

    out = {"path": md_path, "audio": None, "bytes": 0, "prunable": False,
           "reason": "", "code": ""}

    rel = fm.get("audio")
    if not rel:
        out["reason"] = "no audio in frontmatter (thought/imported note)"
        out["code"] = "no-audio"
        return out

    audio = config.ROOT / rel
    out["audio"] = audio
    if not audio.exists():
        out["reason"] = "audio already gone"
        out["code"] = "gone"
        return out
    out["bytes"] = audio.stat().st_size

    if not fm.get("distilled"):
        out["reason"] = "not distilled yet"
        out["code"] = "not-distilled"
        return out

    if len(raw_transcript(body).strip()) < MIN_TRANSCRIPT_CHARS:
        out["reason"] = "transcript too short — looks like a failed transcription"
        out["code"] = "short"
        return out

    if require_diarized and not is_diarized(fm, body):
        out["reason"] = "NOT DIARIZED — audio is the only way to recover speakers"
        out["code"] = "undiarized"
        return out

    age = (datetime.now() - recording_date(fm, audio)).days
    if age < ttl_days:
        out["reason"] = f"only {age}d old (TTL {ttl_days}d)"
        out["code"] = "too-young"
        return out

    out["prunable"] = True
    out["code"] = "prunable"
    out["reason"] = f"transcribed, diarized, distilled, {age}d old"
    return out


def _stamp_pruned(md_path: Path) -> None:
    """Record in the conversation that its audio was intentionally deleted."""
    parsed = parse_conversation(md_path)
    fm = dict(parsed["frontmatter"] or {})
    fm["audio_pruned"] = datetime.now().strftime("%Y-%m-%d")
    fm_yaml = yaml.safe_dump(fm, sort_keys=False, default_flow_style=None, allow_unicode=True)
    md_path.write_text(f"---\n{fm_yaml}---\n{parsed['body']}", encoding="utf-8")


def prune(dry_run: bool = True, ttl_days: int | None = None,
          require_diarized: bool | None = None) -> dict:
    """Prune expired audio. Returns a summary dict; prints one line per decision."""
    ttl_days = config.AUDIO_TTL_DAYS if ttl_days is None else ttl_days
    if require_diarized is None:
        require_diarized = config.AUDIO_TTL_REQUIRE_DIARIZED

    results = [evaluate(p, ttl_days, require_diarized) for p in list_conversations()]
    doomed = [r for r in results if r["prunable"]]
    freed = sum(r["bytes"] for r in doomed)

    # Surface what is being KEPT for a reason worth knowing about, not just what goes.
    NOTEWORTHY = {"undiarized", "short", "not-distilled"}
    for r in results:
        if r["code"] in NOTEWORTHY and r["audio"] and r["audio"].exists():
            mb = r["bytes"] / 1e6
            print(f"[prune] KEEP {r['audio'].name} ({mb:.0f} MB): {r['reason']}")

    for r in doomed:
        mb = r["bytes"] / 1e6
        if dry_run:
            print(f"[prune] would delete {r['audio'].name} ({mb:.0f} MB) — {r['reason']}")
        else:
            r["audio"].unlink()
            _stamp_pruned(r["path"])
            print(f"[prune] deleted {r['audio'].name} ({mb:.0f} MB) — {r['reason']}")

    verb = "would free" if dry_run else "freed"
    print(f"[prune] {len(doomed)} file(s), {verb} {freed / 1e6:.0f} MB")
    return {"count": len(doomed), "bytes": freed, "dry_run": dry_run}
