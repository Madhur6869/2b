"""Ingest pipeline: transcribe -> store -> index.

The markdown file is written BEFORE indexing, so a transcription result is
never lost if indexing deps are missing. We still let indexing errors raise —
fail-hard, no swallowing.
"""
import fcntl
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from secondbrain import config, store, transcribe


@contextmanager
def _one_at_a_time():
    """Serialize the heavy pipeline across processes: Whisper + pyannote peak at several
    GB, and two at once on an 18 GB Mac means a swap storm. The kernel drops the lock when
    the holder exits, so a crash can't leave it stuck."""
    with open(config.ROOT / ".ingest.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("[ingest] another ingest is running; waiting for it...", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        yield


@_one_at_a_time()
def ingest(
    audio_path: Path,
    title: str,
    participants: list[str],
    started_at: datetime,
) -> Path:
    """Transcribe an audio file, write its conversation .md, and index it."""
    audio_path = Path(audio_path)
    title = (title or "").strip() or "conversation"

    print(f"[ingest] transcribing {audio_path}...")
    transcript = transcribe.transcribe(audio_path)

    # Speaker diarization (who-said-what). Labels each segment with SPEAKER_xx and
    # rewrites the transcript body as labelled turns; distillation maps them to names.
    # Diarization is an ENRICHMENT: if it's unavailable (model not yet accessible,
    # etc.) we must NOT lose the transcript — save it plain and warn loudly. This is a
    # deliberate, visible fallback, not silent swallowing.
    diarized_ok = False
    if config.DIARIZE_ENABLED:
        from secondbrain import diarize

        try:
            n_spk = len([p for p in participants if p]) or None
            turns = diarize.diarize(audio_path, num_speakers=n_spk)
            labeled = diarize.label_segments(transcript["segments"], turns)
            transcript["segments"] = labeled
            transcript["text"] = diarize.format_labeled_text(labeled)
            diarized_ok = True
        except Exception as exc:
            print(
                f"[ingest] WARNING: diarization failed ({type(exc).__name__}: {exc}). "
                f"Saving the transcript WITHOUT speaker labels."
            )

    print("[ingest] writing conversation markdown...")
    md_path = store.write_conversation(
        title=title,
        participants=participants,
        audio_path=audio_path,
        transcript=transcript,
        started_at=started_at,
        diarized=diarized_ok,
    )

    # Import lazily and AFTER the md is safely on disk, so a missing indexing
    # dependency can't cost us the transcription. Errors still propagate.
    print("[ingest] indexing...")
    from secondbrain.index import index_file

    index_file(md_path)

    print(f"[ingest] done: {md_path}")
    return md_path


@_one_at_a_time()
def reingest_existing(md_path: Path) -> Path:
    """Re-transcribe (+diarize) an existing conversation from its audio and replace the raw
    transcript. Used to retro-apply diarization / a better model.

    The file goes back to un-distilled, so the distill job re-summarizes it. A diarization
    failure raises and leaves the file as it was: rewriting it unlabelled gains nothing.
    """
    import yaml
    from secondbrain import store

    md_path = Path(md_path)
    parsed = store.parse_conversation(md_path)
    fm = dict(parsed["frontmatter"])
    audio = config.ROOT / fm["audio"]
    if fm.get("audio_pruned"):
        raise FileNotFoundError(
            f"{md_path.name}: audio was pruned on {fm['audio_pruned']} once the conversation "
            f"was transcribed and diarized (see secondbrain/prune.py). Re-ingest is impossible; "
            f"the existing transcript is the only record."
        )
    if not fm.get("audio") or not audio.exists():
        raise FileNotFoundError(f"No audio for {md_path.name} (frontmatter audio={fm.get('audio')!r})")

    print(f"[reingest] {md_path.name}: re-transcribing {audio.name}...")
    transcript = transcribe.transcribe(audio)
    diarized_ok = False
    if config.DIARIZE_ENABLED:
        from secondbrain import diarize

        n_spk = len(fm.get("participants") or []) or None
        turns = diarize.diarize(audio, num_speakers=n_spk)
        labeled = diarize.label_segments(transcript["segments"], turns)
        transcript["text"] = diarize.format_labeled_text(labeled)
        diarized_ok = True

    # Rewrite the file as frontmatter + fresh raw transcript, un-distilled.
    fm.pop("distilled", None)
    fm["diarized"] = diarized_ok
    fm_yaml = yaml.safe_dump(fm, sort_keys=False, default_flow_style=None, allow_unicode=True)
    md_path.write_text(f"---\n{fm_yaml}---\n{transcript['text']}\n", encoding="utf-8")

    from secondbrain.index import index_file
    index_file(md_path)
    print(f"[reingest] done: {md_path}")
    return md_path
