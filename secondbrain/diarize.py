"""Speaker diarization (who-said-what) via pyannote.audio.

Produces speaker turns, aligns them to the Whisper transcript segments by maximal
time overlap, and renders a speaker-labelled transcript. The distillation step then
maps the anonymous SPEAKER_xx labels to real participant names from context.

Fail-hard: if enabled and the model/token is missing, we raise rather than silently
producing an unattributed transcript.
"""
from pathlib import Path

from secondbrain import config

_PIPELINE = None


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from pyannote.audio import Pipeline

        token = config.resolve_hf_token()
        print(f"[diarize] loading {config.DIARIZE_MODEL} (first run downloads the model)...")
        # pyannote/huggingface_hub renamed `use_auth_token` -> `token` in newer versions.
        try:
            pipe = Pipeline.from_pretrained(config.DIARIZE_MODEL, token=token)
        except TypeError:
            pipe = Pipeline.from_pretrained(config.DIARIZE_MODEL, use_auth_token=token)
        if pipe is None:
            raise RuntimeError(
                f"pyannote returned no pipeline for {config.DIARIZE_MODEL!r}. Accept the "
                f"model's terms at https://huggingface.co/{config.DIARIZE_MODEL} with the "
                f"same account as your token."
            )
        _PIPELINE = pipe
    return _PIPELINE


def diarize(audio_path: Path, num_speakers: int | None = None) -> list[dict]:
    """Return speaker turns: [{'start': float, 'end': float, 'speaker': str}, ...].

    Pass num_speakers (e.g. the known participant count) to guide clustering — on
    far-field mono audio the pipeline otherwise under-counts similar voices.
    """
    pipeline = _get_pipeline()
    print(f"[diarize] diarizing {Path(audio_path).name}"
          f"{f' (num_speakers={num_speakers})' if num_speakers else ''}...")
    if num_speakers and num_speakers > 1:
        annotation = pipeline(str(audio_path), num_speakers=num_speakers)
    else:
        annotation = pipeline(str(audio_path))
    # pyannote 4.x wraps the result in a DiarizeOutput; unwrap to the Annotation
    # (which exposes itertracks). 3.x already returns the Annotation directly.
    if hasattr(annotation, "speaker_diarization"):
        annotation = annotation.speaker_diarization
    turns = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    n_speakers = len({t["speaker"] for t in turns})
    print(f"[diarize] done: {len(turns)} turns, {n_speakers} speaker(s)")
    return turns


def _overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def label_segments(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Attach a 'speaker' to each Whisper segment by maximal time overlap."""
    labeled = []
    for seg in segments:
        best_speaker, best_ov = "SPEAKER_?", 0.0
        for t in turns:
            ov = _overlap(seg["start"], seg["end"], t["start"], t["end"])
            if ov > best_ov:
                best_ov, best_speaker = ov, t["speaker"]
        labeled.append({**seg, "speaker": best_speaker})
    return labeled


def format_labeled_text(labeled: list[dict]) -> str:
    """Render 'SPEAKER_xx: text' blocks, merging consecutive same-speaker segments."""
    lines, cur_speaker, buf = [], None, []
    for seg in labeled:
        if seg["speaker"] != cur_speaker:
            if buf:
                lines.append(f"{cur_speaker}: " + " ".join(buf).strip())
            cur_speaker, buf = seg["speaker"], [seg["text"]]
        else:
            buf.append(seg["text"])
    if buf:
        lines.append(f"{cur_speaker}: " + " ".join(buf).strip())
    return "\n".join(lines)
