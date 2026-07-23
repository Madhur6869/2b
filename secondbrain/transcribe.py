"""Transcription with a swappable backend.

- backend "mlx": Apple-Silicon GPU via mlx-whisper (fast).
- backend "faster-whisper": CPU via CTranslate2 (portable, no GPU).

Both return the same dict shape: {text, language, duration, segments}.
Fail-hard: real errors raise.
"""
from pathlib import Path

from secondbrain import config

# Map friendly model names to mlx-community HF repos (used by the mlx backend).
_MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

_FW_MODEL = None  # faster-whisper singleton


def _mlx_repo() -> str:
    m = config.WHISPER_MODEL
    return m if "/" in m else _MLX_REPOS.get(m, _MLX_REPOS["medium"])


def _transcribe_mlx(audio_path: Path) -> dict:
    import mlx_whisper

    repo = _mlx_repo()
    print(f"[transcribe] mlx (Apple GPU) model={repo}...")
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=repo,
        language=config.WHISPER_LANGUAGE,
    )
    segments = [
        {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
        for s in result.get("segments", [])
    ]
    duration = segments[-1]["end"] if segments else 0.0
    return {
        "text": "\n".join(s["text"] for s in segments),
        "language": result.get("language", "en"),
        "duration": float(duration),
        "segments": segments,
    }


def _get_fw_model():
    global _FW_MODEL
    if _FW_MODEL is None:
        from faster_whisper import WhisperModel

        print(f"[transcribe] loading faster-whisper {config.WHISPER_MODEL} "
              f"(compute_type={config.WHISPER_COMPUTE_TYPE})...")
        _FW_MODEL = WhisperModel(
            config.WHISPER_MODEL, device="cpu", compute_type=config.WHISPER_COMPUTE_TYPE
        )
    return _FW_MODEL


def _transcribe_faster_whisper(audio_path: Path) -> dict:
    model = _get_fw_model()
    print(f"[transcribe] faster-whisper transcribing {audio_path.name}...")
    segments_gen, info = model.transcribe(
        str(audio_path),
        language=config.WHISPER_LANGUAGE,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    segments, parts = [], []
    for seg in segments_gen:
        text = seg.text.strip()
        segments.append({"start": seg.start, "end": seg.end, "text": text})
        parts.append(text)
    return {
        "text": "\n".join(parts),
        "language": info.language,
        "duration": float(info.duration),
        "segments": segments,
    }


def transcribe(audio_path: Path) -> dict:
    """Transcribe a WAV. Returns {text, language, duration, segments}."""
    audio_path = Path(audio_path)
    if config.WHISPER_BACKEND == "mlx":
        result = _transcribe_mlx(audio_path)
    elif config.WHISPER_BACKEND == "faster-whisper":
        result = _transcribe_faster_whisper(audio_path)
    else:
        raise ValueError(f"Unknown whisper backend {config.WHISPER_BACKEND!r} (mlx | faster-whisper)")
    print(f"[transcribe] done: {len(result['segments'])} segments, "
          f"language={result['language']}, duration={result['duration']:.1f}s")
    return result
