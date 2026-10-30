"""Collapse degenerate speech-to-text repetition loops.

Whisper (both backends) can fall into a stuck decoding loop on silence, noise or
far-field crosstalk, emitting the same token hundreds of times ("again again
again ..."). Whisper's own guard is `compression_ratio_threshold`: a too-repetitive
window is retried at successively higher temperature. But when every temperature in
the fallback ladder still produces garbage, whisper KEEPS the last result — so the
loop reaches the transcript regardless of decode settings. That is why disabling
`condition_on_previous_text` reduced but did not eliminate these.

So we clean up deterministically after decoding. A collapsed run leaves a visible
marker rather than vanishing, because a loop means audio was LOST there: the reader
(and the distiller) must not read the surrounding text as continuous speech.
"""
import re

MARKER = "[…{n}× STT loop trimmed…]"

# A run must repeat at least this many times to count as degenerate. Real speech
# does repeat ("yeah yeah yeah", "no no no"), so keep this comfortably above that.
MIN_RUN = 6
MAX_NGRAM = 4


def collapse(text: str, min_run: int = MIN_RUN, max_ngram: int = MAX_NGRAM) -> tuple[str, int]:
    """Collapse degenerate n-gram runs. Returns (cleaned_text, words_removed)."""
    if not text:
        return text, 0
    tokens = text.split()
    n_before = len(tokens)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        best = None  # (ngram_len, repeat_count)
        for n in range(1, max_ngram + 1):
            if i + n * min_run > len(tokens):
                continue
            gram = [_norm(t) for t in tokens[i:i + n]]
            # A gram that normalises to nothing is bare punctuation (". . . . ."),
            # which is also loop debris — match those on the raw tokens instead.
            key = gram if any(gram) else tokens[i:i + n]
            reps = 1
            j = i + n
            while j + n <= len(tokens) and (
                [_norm(t) for t in tokens[j:j + n]] if any(gram) else tokens[j:j + n]
            ) == key:
                reps += 1
                j += n
            if reps >= min_run and (best is None or reps * n > best[0] * best[1]):
                best = (n, reps)
        if best:
            n, reps = best
            out.extend(tokens[i:i + n])          # keep ONE instance
            out.append(MARKER.format(n=reps))    # record what was dropped
            i += n * reps
        else:
            out.append(tokens[i])
            i += 1
    cleaned = " ".join(out)
    # words_removed counts only real words, not the markers we added
    removed = n_before - len([t for t in out if not t.startswith("[…")])
    return cleaned, removed


def _norm(tok: str) -> str:
    return re.sub(r"[^\w']", "", tok).lower()


def clean_segments(segments: list[dict]) -> tuple[list[dict], int]:
    """Collapse loops inside each segment, then merge consecutive segments that are
    nothing but the same collapsed loop (a long loop spans many windows)."""
    total = 0
    cleaned = []
    for s in segments:
        text, removed = collapse(s.get("text", ""))
        total += removed
        s = {**s, "text": text}
        if cleaned and _loop_key(text) and _loop_key(cleaned[-1]["text"]) == _loop_key(text):
            cleaned[-1]["end"] = s["end"]          # absorb into the previous loop
            total += len(text.split())
            continue
        cleaned.append(s)
    return cleaned, total


def _loop_key(text: str) -> str | None:
    """If a segment is purely one collapsed loop, return its base token, else None."""
    m = re.fullmatch(r"(.*?)\s*\[…\d+× STT loop trimmed…\]", text.strip())
    return _norm(m.group(1)) if m else None
