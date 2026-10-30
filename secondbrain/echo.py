"""Remove the speaker echo of the system audio from the mic.

On speakers (not headphones) the mic also hears the other side of a call, so mixing in
the system audio would record it twice, a few tens of ms apart. The system-audio track
is exactly what went to the speakers, so the echo is predicted from it and subtracted:
per block, find the delay (GCC-PHAT), fit a short filter per frequency bin (system
audio -> mic) by least squares, and subtract its output from the mic.

With headphones the fitted filter comes out ~0 and the mic is left as it was, so there
is no speakers-vs-headphones decision to get wrong. Least squares over a whole block is
not thrown off by you talking over the call: your voice is uncorrelated with the
system audio.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

N_FFT = 1024  # 64 ms frames at 16 kHz
HOP = 256  # 75 % overlap, so per-bin filters model the echo well
TAPS = 16  # filter length in frames: ~256 ms of room reverb after the direct path
LEAD = 2  # filter frames before the measured delay, for timestamp error
MAX_DELAY_S = 2.5  # output latency + acoustic path; AirPlay reaches ~2 s
BLOCK_S = 20.0  # the filter is refit per block, so it follows a changing echo path
SILENT_RMS = 1e-4  # system audio below -80 dBFS has no echo worth removing
ECHO_SHARE = 0.1  # a block whose echo is >= -10 dB of the mic counts as "on speakers"

_WINDOW = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype("float32")
_OLA = N_FFT / (2 * HOP)  # overlapping squared (sqrt-Hann) windows sum to this


def _stft(x: np.ndarray, first: int, count: int) -> np.ndarray:
    """STFT frames first .. first+count-1 of x, as (count, bins). Frame t covers samples
    [t*HOP, t*HOP + N_FFT); samples outside x are zero."""
    start = first * HOP
    segment = np.zeros((count - 1) * HOP + N_FFT, dtype="float32")
    lo, hi = max(start, 0), min(start + len(segment), len(x))
    if hi > lo:
        segment[lo - start : hi - start] = x[lo:hi]
    frames = sliding_window_view(segment, N_FFT)[::HOP]
    return np.fft.rfft(frames * _WINDOW, axis=1).astype("complex64")


def _delay(mic: np.ndarray, ref: np.ndarray, start: int, end: int, max_lag: int) -> int:
    """Samples by which ref's echo in mic[start:end] trails ref, by GCC-PHAT."""
    lo = max(start - max_lag, 0)
    m, r = mic[lo:end], ref[lo:end]
    n = 1 << int(np.ceil(np.log2(2 * len(m))))
    cross = np.fft.rfft(m, n) * np.conj(np.fft.rfft(r, n))
    correlation = np.fft.irfft(cross / (np.abs(cross) + 1e-12), n)
    lags = np.arange(-LEAD * HOP, max_lag + 1)  # negative lags wrap: circular correlation
    return int(lags[np.argmax(correlation[lags])])


def remove_echo(
    mic: np.ndarray, ref: np.ndarray, rate: int, boundaries: list[int]
) -> tuple[np.ndarray, str]:
    """The mic minus the echo of ref (the system audio) in it.

    mic and ref are mono float32 on one timeline at `rate`. `boundaries` are sample
    indices where the echo path changed (the output device switched); no block spans
    one. Returns (clean mic, a one-line report for the log).
    """
    n = len(mic)
    ref = ref[:n] if len(ref) >= n else np.pad(ref, (0, n - len(ref)))
    max_lag = int(MAX_DELAY_S * rate)
    overlap = N_FFT // HOP
    first = 1 - overlap  # frames from here on cover every sample `overlap` times
    hops = -(-n // HOP)
    block = int(BLOCK_S * rate) // HOP
    cuts = sorted(
        {first, hops}
        | set(range(first, hops, block))
        | {b // HOP for b in boundaries if 0 < b < n}
    )

    echo = np.zeros((hops - first + overlap) * HOP, dtype="float32")  # sample i at i - first*HOP
    echo_hops = echo.reshape(-1, HOP)
    active = speaker = 0
    mic_energy = residual_energy = 0.0
    for t0, t1 in zip(cuts, cuts[1:]):
        s0, s1 = max(t0 * HOP, 0), min(t1 * HOP, n)
        if s1 <= s0 or not mic[s0:s1].any():
            continue
        if np.sqrt(np.mean(np.square(ref[max(s0 - max_lag, 0) : s1]))) < SILENT_RMS:
            continue
        active += 1

        d = _delay(mic, ref, s0, s1, max_lag) // HOP - LEAD
        count = t1 - t0
        M = _stft(mic, t0, count)  # (T, F)
        R = _stft(ref, t0 - d - TAPS + 1, count + TAPS - 1)
        # X[f, t, j] = ref frame (t0 + t) - d - (TAPS - 1 - j): the filter's inputs.
        X = np.ascontiguousarray(sliding_window_view(R, TAPS, axis=0).transpose(1, 0, 2))
        Xh = X.conj().transpose(0, 2, 1)
        A = Xh @ X  # (F, TAPS, TAPS)
        b = Xh @ M.T[:, :, None]  # (F, TAPS, 1)
        load = 1e-3 * np.trace(A, axis1=1, axis2=2).real / TAPS + 1e-9
        A += load[:, None, None] * np.eye(TAPS, dtype=A.dtype)
        E = (X @ np.linalg.solve(A, b))[:, :, 0].T  # (T, F) predicted echo
        # With no echo, TAPS regressors still "explain" ~TAPS/count of each bin by chance;
        # subtracting that would only add a faint copy of the system audio. Keep a bin's
        # prediction only when it explains clearly more than chance.
        explained = np.sum(np.abs(E) ** 2, axis=0) / (np.sum(np.abs(M) ** 2, axis=0) + 1e-20)
        E[:, explained < 4 * TAPS / count] = 0

        frames = np.fft.irfft(E, N_FFT, axis=1).astype("float32") * (_WINDOW / _OLA)
        for q in range(overlap):
            echo_hops[t0 - first + q : t1 - first + q] += frames[:, q * HOP : (q + 1) * HOP]

        block_mic = float(np.sum(np.abs(M) ** 2))
        if np.sum(np.abs(E) ** 2) >= ECHO_SHARE * block_mic:
            speaker += 1
            mic_energy += block_mic
            residual_energy += float(np.sum(np.abs(M - E) ** 2))

    clean = mic - echo[-first * HOP : -first * HOP + n]
    if speaker:
        removed = 10 * np.log10(mic_energy / max(residual_energy, 1e-12))
        report = f"speaker echo in {speaker} of {active} blocks with system audio; mic down {removed:.1f} dB there"
    else:
        report = f"no speaker echo in {active} blocks with system audio"
    return clean, report
