"""Microphone capture to WAV.

Captures at the input device's NATIVE rate (CoreAudio won't let us force another),
then downsamples to config.SAMPLE_RATE mono and writes a 16-bit PCM WAV via
soundfile. Fail-hard: real errors raise.
"""
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from secondbrain import config


def _to_output_mono(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Mix to mono and resample src_rate -> dst_rate, returning a 1-D float32 array.

    Whisper only needs 16 kHz mono, and a 48 kHz built-in mic otherwise yields 3x
    larger files and a much heavier ingest. We capture native (CoreAudio's rule) and
    normalise here. Uses soxr when available, else scipy's polyphase resampler
    (exact 3:1 for 48k->16k); no-op when the rates already match.
    """
    if audio.ndim == 2 and audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype="float32").reshape(-1)

    if src_rate != dst_rate and len(audio):
        try:
            import soxr

            audio = soxr.resample(audio, src_rate, dst_rate)
        except ImportError:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(int(src_rate), int(dst_rate))
            audio = resample_poly(audio, dst_rate // g, src_rate // g)

    return np.asarray(audio, dtype="float32")


class Recorder:
    """Non-blocking microphone recorder backed by a sounddevice InputStream."""

    def __init__(
        self,
        output_path: Path,
        sample_rate: int = config.SAMPLE_RATE,
        channels: int = config.CHANNELS,
    ):
        self.output_path = Path(output_path)
        self.sample_rate = sample_rate
        self.channels = channels

        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._start_time: float | None = None
        self._stop_time: float | None = None
        self._statuses: list[str] = []

        # Set by stop(): whether captured audio fell short of wall-clock time
        # (e.g. a Bluetooth/AirPods mic link dropping out mid-recording).
        self.had_dropout: bool = False
        self.dropout_message: str = ""

    def _callback(self, indata, frames, time_info, status):
        # A non-empty status flags overflow/underflow/device errors — never swallow it.
        if status:
            self._statuses.append(str(status))
        # Copy: sounddevice reuses the underlying buffer between callbacks.
        with self._lock:
            self._frames.append(indata.copy())

    def start(self) -> None:
        if self._stream is not None:
            raise RuntimeError("Recorder already started")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Refresh PortAudio's device list so a mic the user just switched to
        # (built-in, or iPhone via Continuity) is picked up without restarting
        # the app. PortAudio otherwise caches the device list from process start.
        sd._terminate()
        sd._initialize()

        # macOS CoreAudio (AUHAL) rejects sample rates the input device does not
        # natively provide (PaErrorCode -9986 / AudioUnit err -10851), and PortAudio
        # will not resample for us. So capture at the device's NATIVE rate and try a
        # few fallbacks; Whisper resamples to 16 kHz when it reads the file, so the
        # stored rate does not matter for transcription.
        dev = sd.query_devices(kind="input")
        if int(dev["max_input_channels"]) < 1:
            raise RuntimeError(f"Default input device has no input channels: {dev['name']!r}")
        native_rate = int(round(dev["default_samplerate"]))
        max_ch = int(dev["max_input_channels"])

        candidates: list[tuple[int, int]] = []
        for rate in (native_rate, 48000, 44100, 16000):
            for ch in (1, max_ch):
                if (rate, ch) not in candidates:
                    candidates.append((rate, ch))

        self._frames = []
        self._statuses = []
        self.had_dropout = False
        self.dropout_message = ""
        last_err: Exception | None = None
        for rate, ch in candidates:
            try:
                stream = sd.InputStream(
                    samplerate=rate,
                    channels=ch,
                    dtype="float32",
                    callback=self._callback,
                )
                stream.start()
            except sd.PortAudioError as exc:
                last_err = exc
                continue
            self._stream = stream
            self.sample_rate = rate
            self.channels = ch
            print(f"[capture] recording at {rate} Hz, {ch} ch -> {self.output_path.name}")
            break
        else:
            raise RuntimeError(
                f"Could not open the microphone with any supported config "
                f"(tried {candidates}). Last PortAudio error: {last_err}"
            )

        self._start_time = time.monotonic()
        self._stop_time = None

    def stop(self) -> Path:
        if self._stream is None:
            raise RuntimeError("Recorder was never started")
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._stop_time = time.monotonic()

        with self._lock:
            frames = list(self._frames)
        if frames:
            audio = np.concatenate(frames, axis=0)
        else:
            audio = np.zeros((0, self.channels), dtype="float32")

        # Dropout detection runs on the RAW capture (native rate, pre-resample):
        # compare seconds of audio we actually got against wall-clock recording time.
        captured = len(audio) / self.sample_rate if len(audio) else 0.0

        # Downsample to the configured output rate (16 kHz) and mix to mono BEFORE
        # writing, so files stay small and ingest stays light regardless of which
        # mic was the default (a 48 kHz built-in mic otherwise triples file size).
        out_audio = _to_output_mono(audio, self.sample_rate, config.SAMPLE_RATE)
        sf.write(
            str(self.output_path),
            out_audio,
            config.SAMPLE_RATE,
            subtype="PCM_16",
        )
        wall = self.elapsed()
        self.had_dropout = wall - captured > 3.0 and captured < 0.8 * wall
        if self.had_dropout:
            self.dropout_message = (
                f"Captured only {captured:.0f}s of audio but recording ran for "
                f"{wall:.0f}s — the mic input dropped out mid-recording "
                f"(usually a Bluetooth/AirPods link glitch; {len(self._statuses)} "
                f"stream warning(s)). Switch input to the built-in mic for reliable capture."
            )
            print(f"[capture] WARNING: {self.dropout_message}")
        return self.output_path

    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        end = self._stop_time if self._stop_time is not None else time.monotonic()
        return end - self._start_time

    @property
    def is_recording(self) -> bool:
        return self._stream is not None


def record_until_interrupt(output_path: Path) -> Path:
    """Blocking record; press Ctrl-C to stop. Returns the WAV path."""
    recorder = Recorder(output_path)
    recorder.start()
    print("Recording... Ctrl-C to stop")
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    path = recorder.stop()
    from secondbrain import util

    print(f"Stopped after {util.human_duration(recorder.elapsed())}")
    return path
