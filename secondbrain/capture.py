"""Mic + system-audio capture to WAV.

The audiotap helper (audiotap/audiotap.swift) captures two sources and follows devices
as they connect and disconnect:

- the MIC. config capture.mic "auto" = the default input, except a Bluetooth headset
  (AirPods): then the built-in mic. Re-chosen every second, so a mic that disconnects
  or stalls is replaced mid-recording;
- the Mac's audio OUTPUT (capture.system_audio), via a Core Audio process tap. This is
  what makes calls on AirPods recordable: the other side goes straight into your ears
  and never reaches a mic.

Every chunk carries a host-clock timestamp. finish() lays both sources onto one
timeline (a dropout or device switch becomes silence instead of shifting everything
after it earlier), removes the speaker echo of the system audio from the mic
(secondbrain/echo.py: on speakers the mic hears the call too, which would otherwise be
recorded twice), and mixes them into one 16 kHz mono 16-bit WAV.
Fail-hard: real errors raise.
"""
import json
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from secondbrain import config, echo

# A jump this large between a chunk's timestamp and where the previous chunk ended means
# audio is missing (nothing playing, a dropout, a device switch): start a new run at the
# chunk's own timestamp, leaving silence for the gap. Smaller differences are clock
# jitter, and the run continues on sample count, the device's own clock.
_GAP_NS = 200_000_000


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample 1-D float32 audio src_rate -> dst_rate.

    Whisper only needs 16 kHz mono, and a 48 kHz built-in mic otherwise yields 3x
    larger files and a much heavier ingest. Uses soxr when available, else scipy's
    polyphase resampler (exact 3:1 for 48k->16k); no-op when the rates already match.
    """
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


def _timeline(chunks: list[tuple[int, int, np.ndarray]], t0_ns: int, out_rate: int) -> np.ndarray:
    """Lay timestamped mono chunks [(host_ns, rate, samples)] onto one out_rate timeline
    whose sample 0 is host time t0_ns."""
    runs: list[list] = []  # [start_ns, rate, [arrays], n_samples]
    for ns, rate, samples in chunks:
        if runs:
            run = runs[-1]
            run_end_ns = run[0] + run[3] * 1_000_000_000 / run[1]
            if rate == run[1] and ns - run_end_ns < _GAP_NS:
                run[2].append(samples)
                run[3] += len(samples)
                continue
        runs.append([ns, rate, [samples], len(samples)])

    placed: list[tuple[int, np.ndarray]] = []
    end = 0
    for start_ns, rate, arrays, _ in runs:
        audio = _resample(np.concatenate(arrays).astype("float32"), rate, out_rate)
        offset = max(end, round((start_ns - t0_ns) * out_rate / 1_000_000_000))
        placed.append((offset, audio))
        end = offset + len(audio)

    out = np.zeros(end, dtype="float32")
    for offset, audio in placed:
        out[offset : offset + len(audio)] = audio
    return out


def _audiotap(*args: str) -> str:
    if not config.AUDIOTAP_BIN.exists():
        raise RuntimeError(f"{config.AUDIOTAP_BIN} is not built. Run audiotap/build.sh.")
    return subprocess.run(
        [str(config.AUDIOTAP_BIN), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def describe(event: dict) -> str:
    """One readable line for an audiotap event."""
    kind = event["event"]
    if kind == "mic":
        return f"mic -> {event['device']} ({event['transport']}, {event['rate']:.0f} Hz): {event['reason']}"
    if kind == "no_mic":
        return f"no mic: {event['reason']}"
    if kind == "output":
        return f"output -> {event['device']} ({event['transport']})"
    if kind == "tap":
        return f"system audio tap: {event['reason']}"
    return f"audiotap {kind}: {event.get('message', event)}"


class Recorder:
    """Non-blocking recorder of the mic plus, when configured, the system audio.

    start() -> stop() (fast: capture ends) -> finish() (seconds per hour of audio:
    mixing and echo removal, then the WAV is written).
    """

    _HEADER = struct.Struct("<4sQId")  # source, host_ns, frames, sample_rate

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.events: list[dict] = []  # audiotap's device events, in order
        # Set by finish(): problems the user must hear about (a mic gap, missing system
        # audio, the helper dying). The WAV is written regardless.
        self.warnings: list[str] = []

        # float16 halves the memory of a long meeting (48 kHz: ~350 MB/hour per source
        # instead of ~700); far more precision than the 16-bit WAV this ends up in.
        self._chunks: dict[str, list[tuple[int, int, np.ndarray]]] = {"mic": [], "sys": []}
        self._proc: subprocess.Popen | None = None
        self._readers: list[threading.Thread] = []
        self._ready = threading.Event()  # audiotap reported its mic (and tap), or exited
        self._exited = threading.Event()  # audiotap's stderr closed: it has exited
        self._read_error = ""
        self._t0_ns: int | None = None
        self._start_time: float | None = None
        self._stop_time: float | None = None
        self._died_at: float | None = None  # elapsed() when the helper died on its own

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("Recorder already started")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Both permissions belong to the app that launched us. A denied one delivers
        # silence rather than an error, so refuse up front. "unknown" (never asked) is
        # fine: starting capture shows the prompt.
        permission = json.loads(_audiotap("permission"))
        if permission["microphone"] == "denied":
            raise RuntimeError(
                "Microphone access is denied. Enable Python in System Settings > "
                "Privacy & Security > Microphone."
            )
        if config.SYSTEM_AUDIO and permission["system_audio"] == "denied":
            raise RuntimeError(
                "Recording system audio is not allowed. In System Settings > Privacy & "
                "Security > Screen & System Audio Recording, enable Python under "
                "'System Audio Recording Only', or set capture.system_audio: false."
            )

        args = [str(config.AUDIOTAP_BIN), "record", "--mic", config.MIC]
        if not config.SYSTEM_AUDIO:
            args.append("--no-system-audio")
        for bundle_id in config.SYSTEM_AUDIO_EXCLUDE:
            args += ["--exclude-bundle-id", bundle_id]
        self._t0_ns = time.clock_gettime_ns(time.CLOCK_UPTIME_RAW)
        # Own session: a terminal Ctrl-C stops the CLI, which then stops the helper —
        # the helper must not see that SIGINT and look like it quit by itself.
        self._proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
        )
        self._readers = [
            threading.Thread(target=self._read_records, daemon=True),
            threading.Thread(target=self._read_events, daemon=True),
        ]
        for reader in self._readers:
            reader.start()

        # Wait for the helper to report its mic (and tap), so a broken start fails here.
        if not self._ready.wait(timeout=10):
            self._proc.kill()
            raise RuntimeError("audiotap did not start within 10 s")
        if self._exited.is_set():
            self._proc.wait()
            raise RuntimeError(f"audiotap failed to start: {self._helper_error()}")
        self._start_time = time.monotonic()
        self._stop_time = None

    def _read_records(self) -> None:
        stdout = self._proc.stdout
        while header := stdout.read(self._HEADER.size):
            if len(header) < self._HEADER.size:
                self._read_error = f"truncated record header ({len(header)} bytes)"
                return
            source, ns, frames, rate = self._HEADER.unpack(header)
            payload = stdout.read(frames * 4)
            if len(payload) < frames * 4:
                self._read_error = f"truncated record ({len(payload)} of {frames * 4} bytes)"
                return
            samples = np.frombuffer(payload, dtype="<f4").astype("float16")
            self._chunks[source.decode().strip()].append((ns, int(rate), samples))

    def _read_events(self) -> None:
        have_mic = have_tap = False
        for raw in self._proc.stderr:
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            # Everything audiotap writes is a JSON event; anything else is a crash report.
            event = json.loads(line) if line.startswith("{") else {"event": "stderr", "message": line}
            self.events.append(event)
            print(f"[capture] {describe(event)}", flush=True)
            have_mic |= event["event"] in ("mic", "no_mic")
            have_tap |= event["event"] == "tap"
            if have_mic and (have_tap or not config.SYSTEM_AUDIO):
                self._ready.set()
        self._exited.set()
        self._ready.set()

    def _helper_error(self) -> str:
        errors = [e.get("message", "") for e in self.events if e["event"] in ("error", "stderr")]
        return f"exit code {self._proc.returncode}" + (f": {'; '.join(errors)}" if errors else "")

    @property
    def failure(self) -> str:
        """Why capture stopped on its own mid-recording; "" while it is running fine."""
        if not self.is_recording or self._proc.poll() is None:
            return ""
        return f"audiotap {self._helper_error()}"

    def stop(self) -> None:
        """End capture. Fast; call finish() next to build and write the WAV."""
        if not self.is_recording:
            raise RuntimeError("Recorder is not recording")
        if self._proc.poll() is None:
            self._proc.terminate()
        else:
            self._died_at = self.elapsed()
        for reader in self._readers:
            reader.join()
        self._proc.wait()
        self._stop_time = time.monotonic()

    def finish(self) -> Path:
        """Mix the captured sources into the WAV at output_path and set warnings."""
        if self._stop_time is None:
            raise RuntimeError("finish() needs stop() first")
        rate = config.SAMPLE_RATE
        mic = _timeline(self._chunks["mic"], self._t0_ns, rate)
        system = _timeline(self._chunks["sys"], self._t0_ns, rate)
        length = max(len(mic), len(system))
        mic = np.pad(mic, (0, length - len(mic)))
        system = np.pad(system, (0, length - len(system)))
        if system.any():
            # The echo path changes with the output device: never fit across a switch.
            switches = [
                round((e["host_ns"] - self._t0_ns) * rate / 1_000_000_000)
                for e in self.events
                if e["event"] == "output"
            ]
            mic, report = echo.remove_echo(mic, system, rate, switches)
            print(f"[capture] echo: {report}", flush=True)
        sf.write(str(self.output_path), np.clip(mic + system, -1.0, 1.0), rate, subtype="PCM_16")

        self.warnings = []
        if self._died_at is not None:
            m, s = divmod(int(self._died_at), 60)
            self.warnings.append(f"Capture stopped by itself {m}:{s:02d} in ({self._helper_error()}).")
        if self._read_error:
            self.warnings.append(f"Reading audio from audiotap failed: {self._read_error}.")
        captured = sum(len(samples) / r for _, r, samples in self._chunks["mic"])
        wall = self.elapsed()
        if wall - captured > 3.0 and captured < 0.8 * wall:
            mic_events = "; ".join(describe(e) for e in self.events if e["event"] in ("mic", "no_mic"))
            self.warnings.append(
                f"Only {captured:.0f}s of the {wall:.0f}s recording has mic audio — the mic "
                f"dropped out or was missing. Mic changes: {mic_events}"
            )
        if config.SYSTEM_AUDIO and json.loads(_audiotap("permission"))["system_audio"] != "granted":
            self.warnings.append(
                "Recording system audio is not allowed, so the other side of any call is "
                "missing. Allow Python in System Settings > Privacy & Security > Screen & "
                "System Audio Recording."
            )
        for warning in self.warnings:
            print(f"[capture] WARNING: {warning}", flush=True)
        return self.output_path

    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        end = self._stop_time if self._stop_time is not None else time.monotonic()
        return end - self._start_time

    @property
    def is_recording(self) -> bool:
        return self._start_time is not None and self._stop_time is None


def record_until_interrupt(output_path: Path) -> Path:
    """Blocking record; press Ctrl-C to stop. Returns the WAV path."""
    recorder = Recorder(output_path)
    recorder.start()
    print("Recording... Ctrl-C to stop")
    try:
        while not recorder.failure:
            time.sleep(0.25)
        print(f"Capture stopped: {recorder.failure}")
    except KeyboardInterrupt:
        pass
    recorder.stop()
    path = recorder.finish()
    from secondbrain import util

    print(f"Stopped after {util.human_duration(recorder.elapsed())}")
    return path
