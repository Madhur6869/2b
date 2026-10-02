"""macOS menubar app for the Second Brain.

Records the mic plus system audio, tags participants from the People roster, and hands
the WAV to the ingest pipeline (transcribe -> store -> index).

Fail-hard philosophy adapted for a long-running UI: recording/ingest actions
CATCH exceptions so a single bad action doesn't kill the app, but they surface
the error to the user via rumps.alert(...) and re-raise on the worker thread /
log the traceback. There are no bare `except: pass` blocks.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import subprocess
import threading
import traceback

import rumps
from PyObjCTools import AppHelper

from secondbrain import config, util
from secondbrain.capture import Recorder
from secondbrain.ingest import ingest


def _on_main(func, *args):
    """Run an AppKit/UI call on the main thread (safe from worker threads)."""
    AppHelper.callAfter(func, *args)

# Plain-text titles: emoji glyphs render as a "tofu" box on some menu-bar setups,
# making the item impossible to find. Text always renders and is unmistakable.
# The right-side menu bar is clear of the notch, so a widening REC timer is fine.
IDLE_TITLE = "2B"


class SecondBrainApp(rumps.App):
    def __init__(self):
        super().__init__(IDLE_TITLE, quit_button="Quit")

        self.recorder: Recorder | None = None
        self.started_at = None
        self.audio_path: Path | None = None
        self._timer: rumps.Timer | None = None

        # --- status line (informational; timer shown here, not in the menu bar) ---
        self.status_item = rumps.MenuItem("Idle")  # no callback -> disabled/info

        # --- primary action ---
        self.record_item = rumps.MenuItem(
            "Start Recording", callback=self.on_toggle_record
        )

        self.menu = [
            self.status_item,
            self.record_item,
            None,
            rumps.MenuItem("Open Folder", callback=self.on_open_folder),
            rumps.MenuItem("Re-index", callback=self.on_reindex),
        ]

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def on_toggle_record(self, _):
        if self.recorder is not None and self.recorder.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if self.recorder is not None and self.recorder.is_recording:
            return  # already recording — ignore
        try:
            self.audio_path = config.AUDIO_DIR / (util.stamp(util.now()) + ".wav")
            self.started_at = util.now()
            self.recorder = Recorder(self.audio_path)
            self.recorder.start()
        except Exception as exc:
            self.recorder = None
            rumps.alert("Second Brain", f"Failed to start recording:\n{exc}")
            traceback.print_exc()
            raise

        self.record_item.title = "Stop Recording"
        self.title = "REC 0:00"  # plain text, ticks in the menu bar — unmistakable
        self.status_item.title = "Recording — 0:00"
        self._timer = rumps.Timer(self._tick, 1)
        self._timer.start()

    def _tick(self, _):
        if self.recorder is None or not self.recorder.is_recording:
            return
        failure = self.recorder.failure
        if failure:
            # Capture died: save what we have and say so now, rather than keep showing
            # REC over a recording that has stopped.
            self._stop_recording()
            rumps.alert(
                "Second Brain — recording stopped",
                f"Audio capture stopped unexpectedly:\n{failure}\n\n"
                "What was captured is being saved. Start a new recording to continue.",
            )
            return
        seconds = int(self.recorder.elapsed())
        m, s = divmod(seconds, 60)
        self.title = f"REC {m}:{s:02d}"
        self.status_item.title = f"Recording — {m}:{s:02d}"

    def _stop_recording(self):
        if self.recorder is None or not self.recorder.is_recording:
            return  # not recording — ignore

        if self._timer is not None:
            self._timer.stop()
            self._timer = None

        try:
            self.recorder.stop()
        except Exception as exc:
            rumps.alert("Second Brain", f"Failed to stop recording:\n{exc}")
            traceback.print_exc()
            self.title = IDLE_TITLE
            self.status_item.title = "Idle"
            self.record_item.title = "Start Recording"
            self.recorder = None
            raise

        self.title = IDLE_TITLE
        self.status_item.title = "Processing…"
        self.record_item.title = "Start Recording"
        recorder, self.recorder = self.recorder, None
        started_at = self.started_at

        # No title/people prompt — save immediately. The timestamp id names the file and
        # the distiller sets a title from the content. Participants are added afterward
        # (the user tells us who was there and we relabel/re-distill).
        self._ingest_async(recorder, "", [], started_at)

    def _ingest_async(self, recorder, title, participants, started_at):
        def worker():
            try:
                # Mixing + echo removal take seconds per hour of audio: off the UI thread.
                audio_path = recorder.finish()
                if recorder.warnings:
                    # Warn loudly if a source dropped out (don't silently save junk).
                    _on_main(rumps.alert, "Second Brain — audio problem", "\n\n".join(recorder.warnings))
                md_path = ingest(audio_path, title, participants, started_at)
            except Exception as exc:
                # AppKit UI (alerts/notifications/menu edits) MUST run on the main
                # thread — calling them from this worker crashes with NSWindow errors
                # and hides the real failure. Marshal via AppHelper.callAfter.
                _on_main(lambda: setattr(self.status_item, "title", "Idle"))
                _on_main(rumps.alert, "Second Brain", f"Ingest failed:\n{exc}")
                traceback.print_exc()
                return
            _on_main(rumps.notification, "Second Brain", "Saved", Path(md_path).name)
            _on_main(lambda: setattr(self.status_item, "title", "Idle"))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------ #
    # Folder / index
    # ------------------------------------------------------------------ #
    def on_open_folder(self, _):
        subprocess.run(["open", str(config.ROOT)], check=True)

    def on_reindex(self, _):
        def worker():
            try:
                from secondbrain import index

                total = index.reindex_all()
            except Exception as exc:
                _on_main(rumps.alert, "Second Brain", f"Re-index failed:\n{exc}")
                traceback.print_exc()
                return
            _on_main(
                rumps.notification, "Second Brain", "Re-index complete", f"{total} chunks indexed"
            )

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    SecondBrainApp().run()
