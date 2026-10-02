#!/usr/bin/env bash
# One-time setup for the Second Brain. Creates a local venv and installs all deps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "==> Creating virtualenv at $ROOT/.venv"
python3 -m venv .venv
source .venv/bin/activate

echo "==> Upgrading pip"
python -m pip install --upgrade pip wheel

echo "==> Installing requirements (this pulls Whisper + embedding + audio deps)"
pip install -r requirements.txt

echo "==> Pre-downloading the faster-whisper model so first recording isn't slow"
python - <<'PY'
from secondbrain import config
from faster_whisper import WhisperModel
print(f"downloading whisper model: {config.WHISPER_MODEL} ({config.WHISPER_COMPUTE_TYPE})")
WhisperModel(config.WHISPER_MODEL, compute_type=config.WHISPER_COMPUTE_TYPE, device="cpu")
print("whisper model ready")
PY

echo "==> Pre-downloading the embedding model"
python - <<'PY'
from secondbrain import config
from fastembed import TextEmbedding
print(f"downloading embedding model: {config.EMBED_MODEL}")
emb = TextEmbedding(model_name=config.EMBED_MODEL)
list(emb.embed(["warmup"]))
print("embedding model ready")
PY

echo "==> Building the audiotap helper (records the mic + system audio, e.g. calls in AirPods)"
./audiotap/build.sh

echo ""
echo "Setup complete. Next:"
echo "  1. Edit people.yaml with your colleagues."
echo "  2. Grant your terminal Microphone access: System Settings > Privacy & Security > Microphone."
echo "     The first recording also asks to record system audio — allow it (Screen & System Audio Recording)."
echo "  3. Try a recording:   ./.venv/bin/python bin/record       (Ctrl-C to stop)"
echo "  4. Launch the menubar: ./.venv/bin/python bin/menubar"
echo "  5. Register the MCP server with Claude (see README.md)."
