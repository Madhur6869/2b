"""Shared configuration and paths for the Second Brain.

Every module imports from here so paths/settings live in exactly one place.
Fail-hard philosophy: missing config keys raise, they do not silently default.
"""
import os
from pathlib import Path
import yaml

# launchd starts agents with a minimal PATH that excludes Homebrew. MLX-whisper shells
# out to `ffmpeg` and the distiller to `claude`, so ensure the common bin dirs are on
# PATH for every subprocess this app spawns — otherwise transcription fails with
# FileNotFoundError under launchd while working fine from a normal shell.
for _bin in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"):
    if _bin not in os.environ.get("PATH", "").split(":"):
        os.environ["PATH"] = f"{os.environ.get('PATH', '')}:{_bin}".strip(":")

ROOT = Path(__file__).resolve().parent.parent  # ~/second-brain
CONFIG_PATH = ROOT / "config.yaml"
PEOPLE_PATH = ROOT / "people.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


_cfg = _load_yaml(CONFIG_PATH)

# --- directories (created on import so callers never worry about it) ---
AUDIO_DIR = ROOT / _cfg["audio_dir"]
CONVERSATIONS_DIR = ROOT / _cfg["conversations_dir"]
INDEX_DIR = ROOT / _cfg["index_dir"]
for _d in (AUDIO_DIR, CONVERSATIONS_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- capture ---
SAMPLE_RATE = int(_cfg["capture"]["sample_rate"])
CHANNELS = int(_cfg["capture"]["channels"])
MIC = _cfg["capture"]["mic"]
SYSTEM_AUDIO = bool(_cfg["capture"]["system_audio"])
SYSTEM_AUDIO_EXCLUDE = list(_cfg["capture"]["system_audio_exclude"] or [])
AUDIOTAP_BIN = ROOT / "audiotap" / "audiotap"  # built by audiotap/build.sh

# --- whisper ---
WHISPER_BACKEND = _cfg["whisper"].get("backend", "faster-whisper")
WHISPER_MODEL = _cfg["whisper"]["model"]
WHISPER_COMPUTE_TYPE = _cfg["whisper"]["compute_type"]
WHISPER_LANGUAGE = _cfg["whisper"]["language"]  # may be None

# --- embeddings ---
EMBED_MODEL = _cfg["embeddings"]["model"]
CHUNK_SIZE = int(_cfg["embeddings"]["chunk_size"])
CHUNK_OVERLAP = int(_cfg["embeddings"]["chunk_overlap"])

# --- mcp ---
MCP_DEFAULT_K = int(_cfg["mcp"]["default_k"])

# --- gmail ---
_gmail = _cfg.get("gmail", {})
GMAIL_ENABLED = bool(_gmail.get("enabled", False))
GMAIL_ACCOUNTS = _gmail.get("accounts", []) or []

# --- notion ---
_notion = _cfg.get("notion", {})
NOTION_ENABLED = bool(_notion.get("enabled", False))
NOTION_SECRET_FILE = _notion.get("secret_file", ".notion_token")
NOTION_PARENT_PAGE = _notion.get("parent_page", "") or ""

# --- diarize ---
_diar = _cfg.get("diarize", {})
DIARIZE_ENABLED = bool(_diar.get("enabled", False))
DIARIZE_MODEL = _diar.get("model", "pyannote/speaker-diarization-3.1")
DIARIZE_DEVICE = str(_diar.get("device", "auto")).lower()
HF_TOKEN_PATH = ROOT / ".hf_token"

# pyannote on MPS can hit ops Metal doesn't implement. This lets those individual
# ops fall back to CPU instead of raising, so we keep the GPU for everything else.
# Must be set before torch dispatches its first op — config is imported first
# everywhere, so this is the earliest reliable place.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# --- audio retention ---
# Audio expires once its conversation is transcribed, diarized and distilled.
# See secondbrain/prune.py for the full set of conditions.
_audio_ret = _cfg.get("audio_retention", {})
AUDIO_TTL_ENABLED = bool(_audio_ret.get("enabled", False))
AUDIO_TTL_DAYS = int(_audio_ret.get("ttl_days", 30))
AUDIO_TTL_REQUIRE_DIARIZED = bool(_audio_ret.get("require_diarized", True))

# --- distill ---
_CLAUDE_BIN_CFG = _cfg.get("distill", {}).get("claude_bin", "") or ""
DISTILL_POLL_SECONDS = int(_cfg.get("distill", {}).get("poll_seconds", 900))

# Topic notes (living, accumulated knowledge) live here.
TOPICS_DIR = ROOT / "topics"
TOPICS_DIR.mkdir(parents=True, exist_ok=True)

# LanceDB table name for conversation chunks
LANCE_TABLE = "chunks"


def resolve_claude_bin() -> str:
    """Absolute path to the `claude` CLI: the configured path if set, else PATH, else the
    newest nvm install. Raises if none is usable: distillation can't run without it."""
    import re
    import shutil

    if _CLAUDE_BIN_CFG:
        return _CLAUDE_BIN_CFG
    found = shutil.which("claude")
    if found:
        return found
    # launchd's PATH has no nvm. Compare versions as numbers (as text, v24.3.0 outranks
    # v24.22.1) and skip anything not executable: a failed npm postinstall leaves a text
    # stub in place of the binary, which is what stopped distillation from 2026-10-26.
    installs = sorted(
        (Path.home() / ".nvm/versions/node").glob("*/bin/claude"),
        key=lambda c: [int(n) for n in re.findall(r"\d+", c.parent.parent.name)],
        reverse=True,
    )
    for cand in installs:
        if os.access(cand, os.X_OK):
            return str(cand)
    raise RuntimeError(
        f"No usable `claude` CLI (not on PATH; nvm installs checked: {[str(c) for c in installs]}). "
        "Set distill.claude_bin in config.yaml to its absolute path (`which claude`)."
    )


def read_secret(name: str) -> str:
    """Read a secret from a gitignored file under ROOT (e.g. '.notion_token'). Raises if absent."""
    p = ROOT / name
    if not p.exists():
        raise RuntimeError(f"Missing secret file {p}. Create it (chmod 600) with the credential.")
    val = p.read_text().strip()
    if not val:
        raise RuntimeError(f"Secret file {p} is empty.")
    return val


def resolve_hf_token() -> str:
    """HuggingFace token for pyannote: env var, else the .hf_token file. Raises if absent."""
    import os
    for var in ("HUGGINGFACE_TOKEN", "HF_TOKEN"):
        if os.environ.get(var):
            return os.environ[var].strip()
    if HF_TOKEN_PATH.exists():
        tok = HF_TOKEN_PATH.read_text().strip()
        if tok:
            return tok
    raise RuntimeError(
        "No HuggingFace token found. Put it in ~/second-brain/.hf_token or set "
        "HUGGINGFACE_TOKEN. Diarization needs it to download the pyannote model."
    )


def load_people() -> list[dict]:
    """Return the roster from people.yaml (empty list if file absent/empty)."""
    if not PEOPLE_PATH.exists():
        return []
    data = _load_yaml(PEOPLE_PATH)
    return data or []
