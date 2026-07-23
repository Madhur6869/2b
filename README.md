# Second Brain

A fully local, private pipeline that records your desk conversations, transcribes them
on-device, stores them as searchable markdown, and makes them available to Claude Code as
context. Nothing leaves your machine; no paid services.

```
🎙️ capture  →  📝 transcribe (local Whisper)  →  🗂️ markdown  →  🔎 embeddings/RAG  →  🤖 Claude (via MCP)
```

## Layout

```
second-brain/
  config.yaml           # all settings (model, chunk size, dirs)
  people.example.yaml   # roster template (copy to people.yaml; people.yaml is gitignored)
  requirements.txt
  setup.sh              # one-time install
  secondbrain/          # the package
    config.py           # shared paths/settings
    util.py             # slugify, timestamps
    capture.py          # mic -> WAV (sounddevice)
    transcribe.py       # WAV -> text (faster-whisper, local)
    store.py            # text -> markdown conversation file
    index.py            # markdown -> embeddings in LanceDB
    ingest.py           # orchestrates transcribe -> store -> index
    search.py           # result formatting
  bin/
    record              # CLI: record until Ctrl-C, then transcribe+store+index
    ingest              # CLI: (re)ingest an existing WAV
    search              # CLI: semantic search
    menubar             # launch the macOS menubar app
  menubar/app.py        # rumps menubar app (start/stop, People picker)
  mcp_server.py         # MCP server exposing search to Claude
  conversations/        # one .md per conversation  (your brain)
  audio/                # raw WAV recordings
  index/                # LanceDB vector store
```

## Setup (one time)

```bash
git clone git@github.com:Madhur6869/2b.git second-brain
cd second-brain
./setup.sh
```

`setup.sh` creates `.venv`, installs everything, and pre-downloads the Whisper +
embedding models. The private data dirs (`conversations/`, `audio/`, `index/`,
`topics/`) are gitignored and created automatically on first run.

Then:

1. **Roster** — copy the template and fill in the people you talk to:
   ```bash
   cp people.example.yaml people.yaml   # people.yaml is gitignored (real names)
   ```
2. **HuggingFace token** (for speaker diarization) — put a token in `.hf_token`, and
   accept the model terms at
   <https://huggingface.co/pyannote/speaker-diarization-community-1>:
   ```bash
   echo "hf_xxxxxxxx" > .hf_token         # gitignored
   ```
   To skip diarization instead, set `diarize.enabled: false` in `config.yaml`.
3. **Optional integrations** — only if you enable them in `config.yaml`: Gmail sync needs
   an app password per account in `.gmail_work_app_password` / `.gmail_github_app_password`,
   and Notion sync needs `.notion_token`. All `.*token` / `*app_password*` files are
   gitignored.
4. **Microphone access** — grant your terminal (and/or Python)
   System Settings → Privacy & Security → Microphone.

> Secrets and private data never get committed — see `.gitignore`. Only the system
> code and `people.example.yaml` live in the repo.

## Daily use

**Menubar (primary):**
```bash
./.venv/bin/python bin/menubar
```
Click 🧠 → **Start Recording** (icon turns 🔴 with a timer) → tick who's in the room under
**People** → **Stop Recording** → type a title → it transcribes + indexes in the background.

**CLI (fallback):**
```bash
./.venv/bin/python bin/record          # records until Ctrl-C, then prompts title/participants
./.venv/bin/python bin/search "planner latency"
./.venv/bin/python bin/ingest audio/2026-07-16_1430.wav --title "Some chat" --participants "Alex,Priya"
```

## Wire it into Claude Code

Register the MCP server so any Claude session can search your brain. Add to your Claude Code
MCP config (`~/.claude.json` under `mcpServers`, or a project `.mcp.json`):

```json
{
  "mcpServers": {
    "second-brain": {
      "command": "/Users/madhur/second-brain/.venv/bin/python",
      "args": ["/Users/madhur/second-brain/mcp_server.py"]
    }
  }
}
```

Claude then has these tools: `search_conversations`, `get_conversation`,
`list_recent_conversations`, `who_is`.

## Configuration

Edit `config.yaml`:
- `whisper.model` — `tiny`/`base`/`small`/`medium`/`large-v3` (accuracy vs speed). Default `medium`.
- `embeddings.model` — fastembed model id. Default `BAAI/bge-small-en-v1.5`.
- `embeddings.chunk_size` / `chunk_overlap` — retrieval granularity.

## Deferred / future (seams already in place)

- **Speaker diarization** (`pyannote.audio`) — split transcript into Speaker A/B/C turns.
- **Voice ID** (`speechbrain`) — auto-label turns with real names from enrolled voiceprints.
- **Auto-summaries / action items** — currently raw transcripts; Claude summarizes on demand.

## Privacy

Everything runs locally. Audio and transcripts never leave your machine. You're in a
one-party-consent state, but a courtesy heads-up to colleagues is polite.
