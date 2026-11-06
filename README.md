# Second Brain

A private pipeline that records your desk conversations, transcribes them on-device, stores
them as searchable markdown, and makes them available to Claude Code as context. Audio stays
on your machine. Transcripts go to Claude only for distillation (see Privacy).

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
    capture.py          # mic + system audio -> WAV (via audiotap)
    echo.py             # removes the speaker echo of system audio from the mic
    transcribe.py       # WAV -> text (mlx-whisper on the Apple GPU, local)
    diarize.py          # who-said-what (pyannote, local)
    store.py            # text -> markdown conversation file
    index.py            # markdown -> embeddings in LanceDB
    ingest.py           # orchestrates transcribe -> diarize -> store -> index
    distill.py          # transcript -> summary + topic notes, via `claude -p`
    prune.py            # expires audio once it's transcribed, diarized and distilled
    search.py           # result formatting
  bin/
    record              # CLI: record until Ctrl-C, then transcribe+store+index
    ingest              # CLI: (re)ingest an existing WAV
    rediarize           # CLI: recover speaker labels on recordings saved without them
    reindex             # CLI: rebuild the search index
    search              # CLI: semantic search
    menubar             # launch the macOS menubar app
  menubar/app.py        # rumps menubar app (start/stop); runs ingest as a child process
  audiotap/             # Swift helper: mic + system audio, follows device changes (build.sh)
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
5. **System audio** — 2B also records what the Mac plays (the other side of a call in
   AirPods), via `audiotap/` (built by `setup.sh`; rebuild with `audiotap/build.sh`). The
   first recording asks to record system audio: allow it, or enable Python later under
   System Settings → Privacy & Security → Screen & System Audio Recording → *System Audio
   Recording Only*. Without it that side is silent and 2B warns after the recording.
   With AirPods as the input, 2B records from the built-in mic instead (`capture.mic`).
   Devices are followed mid-recording (AirPods in/out, a mic unplugged), and on
   speakers the call's echo is removed from the mic so it isn't recorded twice.

> Secrets and private data never get committed — see `.gitignore`. Only the system
> code and `people.example.yaml` live in the repo.

## Daily use

**Menubar (primary):**
```bash
./.venv/bin/python bin/menubar
```
Click **2B** → **Start Recording** (the title becomes a `REC` timer) → **Stop Recording** → it
transcribes, diarizes and indexes in the background. The distill job then titles and summarizes
it within 15 minutes.

**CLI (fallback):**
```bash
./.venv/bin/python bin/record          # records until Ctrl-C, then prompts title/participants
./.venv/bin/python bin/search "planner latency"
./.venv/bin/python bin/ingest audio/2026-07-16_1430.wav --title "Some chat" --participants "Alex,Priya"
```

## Wire it into Claude Code

Run ONE shared MCP server and point every Claude session at it. Don't register a stdio
server: that starts one copy per session (~160 MB each, because fastembed + lancedb load).
With 51 sessions open on 2026-10-27, that was 8 GB.

```bash
cp com.terran.secondbrain.mcp.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.terran.secondbrain.mcp.plist
claude mcp add --transport http --scope user second-brain http://127.0.0.1:8765/mcp
```

The server runs `mcp_server.py --http`, serves 127.0.0.1 only, and logs to `mcp.log`. After
pulling code changes, restart it with `launchctl kickstart -k gui/$(id -u)/com.terran.secondbrain.mcp`.

Claude then has these tools: `search_conversations`, `get_conversation`,
`list_recent_conversations`, `who_is`.

## Configuration

Edit `config.yaml`:
- `whisper.model` — `tiny`/`base`/`small`/`medium`/`large-v3`/`large-v3-turbo` (accuracy vs speed). Default `large-v3-turbo`.
- `embeddings.model` — fastembed model id. Default `BAAI/bge-small-en-v1.5`.
- `embeddings.chunk_size` / `chunk_overlap` — retrieval granularity.

## Memory

The menu bar app runs for weeks and holds the only copy of a recording in progress, so it stays
small (~50 MB idle, plus ~800 MB per hour while recording) and never loads a model. Ingest runs as
a child process (`bin/ingest`), peaks at several GB, and gives all of it back when it exits. Only
one ingest runs at a time. To check a process, use `footprint -p <pid>` or
`top -stats pid,mem,cmprs`: `ps` RSS hides GPU and compressed memory.

## When something breaks

Failures reach you as an alert (ingest, no speaker labels) or a notification at most once a day
(distill). Logs: `menubar.log`, `distill.log`, `prune.log`.

- **Recordings saved without speaker labels:** fix the cause, then `bin/rediarize` lists them and
  `--apply` redoes them. Their audio is kept until then.
- **Distill failing:** `bin/distill` from a shell shows the error. Under launchd, `claude` is found
  by searching nvm; set `distill.claude_bin` in `config.yaml` to pin it.

## Deferred / future

- **Voice ID** (`speechbrain`) — auto-label turns with real names from enrolled voiceprints.

## Privacy

Audio, transcription, diarization and search all run locally. Distillation sends each transcript
to Claude through the `claude` CLI on your subscription; unload the
`com.terran.secondbrain.distill` agent if you don't want that. You're in a one-party-consent state,
but a courtesy heads-up to colleagues is polite.
