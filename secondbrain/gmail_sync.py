"""Multi-account Gmail ingestion into the brain via IMAP + app passwords.

Each configured account is polled with its own Gmail-search query (run through IMAP
X-GM-RAW). New messages are written as brain notes (type: email), indexed, and — for
accounts with distill=true — left for the distiller to summarise. Dedup is by RFC822
Message-ID persisted in .gmail_state.json.

Fail-hard: connection/auth errors raise. A single un-parseable message is skipped with
a visible warning (never a silent drop of the whole run).
"""
import email
import imaplib
import json
import re
from email.header import decode_header, make_header
from email.utils import parsedate_to_datetime

import yaml

from secondbrain import config, util

IMAP_HOST = "imap.gmail.com"
STATE_PATH = config.ROOT / ".gmail_state.json"


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _decode(value: str) -> str:
    if not value:
        return ""
    return str(make_header(decode_header(value)))


def _plain_body(msg: email.message.Message) -> str:
    """Best-effort plaintext body: prefer text/plain, else strip tags from text/html."""
    if msg.is_multipart():
        # first text/plain part
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and "attachment" not in str(
                part.get("Content-Disposition", "")
            ):
                return part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", "replace")
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                html = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", "replace")
                return _strip_html(html)
        return ""
    payload = msg.get_payload(decode=True)
    text = payload.decode(msg.get_content_charset() or "utf-8", "replace") if payload else ""
    return _strip_html(text) if msg.get_content_type() == "text/html" else text


def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?</\1>", "", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    html = re.sub(r"&nbsp;", " ", html)
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n\s*\n\s*\n+", "\n\n", html)
    return html.strip()


def _gm_search(imap: imaplib.IMAP4_SSL, gmail_query: str) -> list[bytes]:
    """Run a Gmail-syntax query via X-GM-RAW using an IMAP literal (avoids quoting hell)."""
    imap.literal = gmail_query.encode("utf-8")
    typ, data = imap.uid("SEARCH", "X-GM-RAW")
    if typ != "OK":
        raise RuntimeError(f"IMAP search failed: {typ} {data}")
    return data[0].split() if data and data[0] else []


def _github_meta(subject: str) -> dict:
    """Extract repo + issue/PR number from a GitHub notification subject.
    GitHub subjects look like '[org/repo] Title (#123)' (optionally 'Re: ...')."""
    meta = {}
    m = re.search(r"\[([\w.-]+/[\w.-]+)\]", subject)
    if m:
        meta["repo"] = m.group(1)
    m = re.search(r"\(#(\d+)\)|#(\d+)", subject)
    if m:
        meta["issue"] = int(m.group(1) or m.group(2))
    return meta


def _write_email_note(account: str, distill: bool, hdr: dict, body: str) -> "Path":
    from pathlib import Path

    when = hdr["date"]
    title = f"[{account}] {hdr['subject'] or '(no subject)'}"
    frontmatter = {
        "date": util.human_time(when),
        "type": "email",
        "account": account,
        "from": hdr["from"],
        "subject": hdr["subject"],
        "message_id": hdr["message_id"],
        "participants": [],
        "tags": [],
        "title": title,
    }
    # Enrich GitHub notifications so they're queryable by repo/issue for issue-tracking.
    if account == "github":
        gh = _github_meta(hdr["subject"] or "")
        frontmatter.update(gh)
        if gh.get("repo"):
            frontmatter["tags"] = [gh["repo"]] + ([f"#{gh['issue']}"] if gh.get("issue") else [])
    if not distill:
        frontmatter["distilled"] = True  # index-only; skip the distiller

    md_path = (
        config.CONVERSATIONS_DIR
        / f"{util.stamp(when)}_email_{account}_{util.slugify(hdr['subject'] or 'email')}.md"
    )
    fm_yaml = yaml.safe_dump(frontmatter, sort_keys=False, default_flow_style=None, allow_unicode=True)
    md_path.write_text(f"---\n{fm_yaml}---\n{body.strip()}\n", encoding="utf-8")
    return md_path


def sync_account(acct: dict, state: dict) -> int:
    name = acct["name"]
    address = (acct.get("address") or "").strip()
    if not address:
        print(f"[gmail:{name}] no address configured — skipping")
        return 0

    password = config.read_secret(acct["secret_file"])
    query = f'{acct["query"]} newer_than:{int(acct.get("newer_than_days", 30))}d'
    distill = bool(acct.get("distill", False))
    seen = set(state.get(name, []))

    print(f"[gmail:{name}] connecting as {address}...")
    imap = imaplib.IMAP4_SSL(IMAP_HOST)
    imap.login(address, password)
    imap.select("INBOX")
    uids = _gm_search(imap, query)
    print(f"[gmail:{name}] {len(uids)} message(s) match query")

    from secondbrain.index import index_file

    new_count = 0
    for uid in uids:
        typ, msg_data = imap.uid("FETCH", uid, "(RFC822)")
        if typ != "OK" or not msg_data or not msg_data[0]:
            print(f"[gmail:{name}] WARNING: fetch failed for uid {uid!r}; skipping")
            continue
        msg = email.message_from_bytes(msg_data[0][1])
        mid = (msg.get("Message-ID") or f"uid-{uid.decode()}").strip()
        if mid in seen:
            continue
        try:
            when = parsedate_to_datetime(msg.get("Date")) if msg.get("Date") else util.now()
            if when.tzinfo is not None:
                when = when.astimezone().replace(tzinfo=None)
            hdr = {
                "from": _decode(msg.get("From", "")),
                "subject": _decode(msg.get("Subject", "")),
                "message_id": mid,
                "date": when,
            }
            body = _plain_body(msg)
        except Exception as exc:
            print(f"[gmail:{name}] WARNING: could not parse message {mid} ({exc}); skipping")
            continue

        md_path = _write_email_note(name, distill, hdr, body)
        index_file(md_path)
        seen.add(mid)
        new_count += 1
        print(f"[gmail:{name}] + {md_path.name}")

    imap.logout()
    state[name] = sorted(seen)
    return new_count


def sync_all() -> int:
    if not config.GMAIL_ENABLED:
        print("[gmail] disabled")
        return 0
    state = _load_state()
    total = 0
    for acct in config.GMAIL_ACCOUNTS:
        total += sync_account(acct, state)
        _save_state(state)  # persist after each account so a later failure can't lose progress
    print(f"[gmail] ingested {total} new message(s)")
    return total
