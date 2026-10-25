"""Export the current Claude Cowork or Claude Code chat to Markdown: the user's messages and Claude's replies only.

Claude Code: ask Claude in the chat to export to run it (it finds the session by $CLAUDE_CODE_SESSION_ID):

    python3 tools/export_session.py        -> chats/claude - <date>_<title>.md

Cowork: run it inside the Cowork chat to export, in the cloud container (the transcript lives there):

    1. connect ~/armenian-transcriber, stage this file (device_stage_files) ->
       /mnt/user-data/uploads/armenian-transcriber/tools/export_session.py
    2. python3 /mnt/user-data/uploads/armenian-transcriber/tools/export_session.py
       (writes /mnt/user-data/outputs/cowork - <date>_<title>.md)
    3. device_commit_files the result -> ~/armenian-transcriber/chats/<same name>.md

Options: --out PATH, --title "Title", --session path/to/session.jsonl, --final-only (Claude Code).

Kept, in the original order: the user's messages (incl. ones sent while Claude was working), Claude's reply
at the end of each turn, messages Claude sent mid-turn with SendUserMessage, and AskUserQuestion questions
with the user's answers; in Claude Code also Claude's messages between tool calls, which it shows (unless
--final-only). Left out: thinking, tool calls (commands, file reads and edits) and their results, Claude's
notes between tool calls in Cowork (the chat collapses them), and context injected by the harness or the IDE
(<system-reminder>, <ide_opened_file> / <ide_selection>, memory, tool / skill listings, "Continue from where
you left off.", interruption markers, background memory passes).
"""
import argparse, glob, json, os, re, sys

# injected context: harness reminders, and in Claude Code the IDE's opened file / selection
REMINDER = re.compile(r"<(system-reminder|ide_opened_file|ide_selection)>.*?</\1>", re.S)
HARNESS_PROMPTS = {"Continue from where you left off."}
TRANSLIT = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                    "a b v g d e e zh z i y k l m n o p r s t u f kh ts ch sh sch _ y _ e yu ya".split()))
TRANSLIT.update({"ъ": "", "ь": ""})


def clean(t):
    return REMINDER.sub("", t or "").strip()


def slug(title):
    s = "".join(TRANSLIT.get(ch, ch) for ch in title.lower())
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:60] or "chat"


def find_transcript():
    sid = os.environ.get("CLAUDE_CODE_SESSION_ID")
    root = os.path.expanduser("~/.claude/projects")
    if sid:
        hits = glob.glob(os.path.join(root, "*", sid + ".jsonl"))
        if hits:
            return hits[0]
    hits = sorted(glob.glob(os.path.join(root, "*", "*.jsonl")), key=os.path.getmtime)
    if not hits:
        sys.exit("No session transcript under ~/.claude/projects; pass --session")
    return hits[-1]


def user_text(content):
    """Text of a real user prompt, or None for harness records (tool results, markers, injected context)."""
    if isinstance(content, str):
        parts = [content]
    else:
        if any(b.get("type") == "tool_result" for b in content):
            return None
        parts = [b.get("text", "") if b.get("type") == "text" else f"_({b.get('type')} attached)_" for b in content]
    text = clean("\n\n".join(parts))
    if "<command-name>" in text:  # slash command: show it as typed
        name = re.search(r"<command-name>(.*?)</command-name>", text, re.S).group(1).strip()
        args = re.search(r"<command-args>(.*?)</command-args>", text, re.S)
        text = f"{name} {args.group(1).strip() if args else ''}".strip()
    if text.startswith(("<local-command-", "[Request interrupted by user")) or text in HARNESS_PROMPTS:
        return ""
    return text


def ask_block(inp):
    out = []
    for q in inp.get("questions", []):
        out += [f"**{q.get('question', '').strip()}**", ""]
        out += [f"- {o.get('label', '')}" + (f" — {o['description']}" if o.get("description") else "")
                for o in q.get("options", [])]
        out.append("")
    return "\n".join(out).strip()


def ask_answers(questions, record, block):
    res = record.get("toolUseResult")
    answers = res.get("answers") if isinstance(res, dict) else None
    raw = block.get("content")
    raw = raw if isinstance(raw, str) else "\n".join(b.get("text", "") for b in raw or [])
    out = []
    for q in questions:
        qt = q.get("question", "")
        a = (answers or {}).get(qt)
        if a is None:  # fall back to the text result: "...<question>"="<answer>"...
            m = re.search(re.escape(f'"{qt}"="') + r'(.*?)"(?=[.,]\s|\.?$|,\s*")', raw, re.S)
            a = m.group(1) if m else None
        if a is not None:
            out.append(f"{q.get('header') or qt}: {a}" if len(questions) > 1 else str(a))
    return "\n".join(out) if out else clean(raw)


def export(src, notes=False):
    """notes=True also keeps Claude's text between tool calls (shown in Claude Code, collapsed in Cowork)."""
    records, seen = [], set()
    for line in open(src, encoding="utf-8"):
        d = json.loads(line)
        if d.get("uuid") and d["uuid"] in seen:
            continue
        seen.add(d.get("uuid"))
        records.append(d)

    title = next((d["aiTitle"] for d in reversed(records) if d.get("type") == "ai-title" and d.get("aiTitle")), None)
    start = next((d["timestamp"][:10] for d in records if d.get("timestamp")), "")

    # events: ("prompt", text, ts) starts a turn; ("say", text, ts) is shown as is;
    # ("text", text, ts) is Claude's text, shown only if no tool call follows it within the turn; ("tool",)
    events, asks, skipping = [], {}, False
    for d in records:
        t, ts = d.get("type"), (d.get("timestamp") or "")[:16].replace("T", " ")
        if d.get("isSidechain") or d.get("isMeta") or d.get("isCompactSummary"):
            continue
        if t == "attachment":
            a = d.get("attachment") or {}
            text = user_text(a["prompt"]) if a.get("type") == "queued_command" and a.get("prompt") else None
            if text:  # sent while Claude was working; the prompt is a string or a list of blocks
                events.append(("say", "User", text, ts))
            continue
        m = d.get("message")
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if t == "user":
            kind = (d.get("origin") or {}).get("kind", "human")
            text = user_text(c)
            if text is None:  # tool results: only AskUserQuestion answers are shown
                for b in c:
                    if b.get("type") == "tool_result" and b.get("tool_use_id") in asks and not skipping:
                        events.append(("say", "User", ask_answers(asks[b["tool_use_id"]], d, b), ts))
                continue
            if text.startswith("[BACKGROUND MEMORY PASS"):
                skipping = True
                events.append(("prompt", "", ts))
                continue
            skipping = False
            events.append(("prompt", text if kind == "human" else "", ts))
        elif t == "assistant" and not skipping and isinstance(c, list):
            for b in c:
                if b.get("type") == "text" and b.get("text", "").strip():
                    events.append(("text", "Claude", b["text"].strip(), ts))
                elif b.get("type") == "tool_use":
                    events.append(("tool",))
                    inp = b.get("input") or {}
                    if b.get("name") == "SendUserMessage" and inp.get("message", "").strip():
                        events.append(("say", "Claude", inp["message"].strip(), ts))
                    elif b.get("name") == "AskUserQuestion":
                        asks[b.get("id")] = inp.get("questions", [])
                        events.append(("say", "Claude", ask_block(inp), ts))

    body, tool_after = [], False
    for e in reversed(events):  # walk back so we know whether a tool call follows a text block in its turn
        if e[0] == "prompt":
            tool_after = False
            if e[1]:
                body.append(("User", e[1], e[2]))
        elif e[0] == "tool":
            tool_after = True
        elif e[0] == "say" or (e[0] == "text" and (notes or not tool_after)):
            body.append(e[1:])
    body.reverse()

    md, prev = [], None
    for who, text, ts in body:
        if who == prev == "Claude":  # several shown blocks in a row form one reply
            md += [text, ""]
        else:
            md += [f"## {who} · {ts} UTC", "", text, ""]
        prev = who
    return title, start, md


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", help="transcript .jsonl (default: this chat's)")
    ap.add_argument("--out", help="output .md (default: 'chats/claude - <date>_<title>.md', "
                                  "in Cowork '/mnt/user-data/outputs/cowork - <date>_<title>.md')")
    ap.add_argument("--title", help="heading of the export (default: the chat's title)")
    ap.add_argument("--final-only", action="store_true",
                    help="Claude Code: drop Claude's messages between tool calls (always dropped in Cowork)")
    args = ap.parse_args()

    cowork = os.path.isdir("/mnt/user-data")  # the Cowork container; otherwise a local Claude Code session
    notes = not (cowork or args.final_only)
    src = args.session or find_transcript()
    auto_title, start, md = export(src, notes=notes)
    app = "Claude Cowork" if cowork else "Claude Code"
    title = args.title or auto_title or f"{app} chat"
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out or os.path.join("/mnt/user-data/outputs" if cowork else os.path.join(repo, "chats"),
                                   f"{'cowork' if cowork else 'claude'} - {start}_{slug(title)}.md")
    left_out = "tool calls (commands, file reads and edits) with their results" + (
        " are left out; Claude's messages between tool calls are kept, as Claude Code shows them." if notes else
        " and notes between tool calls are left out.")
    head = [f"# {title}", "",
            f"Exported from the {app} chat transcript by `tools/export_session.py`: the user's messages and "
            f"Claude's replies as shown in the chat. Claude's thinking, {left_out}", ""]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(head + md))
    n_user = sum(1 for l in md if l.startswith("## User"))
    n_claude = sum(1 for l in md if l.startswith("## Claude"))
    print(f"{out}\n{src}\n{n_user} user messages, {n_claude} Claude replies")


if __name__ == "__main__":
    main()
