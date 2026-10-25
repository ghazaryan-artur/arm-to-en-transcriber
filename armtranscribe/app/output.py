"""Terminal rendering."""
import shutil
import sys
import threading

from app.models import Result

_lock = threading.Lock()
_writes = 0  # output written so far by Terminal.show / notice; a redraw is safe only if nothing came in between


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def render(r: Result) -> str:
    return f"[{fmt_time(r.t_start)}] {r.speaker}\nHY: {r.hy}\nEN: {r.en}\n"


def render_timings(r: Result) -> str:
    spk = f", spk {r.spk_s:.2f} s" if r.spk_s else ""
    merged = f", {r.merged} segments re-translated" if r.merged > 1 else ""
    info = f"\n  [speaker: {r.spk_info}]" if r.spk_info else ""
    fast = " (fast model)" if r.mt_fast else ""
    return (f"  [{r.audio_s:.1f} s audio: asr {r.asr_s:.2f} s{spk}, mt {r.mt_s:.2f} s{fast}, "
            f"latency {r.latency:.2f} s{merged}]{info}\n")


def notice(msg: str) -> None:
    """Warnings and errors (stderr). Use this instead of print so a later redraw doesn't erase them."""
    global _writes
    with _lock:
        _writes += 1
        print(msg, file=sys.stderr, flush=True)


def rows(text: str, columns: int) -> int:
    """Terminal rows `text` occupies when printed, counting wrapped lines."""
    return sum(max(1, -(-len(line) // columns)) for line in text.split("\n"))


class Terminal:
    """Prints result blocks. A continuation of the line printed last replaces it in place (ANSI
    cursor-up + clear) when stdout is a terminal and nothing was printed since; otherwise the
    continuation is printed as a new block marked ↻ that supersedes the earlier one."""

    def __init__(self, stream=None, redraw: bool | None = None):
        self.stream = stream or sys.stdout
        self.redraw = self.stream.isatty() if redraw is None else redraw
        self._last = None  # (line id, rows printed, _writes after printing)

    def show(self, r: Result, text: str) -> None:
        global _writes
        text = text.rstrip("\n") + "\n\n"  # block + blank separator line
        with _lock:
            columns, height = shutil.get_terminal_size()
            if r.merged > 1:
                last = self._last
                if self.redraw and last and last[0] == r.line and last[2] == _writes and last[1] < height:
                    self.stream.write(f"\x1b[{last[1]}F\x1b[J")  # up to the block's first row, clear below
                else:
                    head, rest = text.split("\n", 1)
                    text = f"{head} ↻\n{rest}"
            self.stream.write(text)
            self.stream.flush()
            _writes += 1
            self._last = (r.line, rows(text[:-1], columns), _writes)


def _pct(values: list[float], q: float) -> float:
    return sorted(values)[min(len(values) - 1, int(q * len(values)))]


def percentile_summary(latencies: list[float]) -> str:
    if not latencies:
        return "no segments"
    return f"{len(latencies)} segments, latency p50 {_pct(latencies, 0.5):.2f} s, p95 {_pct(latencies, 0.95):.2f} s"


def session_summary(results: list[Result]) -> str:
    """--debug summary on exit: latency and per-stage p50 / p95, fallback-MT use."""
    if not results:
        return "no segments"
    lines = [percentile_summary([r.latency for r in results])]
    stages = {"asr": [r.asr_s for r in results], "mt": [r.mt_s for r in results]}
    if spk := [r.spk_s for r in results if r.spk_s]:
        stages["speaker"] = spk
    lines.append("  " + ", ".join(f"{k} p50 {_pct(v, 0.5):.2f} / p95 {_pct(v, 0.95):.2f} s" for k, v in stages.items()))
    if fast := sum(r.mt_fast for r in results):
        lines.append(f"  {fast} of {len(results)} lines translated by the fast model (the app was behind)")
    return "\n".join(lines)
