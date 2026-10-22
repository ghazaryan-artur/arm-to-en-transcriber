"""Terminal rendering."""
from armtranscribe.models import Result


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def render(r: Result) -> str:
    return f"[{fmt_time(r.t_start)}] {r.speaker}\nHY: {r.hy}\nEN: {r.en}\n"
