"""Terminal rendering."""
from armtranscribe.models import Result


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def render(r: Result) -> str:
    return f"[{fmt_time(r.t_start)}] {r.speaker}\nHY: {r.hy}\nEN: {r.en}\n"


def render_timings(r: Result) -> str:
    return f"  [{r.audio_s:.1f} s audio: asr {r.asr_s:.2f} s, mt {r.mt_s:.2f} s, latency {r.latency:.2f} s]\n"


def percentile_summary(latencies: list[float]) -> str:
    if not latencies:
        return "no segments"
    p50, p95 = (sorted(latencies)[min(len(latencies) - 1, int(q * len(latencies)))] for q in (0.5, 0.95))
    return f"{len(latencies)} segments, latency p50 {p50:.2f} s, p95 {p95:.2f} s"
