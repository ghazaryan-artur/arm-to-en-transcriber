from app.models import Result
from app.output import fmt_time, render


def test_fmt_time():
    assert fmt_time(72.4) == "00:01:12"


def test_render():
    r = Result(t_start=75, speaker="Me", hy="Բարև", en="Hi", latency=1.0)
    assert render(r) == "[00:01:15] Me\nHY: Բարև\nEN: Hi\n"
