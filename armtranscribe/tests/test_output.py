from app.models import Result
import io

from app.output import Terminal, fmt_time, notice, render, rows


def test_fmt_time():
    assert fmt_time(72.4) == "00:01:12"


def test_render():
    r = Result(t_start=75, speaker="Me", hy="Բարև", en="Hi", latency=1.0)
    assert render(r) == "[00:01:15] Me\nHY: Բարև\nEN: Hi\n"


def test_rows_counts_wrapped_lines():
    assert rows("ab\ncd\n", 80) == 3
    assert rows("a" * 81 + "\n", 80) == 3
    assert rows("a" * 80, 80) == 1


def block(line, merged, hy):
    r = Result(t_start=0, speaker="Me", hy=hy, en=hy.upper(), latency=0, line=line, merged=merged)
    return r, render(r)


def test_terminal_replaces_its_last_block_in_place():
    out = io.StringIO()
    t = Terminal(out, redraw=True)
    t.show(*block(1, 1, "a"))
    t.show(*block(1, 2, "a b"))
    first = "[00:00:00] Me\nHY: a\nEN: A\n\n"
    assert out.getvalue() == first + "\x1b[4F\x1b[J" + "[00:00:00] Me\nHY: a b\nEN: A B\n\n"


def test_terminal_marks_continuation_when_something_was_printed_in_between(capsys):
    out = io.StringIO()
    t = Terminal(out, redraw=True)
    t.show(*block(1, 1, "a"))
    t.show(*block(2, 1, "x"))      # another source spoke in between
    t.show(*block(1, 2, "a b"))
    assert "\x1b[" not in out.getvalue()
    assert out.getvalue().endswith("[00:00:00] Me ↻\nHY: a b\nEN: A B\n\n")

    out2 = io.StringIO()
    t2 = Terminal(out2, redraw=True)
    t2.show(*block(1, 1, "a"))
    notice("warning")              # stderr output also blocks the redraw
    t2.show(*block(1, 2, "a b"))
    assert "\x1b[" not in out2.getvalue() and "Me ↻" in out2.getvalue()
