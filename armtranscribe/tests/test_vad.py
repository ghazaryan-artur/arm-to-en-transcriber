import numpy as np
import pytest

from app.vad import FRAME, FRAME_S, Segmenter

SPEECH, QUIET_SPEECH, MAYBE, SILENCE = 0.5, 0.1, 0.03, 0.0


def fake_prob(frame):
    """Amplitude stands in for speech: >= 0.05 is speech, 0.02-0.05 is between the thresholds."""
    a = float(np.abs(frame).max())
    return 1.0 if a >= 0.05 else 0.4 if a >= 0.02 else 0.0


def dur(seconds):
    """Duration rounded to whole frames, as signal() produces it."""
    return round(seconds / FRAME_S) * FRAME_S


def signal(*parts):
    """parts: (seconds, amplitude) pairs, each rounded to whole frames."""
    return np.concatenate([np.full(round(s / FRAME_S) * FRAME, a, np.float32) for s, a in parts])


def segment(audio, chunk=None, **kw):
    seg = Segmenter("mic", speech_prob=fake_prob, **kw)
    chunk = chunk or len(audio)
    out = []
    for i in range(0, len(audio), chunk):
        out += seg.feed(audio[i:i + chunk])
    return out + seg.flush()


def test_one_utterance_with_pre_roll_and_pad():
    (s,) = segment(signal((1.0, SILENCE), (1.0, SPEECH), (1.0, SILENCE)))
    assert s.source == "mic"
    assert s.t_start == pytest.approx(1.0 - 0.25, abs=FRAME_S)
    assert s.t_end == pytest.approx(2.0 + 0.2, abs=FRAME_S)
    assert len(s.audio) == pytest.approx((s.t_end - s.t_start) * 16000, abs=1)
    assert s.speech_end == pytest.approx(2.0, abs=FRAME_S)  # latency counts from here, before the pad
    assert s.phrase_end == s.speech_end


def test_segment_ends_only_after_min_silence():
    seg = Segmenter("mic", speech_prob=fake_prob)  # default min_silence_ms=1000
    assert seg.feed(signal((1.0, SPEECH), (0.9, SILENCE))) == []
    assert len(seg.feed(signal((0.2, SILENCE)))) == 1


def test_short_pause_does_not_split():
    assert len(segment(signal((1.0, SPEECH), (0.8, SILENCE), (1.0, SPEECH), (1.5, SILENCE)))) == 1
    assert len(segment(signal((1.0, SPEECH), (1.2, SILENCE), (1.0, SPEECH), (1.5, SILENCE)))) == 2


def test_short_blip_is_dropped():
    assert segment(signal((1.0, SILENCE), (0.2, SPEECH), (1.0, SILENCE))) == []
    assert len(segment(signal((1.0, SILENCE), (0.5, SPEECH), (1.0, SILENCE)))) == 1


def test_between_thresholds_neither_starts_nor_ends():
    assert segment(signal((2.0, MAYBE))) == []
    (s,) = segment(signal((1.0, SPEECH), (2.0, MAYBE), (1.0, SPEECH), (1.0, SILENCE)))
    assert s.t_end == pytest.approx(dur(1.0) + dur(2.0) + dur(1.0) + dur(0.2))


def test_force_cut_in_pause_and_carry_over():
    # a 0.3 s pause (too short to end the segment) 1.5 s before the 12 s limit: cut in its middle
    audio = signal((10.5, SPEECH), (0.3, SILENCE), (4.0, SPEECH), (1.5, SILENCE))
    a, b = segment(audio)
    assert a.t_end == pytest.approx(10.5 + 0.15, abs=FRAME_S)
    assert b.t_start == a.t_end                                        # no gap, no overlap
    assert b.t_end == pytest.approx(10.5 + 0.3 + 4.0 + 0.2, abs=FRAME_S)


def test_force_cut_prefers_longest_pause_in_window():
    audio = signal((9.3, SPEECH), (0.4, SILENCE), (1.5, SPEECH), (0.1, SILENCE), (3.0, SPEECH), (1.5, SILENCE))
    a, _ = segment(audio)
    assert a.t_end == pytest.approx(9.3 + 0.2, abs=FRAME_S)


def test_force_cut_without_pause_takes_least_speechlike_frame():
    # no frame below neg_threshold: cut after the frame between the thresholds, not mid-speech
    audio = signal((10.0, SPEECH), (FRAME_S, MAYBE), (4.0, SPEECH), (1.5, SILENCE))
    a, b = segment(audio)
    assert a.t_end == pytest.approx(dur(10.0) + FRAME_S, abs=FRAME_S / 2)
    assert b.t_start == a.t_end


def test_pause_before_the_cut_window_is_ignored():
    # the only pause is 4 s before the limit, outside the 3 s window: fall back inside the window
    audio = signal((7.5, SPEECH), (0.3, SILENCE), (6.0, SPEECH), (1.5, SILENCE))
    a, _ = segment(audio)
    assert a.t_end > 12.0 - 3.0 - FRAME_S


def test_long_monologue_never_exceeds_max():
    segs = segment(signal((40.0, SPEECH), (1.5, SILENCE)))
    assert all(s.t_end - s.t_start <= 12.0 + FRAME_S for s in segs)
    assert segs[0].t_start == 0 and segs[-1].t_end == pytest.approx(40.2, abs=FRAME_S)
    assert all(x.t_end == y.t_start for x, y in zip(segs, segs[1:]))


def test_flush_emits_segment_in_progress():
    seg = Segmenter("mic", speech_prob=fake_prob)
    assert seg.feed(signal((1.0, SPEECH))) == []
    (s,) = seg.flush()
    assert s.t_end == pytest.approx(1.0, abs=FRAME_S)
    assert seg.flush() == []


def test_chunk_size_does_not_change_result():
    audio = signal((0.7, SILENCE), (2.3, SPEECH), (0.9, SILENCE), (9.1, SPEECH), (0.4, SILENCE))
    whole = segment(audio)
    for chunk in (100, 512, 1000, 4096):
        parts = segment(audio, chunk=chunk)
        assert [(s.t_start, s.t_end) for s in parts] == [(s.t_start, s.t_end) for s in whole]
        assert all(np.array_equal(p.audio, w.audio) for p, w in zip(parts, whole))
