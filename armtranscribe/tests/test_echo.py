import numpy as np

from app.echo import ECHO_XCORR, StreamTail, is_echo, xcorr
from app.pipeline import Pipeline
from app.vad import Segmenter
from tests.test_pipeline import FakeASR, FakeCapture, FakeMT
from tests.test_vad import SILENCE, SPEECH, fake_prob, signal

RNG = np.random.default_rng(0)


def noise(seconds, scale=1.0):
    return (scale * RNG.standard_normal(int(seconds * 16000))).astype(np.float32)


def test_stream_tail_keeps_the_last_seconds_by_stream_time():
    t = StreamTail(seconds=2.0)
    x = np.arange(48000, dtype=np.float32)
    for i in range(0, len(x), 1000):
        t.append(x[i:i + 1000])
    assert t.end_s == 3.0
    assert t.get(1.0, 1.0 + 3 / 16000).tolist() == [16000, 16001, 16002]
    assert len(t.get(0.0, 0.5)) == 0          # already overwritten
    assert len(t.get(2.9, 3.5)) == 1600       # only what has arrived


def test_xcorr_finds_a_delayed_copy():
    a = noise(1.0)
    assert xcorr(np.roll(a, 160)[1000:9000], a) > 0.99
    assert xcorr(noise(0.5), a) < 0.1
    assert xcorr(np.zeros(100, np.float32), a) == 0.0
    assert xcorr(a, a[:100]) == 0.0


def test_leaked_call_audio_is_echo_own_speech_is_not():
    system = noise(5.0)
    tail = StreamTail()
    tail.append(system)
    leak = 0.05 * np.roll(system, 80)[16000:48000] + noise(2.0, 0.005)  # 26 dB down, 5 ms late
    assert is_echo(leak, 1.0, 3.0, tail)[0]
    own_over_call = noise(2.0, 0.2) + leak                               # user talks while they talk
    echo, score = is_echo(own_over_call, 1.0, 3.0, tail)
    assert not echo and score < ECHO_XCORR


def test_pipeline_drops_mic_segments_that_echo_the_system_stream(capsys):
    speech = signal((0.5, SILENCE), (2.0, SPEECH), (2.5, SILENCE))
    system = speech * noise(len(speech) / 16000)              # "speech" on the system stream
    mic = 0.5 * system                                        # the same audio leaking into the mic
    results = []
    p = Pipeline(FakeASR(), FakeMT(), results.append, debug=True)
    p.add_source(FakeCapture(mic), Segmenter("mic", speech_prob=fake_prob))
    p.add_source(FakeCapture(system), Segmenter("system", speech_prob=fake_prob))
    p.start()
    p.stop()
    assert [r.speaker for r in results] == ["System"]
    assert "echo of the call audio" in capsys.readouterr().err
