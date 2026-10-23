import time

from armtranscribe.pipeline import Pipeline
from armtranscribe.vad import Segmenter
from tests.test_vad import SILENCE, SPEECH, fake_prob, signal


class FakeCapture:
    def __init__(self, audio, block=1600):
        self.blocks = [audio[i:i + block] for i in range(0, len(audio), block)]
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def read(self, timeout=0.1):
        if self.blocks:
            return self.blocks.pop(0)
        time.sleep(timeout)
        return None


class FakeASR:
    """Numbers segments in arrival order; raises on the calls listed in `fail_on`."""

    def __init__(self, fail_on=()):
        self.fail_on, self.calls = fail_on, 0

    def transcribe(self, audio):
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError("boom")
        return f"hy{self.calls}"


class FakeMT:
    def translate(self, hy):
        return hy.replace("hy", "en")


def run(audio, asr):
    results = []
    capture = FakeCapture(audio)
    p = Pipeline(asr, FakeMT(), results.append)
    p.add_source(capture, Segmenter("mic", speech_prob=fake_prob))
    p.start()
    p.stop()
    return capture, results


THREE_UTTERANCES = signal((0.5, SILENCE), (1.0, SPEECH), (1.0, SILENCE), (2.0, SPEECH), (1.0, SILENCE),
                          (1.5, SPEECH))  # the last one is still in progress when capture stops


def test_segments_flow_through_in_order_and_last_is_flushed():
    capture, results = run(THREE_UTTERANCES, FakeASR())
    assert capture.started and capture.stopped
    assert [(r.speaker, r.hy, r.en) for r in results] == [("Me", "hy1", "en1"), ("Me", "hy2", "en2"),
                                                          ("Me", "hy3", "en3")]
    assert [r.t_start for r in results] == sorted(r.t_start for r in results)
    assert all(r.audio_s > 0 and r.asr_s >= 0 and r.mt_s >= 0 for r in results)


def test_asr_error_skips_only_that_segment(capsys):
    _, results = run(THREE_UTTERANCES, FakeASR(fail_on={2}))
    assert [r.hy for r in results] == ["hy1", "hy3"]
    assert "ASR error" in capsys.readouterr().err


def test_empty_transcript_is_not_emitted():
    class Silent(FakeASR):
        def transcribe(self, audio):
            return ""

    assert run(THREE_UTTERANCES, Silent())[1] == []

