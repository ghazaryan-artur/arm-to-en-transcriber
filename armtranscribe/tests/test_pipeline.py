import time

import soundfile as sf

from app.audio import Recorder
from app.pipeline import Pipeline
from app.vad import Segmenter
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


def run(audio, asr, recorder=None):
    results = []
    capture = FakeCapture(audio)
    p = Pipeline(asr, FakeMT(), results.append, recorder)
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


def test_recorder_saves_stream_segments_and_manifest(tmp_path):
    rec = Recorder(str(tmp_path))
    _, results = run(THREE_UTTERANCES, FakeASR(), rec)
    stream, _ = sf.read(rec.dir / "mic.wav")
    assert len(stream) == len(THREE_UTTERANCES)
    rows = [line.split("\t") for line in (rec.dir / "segments.tsv").read_text().splitlines()]
    assert rows == [["0001_mic.wav", "hy1", "en1"], ["0002_mic.wav", "hy2", "en2"], ["0003_mic.wav", "hy3", "en3"]]
    for (name, _, _), r in zip(rows, results):
        assert len(sf.read(rec.dir / name)[0]) == round(r.audio_s * 16000)


def test_two_sources_are_labelled_and_both_flushed():
    results = []
    mic, system = FakeCapture(THREE_UTTERANCES), FakeCapture(THREE_UTTERANCES[: 16000 * 3])
    p = Pipeline(FakeASR(), FakeMT(), results.append)
    p.add_source(mic, Segmenter("mic", speech_prob=fake_prob))
    p.add_source(system, Segmenter("system", speech_prob=fake_prob))
    p.start()
    p.stop()
    assert mic.stopped and system.stopped
    assert sorted(r.speaker for r in results) == ["Me", "Me", "Me", "System", "System"]
    assert sorted(r.hy for r in results) == [f"hy{i}" for i in range(1, 6)]


def test_system_segments_get_speaker_labels_mic_stays_me():
    class FakeTracker:
        def __init__(self):
            self.n, self.last = 0, "match"

        def assign(self, audio):
            self.n += 1
            return f"Speaker {self.n}"

    results = []
    mic, system = FakeCapture(THREE_UTTERANCES), FakeCapture(THREE_UTTERANCES[: 16000 * 3])
    p = Pipeline(FakeASR(), FakeMT(), results.append, speakers=FakeTracker())
    p.add_source(mic, Segmenter("mic", speech_prob=fake_prob))
    p.add_source(system, Segmenter("system", speech_prob=fake_prob))
    p.start()
    p.stop()
    assert sorted(r.speaker for r in results) == ["Me", "Me", "Me", "Speaker 1", "Speaker 2"]
    assert all((r.spk_info == "match") == r.speaker.startswith("Speaker") for r in results)


def test_speaker_error_falls_back_to_system_label(capsys):
    class Broken:
        def assign(self, audio):
            raise RuntimeError("boom")

    results = []
    p = Pipeline(FakeASR(), FakeMT(), results.append, speakers=Broken())
    p.add_source(FakeCapture(THREE_UTTERANCES[: 16000 * 3]), Segmenter("system", speech_prob=fake_prob))
    p.start()
    p.stop()
    assert [r.speaker for r in results] == ["System", "System"]
    assert "Speaker ID error" in capsys.readouterr().err


def test_warns_once_when_system_stream_is_silent(capsys):
    import numpy as np

    from app import pipeline as pl

    silence = np.zeros(int(16000 * (pl.SILENT_WARN_S + 2)), np.float32)
    for source, warns in (("system", 1), ("mic", 0)):
        p = Pipeline(FakeASR(), FakeMT(), lambda r: None)
        p.add_source(FakeCapture(silence), Segmenter(source, speech_prob=fake_prob))
        p.start()
        time.sleep(0.3)
        p.stop()
        assert capsys.readouterr().err.count("no sound on the system-audio device") == warns


def test_debug_reports_segments_with_no_text(capsys):
    class Silent(FakeASR):
        def transcribe(self, audio):
            return ""

    p = Pipeline(Silent(), FakeMT(), lambda r: None, debug=True)
    p.add_source(FakeCapture(THREE_UTTERANCES), Segmenter("system", speech_prob=fake_prob))
    p.start()
    p.stop()
    assert capsys.readouterr().err.count("ASR returned no text") == 3
