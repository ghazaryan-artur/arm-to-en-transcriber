"""Live pipeline on plain threads: capture → VAD → ASR worker → MT worker → emit."""
import queue
import sys
import threading
import time
from typing import Callable

from app.asr import Transcriber
from app.models import SAMPLE_RATE, Result, Segment
from app.speakers import SpeakerTracker
from app.translate import Translator
from app.vad import Segmenter

LABELS = {"mic": "Me", "system": "System"}
SILENT_PEAK = 1e-4      # below this a block is digital silence (BlackHole gets nothing)
SILENT_WARN_S = 15.0    # warn once when the system stream stays silent this long
_STOP = None  # queue sentinel


class Pipeline:
    """Sources feed one segment queue; a single ASR worker and an MT worker drain it in order.

    ASR and MT run in separate threads, so one segment can be translated while the next is
    transcribed. Session time starts at start(); segment times are on the same clock.
    """

    def __init__(self, asr: Transcriber, mt: Translator, emit: Callable[[Result], None], recorder=None,
                 speakers: SpeakerTracker | None = None, debug: bool = False):
        """`recorder`: optional audio.Recorder that saves the streams and transcribed segments.
        `speakers`: labels system-audio segments `Speaker N`; without it they are `System`."""
        self.asr, self.mt, self.emit, self.recorder, self.speakers = asr, mt, emit, recorder, speakers
        self.debug = debug  # report segments dropped because ASR returned no text
        self.segments: queue.Queue[Segment | None] = queue.Queue()
        self._texts: queue.Queue[tuple | None] = queue.Queue()
        self._sources: list[tuple] = []
        self._threads: list[threading.Thread] = []
        self._stopping = threading.Event()
        self._t0 = 0.0

    def add_source(self, capture, segmenter: Segmenter) -> None:
        """`capture`: anything with start() / stop() / read(timeout) -> 16 kHz block or None."""
        self._sources.append((capture, segmenter))

    def clock(self) -> float:
        """Seconds since start()."""
        return time.perf_counter() - self._t0

    def start(self) -> None:
        workers = [threading.Thread(target=self._asr_loop, daemon=True),
                   threading.Thread(target=self._mt_loop, daemon=True)]
        sources = [threading.Thread(target=self._source_loop, args=s, daemon=True) for s in self._sources]
        self._threads = sources + workers
        for t in workers:
            t.start()
        self._t0 = time.perf_counter()
        for capture, _ in self._sources:
            capture.start()
        for t in sources:
            t.start()

    def stop(self) -> None:
        """Stop capturing, then finish everything already captured (the segment in progress too)."""
        for capture, _ in self._sources:
            capture.stop()
        self._stopping.set()
        n = len(self._sources)
        for t in self._threads[:n]:
            t.join()
        self.segments.put(_STOP)
        for t in self._threads[n:]:
            t.join()
        if self.recorder:
            self.recorder.close()

    def _source_loop(self, capture, segmenter: Segmenter) -> None:
        silent_s, warned = 0.0, segmenter.source != "system"
        while not self._stopping.is_set():
            block = capture.read(timeout=0.1)
            if block is not None:
                self._feed(segmenter, block)
                if not warned:
                    silent_s = silent_s + len(block) / SAMPLE_RATE if abs(block).max() < SILENT_PEAK else 0.0
                    if silent_s >= SILENT_WARN_S:
                        warned = True
                        print(f"Warning: no sound on the system-audio device for {SILENT_WARN_S:.0f} s. "
                              "Is the Multi-Output Device (with BlackHole) selected as the sound output?",
                              file=sys.stderr, flush=True)
        while (block := capture.read(timeout=0.05)) is not None:  # stream is stopped: drain it
            self._feed(segmenter, block)
        self._put(segmenter.flush())

    def _feed(self, segmenter: Segmenter, block) -> None:
        if self.recorder:
            self.recorder.write_stream(segmenter.source, block)
        self._put(segmenter.feed(block))

    def _put(self, segments: list[Segment]) -> None:
        for seg in segments:
            self.segments.put(seg)

    def _asr_loop(self) -> None:
        while (seg := self.segments.get()) is not _STOP:
            t = time.perf_counter()
            try:
                hy = self.asr.transcribe(seg.audio)
            except Exception as e:  # one bad segment must not stop the session
                print(f"ASR error at {seg.t_start:.1f} s: {e!r}", file=sys.stderr)
                continue
            if hy:  # speaker only for segments with text: noise must not found a speaker
                asr_s = time.perf_counter() - t
                self._texts.put((seg, hy, asr_s, *self._speaker(seg)))
            elif self.debug:
                print(f"  [dropped {seg.source} segment {seg.t_start:.1f}-{seg.t_end:.1f} s: ASR returned no text]",
                      file=sys.stderr, flush=True)
        self._texts.put(_STOP)

    def _speaker(self, seg: Segment) -> tuple[str, float, str]:
        """(label, seconds spent, debug info). Runs in the ASR thread: segments arrive in order."""
        if seg.source != "system" or self.speakers is None:
            return LABELS.get(seg.source, seg.source), 0.0, ""
        t = time.perf_counter()
        try:
            label = self.speakers.assign(seg.audio)
        except Exception as e:
            print(f"Speaker ID error at {seg.t_start:.1f} s: {e!r}", file=sys.stderr)
            return LABELS["system"], time.perf_counter() - t, ""
        return label, time.perf_counter() - t, str(getattr(self.speakers, "last", "") or "")

    def _mt_loop(self) -> None:
        while (item := self._texts.get()) is not _STOP:
            seg, hy, asr_s, speaker, spk_s, spk_info = item
            t = time.perf_counter()
            try:
                en = self.mt.translate(hy)
            except Exception as e:
                print(f"MT error at {seg.t_start:.1f} s: {e!r}", file=sys.stderr)
                en = ""
            if self.recorder:
                self.recorder.write_segment(seg, hy, en)
            self.emit(Result(seg.t_start, speaker, hy, en,
                             latency=self.clock() - seg.t_end, audio_s=seg.t_end - seg.t_start,
                             asr_s=asr_s, mt_s=time.perf_counter() - t, spk_s=spk_s, spk_info=spk_info))
