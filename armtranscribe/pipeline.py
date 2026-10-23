"""Live pipeline on plain threads: capture → VAD → ASR worker → MT worker → emit."""
import queue
import sys
import threading
import time
from typing import Callable

from armtranscribe.asr import Transcriber
from armtranscribe.models import Result, Segment
from armtranscribe.translate import Translator
from armtranscribe.vad import Segmenter

LABELS = {"mic": "Me", "system": "System"}
_STOP = None  # queue sentinel


class Pipeline:
    """Sources feed one segment queue; a single ASR worker and an MT worker drain it in order.

    ASR (GPU) and MT (CPU) run in separate threads, so one segment can be translated while the
    next is transcribed. Session time starts at start(); segment times are on the same clock.
    """

    def __init__(self, asr: Transcriber, mt: Translator, emit: Callable[[Result], None]):
        self.asr, self.mt, self.emit = asr, mt, emit
        self.segments: queue.Queue[Segment | None] = queue.Queue()
        self._texts: queue.Queue[tuple[Segment, str, float] | None] = queue.Queue()
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

    def _source_loop(self, capture, segmenter: Segmenter) -> None:
        while not self._stopping.is_set():
            block = capture.read(timeout=0.1)
            if block is not None:
                self._put(segmenter.feed(block))
        while (block := capture.read(timeout=0.05)) is not None:  # stream is stopped: drain it
            self._put(segmenter.feed(block))
        self._put(segmenter.flush())

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
            if hy:
                self._texts.put((seg, hy, time.perf_counter() - t))
        self._texts.put(_STOP)

    def _mt_loop(self) -> None:
        while (item := self._texts.get()) is not _STOP:
            seg, hy, asr_s = item
            t = time.perf_counter()
            try:
                en = self.mt.translate(hy)
            except Exception as e:
                print(f"MT error at {seg.t_start:.1f} s: {e!r}", file=sys.stderr)
                en = ""
            self.emit(Result(seg.t_start, LABELS.get(seg.source, seg.source), hy, en,
                             latency=self.clock() - seg.t_end, audio_s=seg.t_end - seg.t_start,
                             asr_s=asr_s, mt_s=time.perf_counter() - t))
