"""Live pipeline on plain threads: capture → VAD → ASR worker → MT worker → emit."""
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from app.asr import Transcriber
from app.echo import StreamTail, is_echo
from app.models import SAMPLE_RATE, Result, Segment
from app.output import notice
from app.speakers import SpeakerTracker
from app.translate import Translator
from app.vad import Segmenter

LABELS = {"mic": "Me", "system": "System"}
SILENT_PEAK = 1e-4      # below this a block is digital silence (BlackHole gets nothing)
SILENT_WARN_S = 15.0    # warn once when the system stream stays silent this long
STALL_WARN_S = 3.0      # warn when a device delivers no audio at all this long (unplugged?)
_STOP = None  # queue sentinel

# A segment that follows the previous one of the same source and speaker within MERGE_GAP_S is
# translated together with it (people pause > 1 s mid-sentence; fragments translate into nonsense).
MERGE_GAP_S = 1.5     # segment start - previous segment end; ≈ a pause of up to 2 s (pre-roll/pad)
MERGE_MAX = 3         # segments per line, at most
MERGE_MAX_S = 30.0    # audio per line, at most (MT time grows with the text)

# Backpressure: when a segment reaches MT this long after it ended (both sides talking, the CPU busy
# with the call app), translate with the faster model until the lag is back under CAUGHT_UP_S.
BEHIND_S = 2.5
CAUGHT_UP_S = 1.0


@dataclass
class _Line:
    """The last printed line of one source, which a continuation can extend."""
    id: int
    speaker: str
    t_start: float
    t_end: float
    hy: list[str] = field(default_factory=list)
    audio_s: float = 0.0
    en_done: str = ""     # English of all but the last segment
    en_last: str = ""     # English of the last segment

    @property
    def en(self) -> str:
        return join_en(self.en_done, self.en_last)


def join_en(a: str, b: str) -> str:
    """Join English pieces; a piece starting with punctuation attaches without a space."""
    if not a or not b:
        return a or b
    return a + ("" if b[0] in ",.;:!?" else " ") + b


class Pipeline:
    """Sources feed one segment queue; a single ASR worker and an MT worker drain it in order.

    ASR and MT run in separate threads, so one segment can be translated while the next is
    transcribed. Session time starts at start(); segment times are on the same clock.
    """

    def __init__(self, asr: Transcriber, mt: Translator, emit: Callable[[Result], None], recorder=None,
                 speakers: SpeakerTracker | None = None, debug: bool = False, mt_fast: Translator | None = None):
        """`recorder`: optional audio.Recorder that saves the streams and transcribed segments.
        `speakers`: labels system-audio segments `Speaker N`; without it they are `System`.
        `mt_fast`: faster translator used while the pipeline is behind (see BEHIND_S)."""
        self.asr, self.mt, self.emit, self.recorder, self.speakers = asr, mt, emit, recorder, speakers
        self.mt_fast = mt_fast
        self._behind = False
        self.debug = debug  # report segments dropped because ASR returned no text
        self.segments: queue.Queue[Segment | None] = queue.Queue()
        self._texts: queue.Queue[tuple | None] = queue.Queue()
        self._sources: list[tuple] = []
        self._threads: list[threading.Thread] = []
        self._stopping = threading.Event()
        self._t0 = 0.0
        self._system_tail: StreamTail | None = None  # recent system audio, to spot it leaking into the mic
        self._lines: dict[str, _Line] = {}  # source -> its last line (MT thread only)
        self._n_lines = 0

    def add_source(self, capture, segmenter: Segmenter) -> None:
        """`capture`: anything with start() / stop() / read(timeout) -> 16 kHz block or None."""
        self._sources.append((capture, segmenter))
        if segmenter.source == "system":
            self._system_tail = StreamTail()

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
        last_block, stalled = time.perf_counter(), False
        while not self._stopping.is_set():
            block = capture.read(timeout=0.1)
            now = time.perf_counter()
            if block is None:
                if not stalled and now - last_block >= STALL_WARN_S:
                    stalled = True
                    notice(f"Warning: no audio from the {segmenter.source} device for {STALL_WARN_S:.0f} s. "
                           "Was it unplugged or switched off? Restart the app after reconnecting it.")
                continue
            if stalled:
                notice(f"Audio from the {segmenter.source} device is back.")
            last_block, stalled = now, False
            self._feed(segmenter, block)
            if not warned and len(block):
                silent_s = silent_s + len(block) / SAMPLE_RATE if abs(block).max() < SILENT_PEAK else 0.0
                if silent_s >= SILENT_WARN_S:
                    warned = True
                    notice(f"Warning: no sound on the system-audio device for {SILENT_WARN_S:.0f} s. "
                           "Is the Multi-Output Device (with BlackHole) selected as the sound output?")
        while (block := capture.read(timeout=0.05)) is not None:  # stream is stopped: drain it
            self._feed(segmenter, block)
        self._put(segmenter.flush())

    def _feed(self, segmenter: Segmenter, block) -> None:
        if self.recorder:
            self.recorder.write_stream(segmenter.source, block)
        if segmenter.source == "system" and self._system_tail is not None:
            self._system_tail.append(block)
        self._put(segmenter.feed(block))

    def _put(self, segments: list[Segment]) -> None:
        for seg in segments:
            self.segments.put(seg)

    def _asr_loop(self) -> None:
        while (seg := self.segments.get()) is not _STOP:
            if seg.source == "mic" and self._system_tail is not None:
                echo, score = is_echo(seg.audio, seg.t_start, seg.t_end, self._system_tail)
                if echo:  # the headset leaks the call into its mic: the line is already printed as a speaker
                    if self.debug:
                        notice(f"  [dropped mic segment {seg.t_start:.1f}-{seg.t_end:.1f} s: echo of the call "
                               f"audio, xcorr {score:.2f}]")
                    continue
            t = time.perf_counter()
            try:
                hy = self.asr.transcribe(seg.audio)
            except Exception as e:  # one bad segment must not stop the session
                notice(f"ASR error at {seg.t_start:.1f} s: {e!r}")
                continue
            if hy:  # speaker only for segments with text: noise must not found a speaker
                asr_s = time.perf_counter() - t
                self._texts.put((seg, hy, asr_s, *self._speaker(seg)))
            elif self.debug:
                notice(f"  [dropped {seg.source} segment {seg.t_start:.1f}-{seg.t_end:.1f} s: ASR returned no text]")
        self._texts.put(_STOP)

    def _speaker(self, seg: Segment) -> tuple[str, float, str]:
        """(label, seconds spent, debug info). Runs in the ASR thread: segments arrive in order."""
        if seg.source != "system" or self.speakers is None:
            return LABELS.get(seg.source, seg.source), 0.0, ""
        t = time.perf_counter()
        try:
            label = self.speakers.assign(seg.audio)
        except Exception as e:
            notice(f"Speaker ID error at {seg.t_start:.1f} s: {e!r}")
            return LABELS["system"], time.perf_counter() - t, ""
        return label, time.perf_counter() - t, str(getattr(self.speakers, "last", "") or "")

    def _line(self, seg: Segment, hy: str, speaker: str) -> _Line:
        """The line this segment is printed in: the source's last line extended, or a new one."""
        audio_s = seg.t_end - seg.t_start
        last = self._lines.get(seg.source)
        if (last and last.speaker == speaker and seg.t_start - last.t_end <= MERGE_GAP_S
                and len(last.hy) < MERGE_MAX and last.audio_s + audio_s <= MERGE_MAX_S):
            line = _Line(last.id, speaker, last.t_start, seg.t_end, last.hy + [hy], last.audio_s + audio_s,
                         last.en_done, last.en_last)
        else:
            self._n_lines += 1
            line = _Line(self._n_lines, speaker, seg.t_start, seg.t_end, [hy], audio_s)
        self._lines[seg.source] = line
        return line

    def _pick_mt(self, lag: float) -> Translator:
        """The main translator, or the fast one while behind (with hysteresis); warns on each switch."""
        if not self._behind and lag >= BEHIND_S:
            self._behind = True
            notice(f"Warning: {lag:.1f} s behind"
                   + (", translating with the faster model until caught up" if self.mt_fast else ""))
        elif self._behind and lag <= CAUGHT_UP_S:
            self._behind = False
            if self.mt_fast:
                notice("Caught up, back to the main translation model")
        return self.mt_fast if self._behind and self.mt_fast else self.mt

    def _translate(self, line: _Line, mt: Translator) -> None:
        """Translate the line's last segment. A continuation is translated with the segment before
        it as context (only that one: MT time stays bounded however long the line gets) and keeps
        the English already shown."""
        if len(line.hy) == 1:
            line.en_last = mt.translate(line.hy[0])
            return
        if hasattr(mt, "translate_continuation"):
            head, added = mt.translate_continuation(line.hy[-2], line.en_last, line.hy[-1])
        else:
            head, added = line.en_last, mt.translate(line.hy[-1])
        line.en_done, line.en_last = join_en(line.en_done, head), added

    def _mt_loop(self) -> None:
        while (item := self._texts.get()) is not _STOP:
            seg, hy, asr_s, speaker, spk_s, spk_info = item
            line = self._line(seg, hy, speaker)
            text = " ".join(line.hy)
            mt = self._pick_mt(self.clock() - seg.t_end)
            t = time.perf_counter()
            try:
                self._translate(line, mt)
            except Exception as e:
                notice(f"MT error at {seg.t_start:.1f} s: {e!r}")
            en = line.en
            mt_s = time.perf_counter() - t
            try:
                if self.recorder:
                    self.recorder.write_segment(seg, hy, en)
                self.emit(Result(line.t_start, speaker, text, en,
                                 latency=self.clock() - seg.t_end, audio_s=line.audio_s,
                                 asr_s=asr_s, mt_s=mt_s, spk_s=spk_s, spk_info=spk_info,
                                 line=line.id, merged=len(line.hy), mt_fast=mt is not self.mt))
            except Exception as e:  # a failing recorder / printer must not stop the session
                notice(f"Output error at {seg.t_start:.1f} s: {e!r}")
