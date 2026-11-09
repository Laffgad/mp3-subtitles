"""Dependency-free timing and formatting logic, independently testable."""
from dataclasses import dataclass
from html import escape
from collections import defaultdict
import textwrap


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class Word:
    start: float
    end: float
    text: str
    speaker: str = "UNKNOWN"


@dataclass
class Cue:
    start: float
    end: float
    text: str
    speaker: str


def assign_speakers(words, turns):
    """Choose greatest total overlap per speaker; unmatched words stay UNKNOWN.

    Exclusive turns avoid ambiguity from overlapping talk. A word crossing a
    speaker boundary is assigned as a whole, never duplicated.
    """
    turns = sorted(turns, key=lambda t: t.start)
    cursor = 0
    for word in sorted(words, key=lambda w: w.start):
        while cursor < len(turns) and turns[cursor].end <= word.start:
            cursor += 1
        scores = defaultdict(float)
        for index in range(cursor, len(turns)):
            turn = turns[index]
            if turn.start >= word.end:
                break
            overlap = max(0.0, min(word.end, turn.end) - max(word.start, turn.start))
            if overlap:
                scores[turn.speaker] += overlap
        word.speaker = max(scores, key=scores.get) if scores else "UNKNOWN"
    return words


def make_cues(words, max_duration=6.0, line_width=42, max_gap=0.8):
    """Split on speaker changes, pauses, sentence ends, and reading size.

    Preserve Whisper's leading spaces (important for punctuation and CJK).
    Labels have their own line; body text aims for at most two lines. A single
    exceptionally long word is kept intact rather than inventing word timing.
    """
    cues = []
    group = []

    def flush():
        if group:
            body = "".join(w.text for w in group).strip()
            cues.append(Cue(group[0].start, max(w.end for w in group),
                            body, group[0].speaker))
            group.clear()

    for word in sorted(words, key=lambda w: (w.start, w.end)):
        if not word.text.strip() or word.end <= word.start:
            continue
        proposed = "".join(w.text for w in group) + word.text
        lines = textwrap.wrap(proposed.strip(), width=line_width)
        if group and (word.speaker != group[0].speaker
                      or word.start - group[-1].end > max_gap
                      or word.end - group[0].start > max_duration
                      or len(lines) > 2):
            flush()
        group.append(word)
        if word.text.rstrip().endswith((".", "?", "!", "。", "？", "！")):
            flush()
    flush()
    # Resolve occasional overlapping Whisper word timestamps for subtitle players.
    for current, following in zip(cues, cues[1:]):
        current.end = min(current.end, following.start)
    return [c for c in cues if round(c.end * 1000) > round(c.start * 1000)]


def timestamp(seconds, vtt=False):
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    separator = "." if vtt else ","
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def render_subtitles(cues, kind, line_width=42):
    if kind not in {"srt", "vtt"}:
        raise ValueError("kind must be srt or vtt")
    blocks = []
    for index, cue in enumerate(cues, 1):
        body = "\n".join(textwrap.wrap(cue.text, width=line_width))
        # Escape model text so accidental markup cannot break WebVTT/SRT.
        body = escape(body, quote=False)
        label = escape(cue.speaker, quote=False)
        timing = f"{timestamp(cue.start, kind == 'vtt')} --> {timestamp(cue.end, kind == 'vtt')}"
        blocks.append(f"{index}\n{timing}\n[{label}]\n{body}")
    prefix = "WEBVTT\n\n" if kind == "vtt" else ""
    return prefix + "\n\n".join(blocks) + ("\n" if blocks else "")


def non_speech_gaps(intervals, duration, min_duration=2.0, padding=0.2):
    """Complement of ALL speech evidence, preserving padding beside speech.

    Includes leading/trailing gaps and correctly merges overlapping intervals.
    The minimum duration is applied after padding.
    """
    merged = []
    for start, end in sorted(intervals):
        start, end = max(0.0, start), min(duration, end)
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gaps = []
    cursor = 0.0
    for start, end in merged:
        right = max(0.0, start - padding)
        if right - cursor >= min_duration:
            gaps.append((cursor, right))
        cursor = min(duration, end + padding)
    if duration - cursor >= min_duration:
        gaps.append((cursor, duration))
    return gaps
