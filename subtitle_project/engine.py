"""ML integration. Decode once to mono 16 kHz; preserve the original timeline."""
import gc
import logging
import math
from .core import Word, Turn, assign_speakers, non_speech_gaps

LOG = logging.getLogger(__name__)
SAMPLE_RATE = 16000


def process(args, token):
    import numpy as np
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import get_speech_timestamps, VadOptions

    audio = decode_audio(str(args.input), sampling_rate=SAMPLE_RATE)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio is empty or contains invalid samples.")
    duration = len(audio) / SAMPLE_RATE
    LOG.info("Decoded %.1f seconds; detecting speech", duration)
    speech = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=500))
    evidence = [(s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE) for s in speech]
    words, turns, raw_turns = [], [], []
    language = args.language

    # Avoid language detection failures and model hallucinations on all-silence input.
    if speech:
        LOG.info("Transcribing with %s on %s", args.model, args.device)
        compute = args.compute_type or ("int8" if args.device == "cpu" else "float16")
        model = WhisperModel(args.model, device=args.device, compute_type=compute)
        segments, info = model.transcribe(
            audio, language=args.language, beam_size=5, word_timestamps=True,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False, hallucination_silence_threshold=2.0,
        )
        language = info.language
        # Transcription is lazy: consuming the generator performs inference.
        for segment in segments:
            for word in segment.words or []:
                start, end = max(0.0, word.start), min(duration, word.end)
                if end > start:
                    words.append(Word(start, end, word.word))
            if segment.text.strip() and not segment.words:
                LOG.warning("No word alignment for segment at %.2f; using segment timing", segment.start)
                words.append(Word(max(0.0, segment.start), min(duration, segment.end), segment.text))
        del model, segments
        gc.collect()  # Release Whisper before loading the diarization pipeline.

        if not args.no_diarization:
            import torch
            from pyannote.audio import Pipeline
            LOG.info("Running Community-1 speaker diarization")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1", token=token)
            if pipeline is None:
                raise RuntimeError("Could not load Community-1; check model access and HF_TOKEN.")
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested, but PyTorch cannot access an NVIDIA GPU.")
            pipeline.to(torch.device(args.device))
            options = {"num_speakers": args.num_speakers} if args.num_speakers else {}
            # In-memory input bypasses TorchCodec audio decoding; PyAV decoded above.
            result = pipeline({"waveform": torch.from_numpy(audio.copy()).unsqueeze(0),
                               "sample_rate": SAMPLE_RATE}, **options)
            exclusive = list(result.exclusive_speaker_diarization.itertracks(yield_label=True))
            regular = list(result.speaker_diarization.itertracks(yield_label=True))
            mapping = {}
            for turn, _, speaker in sorted(regular, key=lambda row: row[0].start):
                mapping.setdefault(speaker, f"SPEAKER_{len(mapping) + 1:02d}")
                raw_turns.append(Turn(turn.start, turn.end, mapping[speaker]))
            for turn, _, speaker in exclusive:
                mapping.setdefault(speaker, f"SPEAKER_{len(mapping) + 1:02d}")
                turns.append(Turn(turn.start, turn.end, mapping[speaker]))
            assign_speakers(words, turns)
            del pipeline, result
            gc.collect()
        else:
            for word in words:
                word.speaker = "UNLABELED"
    else:
        LOG.warning("No speech detected; writing empty subtitles.")

    cuts = []
    if args.smart_cut:
        # Protect speech from VAD, diarization, AND transcription, not only ASR gaps.
        evidence += [(t.start, t.end) for t in raw_turns]
        evidence += [(w.start, w.end) for w in words]
        for start, end in non_speech_gaps(evidence, duration, args.min_cut, args.cut_padding):
            samples = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
            rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
            dbfs = 20 * math.log10(max(rms, 1e-12))
            reason = "silence" if dbfs <= args.silence_db else "possible_non_speech_audio"
            cuts.append({"start": round(start, 3), "end": round(end, 3),
                         "duration": round(end - start, 3), "reason": reason,
                         "rms_dbfs": round(dbfs, 2), "review_required": True})
            LOG.info("Suggested cut %.3f–%.3f: %s", start, end, reason)
    return words, raw_turns, cuts, {"duration": duration, "language": language}
