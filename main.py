"""PyCharm entry point. Run `python main.py --help` for all options."""
import argparse
from dataclasses import asdict
import json
import logging
import math
import os
from pathlib import Path
from subtitle_project.core import make_cues, render_subtitles


def main():
    parser = argparse.ArgumentParser(description="MP3 to speaker-labeled SRT/VTT subtitles")
    parser.add_argument("input", type=Path, help="Path to an MP3 (or another PyAV-supported audio file)")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--format", choices=["srt", "vtt", "both"], default="both")
    parser.add_argument("--model", default="large-v3", help="Whisper size or local converted model directory")
    parser.add_argument("--language", help="Language code, e.g. en, ru; default: automatic")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--compute-type", choices=["int8", "float32", "float16", "int8_float16"])
    parser.add_argument("--num-speakers", type=int, help="Known speaker count; default: estimate")
    parser.add_argument("--no-diarization", action="store_true", help="Skip speaker detection and token requirement")
    parser.add_argument("--smart-cut", action="store_true", help="Log candidate cuts; never edit the input")
    parser.add_argument("--min-cut", type=float, default=2.0, help="Minimum suggested cut length in seconds")
    parser.add_argument("--cut-padding", type=float, default=0.2, help="Seconds preserved beside speech")
    parser.add_argument("--silence-db", type=float, default=-40.0, help="Silence RMS threshold in dBFS")
    parser.add_argument("--max-cue-duration", type=float, default=6.0)
    parser.add_argument("--line-width", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing result files")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    for name in ("min_cut", "cut_padding", "silence_db", "max_cue_duration"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"{name} must be finite")
    if args.min_cut <= 0 or args.max_cue_duration <= 0 or args.cut_padding < 0 or args.line_width < 10:
        parser.error("Durations must be positive, padding nonnegative, and line width at least 10")
    if args.num_speakers is not None and args.num_speakers < 1:
        parser.error("--num-speakers must be positive")
    if args.device == "cpu" and args.compute_type in {"float16", "int8_float16"}:
        parser.error("Use int8 or float32 on CPU")
    formats = ["srt", "vtt"] if args.format == "both" else [args.format]
    suffixes = formats + ["transcript.json"] + (["cuts.json"] if args.smart_cut else [])
    paths = {suffix: args.output_dir / f"{args.input.stem}.{suffix}" for suffix in suffixes}
    for path in paths.values():
        if path.resolve() == args.input.resolve():
            parser.error("An output path would overwrite the input")
        if path.exists() and not args.overwrite:
            parser.error(f"Output exists: {path}; choose another folder or use --overwrite")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).with_name(".env"))
        os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
        token = os.getenv("HF_TOKEN")
        if not args.no_diarization and (not token or token == "hf_replace_me"):
            parser.error("Set HF_TOKEN in .env after accepting Community-1 model terms, or use --no-diarization")
        from subtitle_project.engine import process
        words, turns, cuts, metadata = process(args, token)
        cues = make_cues(words, args.max_cue_duration, args.line_width)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for kind in formats:
            paths[kind].write_text(render_subtitles(cues, kind, args.line_width), encoding="utf-8")
        report = {**metadata, "input": str(args.input.resolve()), "model": args.model,
                  "diarization": not args.no_diarization, "words": [asdict(w) for w in words],
                  "speaker_turns": [asdict(t) for t in turns], "cues": [asdict(c) for c in cues]}
        paths["transcript.json"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.smart_cut:
            paths["cuts.json"].write_text(json.dumps({"timeline": "original audio, seconds",
                "notice": "Suggestions only; review before editing. Non-speech audio may be intentional.",
                "candidates": cuts}, indent=2), encoding="utf-8")
        logging.info("Wrote %d cues to %s", len(cues), args.output_dir.resolve())
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logging.error("Processing failed: %s", exc)
        logging.error("See README troubleshooting for dependency, model access, and GPU setup.")
        return 1
    return 0


if __name__ == "__main__":
    import subprocess
    import sys

    # PyCharm's green Run button supplies no arguments. Use the verified local
    # environment even when the IDE still has its older interpreter selected.
    if len(sys.argv) == 1:
        project = Path(__file__).resolve().parent
        python = project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.exists():
            python = Path(sys.executable)
        raise SystemExit(subprocess.run(
            [str(python), str(project / "run_audio_folder.py")], cwd=project
        ).returncode)
    raise SystemExit(main())
