"""Press Run in PyCharm: process each MP3 in audio/ without editing arguments."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    files = sorted(p for p in (ROOT / 'audio').iterdir()
                   if p.is_file() and p.suffix.lower() == '.mp3')
    if not files:
        print(f'Ready. Put your MP3 files in:\n{ROOT / "audio"}\nThen press Run again.')
        return 0
    token = os.getenv('HF_TOKEN', '').strip()
    diarize = bool(token and token != 'hf_replace_me')
    if not diarize:
        print('HF_TOKEN is not configured: producing subtitles without speaker detection.\n'
              'Add your authorized Hugging Face token to .env to enable speaker labels.')
    # Multilingual small balances download size and CPU processing time.
    model = os.getenv('WHISPER_MODEL', 'small')
    failed = 0
    for audio in files:
        destination = ROOT / 'results' / audio.name
        command = [sys.executable, str(ROOT / 'main.py'), str(audio),
                   '--model', model, '--device', 'cpu', '--format', 'both',
                   '--smart-cut', '--output-dir', str(destination), '--overwrite']
        if not diarize:
            command.append('--no-diarization')
        print(f'\nProcessing {audio.name}', flush=True)
        failed += subprocess.run(command, cwd=ROOT).returncode != 0
    print(f'\nFinished. Results: {ROOT / "results"}\nFailed files: {failed}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
