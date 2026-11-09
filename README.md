# MP3 Subtitles

Turn MP3 recordings into **SRT and WebVTT subtitles**, with optional speaker labels
and suggested timestamps for removing long pauses or non-speech audio.
Built with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (OpenAI Whisper)
and [pyannote.audio](https://github.com/pyannote/pyannote-audio).

Inference runs on your computer. No paid transcription API is required. The first
run downloads models; audio, tokens, models, and generated transcripts are not
included in this repository.

## What you need

- **64-bit Python 3.11** from [python.org](https://www.python.org/downloads/).
- PyCharm, VS Code, or a terminal.
- Internet access for installation and model downloads, several GB of free disk
  space, and enough RAM for the models and decoded audio. CPU processing can be slow.
- A free Hugging Face account **only if you want speaker detection**.

The starter configuration uses multilingual Whisper `small` on CPU. NVIDIA GPU
use is optional. Apple Silicon uses CPU in this project; there is no MPS option.
The existing installation was verified on Apple Silicon macOS with Python 3.11.
Windows and Linux instructions are provided but have not been tested here. Older
Intel Macs may lack wheels for this pinned PyTorch stack.

## Option A — PyCharm (beginner-friendly)

1. On this repository's GitHub page, choose **Code → Download ZIP**, then extract it.
   Alternatively clone the repository with Git.
2. In PyCharm choose **File → Open** and select the extracted project folder.
3. In **Settings → Project → Python Interpreter**, add a **local virtual environment**
   using Python 3.11. Set its location to `.venv` inside the project. Select it as
   the project interpreter. Menu wording may vary between PyCharm versions.
4. Open PyCharm's Terminal in the project folder and install the dependencies:

   macOS / Linux:
   ```bash
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install -r requirements.txt
   .venv/bin/python -m pip check
   ```

   Windows PowerShell:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m pip check
   ```

5. Copy `.env.example` to a new file named **`.env`** in the project root.
   Leave `HF_TOKEN` empty for transcription only, or follow the token steps below.
6. Put your MP3 files in the **`audio`** folder.
7. Open **`main.py`** and press the green **Run** button. No arguments are needed.
   It processes all MP3s directly inside `audio` (not subfolders).
8. Find the subtitles in **`results/<audio filename>/`**.

The no-arguments launcher prefers `.venv` automatically. Set PyCharm's interpreter
as described above as well, so editing, debugging, and package checks use it too.

## Option B — Terminal or VS Code

Download/extract the repository, open a terminal in that folder, then run:

macOS / Linux:
```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv/bin/python main.py
```

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe main.py
```

Copy the environment template only once; repeating the copy would replace your
local settings. Add MP3s to `audio`, configure a token if desired, then run the last
command again. In VS Code select `.venv` with **Python: Select Interpreter**.
No environment activation is needed with the explicit commands above.

## Get your own Hugging Face token (speaker labels)

1. Create/sign into your own account at [Hugging Face](https://huggingface.co/join).
   Complete email verification if requested.
2. Open [the Community-1 model page](https://huggingface.co/pyannote/speaker-diarization-community-1)
   while logged in. Read and accept its access conditions. Token creation alone
   does not grant access to this gated model.
3. Open [Access Tokens](https://huggingface.co/settings/tokens), choose **Create new
   token**, give it a name such as `mp3-subtitles`, and choose **Read** access.
   If using a fine-grained token instead, allow read access to the gated model.
   Write/admin permissions are unnecessary.
4. Copy the token into your local `.env`, after `HF_TOKEN=`. Do not paste it into
   Python source code, GitHub, issues, screenshots, or chat. Example structure:
   ```dotenv
   HF_TOKEN=your_own_token_here
   WHISPER_MODEL=small
   ```
   Replace the example value with the actual token; preserve the other settings
   already copied from `.env.example`.
5. Run `main.py` again. The first speaker-detection run downloads Community-1.

Each student uses their own account and token. `.env` is excluded by `.gitignore`.
Without a token the folder runner explicitly reports that speaker detection is
skipped and uses `UNLABELED` captions. Existing environment variables take
precedence over `.env`. See the [official token guide](https://huggingface.co/docs/hub/security-tokens).

## Results and behavior

For `audio/interview.mp3`, the folder runner writes:
```text
results/interview.mp3/
  interview.srt
  interview.vtt
  interview.transcript.json
  interview.cuts.json
```

- SRT/VTT contain the text and labels such as `SPEAKER_01` and `SPEAKER_02`.
- Transcript JSON contains word timings, cues, and diarization turns.
- Cuts JSON contains candidate start/end timestamps and reasons. The original
  audio is **never edited**. Review suggestions before making cuts.
- Re-running the folder runner **replaces that file's generated outputs**. Save
  any manually edited subtitles elsewhere first.
- All timestamps refer to the original audio. Retiming is required after editing it.

Speaker numbers are anonymous clusters within one recording, not real identities.
Short turns, overlapping speakers, background music, and quiet speech can produce
errors. Cut suggestions use detected speech gaps plus amplitude, not a dedicated
noise classifier; intentional music or laughter may be flagged. Whisper can also
misrecognize or hallucinate text. Review important subtitles.

## Optional: one file and advanced settings

Examples below use macOS/Linux paths; on Windows substitute
`.\.venv\Scripts\python.exe` for `.venv/bin/python`.

```bash
.venv/bin/python main.py audio/interview.mp3 --model small --num-speakers 2 --smart-cut
.venv/bin/python main.py audio/interview.mp3 --model small --language en --no-diarization
.venv/bin/python main.py --help
```

Omit `--num-speakers` to estimate the count and `--language` to detect language.
The single-file command writes directly to `results` by default and refuses to
replace existing outputs unless `--overwrite` is supplied. The folder runner
always creates both subtitle formats and cut suggestions.

For a larger model, set `WHISPER_MODEL=large-v3` in `.env` for folder runs, or use
`--model large-v3` for a single file. Expect larger downloads and slower CPU runs.

For NVIDIA CUDA, install matching Torch/torchaudio 2.8.0 builds using the
[PyTorch version instructions](https://pytorch.org/get-started/previous-versions/)
and follow the [faster-whisper GPU setup](https://github.com/SYSTRAN/faster-whisper#gpu).
Then use `--device cuda --compute-type float16` with the single-file command.
Keep Torch 2.8.0, torchaudio 2.8.0, and TorchCodec 0.7.0 together.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| Missing package | Install requirements with the `.venv` interpreter and select it in your editor. |
| No compatible wheel | Use 64-bit Python 3.11 on a supported platform. |
| 401 / 403 / gated model | Accept Community-1 conditions using the account that owns your read token. Check its permissions. |
| No MP3 files found | Put files directly inside `audio`; run again. |
| Output already exists | For single-file commands, choose another `--output-dir` or use `--overwrite`. |
| CPU is slow / insufficient memory | Use `small` or `base`, and start with a short recording. |
| Wrong language | Supply `--language en`, `--language ru`, etc. in single-file mode. |
| TorchCodec/FFmpeg warning | This project decodes via PyAV and passes in-memory audio to pyannote. A decoder warning can be nonfatal; a later traceback still needs investigation. |

PyAV bundles audio decoding libraries, so this implementation does not require a
separate FFmpeg command. It holds decoded audio in memory; long recordings need
more RAM. First runs may spend time downloading models even before progress appears.

## Development and tests

```bash
python -m unittest discover -s tests -v
```

These 11 tests cover timing, speaker alignment, subtitle formatting, and gap logic
without downloading models. An end-to-end test using generated speech previously
passed on the author's Mac; it is not a measurement of transcription accuracy.
The requirements pin major libraries but are not a cross-platform transitive lock.

```text
main.py                 CLI and no-arguments PyCharm launcher
run_audio_folder.py     Batch MP3 runner
subtitle_project/       Model integration, timing, and subtitle formatting
tests/                 Dependency-free logic tests
.env.example            Safe settings template (no credentials)
```

## License and acknowledgments

Project code is available under the [MIT License](LICENSE). Third-party packages
and model weights retain their own licenses and conditions. Model weights are
not distributed in this repository. Thanks to [OpenAI Whisper](https://github.com/openai/whisper),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[pyannote.audio](https://github.com/pyannote/pyannote-audio), and
[Silero VAD](https://github.com/snakers4/silero-vad).
Community-1's model card specifies CC-BY-4.0; consult it for attribution and terms.
