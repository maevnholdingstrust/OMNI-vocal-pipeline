# OMNI-vocal-pipeline

**Local end-to-end AI Vocal Production Pipeline**

```
Lyrics (Suno-style blocks)
  → Bark singing synthesis
  → RVC voice cloning with your .PTH model
  → Audio mastering
  → Final WAV
```

Runs **completely on your local machine** — no Colab, no cloud, no `.index` file needed.  
Just your `.pth` voice model and a text file of lyrics.

---

## Features

- **Suno-style lyric blocks** — paste lyrics with `[Verse]`, `[Chorus]`, `[Bridge]`, etc.
- **Bark TTS** — generates singing-like audio from text (♪ mode)
- **RVC voice cloning** — applies your `.pth` model without any `.index` file
- **Mastering presets** — streaming, vinyl, club, bright_pop, raw
- **CLI + Python API** — use from the command line or import as a library
- **GPU-optional** — runs on CPU (slow) or CUDA/Apple MPS (fast)

---

## Quick Start (Local)

### 1. Install dependencies

```bash
# Python 3.10+ recommended
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> First run downloads Bark model weights (~5 GB) and RVC/HuBERT assets automatically.

### 2. Configure

Edit `config.yaml` and set your `.pth` path:

```yaml
model:
  pth_path: "/absolute/path/to/maevn1.pth"   # ← your .PTH file here
```

No `.index` file is required — RVC runs with `index_rate: 0`.

### 3. Write your lyrics

Create a `song.txt` file in Suno block format:

```
[Verse 1]
I walk alone under neon skies
The city never sleeps, neither do I

[Chorus]
Echoes of the night, calling out my name
We're burning bright, we'll never be the same

[Bridge]
In the silence I can hear you calling
```

### 4. Run the pipeline

```bash
python run_pipeline.py song.txt
```

Output lands in `output/output_mastered.wav`.

#### Other options

```bash
# Custom output name
python run_pipeline.py song.txt --name my_song

# Override mastering preset
python run_pipeline.py song.txt --preset vinyl

# Override the .pth path without editing config.yaml
python run_pipeline.py song.txt --pth /path/to/model.pth

# Skip mastering (raw RVC vocal)
python run_pipeline.py song.txt --no-master

# Force CPU
python run_pipeline.py song.txt --device cpu
```

---

## Supported Lyric Block Tags

| Tag | Description |
|-----|-------------|
| `[Intro]` | Opening section |
| `[Verse]` / `[Verse 1]` | Verse |
| `[Pre-Chorus]` | Build before chorus |
| `[Chorus]` / `[Hook]` | Chorus / Hook |
| `[Bridge]` | Bridge |
| `[Refrain]` / `[Tag]` | Repeated hook / outro tag |
| `[Outro]` | Closing section |
| `[Break]` / `[Instrumental]` | Skipped (no synthesis) |

---

## Mastering Presets

| Preset | LUFS | Character |
|--------|------|-----------|
| `streaming` | -14 | Clean, air on top (default) |
| `vinyl` | -18 | Warm, low-mid body |
| `club` | -9 | Punchy, loud |
| `bright_pop` | -12 | Presence, shimmer |
| `raw` | -14 | Loudness only, no EQ |

---

## Python API

```python
from omni_pipeline import run

output_path = run(
    lyrics=open("song.txt").read(),
    config_path="config.yaml",
    output_name="my_song",
)
print(f"Done: {output_path}")
```

---

## Repository Structure

```
OMNI-vocal-pipeline/
├── run_pipeline.py          ← CLI entry point
├── config.yaml              ← user configuration
├── requirements.txt         ← all dependencies
├── omni_pipeline/
│   ├── lyrics_parser.py     ← Suno block parser
│   ├── tts_engine.py        ← Bark singing TTS
│   ├── rvc_engine.py        ← RVC inference (.PTH only)
│   ├── mastering.py         ← audio post-production
│   └── pipeline.py          ← end-to-end orchestration
└── main_pipeline.ipynb      ← legacy Colab notebook (reference)
```

---

## Requirements

- Python 3.10+
- ~8 GB disk space (Bark model weights + RVC assets on first run)
- GPU strongly recommended (NVIDIA CUDA or Apple MPS)
- Your `.pth` voice model — **no `.index` file required**

---

Made with ❤️ for maevn1 voice cloning
