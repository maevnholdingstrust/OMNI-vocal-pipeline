# OMNI-vocal-pipeline

**End-to-end AI Vocal Cloning Pipeline**  
Full song → Strong stem separation → RVC cloning with **maevn1** model → Preset mastering → Automated release gates + benchmark tracking

Optimized for **Google Colab Free Tier (T4 GPU)**.

### Features
- Generic `new_render_slot` — just drop any audio files
- Stronger vocal isolation and vocal conditioning before RVC
- RVC inference with your `maevn1.pth`
- Preset tiers with strict constraints: `streaming clean`, `radio loud`, `cinematic wide`
- Per-track release gate metrics: LUFS, true peak, SNR, intelligibility, artifact score, stereo correlation, clipping detection
- Automated QA reports that block export when thresholds fail
- A/B blind-test packet generation with rubric CSV for listener preference scoring
- Run-history metadata logging (input/output hashes, model params, metrics, pass/fail)
- Weekly regression summary generation from run history

### Quick Start

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maevnholdingtrust/OMNI-vocal-pipeline/blob/main/main_pipeline.ipynb)

1. Click the badge above to open in Colab
2. Run Cell 1 (Setup)
3. Run Cell 2 (Quality gates + benchmark suite initialization)
4. Run Cell 3 (Upload your `maevn1.pth` + index)
5. Drop your songs into the `new_render_slot` folder
6. Run Cell 4 (stem separation + conditioning + RVC clone)
7. Run Cell 5 (preset mastering + QA gates + A/B packet + run tracking)
8. Run Cell 6 (download only release-gate-passing outputs)

### Quality Program Outputs
- `/content/quality_program/benchmark_suite/benchmark_manifest.json`  
  Locked benchmark manifest (edit with real tracks and keep versioned).
- `/content/quality_program/qa_reports/*.json`  
  Per-track QA gate results and failure reasons.
- `/content/quality_program/run_history.jsonl`  
  Versioned per-run metadata + metrics for reproducibility.
- `/content/quality_program/ab_tests/<track_id>/`  
  Blind A/B WAVs and fixed scoring rubric CSV (add `listener_summary.json` with `mean_preference_score` to satisfy listener gate).
- `/content/quality_program/weekly_regression_report.json`  
  Summary of pass rate and artifact trends for continuous improvement.

### Repository Structure
- `main_pipeline.ipynb` → Complete single notebook
- `README.md` → This file

### Future Plans
- Separate Python modules
- Gradio web UI
- Full objective intelligibility/artifact models
- Statistical significance testing for benchmark promotion decisions

---

Made with ❤️ for maevn1 voice cloning
